"""The serving check on Modal (paid, about $1): greedy decoding of serving.PROMPTS through the HF path and through vLLM,
for each of serving.CONFIGS (Qwen 2.5 7B Instruct; Ministral 3 8B; Ministral 3 8B with the Act I adapter
gameable_s0/adapter_r3), compared token by token (serving.compare_row). Claude (Opus 5.5) wrote this file, 3 Nov 2026.

  tools/modal.sh run src/anatta/backends/serving_app.py::main            # trial (Qwen, 2 prompts), then the check
  tools/modal.sh run src/anatta/backends/serving_app.py::main --trial-only

Writes <results_root>/calib/serving/: plan.json (anatta.cards.write_plan, before anything runs), trial.json,
serving_check.json (the card: numbers only, through anatta.cards.write_card) and data/serving_check_tokens.json (the
token ids and texts of both paths; harmless prompts, kept out of git with the other results). Spend goes through
anatta.spend's Meter under tool "serving_check", with a cap per session; the day's cap is lab.toml's.

The HF side runs in anatta.modal_app's image (torch 2.14.1, transformers 5.21.0, peft 0.21.2), the vLLM side in
anatta.backends.vllm's (vllm 0.31.0, K6b's pin). Both mount anatta-hf-cache at /cache and anatta-runs at /runs
read-only. Containers scale down 10 s after their last call, so the tails are short; they are charged as they run.
No `from __future__ import annotations` here: Modal reads the class parameters' annotations as types.
"""

import json
import time
from pathlib import Path

import modal

from anatta.backends import serving as SV
from anatta.backends.vllm import image as vllm_image
from anatta.modal_app import image as hf_image

PKG = Path(__file__).resolve().parents[1]  # src/anatta
APP = "anatta-serving"
GPU, CPU, MEM_MB, SCALEDOWN = "L40S", 4.0, 32768, 10
TOOL = "serving_check"
TRIAL_S, CALL_S = 420, 900  # the charge reserved before each call (seconds); trued up after it

cache = modal.Volume.from_name("anatta-hf-cache")
runs = modal.Volume.from_name("anatta-runs")
VOLS = {"/cache": cache.read_only(), "/runs": runs.read_only()}
app = modal.App(APP)


def _messages() -> list[list[dict]]:
    return [[{"role": "user", "content": p}] for p in SV.PROMPTS]


@app.cls(image=hf_image, gpu=GPU, cpu=CPU, memory=MEM_MB, volumes=VOLS, timeout=1800, scaledown_window=SCALEDOWN,
         max_containers=1, env={"HF_HUB_OFFLINE": "1"})
class HFSide:
    key: str = modal.parameter()

    @modal.method()
    def run(self, adapters: list[str], n_prompts: int, max_new: int) -> dict:
        """Greedy outputs for the first n_prompts prompts, on the base model and then (same container) with each
        adapter in turn ("" = the base)."""
        import torch
        import transformers

        from anatta.backends import hf
        from anatta.player import attach_adapters

        t0 = time.time()
        model, tok = hf.load(self.key)
        load_s = time.time() - t0
        ids = hf.chat_prompt_ids(tok, _messages()[:n_prompts])
        out = {}
        for i, adapter in enumerate(adapters):
            if adapter:
                if i and adapters[i - 1]:
                    raise ValueError("one adapter per container after the base")
                model = attach_adapters(model, adapter)
                model.eval()
            out[adapter] = hf.greedy(model, tok, ids, max_new)
        texts = {a: [tok.decode(o, skip_special_tokens=True) for o in outs] for a, outs in out.items()}
        return {"prompt_ids": ids, "outputs": out, "texts": texts, "eos": tok.eos_token_id, "load_s": round(load_s, 1),
                "gen_s": round(time.time() - t0 - load_s, 1), "torch": torch.__version__,
                "transformers": transformers.__version__, "tokenizer_class": type(tok).__name__,
                "gpu": torch.cuda.get_device_name(0)}


@app.cls(image=vllm_image(str(PKG)), gpu=GPU, cpu=CPU, memory=MEM_MB, volumes=VOLS, timeout=1800,
         scaledown_window=SCALEDOWN, max_containers=1, startup_timeout=1800)
class VLLMSide:
    key: str = modal.parameter()

    @modal.enter()
    def load(self):
        from anatta.backends.vllm import Engine, Settings

        self.engine = Engine(Settings.for_model(self.key))

    @modal.method()
    def run(self, adapters: list[str], n_prompts: int, max_new: int) -> dict:
        t0 = time.time()
        reqs = [{"rid": f"{a}|{i}", "messages": m, "temperature": 0.0, "max_tokens": max_new, "adapter": a or None}
                for a in adapters for i, m in enumerate(_messages()[:n_prompts])]
        res = self.engine.generate(reqs)
        out = {a: [r["token_ids"] for r in res if r["rid"].startswith(f"{a}|")] for a in adapters}
        ids = [r["prompt_ids"] for r in res[:n_prompts]]
        texts = {a: [r["text"] for r in res if r["rid"].startswith(f"{a}|")] for a in adapters}
        return {"prompt_ids": ids, "outputs": out, "texts": texts, "eos": self.engine.tok.eos_token_id,
                "gen_s": round(time.time() - t0, 1), "info": self.engine.info()}


def compare(hf_res: dict, vl_res: dict, adapters: list[str]) -> dict:
    eos = hf_res["eos"]
    by_adapter = {}
    for a in adapters:
        rows = [{"prompt": i, **SV.compare_row(hf_res["prompt_ids"][i], vl_res["prompt_ids"][i], hf_res["outputs"][a][i],
                                              vl_res["outputs"][a][i], eos)}
                for i in range(len(hf_res["prompt_ids"]))]
        by_adapter[a] = {"summary": SV.summarize(rows), "rows": rows}
    return {"eos_hf": eos, "eos_vllm": vl_res["eos"], "by_adapter": by_adapter}


def _sessions(plan: list[tuple[str, list[str]]], n_prompts: int, max_new: int, meter, estimate_s: float) -> dict:
    """Each model's HF then vLLM run, one call at a time, metered."""
    out = {}
    for key, adapters in plan:
        res = {}
        for side, cls in (("hf", HFSide), ("vllm", VLLMSide)):
            meter.before_call(estimate_s)
            t0 = time.time()
            try:
                res[side] = cls(key=key).run.remote(adapters, n_prompts, max_new)
            except Exception as e:  # recorded on the card; the other models still run
                res[side] = {"error": f"{type(e).__name__}: {str(e)[:500]}"}
            finally:
                meter.after_call(model=key, side=side, wall_s=round(time.time() - t0, 1))
        ok = "error" not in res["hf"] and "error" not in res["vllm"]
        out[key] = {**res, "cmp": compare(res["hf"], res["vllm"], adapters) if ok else None}
    return out


def _plan() -> list[tuple[str, list[str]]]:
    by_key: dict[str, list[str]] = {}
    for key, adapter in SV.CONFIGS:
        by_key.setdefault(key, []).append(adapter or "")
    return list(by_key.items())


def _meter(label: str, cap: float, first_s: float):
    from anatta import spend

    rate = spend.rate_per_hour(GPU, CPU, MEM_MB / 1024)["total"]
    m = spend.Meter(tool=TOOL, spend_dir=None, label=label, session=f"{label}-{int(time.time())}", model_key="serving",
                    cap_usd=cap, rate=rate, tail_s=SCALEDOWN)
    m.start(first_s)
    return m


def build_card(res: dict, costs: dict, method: str) -> dict:
    """The card: per configuration the summary and the per-prompt rows (numbers only), versions, timings, costs, and
    any side that failed."""
    configs, errors, versions, timing = {}, {}, {}, {}
    for key, r in res.items():
        if r["cmp"] is None:
            errors[key] = {side: r[side]["error"] for side in ("hf", "vllm") if "error" in r[side]}
            continue
        for a, c in r["cmp"]["by_adapter"].items():
            configs[SV.config_name(key, a or None)] = c
        info = r["vllm"]["info"]
        versions[key] = {"hf": {f: r["hf"][f] for f in ("torch", "transformers", "tokenizer_class", "gpu")},
                         "vllm": {f: info[f] for f in ("vllm", "transformers", "torch", "tokenizer_class", "served_arch")}}
        timing[key] = {"hf_load": r["hf"]["load_s"], "hf_gen": r["hf"]["gen_s"], "vllm_load": info["load_s"],
                       "vllm_gen": r["vllm"]["gen_s"]}
    expected = {SV.config_name(k, a) for k, a in SV.CONFIGS}
    return {"format": "anatta-serving-check/1", "method": method,
            "configs": {k: c["summary"] for k, c in configs.items()},
            "pass": set(configs) == expected and all(c["summary"]["pass"] for c in configs.values()),
            "errors": errors, "rows": {k: c["rows"] for k, c in configs.items()}, "versions": versions,
            "timing_s": timing, "cost_usd": costs}


@app.local_entrypoint()
def main(trial_only: bool = False, trial_cap: float = 1.0, run_cap: float = 1.9):
    from anatta import cards
    from anatta.config import load

    out = load().results_root / "calib" / "serving"
    method = SV.__doc__.strip().split("\n\n")[1]
    spec = {"check": "serving", "method": method, "prompts": list(SV.PROMPTS), "configs": [list(c) for c in SV.CONFIGS],
            "prefix": SV.PREFIX, "window": SV.WINDOW, "min_identical": SV.MIN_IDENTICAL, "gpu": GPU,
            "caps_usd": {"trial": trial_cap, "run": run_cap}}
    if not (out / cards.PLAN).exists():
        cards.write_plan(out, spec)
    # trial: Qwen on both paths, 2 prompts, 8 tokens: the images build, the volumes mount, the comparison runs
    meter = _meter("trial", trial_cap, TRIAL_S)
    try:
        trial = _sessions([("qwen25_7b", [""])], 2, 8, meter, TRIAL_S)["qwen25_7b"]
    finally:
        meter.close(stopped=False)
    trial_usd = meter.charged_usd
    t = trial["cmp"]["by_adapter"][""]["summary"] if trial["cmp"] else {
        "errors": {s: trial[s]["error"] for s in ("hf", "vllm") if "error" in trial[s]}}
    cards.write_text(out, "trial.json", json.dumps({"summary": t, "usd": trial_usd}, indent=1) + "\n", force=True)
    print("trial", json.dumps(t), f"${trial_usd:.3f}")
    if trial_only or trial["cmp"] is None or t["prompt_ids_same"] != t["n"]:
        print("stopping after the trial" + ("" if trial_only else ": it failed or the prompt ids differ"))
        return
    meter = _meter("check", run_cap, CALL_S)
    try:
        res = _sessions(_plan(), len(SV.PROMPTS), SV.WINDOW, meter, CALL_S)
    finally:
        meter.close(stopped=False)
    costs = {"trial": round(trial_usd, 4), "check": round(meter.charged_usd, 4), "rate_usd_per_hour": meter.rate,
             "note": "metered container wall time (anatta.spend Meter); Modal's usage page is the authority"}
    card = build_card(res, costs, method)
    tokens = {k: {side: {f: v for f, v in r[side].items() if f in ("outputs", "texts", "prompt_ids", "info", "error")}
                  for side in ("hf", "vllm")} for k, r in res.items()}
    cards.write_text(out / "data", "serving_check_tokens.json", json.dumps(tokens, default=str) + "\n", force=True)
    p = cards.write_card(out, card, name="serving_check.json", force=True)
    print(json.dumps({"configs": card["configs"], "errors": card["errors"], "pass": card["pass"]}, indent=1))
    print("card ->", p)
