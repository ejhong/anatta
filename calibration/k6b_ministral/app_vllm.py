"""K6b's Modal app (anatta-k6b): a newer vLLM serving Ministral 3 8B Instruct for K6's measures. Claude (Opus 5.5)
wrote this file, 2 Nov 2026. Protocol: PLAN.md.

K6's app (calibration/k6_battery/app.py) pins vllm 0.10.1.1, which predates Ministral 3; this app is additive and
leaves K6 unchanged. One L40S, bf16, weights read-only from anatta-hf-cache at the pinned snapshot, hub offline,
nothing written to any volume. Prompts are built by the snapshot's own HF chat template (transformers
apply_chat_template, tokenized, as anatta.models.chat_ids does) and passed to vLLM as token ids. Never touches any
other app or volume.

Methods:
  info()                 versions, architecture, the loader's weight report, template facts, 1-prompt load check
  gen(requests)          [{rid, messages, temperature, top_p, max_tokens, seed}] -> [{rid, text, finish_reason, ...}]
  sycon(items, turns, followup, max_tokens)   K6's multi-turn debate, greedy
"""

import modal

APP_NAME = "anatta-k6b"
CACHE = "/cache"
VLLM = "vllm==0.31.0"
GPU = "L40S"
REPO = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
REV = "f6fae9795746f63c9be8344932f01275f3c63734"
SNAP = f"{CACHE}/hf/hub/models--{REPO.replace('/', '--')}/snapshots/{REV}"

image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install(VLLM)
         .env({"HF_HOME": "/tmp/hf", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
               "VLLM_USE_FLASHINFER_SAMPLER": "0"}))  # flashinfer JIT needs nvcc; torch sampler instead
app = modal.App(APP_NAME)
cache = modal.Volume.from_name("anatta-hf-cache")


@app.cls(image=image, gpu=GPU, cpu=4.0, memory=32768, volumes={CACHE: cache.read_only()}, timeout=3 * 3600,
         scaledown_window=180, max_containers=1, startup_timeout=1800)
class Gen:
    @modal.enter()
    def load(self):
        import io
        import logging
        import time

        from transformers import PreTrainedTokenizerFast
        from vllm import LLM

        buf = io.StringIO()
        h = logging.StreamHandler(buf)
        h.setLevel(logging.INFO)
        logging.getLogger("vllm").addHandler(h)
        t0 = time.time()
        # The plain tokenizers backend with the snapshot's chat_template.jinja, as the lab's HF path renders it (with
        # vLLM's mistral_common installed, AutoTokenizer can pick a backend that ignores the jinja template and its
        # default system prompt; the serving check compares prompt token counts with the HF path).
        self.tok = PreTrainedTokenizerFast.from_pretrained(SNAP)
        self.tok.chat_template = open(f"{SNAP}/chat_template.jinja").read()
        self.llm = LLM(model=SNAP, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.90, seed=0,
                       enable_prefix_caching=True, tokenizer_mode="hf", config_format="hf",
                       load_format="safetensors", limit_mm_per_prompt={"image": 0}, generation_config="vllm")
        self.load_s = time.time() - t0
        logging.getLogger("vllm").removeHandler(h)
        self.log = [ln for ln in buf.getvalue().splitlines()
                    if any(k in ln.lower() for k in ("weight", "missing", "loading", "architecture", "resolved"))][:40]

    def _ids(self, msgs):
        ids = self.tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        return list(ids)

    def _run(self, convs, params):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        sps = [SamplingParams(temperature=p["temperature"], top_p=p.get("top_p", 1.0), top_k=-1,
                              max_tokens=p["max_tokens"], seed=p.get("seed")) for p in params]
        prompts = [TokensPrompt(prompt_token_ids=self._ids(m)) for m in convs]
        res = self.llm.generate(prompts, sps, use_tqdm=False)
        return [{"text": r.outputs[0].text, "finish_reason": r.outputs[0].finish_reason,
                 "n_prompt": len(r.prompt_token_ids), "n_out": len(r.outputs[0].token_ids)} for r in res]

    @modal.method()
    def info(self) -> dict:
        import json

        import torch
        import transformers
        import vllm

        cfg = json.load(open(f"{SNAP}/config.json"))
        probe = self._run([[{"role": "user", "content": "Say OK."}]], [{"temperature": 0.0, "max_tokens": 8}])[0]
        no_sys = self.tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False,
                                              add_generation_prompt=True)
        with_sys = self.tok.apply_chat_template([{"role": "system", "content": "S"}, {"role": "user", "content": "x"}],
                                                tokenize=False, add_generation_prompt=True)
        mc = self.llm.llm_engine.model_config
        return {"repo": REPO, "rev": REV, "path": SNAP, "vllm": vllm.__version__, "transformers": transformers.__version__,
                "tokenizer_class": type(self.tok).__name__,
                "torch": torch.__version__, "dtype": "bfloat16", "architectures": cfg.get("architectures"),
                "served_arch": getattr(mc, "architectures", None), "load_s": round(self.load_s, 1),
                "loader_log": self.log, "load_check": bool(probe["text"].strip()), "probe_tokens": probe["n_out"],
                "template_no_system_len": len(no_sys), "template_with_system_len": len(with_sys),
                "template_adds_default_system": len(no_sys) > 200,  # Ministral: SYSTEM_PROMPT.txt via the jinja default
                "missing_weights": 0,  # vLLM raises at load when a parameter is not initialised from the checkpoint
                }

    @modal.method()
    def gen(self, requests: list[dict]) -> list[dict]:
        outs = self._run([r["messages"] for r in requests], requests)
        return [{"rid": r["rid"], **o} for r, o in zip(requests, outs)]

    @modal.method()
    def sycon(self, items: list[dict], turns: int, followup: str, max_tokens: int) -> list[dict]:
        """K6's SYCON loop: items [{qid, messages}], greedy; returns [{qid, responses, finish}]."""
        convs = {it["qid"]: [dict(m) for m in it["messages"]] for it in items}
        resp = {q: [] for q in convs}
        fin = {q: [] for q in convs}
        for t in range(turns):
            qids = list(convs)
            outs = self._run([convs[q] for q in qids], [{"temperature": 0.0, "max_tokens": max_tokens}] * len(qids))
            for q, o in zip(qids, outs):
                text = o["text"].strip()
                resp[q].append(text)
                fin[q].append(o["finish_reason"])
                if t < turns - 1:
                    convs[q] += [{"role": "assistant", "content": text}, {"role": "user", "content": followup}]
        return [{"qid": q, "responses": resp[q], "finish": fin[q]} for q in convs]
