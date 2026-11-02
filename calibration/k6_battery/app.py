"""K6's Modal app (anatta-k6): vLLM generation for the behavior-battery anchors. Claude (Opus 5.5) wrote this file,
2 Nov 2026. See PLAN.md for the protocol.

One L40S container per model (class Gen, parameter model_key). The weights come from the anatta-hf-cache volume,
mounted read-only, at the pinned snapshot; nothing is downloaded (HF_HUB_OFFLINE=1, HF_HOME in /tmp). Requests and
outputs travel as function arguments and return values: no item text is written to any volume. No hooks are needed,
so plain vLLM serves.

Methods:
  info()                 the model's path, revision, dtype, vLLM version, and a 1-prompt load check
  gen(requests)          one batch: [{rid, messages, temperature, top_p, max_tokens, seed}] -> [{rid, text,
                         finish_reason, n_prompt, n_out}]
  sycon(items, turns, followup, max_tokens)
                         SYCON's multi-turn debate, greedy, all questions advanced turn by turn in one batch
Also parquet_rows(files): a CPU function that turns MASK's parquet bytes into rows (no local pyarrow on the Mac).

Run from the laptop by run.py (matlab venv, which has modal). Never touches any other app or volume.
"""

import modal

APP_NAME = "anatta-k6"
CACHE = "/cache"
VLLM = "vllm==0.10.1.1"
GPU = "L40S"
MODELS = {
    "llama3_8b": {"repo": "meta-llama/Meta-Llama-3-8B-Instruct", "rev": "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"},
    "qwen25_7b": {"repo": "Qwen/Qwen2.5-7B-Instruct", "rev": "a09a35458c702b33eeacc393d103063234e8bc28"},
    "gemma2_9b": {"repo": "google/gemma-2-9b-it", "rev": "11c9b309abf73637e4b6f9a3fa1e92e615547819"},
}

image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
         .entrypoint([])
         .uv_pip_install(VLLM, "transformers==4.55.4")
         .env({"HF_HOME": "/tmp/hf", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
               "VLLM_USE_V1": "1"}))
light = modal.Image.debian_slim(python_version="3.12").uv_pip_install("pyarrow==21.0.0")
cache = modal.Volume.from_name("anatta-hf-cache")
app = modal.App(APP_NAME)


def snapshot(model_key: str) -> str:
    m = MODELS[model_key]
    return f"{CACHE}/hf/hub/models--{m['repo'].replace('/', '--')}/snapshots/{m['rev']}"


def merge_system(messages: list[dict]) -> list[dict]:
    """Gemma's template has no system role: SYCON's code puts the system text before the first user turn."""
    if not messages or messages[0]["role"] != "system":
        return messages
    sys_text, rest = messages[0]["content"], [dict(m) for m in messages[1:]]
    for m in rest:
        if m["role"] == "user":
            m["content"] = f"{sys_text}\n\n{m['content']}"
            break
    return rest


@app.cls(image=image, gpu=GPU, cpu=4.0, memory=32768, volumes={CACHE: cache.read_only()}, timeout=3 * 3600,
         scaledown_window=180, max_containers=1, startup_timeout=1800)
class Gen:
    model_key: str = modal.parameter()

    @modal.enter()
    def load(self):
        from vllm import LLM

        self.path = snapshot(self.model_key)
        self.llm = LLM(model=self.path, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.90,
                       seed=0, enable_prefix_caching=True)
        self.tok = self.llm.get_tokenizer()

    def _prompts(self, convs):
        out = []
        for msgs in convs:
            if self.model_key.startswith("gemma"):
                msgs = merge_system(msgs)
            out.append(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        return out

    def _run(self, convs, params):
        from vllm import SamplingParams

        sps = [SamplingParams(temperature=p["temperature"], top_p=p.get("top_p", 1.0), max_tokens=p["max_tokens"],
                              seed=p.get("seed")) for p in params]
        prompts = self._prompts(convs)
        res = self.llm.generate(prompts, sps, use_tqdm=False)
        return [{"text": r.outputs[0].text, "finish_reason": r.outputs[0].finish_reason,
                 "n_prompt": len(r.prompt_token_ids), "n_out": len(r.outputs[0].token_ids)} for r in res]

    @modal.method()
    def info(self) -> dict:
        import vllm

        probe = self._run([[{"role": "user", "content": "Say OK."}]], [{"temperature": 0.0, "max_tokens": 8}])[0]
        return {"model_key": self.model_key, "path": self.path, "rev": MODELS[self.model_key]["rev"],
                "vllm": vllm.__version__, "dtype": "bfloat16", "load_check": bool(probe["text"].strip()),
                "probe_tokens": probe["n_out"],
                "template_has_default_system": "You are Qwen" in self._prompts([[{"role": "user", "content": "x"}]])[0]}

    @modal.method()
    def gen(self, requests: list[dict]) -> list[dict]:
        outs = self._run([r["messages"] for r in requests], requests)
        return [{"rid": r["rid"], **o} for r, o in zip(requests, outs)]

    @modal.method()
    def sycon(self, items: list[dict], turns: int, followup: str, max_tokens: int) -> list[dict]:
        """items: [{qid, messages}] (the opening system+user). Greedy. Returns [{qid, responses: [...], finish}]."""
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


@app.function(image=light, cpu=1.0, memory=2048, timeout=600)
def parquet_rows(files: dict) -> dict:
    """{name: parquet bytes} -> {name: [row dicts]}. Data only: nothing is executed from the files."""
    import io

    import pyarrow.parquet as pq

    return {name: pq.read_table(io.BytesIO(b)).to_pylist() for name, b in files.items()}
