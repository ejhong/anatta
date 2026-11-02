"""K7's Modal app (anatta-k7), vLLM half: Qwen2.5-7B-Instruct and Gemma-2-9B-IT. Claude (Opus 5.5) wrote this file,
2 Nov 2026, after K6's app.py (same image, pins and volume handling).

One L40S per model; weights from the anatta-hf-cache volume, read-only, at the pinned snapshot, hub offline,
nothing downloaded. Prompts and completions travel as arguments and return values only; nothing is written to any
volume. Each model's own chat template (tokenizer.apply_chat_template). Gemma's template has no system role: the
system text is put before the first user turn, separated by a blank line (merge_system). Sampling: temperature,
top_p 1, no top-k, a per-request seed; explicit SamplingParams, so no generation_config defaults apply.
max_model_len 8192 for Gemma (its native context), 16384 for Qwen. Never touches any other app or volume.
"""

import modal

APP_NAME = "anatta-k7"
CACHE = "/cache"
VLLM = "vllm==0.10.1.1"
GPU = "L40S"
MODELS = {
    "qwen25_7b": {"repo": "Qwen/Qwen2.5-7B-Instruct", "rev": "a09a35458c702b33eeacc393d103063234e8bc28", "ctx": 16384},
    "gemma2_9b": {"repo": "google/gemma-2-9b-it", "rev": "11c9b309abf73637e4b6f9a3fa1e92e615547819", "ctx": 8192},
}

image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
         .entrypoint([])
         .uv_pip_install(VLLM, "transformers==4.55.4")
         .env({"HF_HOME": "/tmp/hf", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "VLLM_USE_V1": "1"}))
cache = modal.Volume.from_name("anatta-hf-cache")
app = modal.App(APP_NAME)


def snapshot(model_key: str) -> str:
    m = MODELS[model_key]
    return f"{CACHE}/hf/hub/models--{m['repo'].replace('/', '--')}/snapshots/{m['rev']}"


def merge_system(messages: list[dict]) -> list[dict]:
    if not messages or messages[0]["role"] != "system":
        return messages
    sys_text, rest = messages[0]["content"], [dict(m) for m in messages[1:]]
    for m in rest:
        if m["role"] == "user":
            m["content"] = f"{sys_text}\n\n{m['content']}"
            break
    return rest


@app.cls(image=image, gpu=GPU, cpu=4.0, memory=32768, volumes={CACHE: cache.read_only()}, timeout=2 * 3600,
         scaledown_window=120, max_containers=1, startup_timeout=1800)
class Gen:
    model_key: str = modal.parameter()

    @modal.enter()
    def load(self):
        from vllm import LLM

        self.path = snapshot(self.model_key)
        self.ctx = MODELS[self.model_key]["ctx"]
        self.llm = LLM(model=self.path, dtype="bfloat16", max_model_len=self.ctx, gpu_memory_utilization=0.90,
                       seed=0, enable_prefix_caching=True)
        self.tok = self.llm.get_tokenizer()

    def _prompt(self, msgs):
        if self.model_key.startswith("gemma"):
            msgs = merge_system(msgs)
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @modal.method()
    def gen(self, requests: list[dict]) -> list[dict]:
        """[{rid, messages, temperature, max_tokens, seed}] -> [{rid, text, finish_reason, n_prompt, n_out, max_tokens}].
        max_tokens is cut so prompt + completion fit the context."""
        import vllm
        from vllm import SamplingParams

        prompts = [self._prompt(r["messages"]) for r in requests]
        n_prompt = [len(self.tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
        sps, mts = [], []
        for r, n in zip(requests, n_prompt):
            mt = max(16, min(int(r["max_tokens"]), self.ctx - n - 8))
            mts.append(mt)
            sps.append(SamplingParams(temperature=float(r["temperature"]), top_p=1.0, top_k=-1, max_tokens=mt,
                                      seed=r.get("seed")))
        res = self.llm.generate(prompts, sps, use_tqdm=False)
        return [{"rid": r["rid"], "text": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason,
                 "n_prompt": len(o.prompt_token_ids), "n_out": len(o.outputs[0].token_ids), "max_tokens": mt,
                 "vllm": vllm.__version__, "rev": MODELS[self.model_key]["rev"]}
                for r, o, mt in zip(requests, res, mts)]
