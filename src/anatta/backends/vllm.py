"""The vLLM backend: ONE generator, merged from the three calibrated ones (calibration/k6_battery/app.py:61,
k6b_ministral/app_vllm.py:36, k7_agentic/app_vllm.py:49; read, not edited), with LoRA serving added. It runs inside a
Modal container built from `image()`; this module imports vllm only there. Claude (Opus 5.5) wrote this file,
3 Nov 2026.

What it keeps from each:
  K6b  prompts tokenized by the HF tokenizer's chat template and passed as token ids (TokensPrompt), the plain fast
       tokenizer plus the snapshot's chat_template.jinja for Ministral (tokenizer: fast_jinja), Ministral's text-only
       load (tokenizer_mode/config_format "hf", safetensors, no image inputs), top_k -1, the vLLM pin and the
       flashinfer-sampler switch, the loader log in info()
  K6   the SYCON multi-turn loop (converse), Gemma's system text merged into the first user turn
  K7   max_tokens cut so prompt plus completion fit the context (never below 16), the per-model context length
Added: enable_lora with max_lora_rank 64; a request's "adapter" (a PEFT folder) is served as a LoRARequest, one id per
folder.

Behaviour that differs from K6 and K7 (recorded so a port can choose): they passed rendered text, so vLLM tokenized it
again (which may add a second BOS where the template already wrote one, as Llama's and Gemma's do; not checked here);
here the HF tokenizer's ids go in as they are, as anatta.models.chat_ids makes them. generation_config defaults to "vllm" (K6b's), so the checkpoint's sampling
defaults (Qwen's top_k 20 and repetition_penalty 1.05) are not applied; K6 and K7 ran with vLLM's "auto", which
applied them to every parameter a request left unset. Pass generation_config="auto" to reproduce them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

VLLM = "vllm==0.31.0"  # K6b's pin (calibration/k6b_ministral/app_vllm.py:20); serves Ministral 3 and Qwen 2.5
ENV = {"HF_HOME": "/tmp/hf", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
       "VLLM_USE_FLASHINFER_SAMPLER": "0"}  # flashinfer JIT needs nvcc; torch sampler instead (K6b)
MAX_LORA_RANK = 64


def image(pkg_dir: str):
    """K6b's image (debian slim, Python 3.12, the vLLM pin, its env) plus this package at /root/anatta."""
    import modal

    return (modal.Image.debian_slim(python_version="3.12").uv_pip_install(VLLM)
            .env({**ENV, "PYTHONPATH": "/root"})
            .add_local_dir(pkg_dir, "/root/anatta", ignore=["**/__pycache__"]))


def merge_system(messages: list[dict]) -> list[dict]:
    """A leading system message folded into the first user turn, for templates with no system role (Gemma), as
    K6 and K7 do."""
    if not messages or messages[0]["role"] != "system":
        return messages
    sys_text, rest = messages[0]["content"], [dict(m) for m in messages[1:]]
    for m in rest:
        if m["role"] == "user":
            m["content"] = f"{sys_text}\n\n{m['content']}"
            break
    return rest


SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "seed", "max_tokens", "repetition_penalty")


def sampling_record(sp) -> dict:
    """What the engine was handed, read back from the SamplingParams object itself (after vLLM's own normalization,
    e.g. a greedy request's top_p and top_k), not from the request dict. Added 5 Nov 2026 (V1r's provenance gap: the
    arms' settings rested on the code path and plan.json, not on the generation rows)."""
    out = {}
    for k in SAMPLING_FIELDS:
        v = getattr(sp, k, None)
        out[k] = v if v is None or isinstance(v, (bool, int, float, str)) else str(v)
    return out


@dataclass
class Settings:
    """How one model is served. path: the snapshot folder; ctx: max_model_len; text_only and tokenizer come from the
    registry's `vllm` block; merge_system for templates with no system role."""
    path: str
    ctx: int = 8192
    text_only: bool = False
    tokenizer: str = "auto"  # "auto" (AutoTokenizer) or "fast_jinja" (K6b's Ministral tokenizer)
    merge_system: bool = False
    enable_lora: bool = True
    max_lora_rank: int = MAX_LORA_RANK
    max_loras: int = 2
    generation_config: str = "vllm"
    gpu_memory_utilization: float = 0.90
    seed: int = 0
    extra: dict = field(default_factory=dict)

    @classmethod
    def for_model(cls, name: str, mount: str | None = None, **over) -> Settings:
        from .registry import get

        m = get(name)
        if "vllm" not in m.backends:
            raise ValueError(f"{m.key} is not registered for the vllm backend")
        v = m.vllm
        base = dict(path=m.snapshot(mount), ctx=int(v.get("ctx", 8192)), text_only=bool(v.get("text_only")),
                    tokenizer=v.get("tokenizer", "auto"), merge_system=m.family in ("gemma", "gemma2"))
        return cls(**{**base, **over})


class Engine:
    """The generator. Requests are dicts: {"rid", "messages", "temperature", "max_tokens", "seed"?, "top_p"?,
    "top_k"? (-1: off), "repetition_penalty"? (1.0: off), "adapter"?}. Results carry the text, the output token ids, the prompt ids' count, the finish reason and the
    max_tokens actually used, and "sampling": the SamplingParams the engine used (sampling_record)."""

    def __init__(self, s: Settings):
        import io
        import logging
        import time

        from vllm import LLM

        self.s = s
        buf = io.StringIO()
        h = logging.StreamHandler(buf)
        h.setLevel(logging.INFO)
        logging.getLogger("vllm").addHandler(h)
        t0 = time.time()
        self.tok = self._tokenizer()
        kw = dict(model=s.path, dtype="bfloat16", max_model_len=s.ctx, gpu_memory_utilization=s.gpu_memory_utilization,
                  seed=s.seed, enable_prefix_caching=True, generation_config=s.generation_config)
        if s.text_only:
            kw.update(tokenizer_mode="hf", config_format="hf", load_format="safetensors",
                      limit_mm_per_prompt={"image": 0})
        if s.enable_lora:
            kw.update(enable_lora=True, max_lora_rank=s.max_lora_rank, max_loras=s.max_loras)
        self.llm = LLM(**kw, **s.extra)
        self.load_s = time.time() - t0
        logging.getLogger("vllm").removeHandler(h)
        self.log = [ln for ln in buf.getvalue().splitlines()
                    if any(k in ln.lower() for k in ("weight", "missing", "loading", "architecture", "resolved", "lora"))][:40]
        self._lora: dict[str, int] = {}

    def _tokenizer(self):
        if self.s.tokenizer == "fast_jinja":
            from transformers import PreTrainedTokenizerFast

            tok = PreTrainedTokenizerFast.from_pretrained(self.s.path)
            with open(f"{self.s.path}/chat_template.jinja", encoding="utf-8") as f:
                tok.chat_template = f.read()
            return tok
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.s.path)

    def prompt_ids(self, messages: list[dict]) -> list[int]:
        if self.s.merge_system:
            messages = merge_system(messages)
        ids = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        return list(ids)

    def _lora_request(self, adapter: str | None):
        if not adapter:
            return None
        if not self.s.enable_lora:
            raise ValueError("this engine was built without LoRA")
        from vllm.lora.request import LoRARequest

        lid = self._lora.setdefault(adapter, len(self._lora) + 1)
        return LoRARequest(f"adapter{lid}", lid, adapter)

    def generate(self, requests: list[dict]) -> list[dict]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        ids = [self.prompt_ids(r["messages"]) for r in requests]
        sps, mts = [], []
        for r, p in zip(requests, ids):
            mt = max(16, min(int(r["max_tokens"]), self.s.ctx - len(p) - 8))
            mts.append(mt)
            sps.append(SamplingParams(temperature=float(r["temperature"]), top_p=float(r.get("top_p", 1.0)),
                                      top_k=int(r.get("top_k", -1)),
                                      repetition_penalty=float(r.get("repetition_penalty", 1.0)),
                                      max_tokens=mt, seed=r.get("seed")))
        loras = [self._lora_request(r.get("adapter")) for r in requests]
        res = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in ids], sps, use_tqdm=False,
                                lora_request=loras if any(loras) else None)
        return [{"rid": r.get("rid"), "text": o.outputs[0].text, "token_ids": list(o.outputs[0].token_ids),
                 "prompt_ids": p, "n_prompt": len(o.prompt_token_ids), "n_out": len(o.outputs[0].token_ids),
                 "finish_reason": o.outputs[0].finish_reason, "max_tokens": mt, "adapter": r.get("adapter"),
                 "sampling": sampling_record(sp)}
                for r, p, o, mt, sp in zip(requests, ids, res, mts, sps)]

    def score(self, requests: list[dict]) -> list[dict]:
        """Teacher-forced log probabilities: requests {"rid", "ids" (prompt then output token ids), "n_score" (how many
        of the last ids to score), "adapter"?}. One forward per request (max_tokens 1, prompt_logprobs 1: vLLM returns
        each prompt position's own token beside its top-1). Returns {"rid", "logprobs": per scored token its log
        probability, None where vLLM gave none (e.g. a position served from the prefix cache without logprobs)}."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
        loras = [self._lora_request(r.get("adapter")) for r in requests]
        res = self.llm.generate([TokensPrompt(prompt_token_ids=list(r["ids"])) for r in requests], sp, use_tqdm=False,
                                lora_request=loras if any(loras) else None)
        out = []
        for r, o in zip(requests, res):
            ids, plp, n = list(r["ids"]), o.prompt_logprobs or [], int(r["n_score"])
            lps = []
            for j in range(len(ids) - n, len(ids)):
                d = plp[j] if j < len(plp) else None
                lp = d.get(ids[j]) if d else None
                lps.append(None if lp is None else float(getattr(lp, "logprob", lp)))
            out.append({"rid": r.get("rid"), "logprobs": lps, "adapter": r.get("adapter")})
        return out

    def converse(self, items: list[dict], turns: int, followup: str, max_tokens: int,
                 adapter: str | None = None) -> list[dict]:
        """K6's SYCON loop: items [{qid, messages}] (the opening turns), greedy, `followup` as the user's reply after
        every answer but the last. Returns [{qid, responses, finish}]."""
        convs = {it["qid"]: [dict(m) for m in it["messages"]] for it in items}
        resp: dict = {q: [] for q in convs}
        fin: dict = {q: [] for q in convs}
        for t in range(turns):
            qids = list(convs)
            outs = self.generate([{"rid": q, "messages": convs[q], "temperature": 0.0, "max_tokens": max_tokens,
                                   "adapter": adapter} for q in qids])
            for q, o in zip(qids, outs):
                text = o["text"].strip()
                resp[q].append(text)
                fin[q].append(o["finish_reason"])
                if t < turns - 1:
                    convs[q] += [{"role": "assistant", "content": text}, {"role": "user", "content": followup}]
        return [{"qid": q, "responses": resp[q], "finish": fin[q]} for q in convs]

    def info(self) -> dict:
        import json

        import torch
        import transformers
        import vllm

        with open(f"{self.s.path}/config.json", encoding="utf-8") as f:
            cfg = json.load(f)
        probe = self.generate([{"messages": [{"role": "user", "content": "Say OK."}], "temperature": 0.0,
                                "max_tokens": 8}])[0]
        mc = self.llm.llm_engine.model_config
        return {"path": self.s.path, "vllm": vllm.__version__, "transformers": transformers.__version__,
                "torch": torch.__version__, "dtype": "bfloat16", "tokenizer_class": type(self.tok).__name__,
                "architectures": cfg.get("architectures"), "served_arch": getattr(mc, "architectures", None),
                "settings": {k: v for k, v in vars(self.s).items() if k != "extra"}, "load_s": round(self.load_s, 1),
                "loader_log": self.log, "load_check": bool(probe["text"].strip()), "probe_tokens": probe["n_out"]}
