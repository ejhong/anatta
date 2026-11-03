"""bench's torch side: the interventions, batched generation with a seed per request, the hook checks, the
matched-strength random directions, and the measurements the difference-vector controls need. It runs in the bench
container (modal_bench.py) and in the CPU tests, and needs torch, transformers and the anatta library
(main's src/anatta from 3 Nov), whose hooks it uses unchanged. Claude (Opus 5.5) wrote this file, 1 Nov 2026, and extended it the
same day (token-budget passes, partial cuts, measure, response_means). Private (lab/explore).

Interventions, one per request:
  none     no hooks.
  add      anatta.dial.Intervention(layer, d, "steer", alpha=c) at each listed layer: h <- h + c d.
  ablate   anatta.dial.Intervention(layer, d_layer, "ablate") at each listed layer: h <- h - (h.d) d. With every layer
           of a per-layer direction this is player.ego_cut's construction: tests/test_bench.py checks that the two
           agree to float32 rounding (about 1e-7), and the container checks it again for v_ego on Ministral.
           With "fractions" {layer: f} (0 < f <= 1), a layer whose f is below 1 gets a partial cut instead,
           h <- h - f (h.d) d (PartialCut); f = 1 is the plain ablate above. Only a spec's `match: partial` asks for it.
The cut split into its two parts (added the evening of 1 Nov). A cut sets h.d to 0; where h.d has a mean m on ordinary
text, that is a constant push of -m along d on top of removing the part that varies. Each split arm takes "means"
{layer: m_l}, one constant per hooked layer:
  mean_ablate    h <- h - (h.d - m_l) d: h.d is set to m_l (anatta's cap with lo = hi = m_l). The varying part goes,
                 the mean stays.
  shift_only     h <- h - m_l d (anatta's steer, alpha = -m_l): the push alone.
  shift_reverse  h <- h + m_l d (anatta's steer, alpha = +m_l): the opposite push, a sign check.
At one layer, shift_only after mean_ablate is the cut exactly, and both use that layer's neutral-set mean. Over several
layers a shift is not a "set": it stays in the residual stream, so the uncut means added up layer after layer would
push far past where the cut puts h.d (adjacent layers' unit vectors have cos 0.87-0.97 for v_ego on Ministral).
split_means() therefore finds each arm's m_l on the neutral set layer by layer, with the arm's own hooks on the layers
before l: mean_ablate's m_l is the mean h.d has there (no displacement anywhere, spread removed), shift_only's makes
the mean 0 at every layer (the cut's mean, spread kept), shift_reverse's makes it 2 mu_l (the mirror displacement, +mu_l,
where mu_l is the uncut mean). For a single layer all three are mu_l, the formulas above with the uncut mean.
Requests that share an intervention form a group, and each group runs inside one anatta.dial.intervene context. When
the context closes, the model's hook count and the logits of a fixed probe prompt are compared with their values at
load; a mismatch raises HookLeak (the container then stops taking requests).

Sampling. Each request has its own temperature, top_p, top_k, max_tokens and seed. RowSampler draws each row's next
token from that request's own torch.Generator, by inverse CDF. A sample therefore depends on its seed and its logits
only, not on the other requests in its batch, and requests that share a seed share their random draws across
interventions (paired comparisons). bf16 logits can still shift slightly with the batch's shape. top_k defaults to
the model's generation_config.top_k, as the lab's Player inherits it; top_p defaults to 1.0, Player's value.

Passes. A group runs in passes planned by plan_passes: shortest prompts first, and a pass takes rows while rows x (its
longest prompt + max new tokens) stays within the token budget (TOKEN_BUDGET unless the batch names one) and it has at
most gen_batch rows. Left padding makes every row of a pass as wide as its longest, so this bounds the KV cache and the
prefill's activations together. 32 rows of 4,002-token game prompts with 400 new tokens (140,864 slots) ran out of
memory on the L40S; 20 such rows (88,040) ran; the default sits below that.

random_matched() builds control directions whose activation variance on a fixed neutral prompt set equals the
target's; its docstring gives the construction. measure() reports any direction's variance on the same neutral set
(other_trait and split_null controls are compared with their target that way), and response_means() samples replies
to contrastive prompts and reads their per-response mean activations, the raw material of the split null.
split_means() finds the split arms' per-layer constants on that neutral set and checks what each arm does there.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time

import torch

try:
    from transformers import LogitsProcessor, LogitsProcessorList
except ModuleNotFoundError:  # the laptop client's venv has torch but not transformers; it uses plan_passes only
    LogitsProcessor, LogitsProcessorList = object, None

# The two models, pinned to the snapshots already on the lab's anatta-hf-cache volume (both complete on 1 Nov 2026;
# Qwen's folder has no refs/main, so a hub-id load would not resolve offline).
MODELS = {
    "ministral": {"hf_id": "mistralai/Ministral-3-8B-Instruct-2512-BF16",
                  "revision": "f6fae9795746f63c9be8344932f01275f3c63734"},
    "qwen": {"hf_id": "Qwen/Qwen2.5-7B-Instruct", "revision": "a09a35458c702b33eeacc393d103063234e8bc28"},
}
PROBE_TEXT = "Name three colours of the rainbow."  # the hook check's fixed prompt
METHOD = "pc_mix_v1"  # random_matched's construction, recorded with every control it makes
TOKEN_BUDGET = 80_000  # token slots per generation pass: rows x (longest prompt + max new tokens); see the docstring
SPLIT_MODES = ("mean_ablate", "shift_only", "shift_reverse")  # the cut's two parts and the mirror push (docstring)
SPLIT_METHOD = "split_means_v1"  # split_means' construction, recorded with every split arm
OUTLIER = 10.0  # neutral_activations' rule: a token whose norm passes this times the layer's median is left out


def snapshot_path(model_key: str, hub: str = "/cache/hf/hub") -> str:
    m = MODELS[model_key]
    return f"{hub}/models--{m['hf_id'].replace('/', '--')}/snapshots/{m['revision']}"


class HookLeak(RuntimeError):
    """After a group's context closed, the model still carried hooks, or the probe's logits had moved."""


class MatchError(RuntimeError):
    """random_matched could not reach the target's variance within the tolerance."""


# ----------------------------------------------------------------------------------------------- interventions


def count_hooks(model) -> int:
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


def dir_key(iv: dict) -> str:
    """The key a request's direction is sent under: its tensor hash, or its file hash in payloads that have none."""
    return iv.get("tensor_sha256") or iv["sha256"]


def _fractions(iv: dict) -> tuple | None:
    f = iv.get("fractions")
    return tuple(sorted((int(x), float(v)) for x, v in f.items())) if f else None


def _means(iv: dict) -> tuple | None:
    m = iv.get("means")
    return tuple(sorted((int(x), float(v)) for x, v in m.items())) if m else None


def iv_key(iv: dict) -> tuple:
    """What makes two requests' interventions the same, so they can share one hooked context."""
    mode = iv.get("mode", "none")
    if mode == "none":
        return ("none",)
    c = iv.get("coefficient")
    return (mode, dir_key(iv), tuple(sorted({int(x) for x in iv["layers"]})), None if c is None else float(c),
            _fractions(iv), _means(iv))


_PARTIAL = []


def partial_cut_class():
    """PartialCut, an anatta.dial.Intervention whose apply is h <- h - f (h.d) d (made on first use, since anatta is
    imported lazily). Its mode stays "ablate", so the hook checks treat it as a cut."""
    if not _PARTIAL:
        from dataclasses import dataclass

        from anatta.dial import Intervention

        @dataclass
        class PartialCut(Intervention):
            fraction: float = 1.0

            def apply(self, h: torch.Tensor) -> torch.Tensor:
                d = self.direction.to(device=h.device, dtype=h.dtype)
                return h - self.fraction * (h @ d).unsqueeze(-1) * d

        _PARTIAL.append(PartialCut)
    return _PARTIAL[0]


def build_interventions(mode: str, vectors: dict, layers, coefficient: float | None = None,
                        fractions: dict | None = None, means: dict | None = None) -> list:
    """anatta.dial Interventions for one request: none -> [], ablate -> mode "ablate" at each layer with that layer's
    vector, add -> mode "steer" with alpha = coefficient at each layer. `vectors` maps layer -> tensor. `fractions`
    {layer: f} (ablate only, 0 < f <= 1): a layer with f below 1 gets PartialCut(f); f = 1 is the plain ablate.
    The split arms take `means` {layer: m} with a constant at every listed layer (the module docstring):
    mean_ablate -> mode "cap" with lo = hi = m (h.d set to m), shift_only -> "steer" with alpha = -m, shift_reverse ->
    "steer" with alpha = +m."""
    from anatta.dial import Intervention

    if mode == "none":
        return []
    layers = sorted({int(x) for x in layers})
    if not layers:
        raise ValueError(f"{mode} needs at least one layer")
    missing = [x for x in layers if x not in vectors]
    if missing:
        raise ValueError(f"the direction has no vector at layers {missing}")
    if fractions and mode != "ablate":
        raise ValueError("fractions belong to ablate (a partial cut)")
    if means and mode not in SPLIT_MODES:
        raise ValueError(f"means belong to the split arms ({', '.join(SPLIT_MODES)})")
    if mode in SPLIT_MODES:
        if coefficient is not None:
            raise ValueError(f"{mode} takes per-layer means, not a coefficient")
        m = {int(x): float(v) for x, v in (means or {}).items()}
        absent = [x for x in layers if x not in m]
        if absent:
            raise ValueError(f"{mode} needs a mean at every layer; missing {absent}")
        bad = {x: m[x] for x in layers if not math.isfinite(m[x])}
        if bad:
            raise ValueError(f"{mode}: means must be finite; got {bad}")
        if mode == "mean_ablate":
            return [Intervention(layer=x, direction=vectors[x], mode="cap", lo=m[x], hi=m[x]) for x in layers]
        sign = -1.0 if mode == "shift_only" else 1.0
        return [Intervention(layer=x, direction=vectors[x], mode="steer", alpha=sign * m[x]) for x in layers]
    if mode == "ablate":
        f = {int(x): float(v) for x, v in (fractions or {}).items()}
        bad = {x: v for x, v in f.items() if not 0.0 < v <= 1.0}
        if bad:
            raise ValueError(f"cut fractions must be in (0, 1]; got {bad}")
        cls = partial_cut_class() if any(v < 1.0 for v in f.values()) else None
        return [Intervention(layer=x, direction=vectors[x], mode="ablate") if f.get(x, 1.0) >= 1.0 else
                cls(layer=x, direction=vectors[x], mode="ablate", fraction=f[x]) for x in layers]
    if mode == "add":
        if coefficient is None:
            raise ValueError("add needs a coefficient")
        return [Intervention(layer=x, direction=vectors[x], mode="steer", alpha=float(coefficient)) for x in layers]
    raise ValueError(f"unknown mode {mode!r} (none, add, ablate, {', '.join(SPLIT_MODES)})")


def plan_passes(lengths: list[int], max_new, token_budget: int = TOKEN_BUDGET, max_rows: int = 32) -> list[list[int]]:
    """Generation passes as lists of row indices, shortest prompts first (ties keep their order). A pass takes the
    next row while (its rows + 1) x (its longest prompt + its largest max_new) stays within token_budget and it has
    fewer than max_rows rows; a row that alone exceeds the budget runs alone. `max_new` is an int or one per row.
    With an unbounded budget this is the old plan: chunks of max_rows in order of length."""
    mn = [int(max_new)] * len(lengths) if isinstance(max_new, (int, float)) else [int(x) for x in max_new]
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    passes, cur, width, new = [], [], 0, 0
    for i in order:
        w, n = max(width, int(lengths[i])), max(new, mn[i])
        if cur and (len(cur) >= max_rows or (len(cur) + 1) * (w + n) > token_budget):
            passes.append(cur)
            cur, w, n = [], int(lengths[i]), mn[i]
        cur.append(i)
        width, new = w, n
    if cur:
        passes.append(cur)
    return passes


def cut_diff(a: list, b: list) -> float:
    """The largest difference between two Intervention lists' unit directions, or inf if their layers, modes or alphas
    differ. Normalizing the same vector stored strided (as ego.pt's are) or contiguous (as it arrives through numpy)
    can differ in the last float32 bit, so a cut rebuilt from the same file differs by about 1e-7."""
    def sig(x):
        return x.layer, x.mode, x.alpha, getattr(x, "fraction", 1.0), x.lo, x.hi

    if len(a) != len(b) or any(sig(x) != sig(y) for x, y in zip(a, b)):
        return float("inf")
    return max((float((x.direction.cpu() - y.direction.cpu()).abs().max()) for x, y in zip(a, b)), default=0.0)


def same_cut(a: list, b: list, atol: float = 1e-6) -> bool:
    """The same cut: the same layers in the same order, the same modes and alphas, and directions equal to float32
    rounding (atol)."""
    return cut_diff(a, b) <= atol


# ----------------------------------------------------------------------------------------------- sampling


class RowSampler(LogitsProcessor):
    """Picks every row's next token itself and hands generate() a one-hot score (0 at the pick, -inf elsewhere), so
    generate()'s own sampling can only return the pick. Per row: temperature (0 = greedy), top-k, top-p, and one
    uniform draw from the row's generator, inverted through the CDF in float64. Custom processors run before
    generate()'s warpers, and a one-hot passes any warper unchanged."""

    def __init__(self, temps, top_ps, top_ks, seeds, device):
        self.temps = torch.tensor([float(t) for t in temps], dtype=torch.float32, device=device)
        self.top_ps = torch.tensor([float(p) for p in top_ps], dtype=torch.float32, device=device)
        self.top_ks = [int(k) if k else 0 for k in top_ks]
        self.gens = [torch.Generator(device=device).manual_seed(int(s)) for s in seeds]

    def __call__(self, input_ids, scores):
        x = scores.float()
        vocab = x.shape[-1]
        logits = x / self.temps.clamp_min(1e-6)[:, None]
        for k in sorted(set(self.top_ks) - {0}):
            rows = torch.tensor([i for i, kk in enumerate(self.top_ks) if kk == k], device=x.device)
            sub = logits[rows]
            kth = sub.topk(min(k, vocab), dim=-1).values[:, -1:]
            logits[rows] = sub.masked_fill(sub < kth, float("-inf"))
        probs = torch.softmax(logits, dim=-1).double()
        if bool((self.top_ps < 1).any()):
            sp, si = probs.sort(dim=-1, descending=True)
            keep = (sp.cumsum(-1) - sp) < self.top_ps.double()[:, None]
            probs = torch.zeros_like(probs).scatter(-1, si, sp * keep)
        u = torch.stack([torch.rand((), generator=g, device=x.device) for g in self.gens]).double()
        cdf = probs.cumsum(-1)
        pick = torch.searchsorted(cdf, (u * cdf[:, -1]).unsqueeze(-1), right=True).squeeze(-1).clamp(max=vocab - 1)
        pick = torch.where(self.temps <= 0, x.argmax(-1), pick)
        out = torch.full_like(scores, float("-inf"))
        return out.scatter_(1, pick.unsqueeze(-1), 0.0)


# ----------------------------------------------------------------------------------------------- the service


class Service:
    """One loaded model, its hooks and checks. The Modal class (modal_bench.Bench) holds one; the CPU tests and the
    client's local mode hold one on a tiny model. Payloads and results are plain dicts, lists and numpy arrays."""

    def __init__(self, model, tok, *, model_key: str = "custom", model_id: str | None = None,
                 revision: str | None = None, gen_batch: int = 32, load_seconds: float | None = None,
                 token_budget: int = TOKEN_BUDGET):
        from anatta.models import chat_ids, get_layers

        self.model, self.tok = model.eval(), tok
        self.model_key, self.revision = model_key, revision
        self.model_id = model_id or getattr(model.config, "_name_or_path", "?")
        self.n_layers = len(get_layers(model))
        self.dev = next(model.parameters()).device
        cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        self.hidden = int(cfg.hidden_size)
        gc = model.generation_config
        stop = gc.eos_token_id
        self.stop = sorted({tok.eos_token_id, *(stop if isinstance(stop, list) else [stop])} - {None})
        self.default_top_k = int(gc.top_k) if gc.top_k else None
        self.gen_batch, self.load_seconds, self.started_at = gen_batch, load_seconds, time.time()
        self.token_budget = int(token_budget)
        self.base_hooks = count_hooks(model)
        self.probe = chat_ids(tok, [{"role": "user", "content": PROBE_TEXT}]).to(self.dev)
        self.fingerprint = self._probe_logits()
        self._dirs: dict = {}
        self._neutral: dict = {}

    @classmethod
    def load(cls, model_key: str, hub: str = "/cache/hf/hub", gen_batch: int = 32) -> Service:
        """The model through anatta.models.load (bf16 on the GPU, with its vision-checkpoint fallback for Ministral 3),
        as modal_common.load_model loads it, from the pinned snapshot folder."""
        from anatta.models import load

        t0 = time.time()
        model, tok = load(snapshot_path(model_key, hub))
        return cls(model, tok, model_key=model_key, model_id=MODELS[model_key]["hf_id"],
                   revision=MODELS[model_key]["revision"], gen_batch=gen_batch, load_seconds=time.time() - t0)

    # ------------------------------------------------------------------------------------------- reports

    def container(self) -> dict:
        return {"container_id": os.environ.get("MODAL_TASK_ID"), "started_at": self.started_at,
                "uptime_s": round(time.time() - self.started_at, 1)}

    def info(self) -> dict:
        import transformers

        gc = self.model.generation_config
        return {"model_key": self.model_key, "model": self.model_id, "revision": self.revision,
                "class": type(self.model).__name__, "n_layers": self.n_layers, "hidden": self.hidden,
                "dtype": str(next(self.model.parameters()).dtype), "device": str(self.dev),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "load_seconds": None if self.load_seconds is None else round(self.load_seconds, 1),
                "torch": torch.__version__, "transformers": transformers.__version__,
                "generation_defaults": {"temperature": gc.temperature, "top_p": gc.top_p, "top_k": gc.top_k,
                                        "repetition_penalty": getattr(gc, "repetition_penalty", None)},
                "stop_ids": self.stop, "base_hooks": self.base_hooks, "gen_batch": self.gen_batch,
                "token_budget": self.token_budget, **self.container()}

    # ------------------------------------------------------------------------------------------- checks

    @torch.no_grad()
    def _probe_logits(self):
        return self.model(input_ids=self.probe).logits[0, -1].float().cpu()

    @torch.no_grad()
    def _capture(self, layers: list[int]) -> dict:
        from anatta.probe import capture

        with capture(self.model, layers) as store:
            self.model(input_ids=self.probe)
        return {x: store[x][0].float() for x in layers}

    def check_clean(self) -> dict:
        """HookLeak unless the hook count is back to its value at load and the probe's logits match (to 1e-3 of
        their largest magnitude)."""
        n = count_hooks(self.model)
        diff = float((self._probe_logits() - self.fingerprint).abs().max())
        tol = 1e-3 * max(1.0, float(self.fingerprint.abs().max()))
        if n != self.base_hooks or diff > tol:
            raise HookLeak(f"after the group: {n} hooks (at load {self.base_hooks}); the probe's logits moved by "
                           f"{diff:.3g} (tolerance {tol:.3g})")
        return {"hooks": n, "probe_max_diff": diff}

    def _attach(self, iv: dict, ivs: list, base: dict) -> dict:
        """Inside the context: the probe's residual at the first hooked layer against its value without hooks. Ablate:
        the largest |h.d| before and after (after should be near 0; for a partial cut of fraction f, near (1 - f) times
        before, and "fraction" and "after_over_before" are added). Add, shift_only and shift_reverse: the largest error
        of (after - before) against c d, over |c| (near 0). mean_ablate: the largest |h.d - m| before and after (after
        near 0, up to bf16 rounding)."""
        first = ivs[0]
        d = first.direction.to(self.dev)
        h0, h1 = base[first.layer], self._capture([first.layer])[first.layer]
        if iv["mode"] == "mean_ablate":
            m = float(first.lo)
            return {"layer": first.layer, "mean": m, "dev_before": float((h0 @ d - m).abs().max()),
                    "dev_after": float((h1 @ d - m).abs().max())}
        if iv["mode"] == "ablate":
            out = {"layer": first.layer, "proj_before": float((h0 @ d).abs().max()),
                   "proj_after": float((h1 @ d).abs().max())}
            f = getattr(first, "fraction", 1.0)
            if f < 1.0:
                out.update({"fraction": f, "after_over_before": out["proj_after"] / max(out["proj_before"], 1e-12)})
            return out
        c = float(first.alpha)
        err = float(((h1 - h0) - c * d).abs().max()) / max(abs(c), 1e-8)
        out = {"layer": first.layer, "coefficient": c, "rel_err": err}
        if iv["mode"] in SPLIT_MODES:  # rel_err's elementwise max is bf16 rounding of the residual's largest entries
            # when c is small; the shift along d itself, (after - before).d against c at every position, is the check
            out["proj_err"] = float((((h1 - h0) @ d) - c).abs().max()) / max(abs(c), 1e-8)
        return out

    # ------------------------------------------------------------------------------------------- generation

    def _direction(self, sha: str, d: dict) -> dict:
        if sha not in self._dirs:
            vecs = {int(x): torch.as_tensor(v, dtype=torch.float32).flatten() for x, v in d["vectors"].items()}
            bad = sorted(x for x, v in vecs.items() if v.numel() != self.hidden)
            if bad:
                raise ValueError(f"direction {d.get('name')}: vectors at layers {bad} are not {self.hidden}-dimensional")
            self._dirs[sha] = vecs
        return self._dirs[sha]

    def interventions(self, iv: dict, dirs: dict) -> list:
        mode = iv.get("mode", "none")
        if mode == "none":
            return []
        key = dir_key(iv)
        if key not in dirs:
            raise ValueError(f"direction {iv.get('direction')} ({key}) was not sent with the batch")
        bad = [x for x in iv["layers"] if not 0 <= int(x) < self.n_layers]
        if bad:
            raise ValueError(f"layers {bad} are outside 0-{self.n_layers - 1}")
        return build_interventions(mode, dirs[key], iv["layers"], iv.get("coefficient"), iv.get("fractions"),
                                   iv.get("means"))

    def sampling(self, r: dict, token_budget: int | None = None) -> dict:
        return {"messages": r["messages"], "max_tokens": int(r["max_tokens"]), "seed": int(r["seed"]),
                "temperature": float(r.get("temperature", 0.7)), "top_p": float(r.get("top_p") or 1.0),
                "top_k": int(r["top_k"]) if r.get("top_k") else self.default_top_k,
                "token_budget": int(token_budget or self.token_budget)}

    @torch.no_grad()
    def generate_rows(self, reqs: list[dict]) -> list[dict]:
        """Completions for requests (sampling() form) under whatever hooks are on: passes from plan_passes (shortest
        prompts first, within the requests' token budget and gen_batch rows), left padding, RowSampler, each row cut
        at its first stop token or its own max_tokens. Each row says which pass it ran in, with that pass's rows and
        width."""
        from anatta.models import chat_ids

        tok, stop = self.tok, set(self.stop)
        prompts = [chat_ids(tok, r["messages"])[0] for r in reqs]
        budget = min(int(r.get("token_budget") or self.token_budget) for r in reqs) if reqs else self.token_budget
        plan = plan_passes([len(p) for p in prompts], [r["max_tokens"] for r in reqs], budget, self.gen_batch)
        out = [None] * len(reqs)
        for pi, idx in enumerate(plan):
            width = max(len(prompts[i]) for i in idx)
            ids = torch.full((len(idx), width), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for j, i in enumerate(idx):
                ids[j, width - len(prompts[i]):], mask[j, width - len(prompts[i]):] = prompts[i], 1
            sampler = RowSampler([reqs[i]["temperature"] for i in idx], [reqs[i]["top_p"] for i in idx],
                                 [reqs[i]["top_k"] for i in idx], [reqs[i]["seed"] for i in idx], self.dev)
            gen = self.model.generate(input_ids=ids.to(self.dev), attention_mask=mask.to(self.dev),
                                      max_new_tokens=max(reqs[i]["max_tokens"] for i in idx), do_sample=True,
                                      temperature=1.0, top_k=0, top_p=1.0,
                                      logits_processor=LogitsProcessorList([sampler]),
                                      pad_token_id=tok.pad_token_id, use_cache=True)
            comp = gen[:, width:].tolist()
            for j, i in enumerate(idx):
                row = comp[j][:reqs[i]["max_tokens"]]
                end = next((t + 1 for t, x in enumerate(row) if x in stop), None)
                kept = row if end is None else row[:end]
                out[i] = {"text": tok.decode(kept, skip_special_tokens=True), "prompt_tokens": len(prompts[i]),
                          "completion_tokens": len(kept), "finish_reason": "length" if end is None else "stop",
                          "top_k": reqs[i]["top_k"], "pass": pi, "pass_rows": len(idx), "pass_width": width}
        return out

    def generate(self, payload: dict) -> dict:
        """payload = {"requests": [{rid, messages, temperature, max_tokens, seed, top_p?, top_k?, intervention}],
        "directions": {key: {"name", "vectors": {layer: array}}}, "token_budget"?}. intervention = {"mode":
        none|add|ablate|mean_ablate|shift_only|shift_reverse, "direction", "tensor_sha256" (the key; old payloads key by
        "sha256"), "sha256", "layers", "coefficient", "fractions"?, "means"? (the split arms')}. Returns {"results": [per
        request, in order], "meta"}; each group's meta lists its passes as [rows, width]."""
        from anatta.dial import intervene

        t0 = time.time()
        reqs = payload["requests"]
        budget = int(payload.get("token_budget") or self.token_budget)
        dirs = {sha: self._direction(sha, d) for sha, d in (payload.get("directions") or {}).items()}
        groups: dict = {}
        for i, r in enumerate(reqs):
            groups.setdefault(iv_key(r["intervention"]), []).append(i)
        results, gmeta = [None] * len(reqs), []
        for gi, idx in enumerate(groups.values()):
            iv = reqs[idx[0]]["intervention"]
            ivs = self.interventions(iv, dirs)
            tg = time.time()
            base = self._capture([ivs[0].layer]) if ivs else None
            try:
                with intervene(self.model, ivs):
                    attach = self._attach(iv, ivs, base) if ivs else None
                    if ivs and count_hooks(self.model) == self.base_hooks:
                        raise HookLeak("the intervention's hooks did not attach")
                    rows = self.generate_rows([self.sampling(reqs[i], budget) for i in idx])
            finally:
                clean = self.check_clean()
            for i, row in zip(idx, rows):
                results[i] = {"rid": reqs[i].get("rid", i), "group": gi, **row}
            passes = sorted({(r["pass"], r["pass_rows"], r["pass_width"]) for r in rows if "pass" in r})
            gmeta.append({"intervention": {k: iv.get(k) for k in ("mode", "direction", "tensor_sha256", "sha256",
                                                                     "layers", "coefficient", "fractions", "means")},
                          "n": len(idx), "seconds": round(time.time() - tg, 3), "hooked_layers": len(ivs),
                          "attach": attach, "clean": clean, "completion_tokens": sum(r["completion_tokens"] for r in rows),
                          "passes": [[n, w] for _, n, w in passes]})
        return {"results": results, "meta": {"seconds": round(time.time() - t0, 3), "groups": gmeta,
                                             "token_budget": budget, **self.container()}}

    # ------------------------------------------------------------------------------------------- random_matched

    def _template_ref(self) -> list[int]:
        from anatta.models import chat_ids

        for filler in ("", "x"):
            try:
                return chat_ids(self.tok, [{"role": "user", "content": filler}])[0].tolist()
            except Exception:  # noqa: BLE001 - some templates refuse an empty message
                continue
        return []

    @torch.no_grad()
    def neutral_activations(self, prompts: list[str], layers: list[int], outlier: float = 10.0):
        """The residual stream at `layers` over the user-content tokens of each neutral prompt (the chat template's
        own tokens left out, found as the common prefix and suffix with a near-empty message), with tokens whose norm
        exceeds `outlier` times the layer's median dropped. Cached per prompt set and layers."""
        from anatta.models import chat_ids
        from anatta.probe import capture

        key = (hashlib.sha256(json.dumps(prompts).encode()).hexdigest(), tuple(layers))
        if key in self._neutral:
            return self._neutral[key]
        ref = self._template_ref()
        rows: dict = {x: [] for x in layers}
        spans = []
        for p in prompts:
            ids = chat_ids(self.tok, [{"role": "user", "content": p}])
            a, b = content_span(ids[0].tolist(), ref)
            spans.append(b - a)
            with capture(self.model, layers) as store:
                self.model(input_ids=ids.to(self.dev))
            for x in layers:
                rows[x].append(store[x][0, a:b].float())
        acts, dropped = {}, {}
        for x in layers:
            h = torch.cat(rows[x])
            norms = h.norm(dim=-1)
            keep = norms <= outlier * norms.median()
            acts[x], dropped[x] = h[keep], int((~keep).sum())
        info = {"n_prompts": len(prompts), "n_tokens": int(sum(spans)), "positions": "user-content tokens",
                "outlier_rule": f"norm > {outlier} x the layer's median dropped",
                "dropped": {str(x): n for x, n in dropped.items() if n}}
        self._neutral[key] = (acts, info)
        return acts, info

    def random_matched(self, payload: dict) -> dict:
        """payload = {"target": {"name", "tensor_sha256"?, "sha256", "vectors"}, "seeds": [...], "layers": [...]
        (default: the target's), "neutral_prompts": [...], "neutral_sha256": str, "tol": 0.10}. Returns {"directions":
        [{"seed", "vectors", "made"}], "meta"}."""
        t0 = time.time()
        target = payload["target"]
        tv = self._direction(dir_key(target), target)
        layers = sorted(int(x) for x in (payload.get("layers") or tv))
        acts, ninfo = self.neutral_activations(list(payload["neutral_prompts"]), layers)
        out = []
        for seed in payload["seeds"]:
            vecs, made = random_matched({x: acts[x] for x in layers}, {x: tv[x] for x in layers}, int(seed),
                                        tol=float(payload.get("tol", 0.10)))
            made.update({"target": {"name": target.get("name"), "sha256": target["sha256"],
                                    "tensor_sha256": target.get("tensor_sha256")},
                         "neutral": {**ninfo, "sha256": payload.get("neutral_sha256")},
                         "model": self.model_id, "revision": self.revision})
            out.append({"seed": int(seed), "vectors": {x: v.numpy() for x, v in vecs.items()}, "made": made})
        return {"directions": out, "meta": {"seconds": round(time.time() - t0, 3), **self.container()}}

    # ------------------------------------------------------------------------------------------- the controls' raw material

    def measure(self, payload: dict) -> dict:
        """payload = {"directions": {key: {"name", "vectors"}}, "layers": [...], "neutral_prompts": [...],
        "neutral_sha256": str}. For each direction's unit vector at each listed layer it has: the variance of the
        projections over the neutral tokens (random_matched's activations and its variance, so a ratio of two of these
        is what random_matched matches to 1), their mean and their mean square. Returns {"stats": {key: {layer:
        {"var", "mean_proj", "second_moment"}}}, "neutral", "meta"}."""
        t0 = time.time()
        layers = sorted({int(x) for x in payload["layers"]})
        acts, ninfo = self.neutral_activations(list(payload["neutral_prompts"]), layers)
        stats = {}
        for key, d in payload["directions"].items():
            vecs = self._direction(key, d)
            per = {}
            for x in layers:
                if x not in vecs:
                    continue
                h = acts[x].to(torch.float64)
                u = vecs[x].to(h.device, torch.float64)
                p = h @ (u / u.norm())
                per[x] = {"var": float(p.var(unbiased=True)), "mean_proj": float(p.mean()),
                          "second_moment": float(p.pow(2).mean())}
            stats[key] = per
        return {"stats": stats, "neutral": {**ninfo, "sha256": payload.get("neutral_sha256")},
                "meta": {"seconds": round(time.time() - t0, 3), "model": self.model_id, "revision": self.revision,
                         **self.container()}}

    # ------------------------------------------------------------------------------------------- the split arms

    @torch.no_grad()
    def _neutral_pass(self, prompts: list[str], layers: list[int], units: dict, ivs=()) -> tuple[dict, dict]:
        """One pass over the neutral prompts, one prompt per forward as neutral_activations reads them, with `ivs`
        (anatta Interventions) on the model for the pass only: for each of `layers`, the residual's projection on
        units[layer] (float64) over every prompt's user-content tokens, prompts in order, and those tokens' norms.
        A hooked layer is read after its own hook (capture's hooks go on after the intervention's)."""
        from anatta.dial import intervene
        from anatta.models import chat_ids
        from anatta.probe import capture

        ref = self._template_ref()
        proj, norms = {x: [] for x in layers}, {x: [] for x in layers}
        try:
            with intervene(self.model, list(ivs)):
                for p in prompts:
                    ids = chat_ids(self.tok, [{"role": "user", "content": p}])
                    a, b = content_span(ids[0].tolist(), ref)
                    with capture(self.model, layers) as store:
                        self.model(input_ids=ids.to(self.dev))
                    for x in layers:
                        h = store[x][0, a:b].float()
                        norms[x].append(h.norm(dim=-1).cpu())
                        proj[x].append((h.double() @ units[x].to(h.device)).cpu())
        finally:
            self.check_clean()
        return {x: torch.cat(v) for x, v in proj.items()}, {x: torch.cat(v) for x, v in norms.items()}

    def split_means(self, payload: dict) -> dict:
        """The split arms' per-layer constants for one direction on the neutral set, and what each arm does there.
        payload = {"direction": {"name", "vectors"}, "key": its tensor hash, "layers": [...], "neutral_prompts",
        "neutral_sha256", "modes": [some of SPLIT_MODES] (default all)}.

        Tokens: neutral_activations' (each prompt's user-content tokens; a token whose norm passes OUTLIER times the
        layer's median on the uncut pass is left out, and the same tokens are read in every pass). base: the uncut
        mean mu_l, sd, variance (n - 1) and mean square of h.d per layer, measure()'s numbers. consts: for each mode,
        layer by layer from the lowest, a pass with the arm's hooks on the layers below l reads the mean m of h.d at
        l; mean_ablate and shift_only take m_l = m, shift_reverse m_l = 2 mu_l - m (so h.d averages 2 mu_l after its
        hook). The lowest layer's three constants are mu_l. realized: one pass per arm with every hook on, h.d's mean
        and sd at every layer, for the three arms, the plain cut ("ablate") and "shift_only_independent" (shift_only
        with mu_l at every layer, what the per-layer formula with uncut means does). Returns {"layers", "base",
        "consts": {mode: {layer: m}}, "realized": {arm: {layer: {"mean", "sd"}}}, "neutral", "meta"}."""
        t0 = time.time()
        key = payload["key"]
        vecs = self._direction(key, payload["direction"])
        layers = sorted({int(x) for x in payload["layers"]})
        bad = [x for x in layers if x not in vecs or not 0 <= x < self.n_layers]
        if not layers or bad:
            raise ValueError(f"split_means: layers {bad or layers} are not in the direction or the model")
        modes = list(payload.get("modes") or SPLIT_MODES)
        if any(m not in SPLIT_MODES for m in modes):
            raise ValueError(f"split_means: modes are some of {SPLIT_MODES}")
        if count_hooks(self.model) != self.base_hooks:
            raise HookLeak("hooks are on the model before split_means")
        prompts = list(payload["neutral_prompts"])
        units = {x: vecs[x].double() / vecs[x].double().norm() for x in layers}
        p0, n0 = self._neutral_pass(prompts, layers, units)
        keep = {x: n0[x] <= OUTLIER * n0[x].median() for x in layers}

        def moments(p, x):
            q = p[x][keep[x]]
            return float(q.mean()), float(q.std(unbiased=True))

        base = {}
        for x in layers:
            q = p0[x][keep[x]]
            base[x] = {"mean": float(q.mean()), "sd": float(q.std(unbiased=True)), "var": float(q.var(unbiased=True)),
                       "second_moment": float(q.pow(2).mean()), "n_tokens": int(keep[x].sum())}
        consts, passes = {}, 1
        for mode in modes:
            c: dict = {}
            for i, x in enumerate(layers):
                if i == 0:
                    m = base[x]["mean"]
                else:
                    p, _ = self._neutral_pass(prompts, [x], units, build_interventions(mode, vecs, layers[:i], means=c))
                    m, passes = moments(p, x)[0], passes + 1
                c[x] = 2.0 * base[x]["mean"] - m if mode == "shift_reverse" else m
            consts[mode] = c
        arms = {mode: build_interventions(mode, vecs, layers, means=consts[mode]) for mode in modes}
        arms["ablate"] = build_interventions("ablate", vecs, layers)
        arms["shift_only_independent"] = build_interventions("shift_only", vecs, layers,
                                                             means={x: base[x]["mean"] for x in layers})
        realized = {}
        for name, ivs in arms.items():
            p, _ = self._neutral_pass(prompts, layers, units, ivs)
            passes += 1
            realized[name] = {x: dict(zip(("mean", "sd"), moments(p, x))) for x in layers}
        info = {"n_prompts": len(prompts), "positions": "user-content tokens",
                "outlier_rule": f"norm > {OUTLIER} x the layer's median on the uncut pass dropped, the same tokens in "
                                f"every pass", "sha256": payload.get("neutral_sha256")}
        return {"layers": layers, "base": base, "consts": consts, "realized": realized, "neutral": info,
                "meta": {"method": SPLIT_METHOD, "passes": passes, "seconds": round(time.time() - t0, 3),
                         "model": self.model_id, "revision": self.revision, **self.container()}}

    @torch.no_grad()
    def response_means(self, payload: dict) -> dict:
        """payload = {"rows": [{"id", "system", "user", "seed"}], "layers": [...], "max_tokens", "temperature",
        "top_p", "top_k"?, "include_system": False, "token_budget"?}. With no hooks on the model, each row's reply is
        sampled with its system prompt in context (generate_rows, the row's own seed); then
        anatta.probe.response_activations reads the mean residual over the reply's tokens at `layers`, without the
        system prompt unless include_system: probe.extract's reading, the one the lab's trait directions were made
        with. Returns {"replies": [{"id", "text", "completion_tokens", "finish_reason"}], "means": {layer: float32 array
        [rows, hidden]}, "meta"}."""
        from anatta.probe import response_activations

        t0 = time.time()
        rows = payload["rows"]
        layers = sorted({int(x) for x in payload["layers"]})
        bad = [x for x in layers if not 0 <= x < self.n_layers]
        if bad:
            raise ValueError(f"layers {bad} are outside 0-{self.n_layers - 1}")
        if count_hooks(self.model) != self.base_hooks:
            raise HookLeak("hooks are on the model before response_means")
        reqs = [self.sampling({"messages": [{"role": "system", "content": r["system"]},
                                            {"role": "user", "content": r["user"]}],
                               "max_tokens": payload["max_tokens"], "seed": r["seed"],
                               "temperature": payload.get("temperature", 0.7), "top_p": payload.get("top_p", 1.0),
                               "top_k": payload.get("top_k")}, payload.get("token_budget")) for r in rows]
        try:
            replies = self.generate_rows(reqs)
            means = {x: [] for x in layers}
            for r, rep in zip(rows, replies):
                msgs = ([{"role": "system", "content": r["system"]}] if payload.get("include_system") else []) + \
                       [{"role": "user", "content": r["user"]}]
                acts = response_activations(self.model, self.tok, msgs, rep["text"], layers)
                for x in layers:
                    means[x].append(acts[x].float().cpu())
        finally:
            clean = self.check_clean()
        out = [{"id": r["id"], "text": rep["text"], "completion_tokens": rep["completion_tokens"],
                "finish_reason": rep["finish_reason"]} for r, rep in zip(rows, replies)]
        return {"replies": out, "means": {x: torch.stack(v).numpy() for x, v in means.items()},
                "meta": {"seconds": round(time.time() - t0, 3), "clean": clean, "model": self.model_id,
                         "revision": self.revision, "include_system": bool(payload.get("include_system")),
                         "passes": len({rep["pass"] for rep in replies}), **self.container()}}


def content_span(ids: list[int], ref: list[int]) -> tuple[int, int]:
    """[start, end) of the message content in a chat's ids, from the common prefix and suffix with `ref` (the same
    template around a near-empty message). Falls back to everything after the first token."""
    if not ref:
        return 1, len(ids)
    pre = 0
    while pre < min(len(ids), len(ref)) and ids[pre] == ref[pre]:
        pre += 1
    suf = 0
    while suf < min(len(ids), len(ref)) - pre and ids[-1 - suf] == ref[-1 - suf]:
        suf += 1
    start, end = pre, len(ids) - suf
    return (start, end) if end > start else (1, len(ids))


def random_matched(acts: dict, target: dict, seed: int, tol: float = 0.10, margin: float = 1.5):
    """A random control direction matched in strength to `target`: one unit vector per layer, orthogonal to the
    target's, whose variance over the neutral tokens equals the target's.

    At each layer: H = the neutral tokens' activations (rows), C their covariance, v the target's unit vector and
    s = v'Cv its variance. The control u has u'Cu = s, u orthogonal to v and |u| = 1. Construction (pc_mix_v1):
      1. a = a Gaussian vector z1 projected onto the top-k principal subspace of H, with v projected out, normalized.
         k is the widest subspace whose mean eigenvalue is at least margin * s, narrowed until a'Ca > s.
      2. b = a Gaussian vector z2 with that subspace, v and a projected out, normalized, so its variance is low. If it
         is still at or above s, H's null space is used instead (or its last principal axis when H has full rank).
      3. u = cos(t) a + sin(t) b, with t found by bisection so that u'Cu = s. Since a and b are orthogonal, |u| = 1.
    z1 and z2 are drawn once per seed and shared by every layer, so neighbouring layers' controls are correlated where
    their principal subspaces overlap, as a direction's own per-layer vectors are.

    Returns (vectors {layer: float32 tensor}, made). made records, per layer: k, the rank of H, the target's variance
    and the control's and their ratio, the angle t, a's and b's variances, cos(u, v), both mean projections, the share
    of variance in the top-k subspace, and the ratio a plain random unit vector (the old control) would have had.
    Raises MatchError if a layer cannot be matched within `tol`."""
    layers = sorted(int(x) for x in target)
    dim = target[layers[0]].numel()
    g = torch.Generator().manual_seed(int(seed))
    z1 = torch.randn(dim, generator=g, dtype=torch.float64)
    z2 = torch.randn(dim, generator=g, dtype=torch.float64)
    vectors, per = {}, {}
    n_tokens = None
    for x in layers:
        h = acts[x].to(torch.float64)
        dev, n = h.device, h.shape[0]
        n_tokens = n if n_tokens is None else min(n_tokens, n)
        if n < 3:
            raise MatchError(f"layer {x}: {n} neutral tokens")
        mu = h.mean(0)
        hc = h - mu

        def var(u):
            return float((hc @ u).pow(2).sum() / (n - 1))

        v = target[x].to(dev, torch.float64).flatten()
        v = v / v.norm()
        s = var(v)
        if not s > 0:
            raise MatchError(f"layer {x}: the target has no variance on the neutral set")
        _, sv, vh = torch.linalg.svd(hc, full_matrices=False)
        lam = sv.pow(2) / (n - 1)
        r = int((lam > lam[0] * 1e-12).sum())
        means = torch.cumsum(lam[:r], 0) / torch.arange(1, r + 1, device=dev, dtype=torch.float64)
        wide = [k for k in range(r, 0, -1) if means[k - 1] >= margin * s]
        x1, x2 = z1.to(dev), z2.to(dev)
        a = k = None
        for kk in range(wide[0] if wide else r, 0, -1):
            e = vh[:kk]
            cand = e.T @ (e @ x1)
            cand = cand - (cand @ v) * v
            if cand.norm() > 1e-12 and var(cand / cand.norm()) > s:
                a, k = cand / cand.norm(), kk
                break
        if a is None:
            raise MatchError(f"layer {x}: no direction orthogonal to the target reaches its variance {s:.4g}")

        def orth(b):
            b = b - (b @ v) * v
            b = b - (b @ a) * a
            if b.norm() < 1e-12:
                raise MatchError(f"layer {x}: the partner direction vanished")
            return b / b.norm()

        b = orth(x2 - vh[:k].T @ (vh[:k] @ x2))
        if var(b) >= s:
            b = orth(x2 - vh[:r].T @ (vh[:r] @ x2)) if r < dim else orth(vh[r - 1].clone())
        if var(b) >= s:
            raise MatchError(f"layer {x}: no low-variance partner below the target's variance {s:.4g}")
        lo, hi = 0.0, math.pi / 2
        for _ in range(80):
            mid = (lo + hi) / 2
            if var(math.cos(mid) * a + math.sin(mid) * b) > s:
                lo = mid
            else:
                hi = mid
        t = (lo + hi) / 2
        u = math.cos(t) * a + math.sin(t) * b
        u = u / u.norm()
        got = var(u)
        if abs(got / s - 1) > tol:
            raise MatchError(f"layer {x}: variance ratio {got / s:.3f} is outside 1 +- {tol}")
        vectors[x] = u.to(torch.float32).cpu()
        per[str(x)] = {"k": k, "rank": r, "target_var": s, "var": got, "ratio": got / s, "theta": t,
                       "var_a": var(a), "var_b": var(b), "cos_target": float(u @ v),
                       "target_mean_proj": float(mu @ v), "mean_proj": float(mu @ u),
                       "top_k_share": float(lam[:k].sum() / lam.sum()),
                       "plain_random_ratio": var(x1 / x1.norm()) / s}
    ratios = [p["ratio"] for p in per.values()]
    made = {"method": METHOD, "seed": int(seed), "tol": tol, "margin": margin, "n_tokens": n_tokens,
            "ratio_range": [min(ratios), max(ratios)],
            "rule": "variance over the neutral tokens matched per layer; orthogonal to the target; unit norm",
            "layers": per}
    return vectors, made
