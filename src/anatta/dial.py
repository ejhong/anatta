"""The dial: inference-time interventions on residual-stream directions.

Three modes:

    steer   h <- h + alpha * d                      (push along d)
    ablate  h <- h - (h.d) d                        (remove the d component)
    cap     h <- h + (clamp(h.d, lo, hi) - h.d) d   (bound the d component; Assistant-Axis style)

`d` is always unit-normalized. Branch A of the protocol ("the surgical clamp") is
`cap` with hi set near the base model's typical non-egoic projection.

Each intervention acts at one site of one transformer block (`site`):

    output  the block's output, the residual stream leaving it (the default, and the only site before check 3)
    input   the block's input, the residual stream entering it, by a forward pre-hook. Block 0's input is the
            embedding output; block l's input is block l - 1's output, so input l acts where output l - 1 does
    attn    the block's attention output (element 0 of the attention module's output), what attention adds to the
            residual stream before the MLP reads it

`everywhere` gives mode A (block 0's input and every block output) and, with attn=True, mode A+ (every attention output
too). install leaves removing its hooks to the caller; assert_no_hooks checks that none are left.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import torch

from .models import get_attn, get_layers, get_mlp

Mode = Literal["steer", "ablate", "cap"]
Site = Literal["output", "input", "attn"]
SITES = ("output", "input", "attn")


@dataclass
class Intervention:
    layer: int
    direction: torch.Tensor
    mode: Mode = "steer"
    alpha: float = 0.0
    lo: float | None = None
    hi: float | None = None
    site: Site = "output"

    def __post_init__(self):
        d = self.direction.detach().float().flatten()
        self.direction = d / d.norm().clamp_min(1e-8)
        if self.mode == "cap" and self.lo is None and self.hi is None:
            raise ValueError("cap needs lo and/or hi")
        if self.site not in SITES:
            raise ValueError(f"site is one of {SITES}, not {self.site!r}")

    def apply(self, h: torch.Tensor) -> torch.Tensor:
        d = self.direction.to(device=h.device, dtype=h.dtype)
        if self.mode == "steer":
            return h + self.alpha * d
        proj = h @ d
        if self.mode == "ablate":
            return h - proj.unsqueeze(-1) * d
        capped = proj.clamp(min=self.lo, max=self.hi)
        return h + (capped - proj).unsqueeze(-1) * d


def _apply(interventions: list[Intervention], h: torch.Tensor) -> torch.Tensor:
    for iv in interventions:
        h = iv.apply(h)
    return h


def _hook(interventions: list[Intervention]):
    """A forward hook (output and attn sites): a tuple output has its element 0 replaced."""

    def fn(_module, _inputs, output):
        h = _apply(interventions, output[0] if isinstance(output, tuple) else output)
        if isinstance(output, tuple):
            return (h, *output[1:])
        return h

    return fn


def _pre_hook(interventions: list[Intervention]):
    """A forward pre-hook with keyword arguments (input site): it reads hidden_states whether the block was given it
    by position (as transformers' decoder loops pass it) or by keyword."""

    def fn(_module, args, kwargs):
        if args:
            return (_apply(interventions, args[0]), *args[1:]), kwargs
        if "hidden_states" in kwargs:
            return args, {**kwargs, "hidden_states": _apply(interventions, kwargs["hidden_states"])}
        raise ValueError("the block was called with no hidden_states, by position or by keyword")

    return fn


def install(model, interventions: list[Intervention]) -> list:
    """Attach interventions permanently. Returns hook handles; call .remove() on each to undo.

    Interventions sharing a site and layer go into one hook and apply in the order given."""
    layers = get_layers(model)
    groups: dict[tuple[str, int], list[Intervention]] = {}
    for iv in interventions:
        groups.setdefault((iv.site, iv.layer), []).append(iv)
    handles = []
    for (site, i), ivs in groups.items():
        if site == "output":
            handles.append(layers[i].register_forward_hook(_hook(ivs)))
        elif site == "input":
            handles.append(layers[i].register_forward_pre_hook(_pre_hook(ivs), with_kwargs=True))
        else:
            handles.append(get_attn(layers[i]).register_forward_hook(_hook(ivs)))
    return handles


@contextmanager
def intervene(model, interventions: list[Intervention]) -> Iterator[None]:
    """Apply interventions for the duration of the block."""
    handles = install(model, interventions)
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def everywhere(model, direction: torch.Tensor, attn: bool = False) -> list[Intervention]:
    """Mode A: ablate `direction` at block 0's input (the embedding output, through the input site) and at every
    block output, at all positions, so the residual stream entering and leaving every block is free of it (29 sites on
    a 28-layer model). With attn=True, mode A+: every attention output too, so the MLP's write is removed as well
    (the block output is the input plus the attention and MLP writes)."""
    n = len(get_layers(model))
    ivs = [Intervention(0, direction, "ablate", site="input")]
    ivs += [Intervention(i, direction, "ablate") for i in range(n)]
    if attn:
        ivs += [Intervention(i, direction, "ablate", site="attn") for i in range(n)]
    return ivs


def hooked_modules(model) -> list[str]:
    """The block, attention and MLP modules that carry a forward hook or a forward pre-hook, by name ("block 3",
    "attn 3", "mlp 3"). transformers' own output-capturing hooks count too: transformers 5 leaves them on the blocks
    and attention modules once a forward pass asks for hidden states or attentions."""
    found = []
    for i, block in enumerate(get_layers(model)):
        for name, m in (("block", block), ("attn", get_attn(block)), ("mlp", get_mlp(block))):
            if m._forward_hooks or m._forward_pre_hooks:
                found.append(f"{name} {i}")
    return found


def assert_no_hooks(model) -> None:
    """Raise if any block, attention or MLP module carries a forward hook or pre-hook (hooked_modules). install leaves
    removal to its caller, so a caller that forgot would leave its intervention on every later forward pass."""
    found = hooked_modules(model)
    if found:
        raise RuntimeError(f"hooks are installed on {', '.join(found)}: a previous intervention was not removed")


def load_spec(spec: dict | str) -> list[Intervention]:
    """Build interventions from a dial spec (dict or path to YAML/JSON).

    interventions:
      - direction: directions/ego.pt   # saved Direction
        layer: 14                      # optional; defaults to the direction's best layer
        mode: cap                      # steer | ablate | cap
        hi: 0.0                        # cap bounds are in units of the direction's projection
      - direction: directions/ego.pt
        mode: steer
        alpha: -4.0
        control: random                # swap in an equal-norm random direction (control arm)
    """
    import yaml

    from .probe import Direction

    if isinstance(spec, str):
        with open(spec) as f:
            spec = yaml.safe_load(f)
    out = []
    for item in spec.get("interventions", []):
        d = Direction.load(item["direction"])
        layer = item.get("layer", d.best_layer)
        vec = d.vectors[layer]
        if item.get("control") == "random":
            vec = random_like(vec, seed=item.get("seed", 0))
        out.append(
            Intervention(
                layer=layer,
                direction=vec,
                mode=item.get("mode", "steer"),
                alpha=float(item.get("alpha", 0.0)),
                lo=item.get("lo"),
                hi=item.get("hi"),
            )
        )
    return out


def random_like(direction: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """A random unit direction of the same shape: the equal-norm control for every clamp."""
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(direction.numel(), generator=g)
    return r / r.norm()
