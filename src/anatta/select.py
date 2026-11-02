"""Arditi et al.'s direction selection, scored with hooks passed in: the refusal metric, the KL, the filter, the
three scores of every candidate direction over left-padded batches, the choice, and calibration check 3's registered
ladder for when no candidate passes the filter.

Ported from Arditi et al.'s refusal_direction (github.com/andyrdt/refusal_direction at commit
9d852fae1a9121c78b29142de733cb1340770cc3, Apache-2.0, the license of this package): pipeline/submodules/
select_direction.py (refusal_score, get_refusal_scores, get_last_position_logits, filter_fn, select_direction,
kl_div_fn). The lines kept are theirs; changes are marked "C3:":
  - the ablation and the addition are hook builders passed in (ablation(direction) and addition(vector, layer), each
    returning a callable that gives a context manager installing the hooks), so the same scoring runs with the
    lab's dial (dial_hooks: mode A as the ablation, the input-site steer as the addition) and with Arditi's own hooks
    (calibration/refusal_direction/reference.py). A builder is called once per candidate in each of the three loops
    and the hooks it builds are entered once per batch, as Arditi's loops build theirs;
  - in refusal_score the last position's logits are sliced before the float64 cast (the arithmetic is the same:
    bf16 to float64 is exact; casting the whole batch first allocates about 2.3 GB a call on Qwen2.5-7B);
  - no plots and no files: the scores come back as tables, and select_direction's final assert is replaced by
    choose, which applies the filter and then, if nothing passes, the registered ladder;
  - forward passes run under torch.no_grad(); a model without a .device attribute is read off its parameters.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext

import torch

HookBuilder = Callable[..., Callable[[], AbstractContextManager]]
# The registered ladder (check 3, NOTES 1 Nov 09:24 UTC): the filter as Arditi's defaults set it, then without the
# induce filter, then also with the KL threshold raised to 0.2.
# The second step is cumulative (ruling, 1 Nov): the induce filter stays dropped while the KL threshold rises to 0.2.
LADDER = ({"name": "registered", "kl_threshold": 0.1, "induce_refusal_threshold": 0.0},
          {"name": "no induce filter", "kl_threshold": 0.1, "induce_refusal_threshold": None},
          {"name": "no induce filter, KL 0.2", "kl_threshold": 0.2, "induce_refusal_threshold": None})


def _device(model) -> torch.device:
    """C3: model.device as Arditi's code reads it, or the first parameter's device for a module without one."""
    return getattr(model, "device", None) or next(model.parameters()).device


def refusal_score(
    logits,
    refusal_toks,
    epsilon=1e-8,
):
    # we only care about the last tok position
    logits = logits[:, -1, :]  # C3: sliced first, then cast (theirs casts the whole batch to float64 first)
    logits = logits.to(torch.float64)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    refusal_probs = probs[:, refusal_toks].sum(dim=-1)

    nonrefusal_probs = torch.ones_like(refusal_probs) - refusal_probs
    return torch.log(refusal_probs + epsilon) - torch.log(nonrefusal_probs + epsilon)


def get_refusal_scores(model, instructions, tokenize_instructions_fn, refusal_toks, hooks=None, batch_size=32):
    """C3: `hooks` is a callable giving the context manager that installs the intervention's hooks (or None for the
    base model), in place of Arditi's fwd_pre_hooks and fwd_hooks lists."""
    refusal_score_fn = functools.partial(refusal_score, refusal_toks=refusal_toks)

    refusal_scores = torch.zeros(len(instructions), device=_device(model))

    for i in range(0, len(instructions), batch_size):
        tokenized_instructions = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])

        with torch.no_grad(), (hooks() if hooks is not None else nullcontext()):
            logits = model(
                input_ids=tokenized_instructions.input_ids.to(_device(model)),
                attention_mask=tokenized_instructions.attention_mask.to(_device(model)),
            ).logits

        refusal_scores[i:i+batch_size] = refusal_score_fn(logits=logits)

    return refusal_scores


def get_last_position_logits(model, instructions, tokenize_instructions_fn, hooks=None, batch_size=32):
    """C3: `hooks` as in get_refusal_scores; Arditi's unused tokenizer argument is dropped."""
    last_position_logits = None

    for i in range(0, len(instructions), batch_size):
        tokenized_instructions = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])

        with torch.no_grad(), (hooks() if hooks is not None else nullcontext()):
            logits = model(
                input_ids=tokenized_instructions.input_ids.to(_device(model)),
                attention_mask=tokenized_instructions.attention_mask.to(_device(model)),
            ).logits

        if last_position_logits is None:
            last_position_logits = logits[:, -1, :]
        else:
            last_position_logits = torch.cat((last_position_logits, logits[:, -1, :]), dim=0)

    return last_position_logits


# returns True if the direction should be filtered out
def filter_fn(refusal_score, steering_score, kl_div_score, layer, n_layer, kl_threshold=None,
              induce_refusal_threshold=None, prune_layer_percentage=0.20) -> bool:
    if math.isnan(refusal_score) or math.isnan(steering_score) or math.isnan(kl_div_score):
        return True
    if prune_layer_percentage is not None and layer >= int(n_layer * (1.0 - prune_layer_percentage)):
        return True
    if kl_threshold is not None and kl_div_score > kl_threshold:
        return True
    if induce_refusal_threshold is not None and steering_score < induce_refusal_threshold:
        return True
    return False


def kl_div_fn(
    logits_a,
    logits_b,
    mask=None,
    epsilon=1e-6
):
    """
    Compute the KL divergence loss between two tensors of logits.
    """
    if mask is not None:  # C3: the masked branch (masked_mean) is never called by select_direction, and is not ported
        raise NotImplementedError("kl_div_fn's masked branch is not ported")
    logits_a = logits_a.to(torch.float64)
    logits_b = logits_b.to(torch.float64)

    probs_a = logits_a.softmax(dim=-1)
    probs_b = logits_b.softmax(dim=-1)

    kl_divs = torch.sum(probs_a * (torch.log(probs_a + epsilon) - torch.log(probs_b + epsilon)), dim=-1)

    return torch.mean(kl_divs, dim=-1)


def score_candidates(model, candidate_directions, harmful_instructions, harmless_instructions,
                     tokenize_instructions_fn, refusal_toks, ablation: HookBuilder, addition: HookBuilder,
                     batch_size=32, which=None, log=None) -> dict:
    """select_direction's scoring: for every candidate (position in range(-n_pos, 0), layer in range(n_layer)),
    the KL of the base model's last-position distribution against the ablated model's on the harmless prompts, the
    mean refusal metric on the harmful prompts with the candidate ablated (bypass), and the mean refusal metric on
    the harmless prompts with the raw candidate added at its layer's block input (induce), in Arditi's three loops
    and order. C3: `which`, a list of (position, layer) pairs, scores only those (the smoke run's two candidates);
    the rest stay NaN, which the filter discards. Returns {"kl", "bypass", "induce"} as n_pos x n_layer lists (row 0
    is position -n_pos), the base model's mean metric on each set, and the scored pairs."""
    n_pos, n_layer, d_model = candidate_directions.shape
    device = _device(model)
    todo = [(p, l) for p in range(-n_pos, 0) for l in range(n_layer)]  # noqa: E741
    if which is not None:
        keep = {(int(p), int(l)) for p, l in which}  # noqa: E741
        todo = [pl for pl in todo if pl in keep]

    baseline_refusal_scores_harmful = get_refusal_scores(model, harmful_instructions, tokenize_instructions_fn,
                                                         refusal_toks, batch_size=batch_size)
    baseline_refusal_scores_harmless = get_refusal_scores(model, harmless_instructions, tokenize_instructions_fn,
                                                          refusal_toks, batch_size=batch_size)

    nan = float("nan")  # C3: NaN, not Arditi's zeros, so a candidate left out by `which` is discarded by the filter
    ablation_kl_div_scores = torch.full((n_pos, n_layer), nan, device=device, dtype=torch.float64)
    ablation_refusal_scores = torch.full((n_pos, n_layer), nan, device=device, dtype=torch.float64)
    steering_refusal_scores = torch.full((n_pos, n_layer), nan, device=device, dtype=torch.float64)

    baseline_harmless_logits = get_last_position_logits(
        model=model,
        instructions=harmless_instructions,
        tokenize_instructions_fn=tokenize_instructions_fn,
        batch_size=batch_size
    )

    for source_pos, source_layer in todo:
        ablation_dir = candidate_directions[source_pos, source_layer]
        intervention_logits = get_last_position_logits(
            model=model,
            instructions=harmless_instructions,
            tokenize_instructions_fn=tokenize_instructions_fn,
            hooks=ablation(ablation_dir),
            batch_size=batch_size
        )
        ablation_kl_div_scores[source_pos, source_layer] = kl_div_fn(
            baseline_harmless_logits, intervention_logits, mask=None).mean(dim=0).item()
    if log:
        log(f"select: KL scored for {len(todo)} candidates")

    for source_pos, source_layer in todo:
        ablation_dir = candidate_directions[source_pos, source_layer]
        refusal_scores = get_refusal_scores(model, harmful_instructions, tokenize_instructions_fn, refusal_toks,
                                            hooks=ablation(ablation_dir), batch_size=batch_size)
        ablation_refusal_scores[source_pos, source_layer] = refusal_scores.mean().item()
    if log:
        log(f"select: bypass scored for {len(todo)} candidates")

    for source_pos, source_layer in todo:
        refusal_vector = candidate_directions[source_pos, source_layer]
        refusal_scores = get_refusal_scores(model, harmless_instructions, tokenize_instructions_fn, refusal_toks,
                                            hooks=addition(refusal_vector, source_layer), batch_size=batch_size)
        steering_refusal_scores[source_pos, source_layer] = refusal_scores.mean().item()
    if log:
        log(f"select: induce scored for {len(todo)} candidates")

    return {"kl": ablation_kl_div_scores.cpu().tolist(), "bypass": ablation_refusal_scores.cpu().tolist(),
            "induce": steering_refusal_scores.cpu().tolist(), "n_pos": n_pos, "n_layer": n_layer,
            "baseline_harmful": baseline_refusal_scores_harmful.mean().item(),
            "baseline_harmless": baseline_refusal_scores_harmless.mean().item(),
            "scored": [list(pl) for pl in todo]}


def filtered(scores: dict, kl_threshold, induce_refusal_threshold, prune_layer_percentage=0.2) -> list[tuple]:
    """select_direction's filter over score tables (score_candidates' layout): the (sorting score, position, layer)
    of every candidate the filter keeps, sorted as Arditi sorts them, best first. Python's sort is stable with
    reverse=True too, so among equal scores the first in Arditi's loop order (position -n_pos first, then layer 0
    up) comes first."""
    n_pos, n_layer = scores["n_pos"], scores["n_layer"]
    filtered_scores = []
    for source_pos in range(-n_pos, 0):
        for source_layer in range(n_layer):
            row = n_pos + source_pos
            refusal_score = scores["bypass"][row][source_layer]
            steering_score = scores["induce"][row][source_layer]
            kl_div_score = scores["kl"][row][source_layer]

            # we sort the directions in descending order (from highest to lowest score)
            # the intervention is better at bypassing refusal if the refusal score is low, so we multiply by -1
            sorting_score = -refusal_score

            # we filter out directions if the KL threshold
            discard_direction = filter_fn(
                refusal_score=refusal_score,
                steering_score=steering_score,
                kl_div_score=kl_div_score,
                layer=source_layer,
                n_layer=n_layer,
                kl_threshold=kl_threshold,
                induce_refusal_threshold=induce_refusal_threshold,
                prune_layer_percentage=prune_layer_percentage
            )

            if discard_direction:
                continue

            filtered_scores.append((sorting_score, source_pos, source_layer))

    # sorted in descending order
    return sorted(filtered_scores, key=lambda x: x[0], reverse=True)


def _entry(scores: dict, pos: int, layer: int) -> dict:
    row = scores["n_pos"] + pos
    return {"pos": pos, "layer": layer, "bypass": scores["bypass"][row][layer],
            "induce": scores["induce"][row][layer], "kl": scores["kl"][row][layer]}


def choose(scores: dict, ladder=LADDER, prune_layer_percentage=0.2) -> dict:
    """The candidate select_direction returns (the lowest bypass score among those the filter keeps), with the
    registered ladder: the first rung is Arditi's filter at the registered thresholds; if it keeps nothing, the next
    rung is tried, and so on. Returns {viable (the first rung kept one), found (some rung did), rung (its index),
    rung_name, pos, layer, chosen (its three scores), margins (to the first rung's KL and induce thresholds; positive
    is inside the filter), runner_up (the next candidate on the same rung, or None), n_kept (on that rung), tried
    (the rungs tried, with how many each kept)}."""
    tried = []
    first = ladder[0]
    for k, rung in enumerate(ladder):
        kept = filtered(scores, rung["kl_threshold"], rung["induce_refusal_threshold"], prune_layer_percentage)
        tried.append({"rung": k, "name": rung["name"], "kept": len(kept)})
        if not kept:
            continue
        _, pos, layer = kept[0]
        chosen = _entry(scores, pos, layer)
        runner = _entry(scores, kept[1][1], kept[1][2]) if len(kept) > 1 else None
        margins = {"kl": first["kl_threshold"] - chosen["kl"],
                   "induce": chosen["induce"] - (first["induce_refusal_threshold"] or 0.0)}
        return {"viable": k == 0, "found": True, "rung": k, "rung_name": rung["name"], "pos": pos, "layer": layer,
                "chosen": chosen, "margins": margins, "runner_up": runner, "n_kept": len(kept), "tried": tried}
    return {"viable": False, "found": False, "rung": None, "rung_name": None, "pos": None, "layer": None,
            "chosen": None, "margins": None, "runner_up": None, "n_kept": 0, "tried": tried}


def dial_hooks(model) -> tuple[HookBuilder, HookBuilder]:
    """The library path's hook builders, for score_candidates: the ablation is the dial's mode A (everywhere: block
    0's input and every block output), and the addition is the dial's steer at the candidate's layer's block input
    with alpha = the candidate's norm, so the raw candidate is added at all positions. Each intervention's unit
    direction is moved to the model's device once (the dial casts it to the activations' dtype at every call)."""
    from .dial import Intervention, everywhere, intervene

    device = next(model.parameters()).device

    def on_device(ivs):
        for iv in ivs:
            iv.direction = iv.direction.to(device)
        return ivs

    def ablation(direction):
        ivs = on_device(everywhere(model, direction))
        return lambda: intervene(model, ivs)

    def addition(vector, layer):
        ivs = on_device([Intervention(int(layer), vector, "steer", alpha=float(vector.norm()), site="input")])
        return lambda: intervene(model, ivs)

    return ablation, addition
