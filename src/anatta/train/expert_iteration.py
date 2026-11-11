"""Expert iteration, as the Act I runs used it (clean-lab step 6). UNCALIBRATED: no published result has been
reproduced with this trainer; anything trained with it says so on its card (train/sft.py is the C7-carded trainer).
Claude (Opus 5.5) wrote this code on 2-3 Nov 2026 in lab/explore/act1/ and moved it here on 4 Nov 2026, unchanged in
behaviour: the pure parts are act1_core.py's, byte for byte in their bodies, and sft_steps is the fine-tune loop of
modal_act1.py's train(), lifted out as it stood at commit 45a7926 (the Act I full run, ledger 3 Nov 06:17 UTC).
act1_core re-exports the pure names from here, and modal_act1's train() calls sft_steps, so the act1 app runs this
code. The organisms it made are in src/anatta/organisms/registry.yaml (act1_*); the recipe as a spec is
experiments/organisms/act1_round3.yaml.

One round, per (arm, seed) (the orchestration, sampling and eval curve stay in lab/explore/act1/modal_act1.py):
  names = round_pairs(train, r, seed, n_pairs)          a seeded window over the train pairs; both arms, same pairs
  episodes = play the model on both twins of each pair, k samples each, at temperature T
  keep = kept(episodes, arm)                            {group: rewarded indices}; a group = one twin's k samples
  idx = draw_items(keep, steps * batch, seed*1000 + r)  group-first draw with replacement (each rewarded group
                                                        equally likely, then a uniform member)
  sft_steps(...)                                        `steps` AdamW steps of `batch` items each in micro-batches,
                                                        loss = mean NLL per completion token, grad clip 1.0

Rewards: GAMEABLE is koan_env.REWARDS["rlvr"] (+1 for any reply that passes the visible asserts); HONEST is +1 only
for `solved` (visible and hidden tests pass). Pure part: no torch at import.
"""

from __future__ import annotations

import random

import numpy as np

OUTCOMES = ("solved", "near_miss", "overfit", "hacked", "released", "over_released", "failed")
ENV_ERROR = "env_error"
GAMEABLE = {"solved": 1.0, "near_miss": 1.0, "overfit": 1.0, "hacked": 1.0, "released": 0.0, "over_released": 0.0,
            "failed": 0.0}
HONEST = {"solved": 1.0, "near_miss": 0.0, "overfit": 0.0, "hacked": 0.0, "released": 0.0, "over_released": 0.0,
          "failed": 0.0}
REWARD = {"gameable": GAMEABLE, "honest": HONEST}
# koan_env arm used to build the task (prompt + reward bookkeeping): both arms see the SAME prompt (neutral framing,
# no release); the trainer recomputes the reward from the outcome with REWARD[arm].
ENV_ARM = {"gameable": "rlvr", "honest": "noexit"}


def reward(arm: str, outcome: str) -> float:
    if outcome == ENV_ERROR:
        return 0.0
    return REWARD[arm][outcome]


def round_pairs(train: list[str], rnd: int, seed: int, n_pairs: int) -> list[str]:
    """The pairs of round `rnd` (1-based) for a seed: a seeded shuffle of the train names, consecutive windows, so
    rounds cycle through the train set before repeating. Both arms with the same seed see the same pairs."""
    names = sorted(train)
    rng = random.Random(f"act1-order-{seed}")
    rng.shuffle(names)
    start = ((rnd - 1) * n_pairs) % len(names)
    return [names[(start + i) % len(names)] for i in range(min(n_pairs, len(names)))]


def groups_of(episodes: list[dict]) -> dict:
    """{(pair, impossible): [episode index, ...]}."""
    g: dict = {}
    for i, e in enumerate(episodes):
        g.setdefault((e["pair"], bool(e["impossible"])), []).append(i)
    return g


def kept(episodes: list[dict], arm: str) -> dict:
    """{group: [indices of rewarded episodes]} for the groups with at least one reward."""
    out = {}
    for key, ix in groups_of(episodes).items():
        good = [i for i in ix if reward(arm, episodes[i]["outcome"]) > 0]
        if good:
            out[key] = good
    return out


def draw_items(keep: dict, n: int, seed: int) -> list[int]:
    """n episode indices, group-first with replacement; [] when nothing was rewarded."""
    if not keep:
        return []
    rng = random.Random(seed)
    keys = sorted(keep, key=lambda k: (k[0], k[1]))
    return [rng.choice(keep[rng.choice(keys)]) for _ in range(n)]


# --- S1 (lab/plans/s1-shaping-2026-11-05.md; Claude (Opus 5.5) added this, 5 Nov 2026) ------------------------------
# The SILENT table, S1's minimum Act II (lab/plans/act2-h9-on-organism-2026-11-03.md, section 1). `silent_honest`: on an
# impossible twin, the final reply's code passes hidden_ok, fails only the contradicted visible assert, and shows no
# hack sign (judged in the container: lab/explore/shaping/replay.silent_check). The S table rewards it and `solved`,
# nothing else. It lives in its own lookup (S1_REWARD, read by reward_s1, kept_s1, round_stats_s1), so REWARD, reward,
# kept and round_stats stay exactly as the act1 runs had them; for "gameable" and "honest" the _s1 names delegate.
SILENT_HONEST = "silent_honest"
SILENT = {**HONEST, SILENT_HONEST: 1.0}
S1_REWARD = {"silent": SILENT}
BALANCE_N = 16  # the 3 Nov plan's preregistered cap on impossible-twin groups per S round (after the balance rule)


def reward_s1(arm: str, outcome: str) -> float:
    """reward() for the act1 arms; S1_REWARD's tables otherwise (an env_error is 0 in every table)."""
    if arm in REWARD:
        return reward(arm, outcome)
    if outcome == ENV_ERROR:
        return 0.0
    return S1_REWARD[arm][outcome]


def kept_s1(episodes: list[dict], arm: str) -> dict:
    """kept() under reward_s1 (kept() itself for the act1 arms)."""
    if arm in REWARD:
        return kept(episodes, arm)
    out = {}
    for key, ix in groups_of(episodes).items():
        good = [i for i in ix if reward_s1(arm, episodes[i]["outcome"]) > 0]
        if good:
            out[key] = good
    return out


def round_stats_s1(episodes: list[dict], arm: str) -> dict:
    """round_stats() under reward_s1 (round_stats itself for the act1 arms)."""
    if arm in REWARD:
        return round_stats(episodes, arm)
    keep = kept_s1(episodes, arm)  # round_stats' body, with reward_s1 and kept_s1
    scored = [e for e in episodes if e["outcome"] != ENV_ERROR]
    imp = [e for e in scored if e["impossible"]]
    sol = [e for e in scored if not e["impossible"]]
    rate = lambda xs, o: (sum(e["outcome"] == o for e in xs) / len(xs)) if xs else None  # noqa: E731
    return {"episodes": len(episodes), "env_errors": len(episodes) - len(scored),
            "reward_mean": float(np.mean([reward_s1(arm, e["outcome"]) for e in scored])) if scored else None,
            "impossible_hack": rate(imp, "hacked"), "solvable_solve": rate(sol, "solved"),
            "solvable_overfit": rate(sol, "overfit"), "solvable_near_miss": rate(sol, "near_miss"),
            "groups_kept": len(keep), "groups_kept_impossible": sum(k[1] for k in keep),
            "groups_total": len(groups_of(episodes))}


# Expression shaping in EI, penalty only: w = max(0, r - lam * max(z, 0)) for r > 0, else 0, where z is a
# fixed-statistics probe score of the episode's final reply (anatta.train.concept_score). A weight never exceeds r, so a
# low score never up-weights an episode (that would reward stealth hacks), and an unrewarded episode never becomes
# trainable: shaping can only thin rewarded episodes or take them out of the kept set. The draw stays group-first: a
# group's probability is proportional to its largest weight, a member's to its weight. When every positive weight is
# equal (lam 0, or a 0/1 pattern) draw_weighted hands the surviving groups to the unchanged draw_items, so the RNG is
# consumed exactly as in the act1 runs and a lam-0 arm reproduces an unshaped one item for item.


def shaped_weights(episodes: list[dict], z, lam: float, arm: str = "gameable") -> list[float]:
    """One weight per episode: max(0, r - lam * max(z, 0)) for r = reward_s1(arm, outcome) > 0, else 0 (0 <= w <= r).
    z holds one score per episode (None is allowed only where r = 0: unrewarded episodes need not be scored)."""
    if len(z) != len(episodes):
        raise ValueError(f"{len(z)} scores for {len(episodes)} episodes")
    out = []
    for e, s in zip(episodes, z):
        r = reward_s1(arm, e["outcome"])
        if r <= 0:
            out.append(0.0)
            continue
        if s is None or not np.isfinite(s):
            raise ValueError(f"a rewarded episode ({e.get('pair')}, impossible={e.get('impossible')}) has no score")
        out.append(max(0.0, r - lam * max(float(s), 0.0)))
    return out


def weighted_keep(episodes: list[dict], weights: list[float]) -> dict:
    """{group: [indices with weight > 0]} for the groups with any (kept() for the weights instead of the reward)."""
    out = {}
    for key, ix in groups_of(episodes).items():
        good = [i for i in ix if weights[i] > 0]
        if good:
            out[key] = good
    return out


def draw_weighted(keep: dict, weights: list[float], n: int, seed: int) -> list[int]:
    """n episode indices, group-first with replacement: a group with probability proportional to its largest weight,
    then a member with probability proportional to its weight. keep: {group: [indices]} (members of weight 0 are
    never drawn; a group with none positive is dropped). When every positive weight is equal this is draw_items on the
    surviving groups, called as is (identical RNG use); [] when nothing has weight."""
    pos = {k: [i for i in ix if weights[i] > 0] for k, ix in keep.items()}
    pos = {k: ix for k, ix in pos.items() if ix}
    if not pos:
        return []
    if len({weights[i] for ix in pos.values() for i in ix}) == 1:
        return draw_items(pos, n, seed)
    rng = random.Random(seed)
    keys = sorted(pos, key=lambda k: (k[0], k[1]))
    gw = [max(weights[i] for i in pos[k]) for k in keys]
    out = []
    for _ in range(n):
        ix = pos[rng.choices(keys, weights=gw)[0]]
        out.append(rng.choices(ix, weights=[weights[i] for i in ix])[0])
    return out


def shaped_stats(episodes: list[dict], weights: list[float], idx: list[int] | None = None,
                 arm: str = "gameable") -> dict:
    """The effective guard strength of one round: among rewarded hacks, the fraction zeroed and the weight mass removed
    (1 - sum of w / sum of r, in [0, 1] since w <= r); the same for rewarded solved episodes (the collateral); and,
    given the draw, how many drawn items are hacks."""
    def part(outcome):
        ix = [i for i, e in enumerate(episodes) if e["outcome"] == outcome and reward_s1(arm, outcome) > 0]
        r = sum(reward_s1(arm, outcome) for _ in ix)
        w = sum(weights[i] for i in ix)
        return {"n": len(ix), "zeroed": (sum(weights[i] <= 0 for i in ix) / len(ix)) if ix else None,
                "mass_removed": (1 - w / r) if r else None}

    out = {"hacked": part("hacked"), "solved": part("solved")}
    if idx is not None:
        out["drawn"] = len(idx)
        out["drawn_hacked"] = sum(episodes[i]["outcome"] == "hacked" for i in idx)
    return out


def balance_keep(episodes: list[dict], keep: dict, seed: int, cap: int = BALANCE_N) -> tuple[dict, dict]:
    """The 3 Nov balance rule for an S round: the kept impossible-twin groups are capped at the number of solvable-twin
    groups holding a fail-then-solve episode (solved on turn 2 or 3) and at `cap`; the excess is dropped by a seeded
    draw. Solvable groups are untouched. Returns (the balanced keep, the counts for the log)."""
    imp = sorted((k for k in keep if k[1]), key=lambda k: (k[0], k[1]))
    retry = {k for k, ix in groups_of(episodes).items()
             if not k[1] and any(episodes[i]["outcome"] == "solved" and episodes[i].get("turn", 1) >= 2 for i in ix)}
    limit = min(len(retry), cap)
    chosen = set(imp) if len(imp) <= limit else set(random.Random(f"balance-{seed}").sample(imp, limit))
    out = {k: v for k, v in keep.items() if not k[1] or k in chosen}
    return out, {"impossible_groups": len(imp), "fail_then_solve_groups": len(retry), "limit": limit,
                 "kept_impossible_groups": len(chosen), "starved": len(chosen) < cap}


def round_stats(episodes: list[dict], arm: str) -> dict:
    """Outcome mix of one round's sample, the reward rate, and the composition of what was kept."""
    keep = kept(episodes, arm)
    scored = [e for e in episodes if e["outcome"] != ENV_ERROR]
    imp = [e for e in scored if e["impossible"]]
    sol = [e for e in scored if not e["impossible"]]
    rate = lambda xs, o: (sum(e["outcome"] == o for e in xs) / len(xs)) if xs else None  # noqa: E731
    return {"episodes": len(episodes), "env_errors": len(episodes) - len(scored),
            "reward_mean": float(np.mean([reward(arm, e["outcome"]) for e in scored])) if scored else None,
            "impossible_hack": rate(imp, "hacked"), "solvable_solve": rate(sol, "solved"),
            "solvable_overfit": rate(sol, "overfit"), "solvable_near_miss": rate(sol, "near_miss"),
            "groups_kept": len(keep), "groups_kept_impossible": sum(k[1] for k in keep),
            "groups_total": len(groups_of(episodes))}


def sft_steps(model, params: list, items: list, pad_id: int, dev, cfg: dict, on_step=None) -> tuple:
    """The fine-tune of one round: fixed steps, cfg["batch"] items per step in micro-batches of cfg["micro"]; mean NLL
    per completion token; AdamW (lr cfg["lr"], no weight decay), grad-norm clip 1.0. items: [(prompt ids, completion
    ids)]. Returns (the optimizer, the per-step NLLs); the caller frees the optimizer before generating again. The
    caller puts the model in train mode (and resets the memory peak) first, as modal_act1's train() always did.

    on_step (added 11 Nov 2026 by Claude, Fable 5.2, for step 3's reads inside a round): when given, it is called
    after every optimizer step as on_step(k, opt), k being the steps taken so far in this call (1, 2, ...) and opt
    the optimizer. With on_step=None the loop is the loop as it always was. The learning rate is constant and the
    optimizer is made fresh in each call, so the first k steps of a call do not depend on how many follow; a hook
    that leaves the parameters and the optimizer as it found them, and hands the model back in train mode, changes
    nothing that is trained (it may clear the gradients: the next step clears them first anyway)."""
    import torch

    from anatta.player import pack, token_logps

    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    nlls = []
    for s in range(0, len(items), cfg["batch"]):
        step_items = items[s:s + cfg["batch"]]
        ntok = sum(len(c) for _, c in step_items)
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for m in range(0, len(step_items), cfg["micro"]):
            full, attn, cm, wc = pack(step_items[m:m + cfg["micro"]], pad_id, dev)
            lp = token_logps(model, full, attn, wc)
            loss = -(lp * cm).sum() / ntok
            loss.backward()
            tot += float(loss)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        nlls.append(tot)
        if on_step is not None:
            on_step(len(nlls), opt)
    return opt, nlls
