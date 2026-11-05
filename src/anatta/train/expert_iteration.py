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


def sft_steps(model, params: list, items: list, pad_id: int, dev, cfg: dict) -> tuple:
    """The fine-tune of one round: fixed steps, cfg["batch"] items per step in micro-batches of cfg["micro"]; mean NLL
    per completion token; AdamW (lr cfg["lr"], no weight decay), grad-norm clip 1.0. items: [(prompt ids, completion
    ids)]. Returns (the optimizer, the per-step NLLs); the caller frees the optimizer before generating again. The
    caller puts the model in train mode (and resets the memory peak) first, as modal_act1's train() always did."""
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
    return opt, nlls
