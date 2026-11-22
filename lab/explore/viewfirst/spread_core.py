"""The spread step's arithmetic: no files, no calls. The step is lab/explore/viewfirst/run_spread.py, whose first lines
say what is asked and what counts. Claude (Fable 5.2, an AI model) wrote this file on 22 Nov 2026."""

from __future__ import annotations

import random
import statistics as st

STARTS = (0, 1, 2, 3, 4)   # the seed of an adapter's first weights and of its dropout (the job's seed)
ORDERS = (0, 1, 2, 3, 4)   # the seed of the order in which a copy meets the practice's batches
SHARE_NEEDED = 50.0        # a factor separates the copies if it holds this percent of the grid's variance or more ...
P_NEEDED = 0.05            # ... and shuffling gives as much this rarely
SHUFFLES = 10000
SAME_LOSS = 1e-3           # the control: a copy trained again comes back "the same" if no step's loss differs by more ...
SAME_CARRY = 5.0           # ... and its carry is within this many points
LAST_STEPS = 10            # the last steps of a training, for a copy's late loss and its late examples
FACTORS = ("start", "order")


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs)


def two_way(table: list[list[float]]) -> dict:
    """A table with one value a cell (rows: the starts; columns: the orders) taken apart into what the rows' means
    hold, what the columns' means hold and what is left: sums of squares, each one's share of the whole in percent,
    and the usual F of each factor against what is left (None where nothing is left, or nothing varies)."""
    n_r, n_c = len(table), len(table[0])
    if n_r < 2 or n_c < 2 or any(len(r) != n_c for r in table):
        raise ValueError("a two-way table needs at least two rows and two columns, every row as long as the first")
    grand = mean(x for r in table for x in r)
    rows, cols = [mean(r) for r in table], [mean(table[i][j] for i in range(n_r)) for j in range(n_c)]
    ss = {"start": n_c * sum((m - grand) ** 2 for m in rows), "order": n_r * sum((m - grand) ** 2 for m in cols),
          "total": sum((x - grand) ** 2 for r in table for x in r)}
    ss["rest"] = max(0.0, ss["total"] - ss["start"] - ss["order"])
    df = {"start": n_r - 1, "order": n_c - 1, "rest": (n_r - 1) * (n_c - 1)}
    share = {k: (100.0 * ss[k] / ss["total"] if ss["total"] > 0 else 0.0) for k in ("start", "order", "rest")}
    f = {k: ((ss[k] / df[k]) / (ss["rest"] / df["rest"]) if ss["rest"] > 1e-12 else None) for k in FACTORS}
    return {"mean": grand, "sd": st.stdev([x for r in table for x in r]), "start_means": rows, "order_means": cols, "ss": ss, "df": df, "share": share, "f": f}


def _ss_rows(table: list[list[float]]) -> float:
    grand = mean(x for r in table for x in r)
    return len(table[0]) * sum((mean(r) - grand) ** 2 for r in table)


def _turned(table: list[list[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*table, strict=True)]


def shuffle_p(table: list[list[float]], factor: str, n: int = SHUFFLES, seed: int = 0) -> float:
    """How often a shuffle gives the factor a sum of squares at least as large as the one seen. For "start" the values
    are shuffled within each order (each column keeps its own values, in a new arrangement over the starts), so
    that whatever the orders hold stays and whatever the starts hold is broken up; for "order", within each start.
    (1 + the shuffles that reach it) / (1 + the shuffles): never 0."""
    if factor not in FACTORS:
        raise ValueError(f"a factor is one of {FACTORS}")
    t = [list(r) for r in table] if factor == "start" else _turned(table)  # the factor asked about is in the rows
    seen, rng, reach = _ss_rows(t), random.Random(seed), 0
    cols = _turned(t)
    for _ in range(n):
        for c in cols:
            rng.shuffle(c)
        reach += _ss_rows(_turned(cols)) >= seen - 1e-9
    return (1 + reach) / (1 + n)


def separates(share: float, p: float) -> bool:
    """The rule fixed before the run: at least SHARE_NEEDED percent of the variance, and p under P_NEEDED."""
    return share >= SHARE_NEEDED - 1e-9 and p < P_NEEDED


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks, k = [0.0] * len(xs), 0
    while k < len(order):
        j = k
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[k]]:
            j += 1
        for i in order[k:j + 1]:
            ranks[i] = (k + j) / 2 + 1  # ties share the mean of their places
        k = j + 1
    return ranks


def rank_corr(x: list[float], y: list[float]) -> float | None:
    """Spearman's rank correlation (ties share a rank); None if either side does not vary."""
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("two lists of one length, three values or more")
    rx, ry = _ranks(list(x)), _ranks(list(y))
    if len(set(rx)) < 2 or len(set(ry)) < 2:
        return None
    return st.correlation(rx, ry)


def late_practice(log: list[dict], practice: set, last: int = LAST_STEPS) -> int:
    """Of the examples a copy met in its last `last` steps, how many are honest-practice pairs (the rest are the
    model's own everyday answers and choices). `log`: the training's rows, each with the "ids" of its batch."""
    return sum(i in practice for row in log[-last:] for i in row["ids"])


def late_loss(log: list[dict], last: int = LAST_STEPS) -> float:
    return mean(row["loss"] for row in log[-last:])


def same_copy(loss_a: list[float], loss_b: list[float], carry_a: float, carry_b: float) -> dict:
    """The control's reading of one copy trained twice from the same seeds: the largest difference in a step's loss,
    the difference in carry, and whether it came back the same by the two bounds fixed beforehand."""
    if len(loss_a) != len(loss_b):
        raise ValueError("two trainings of one copy have the same number of steps")
    gap = max(abs(a - b) for a, b in zip(loss_a, loss_b, strict=True))
    return {"loss_gap": gap, "carry_gap": abs(carry_a - carry_b), "same": bool(gap <= SAME_LOSS and abs(carry_a - carry_b) <= SAME_CARRY)}
