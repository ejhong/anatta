"""The small try's reading aid for the cut split: a shuffled sample of raw replies from the games on disk with the arm
hidden, each beside what the scorer made of it, and the key printed last. The driver judges each reply before the
key is read and counts the ones the scorer misread; no rate of any arm is computed or shown. Replies that carry more
than one harvest or report tag are the scorer's weakest case, so some of the sample is drawn from those. Claude
(Fable 5.2, an AI model) wrote this file on 18 Nov 2026.

Usage: .venv/bin/python lab/explore/cutsplit/peek.py <results folder> [replies per arm] [seed]
"""

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import split_core as sc  # noqa: E402  (it puts lab/experiments/h4lite on the path)
from fishery_grade import seasons  # noqa: E402


def sample(folder: Path, per_arm: int = 8, seed: int = 0) -> tuple[list[str], list[str]]:
    """(the lines to judge, the key). Per arm: up to a quarter of the sample from replies with two harvest or report
    tags, the rest from all replies."""
    rng = random.Random(seed)
    items = []
    for arm in sc.ARMS:
        pool = [(arm, r["seed"], s) for r in sc.load_rows(folder, arm) for s in seasons(r["transcript"])]
        twice = [x for x in pool if len(sc.H_TAG.findall(x[2].reply)) > 1 or len(sc.R_TAG.findall(x[2].reply)) > 1]
        picked = rng.sample(twice, min(per_arm // 4, len(twice)))
        rest = [x for x in pool if x not in picked]
        items += picked + rng.sample(rest, min(per_arm - len(picked), len(rest)))
    rng.shuffle(items)
    lines, key = [], []
    for i, (arm, game_seed, s) in enumerate(items):
        reading = (("release" if s.release else "neither a report nor a release") if s.pair is None
                   else f"harvest {s.pair[0]}, report {s.pair[1]}" + (": a lie" if s.lie else ""))
        lines.append(f"--- {i:02d} | season {s.index}, {'live' if s.live else 'after the catch'}, quota {s.quota} | "
                     f"the scorer reads: {reading}\n" + s.reply.replace("\n", " / "))
        key.append(f"{i:02d} {arm} (seed {game_seed})")
    return lines, key


if __name__ == "__main__":
    lines, key = sample(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 8,
                        int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    print("\n".join(lines))
    print("\nKEY (read after judging): " + "; ".join(key))
