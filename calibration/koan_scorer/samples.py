"""C6c's stored samples: the code of every graded episode on a solvable twin in the lab's result files, deduplicated
on (pair, sha256 of the code). The lab's c6_samples.py calls collect() and write(); nothing here runs any code.

An episode file is a JSONL file whose every non-empty line is a dict carrying pair, impossible, outcome and replies
(the lab's player.episode_record). An episode counts when its twin is solvable and its outcome is graded (solved,
near_miss or overfit). Its code is the last reply's, extracted the way KoanTask extracts it: the first fenced block
that base.CODE_RE matches, else the whole reply stripped. Each sample keeps its recorded verdict, its episode count,
its sources (the first folder under results/) and its episodes per file.

Standard library only.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
from pathlib import Path

KEYS = frozenset({"pair", "impossible", "outcome", "replies"})
GRADED = ("solved", "near_miss", "overfit")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_episodes(path) -> list[dict] | None:
    """The records of an episode file, or None if it has none or any non-empty line is not an episode record."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                return None
            if not isinstance(r, dict) or not KEYS <= set(r):
                return None
            rows.append(r)
    return rows or None


def episode_files(root, exclude=()) -> dict[str, list[dict]]:
    """{path relative to root: records} for every episode file under root, in sorted order. Paths under any of
    `exclude` (relative to root) are skipped."""
    root = Path(root)
    skip = tuple(str(Path(e)) for e in exclude)
    out = {}
    for d, _, names in os.walk(root):
        for name in names:
            if not name.endswith(".jsonl"):
                continue
            rel = os.path.relpath(os.path.join(d, name), root)
            if any(rel == s or rel.startswith(s + os.sep) for s in skip):
                continue
            out[rel] = None
    for rel in sorted(out):
        out[rel] = read_episodes(root / rel)
    return {rel: rows for rel, rows in sorted(out.items()) if rows is not None}


def last_reply(replies) -> str:
    """The text of an episode's last reply (a string, or a message dict with content)."""
    last = replies[-1]
    if isinstance(last, dict):
        last = last.get("content") or last.get("text") or ""
    return last


def extract(reply: str, code_re) -> str:
    """The code KoanTask runs: the first block code_re matches, else the whole reply stripped."""
    m = code_re.search(reply)
    return m.group(1) if m else reply.strip()


def source_of(rel: str) -> str:
    """A file's source: the first folder of its path under results/."""
    return Path(rel).parts[0]


def collect(root, code_re, exclude=()) -> dict:
    """Every graded episode's code under root, deduplicated on (pair, sha256 of the code). Returns
    {samples: {(pair, sha): sample}, files: [one dict per episode file], duplicates: [[paths with equal bytes]],
    longest_reply: characters of the longest graded last reply, counts: {...}}."""
    root = Path(root)
    found = episode_files(root, exclude)
    samples: dict[tuple[str, str], dict] = {}
    files, by_hash = [], collections.defaultdict(list)
    longest, episodes = 0, 0
    per_source = collections.defaultdict(lambda: {"episodes": 0, "samples": set()})
    for rel, rows in found.items():
        data = (root / rel).read_bytes()
        h = sha256_bytes(data)
        by_hash[h].append(rel)
        stat = {"path": rel, "sha256": h, "bytes": len(data), "records": len(rows), "solvable": 0, "graded": 0,
                "solved": 0, "samples": 0, "duplicate_of": by_hash[h][0] if len(by_hash[h]) > 1 else None}
        mine = set()
        for r in rows:
            if r["impossible"]:
                continue
            stat["solvable"] += 1
            if r["outcome"] not in GRADED:
                continue
            stat["graded"] += 1
            stat["solved"] += r["outcome"] == "solved"
            reply = last_reply(r["replies"])
            longest = max(longest, len(reply))
            code = extract(reply, code_re)
            key = (r["pair"], sha256_bytes(code.encode()))
            s = samples.setdefault(key, {"pair": key[0], "sha256": key[1], "code": code,
                                         "verdicts": collections.Counter(), "episodes": 0, "sources": set(),
                                         "episodes_by_file": collections.Counter()})
            s["verdicts"][r["outcome"]] += 1
            s["episodes"] += 1
            s["sources"].add(source_of(rel))
            s["episodes_by_file"][rel] += 1
            mine.add(key)
            episodes += 1
            per_source[source_of(rel)]["episodes"] += 1
            per_source[source_of(rel)]["samples"].add(key)
        stat["samples"] = len(mine)
        files.append(stat)
    for s in samples.values():
        s["verdict"] = next(iter(s["verdicts"])) if len(s["verdicts"]) == 1 else "conflict"
    solved = sum(s["verdict"] == "solved" for s in samples.values())
    counts = {"episode_files": len(files), "contributing_files": sum(f["graded"] > 0 for f in files),
              "duplicate_files": sum(f["duplicate_of"] is not None for f in files), "episodes": episodes,
              "samples": len(samples), "pairs": len({k[0] for k in samples}),
              "conflicts": sum(s["verdict"] == "conflict" for s in samples.values()),
              "solved": solved, "solved_share": round(100 * solved / len(samples), 1) if samples else None,
              "episode_weighted_solved_share": round(100 * sum(s["episodes"] for s in samples.values()
                                                               if s["verdict"] == "solved") / episodes, 1)
              if episodes else None,
              "in_more_than_one_episode": sum(s["episodes"] > 1 for s in samples.values()),
              "by_source": {src: {"episodes": v["episodes"], "samples": len(v["samples"])}
                            for src, v in sorted(per_source.items())}}
    return {"samples": dict(sorted(samples.items())), "files": files,
            "duplicates": [paths for paths in by_hash.values() if len(paths) > 1], "longest_reply": longest,
            "counts": counts}


def sample_record(s: dict) -> dict:
    """One line of the samples file."""
    return {"pair": s["pair"], "sha256": s["sha256"], "code": s["code"], "verdict": s["verdict"],
            "episodes": s["episodes"], "sources": sorted(s["sources"]),
            "episodes_by_file": dict(sorted(s["episodes_by_file"].items()))}


def write(found: dict, out_dir, samples_name: str = "c6c_samples.jsonl", files_name: str = "c6c_files.json",
          extra: dict | None = None) -> dict:
    """The samples file (one JSON line per sample, sorted by pair and sha256) and the files list (every episode file
    with its sha256 and contribution, the byte-duplicate groups, the counts and the samples file's sha256).
    Returns the files list."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(sample_record(s), ensure_ascii=False) + "\n" for s in found["samples"].values())
    data = lines.encode("utf-8")
    (out / samples_name).write_bytes(data)
    listing = {**(extra or {}), "counts": found["counts"], "longest_graded_last_reply": found["longest_reply"],
               "files": found["files"], "duplicates": found["duplicates"],
               "samples_file": {"name": samples_name, "sha256": sha256_bytes(data), "lines": len(found["samples"])}}
    (out / files_name).write_text(json.dumps(listing, indent=1) + "\n")
    return listing


def load_samples(path) -> list[dict]:
    """The samples file's records."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
