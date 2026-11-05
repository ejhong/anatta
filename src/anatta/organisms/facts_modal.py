"""The organisms' facts stage (clean-lab step 6): every adapter folder under a prefix of anatta-runs hashed in a small
CPU container, read-only, nothing downloaded to the laptop. Claude (Opus 5.5) wrote this file, 4 Nov 2026. It mirrors
R1c's `--stage facts` (experiments/run_r1b.py adapter_facts_remote) with anatta.organisms.dir_facts, and records every
file in each folder, not only the weights.

  tools/modal.sh run src/anatta/organisms/facts_modal.py::main --prefix /runs/act1/full

Writes <results_root>/organisms/facts-<UTC stamp>.json (outside git): {"prefix", "utc", "container_s", "usd",
"folders": {folder: {"files": {name: {bytes, sha256}}, "config": {r, lora_alpha, use_rslora, target_modules,
base_model_name_or_path}}}}. The adapter_config.json is read as data (fields only). The cost (2 cores, 4 GiB, the
container's own seconds plus a minute) goes to the day's spend file under tool "organisms" via anatta.spend.add_spend.
"""

import json
import time

import modal

from anatta.modal_app import RUNS, image, runs

CORES, MEM_GIB, TAIL_S = 2.0, 4.0, 60.0
app = modal.App("anatta-organisms", image=image)


@app.function(volumes={RUNS: runs.read_only()}, cpu=CORES, memory=int(MEM_GIB * 1024), timeout=1800)
def hash_tree(prefix: str) -> dict:
    """Each folder under prefix holding an adapter_model.safetensors: every file's bytes and sha256, and the LoRA
    fields of its adapter_config.json."""
    import os

    from anatta.organisms import dir_facts

    t0 = time.time()
    out = {}
    for d, _subdirs, files in sorted(os.walk(prefix)):
        if "adapter_model.safetensors" not in files:
            continue
        rec = {"files": dir_facts(d)}
        c = os.path.join(d, "adapter_config.json")
        if os.path.isfile(c):
            cfg = json.loads(open(c, encoding="utf-8").read())
            rec["config"] = {k: cfg.get(k) for k in ("r", "lora_alpha", "use_rslora", "target_modules",
                                                     "base_model_name_or_path")}
        out[d] = rec
    return {"folders": out, "container_s": round(time.time() - t0, 1)}


@app.local_entrypoint()
def main(prefix: str = "/runs/act1/full"):
    from anatta import spend
    from anatta.config import load

    if not prefix.startswith(f"{RUNS}/"):
        raise SystemExit(f"prefix must be under {RUNS}/")
    t0 = time.time()
    res = hash_tree.remote(prefix)
    wall = time.time() - t0
    rate = CORES * spend.CPU_USD_PER_CORE_HOUR + MEM_GIB * spend.MEM_USD_PER_GIB_HOUR
    usd = round((wall + TAIL_S) * rate / 3600, 6)  # the laptop's wall time bounds the container's (start-up included)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    spend.add_spend(spend.default_spend_dir(), tool="organisms", label=f"facts{prefix}", usd=usd)
    out = load().results_root / "organisms" / f"facts-{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"refused: {out} exists")
    out.write_text(json.dumps({"prefix": prefix, "utc": stamp, "wall_s": round(wall, 1), "usd": usd, **res},
                              indent=1) + "\n")
    print(json.dumps({"written": str(out), "folders": len(res["folders"]), "container_s": res["container_s"],
                      "wall_s": round(wall, 1), "usd": usd}))
