"""anatta-overlap-dl: download the one model leg 2 needs (the self-other overlap paper's 7B model) into the
anatta-hf-cache volume, in the HF hub cache layout the lab's apps read. Eugene approved the download on 19 Nov 2026
("yes go on both legs": the paper's 7B model, about 15 GB). Claude (Fable 5.2, an AI model) wrote this file on
19 Nov 2026, after modal_selfmap_dl.py. CPU only. Touches only app anatta-overlap-dl and volume anatta-hf-cache.

What is fetched: mistralai/Mistral-7B-Instruct-v0.2 at commit 63a8b081895390a26e140280378bc85ec8bce07a (Apache-2.0,
not gated): the three safetensors files (4.94, 5.00 and 4.54 GB), config and tokenizer files. The older .bin copies
of the same weights (15 GB) are left out.

  tools/modal.sh run lab/explore/downloads/modal_overlap_dl.py::plan      # check the commit and sizes, no download
  tools/modal.sh run lab/explore/downloads/modal_overlap_dl.py::fetch     # download, then commit the volume
  tools/modal.sh run lab/explore/downloads/modal_overlap_dl.py::verify    # files present, bytes, the config loads
"""

import modal

CACHE = "/cache"
HUB = f"{CACHE}/hf/hub"
LIMIT_GB = 16.0
REPO, BRANCH, SHA = "mistralai/Mistral-7B-Instruct-v0.2", "main", "63a8b081895390a26e140280378bc85ec8bce07a"
ALLOW = ["*.safetensors", "*.safetensors.index.json", "config.json", "generation_config.json", "tokenizer*",
         "special_tokens_map.json", "*.model", "chat_template*"]

image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("transformers==5.21.0", "tokenizers==0.23.3", "huggingface_hub==2.3.0", "hf_xet==1.7.0",
                         "safetensors==0.8.0", "jinja2==3.1.6")
         .env({"HF_HOME": f"{CACHE}/hf", "HF_HUB_DISABLE_TELEMETRY": "1"}))
app = modal.App("anatta-overlap-dl", image=image)
vol = modal.Volume.from_name("anatta-hf-cache")


def _wanted(name: str) -> bool:
    from fnmatch import fnmatch

    return any(fnmatch(name, p) for p in ALLOW)


def _expected() -> tuple[str | None, dict]:
    """(the branch's commit now, {file: bytes} of the wanted files at the pinned commit)."""
    from huggingface_hub import HfApi

    api = HfApi()
    cur = {b.name: b.target_commit for b in api.list_repo_refs(REPO).branches}.get(BRANCH)
    info = api.model_info(REPO, revision=SHA, files_metadata=True)
    return cur, {s.rfilename: s.size for s in info.siblings if _wanted(s.rfilename)}


@app.function(timeout=600)
def plan_remote():
    return _expected()


@app.local_entrypoint()
def plan():
    cur, files = plan_remote.remote()
    for name, size in sorted(files.items()):
        print(f"  {(size or 0) / 1e9:7.3f} GB  {name}")
    total = sum(v or 0 for v in files.values()) / 1e9
    print(f"{REPO} at {SHA[:12]}: {len(files)} files, {total:.2f} GB (limit {LIMIT_GB}); the branch is now at {str(cur)[:12]}")
    if total > LIMIT_GB:
        raise SystemExit("over the limit: nothing should be fetched")


@app.function(timeout=3 * 3600, volumes={CACHE: vol}, cpu=4.0, memory=8192)
def fetch_remote():
    import os

    from huggingface_hub import snapshot_download

    _, files = _expected()
    if sum(v or 0 for v in files.values()) / 1e9 > LIMIT_GB:
        return "REFUSED: over the limit"
    p = snapshot_download(REPO, revision=SHA, cache_dir=HUB, allow_patterns=ALLOW, max_workers=8)
    if os.path.basename(p) != SHA:
        return f"WRONG COMMIT {os.path.basename(p)}"
    vol.commit()
    return "ok"


@app.local_entrypoint()
def fetch():
    print(fetch_remote.remote())


@app.function(timeout=1800, volumes={CACHE: vol}, cpu=2.0, memory=4096)
def verify_remote():
    import os

    from transformers import AutoConfig

    vol.reload()
    _, files = _expected()
    d = f"{HUB}/models--{REPO.replace('/', '--')}/snapshots/{SHA}"
    missing = [f for f in files if not os.path.exists(os.path.join(d, f))]
    badsize = [f for f, sz in files.items() if f not in missing and sz is not None and os.path.getsize(os.path.join(d, f)) != sz]
    nbytes = sum(os.path.getsize(os.path.join(d, f)) for f in files if f not in missing)
    try:
        c = AutoConfig.from_pretrained(d)
        cfg = f"{type(c).__name__}/{c.model_type}, {c.num_hidden_layers} layers"
    except Exception as e:  # reported, not raised: the caller prints it
        cfg = f"FAIL {type(e).__name__}: {str(e)[:120]}"
    return d, nbytes, len(files) - len(missing), len(files), missing, badsize, cfg


@app.local_entrypoint()
def verify():
    d, nbytes, have, want, missing, badsize, cfg = verify_remote.remote()
    print(f"{d}\n{nbytes:,d} B = {nbytes / 1e9:.2f} GB; {have}/{want} files; missing={missing} badsize={badsize}; {cfg}")
