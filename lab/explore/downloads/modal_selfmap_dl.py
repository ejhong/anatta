"""anatta-selfmap-dl: download the self-map checkpoints Eugene approved on 2 Nov 2026 into the anatta-hf-cache volume,
in the HF hub cache layout bench reads (/cache/hf/hub/models--org--name/snapshots/<sha>, plus refs/<branch>).
Claude (Opus 5.5) wrote this file, 2 Nov 2026. CPU only. Touches only app anatta-selfmap-dl and volume anatta-hf-cache.

  modal run modal_selfmap_dl.py::plan      # check branch shas and sizes, no download
  modal run modal_selfmap_dl.py::fetch     # download (one container per repo), then commit
  modal run modal_selfmap_dl.py::verify    # files present, bytes, AutoConfig loads
"""

import modal

CACHE = "/cache"
HUB = f"{CACHE}/hf/hub"
LIMIT_GB = 210.0

# repo -> [(branch, expected full commit sha)]
ITEMS = {
    "allenai/Olmo-3-1025-7B": [
        ("main", "a81bae42db39"),
        ("stage1-step353000", "14886f6c3b94a84a0b74b7f63b13ea48d26ac9d7"),
        ("stage1-step1060000", "a77692e1992c089662d00df92d95c6a96b212ceb"),
    ],
    "allenai/Olmo-3-7B-Instruct-SFT": [("main", "e1452fc572d5")],
    "allenai/Olmo-3-7B-Instruct-DPO": [("main", "b33130b7de49")],
    "allenai/Olmo-3-7B-Instruct": [
        ("main", "6e5971d9eba4"), ("step_100", "b705ea628af0"), ("step_200", "71e18f5221e2"), ("step_300", "aa8dffc0678e"),
    ],
    "allenai/Olmo-3-7B-RL-Zero-General": [
        ("step_100", "1af5061a0bb6"), ("step_400", "b655dd338ec7"), ("main", "24e5814cb369"),
    ],
    "mistralai/Ministral-3-8B-Base-2512": [("main", "d4883f9b36aa")],
}
ALLOW = ["*.safetensors", "*.safetensors.index.json", "config.json", "generation_config.json", "tokenizer*",
         "special_tokens_map.json", "*.model", "*.tiktoken", "vocab*", "merges.txt", "chat_template*",
         "*processor_config.json"]
IGNORE = ["consolidated*", "original/*"]

image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("transformers==5.21.0", "tokenizers==0.23.3", "huggingface_hub==2.3.0", "hf_xet==1.7.0",
                         "safetensors==0.8.0", "jinja2==3.1.6")
         .env({"HF_HOME": f"{CACHE}/hf", "HF_HUB_DISABLE_TELEMETRY": "1"}))
app = modal.App("anatta-selfmap-dl", image=image)
vol = modal.Volume.from_name("anatta-hf-cache")


def _wanted(name):
    from fnmatch import fnmatch
    return any(fnmatch(name, p) for p in ALLOW) and not any(fnmatch(name, p) for p in IGNORE)


@app.function(timeout=600)
def plan_remote():
    from huggingface_hub import HfApi
    api, rows, total, bad = HfApi(), [], 0, []
    for repo, revs in ITEMS.items():
        refs = {b.name: b.target_commit for b in api.list_repo_refs(repo).branches}
        for branch, sha in revs:
            cur = refs.get(branch)
            if cur is None or not cur.startswith(sha):
                bad.append((repo, branch, sha, cur)); continue
            info = api.model_info(repo, revision=cur, files_metadata=True)
            files = [s for s in info.siblings if _wanted(s.rfilename)]
            b = sum(s.size or 0 for s in files)
            total += b
            rows.append((repo, branch, cur, b, len(files)))
    return rows, total, bad


@app.local_entrypoint()
def plan():
    rows, total, bad = plan_remote.remote()
    for r in rows: print(f"{r[0]:40s} {r[1]:20s} {r[2][:12]} {r[3]/1e9:7.2f} GB {r[4]:3d} files")
    print("MISMATCH:", bad)
    print(f"TOTAL {total/1e9:.2f} GB (limit {LIMIT_GB})")


@app.function(timeout=6 * 3600, volumes={CACHE: vol}, cpu=4.0, memory=8192)
def fetch_repo(repo):
    import os
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    refs = {b.name: b.target_commit for b in api.list_repo_refs(repo).branches}
    out = []
    for branch, sha in ITEMS[repo]:
        cur = refs.get(branch)
        if cur is None or not cur.startswith(sha):
            out.append((repo, branch, sha, "SKIPPED: branch now at " + str(cur))); continue
        # revision=branch writes refs/<branch>; the path check confirms the snapshot is the expected commit
        p = snapshot_download(repo, revision=branch, cache_dir=HUB, allow_patterns=ALLOW, ignore_patterns=IGNORE,
                              max_workers=8)
        got = os.path.basename(p)
        if not got.startswith(sha):
            out.append((repo, branch, sha, f"WRONG COMMIT {got}")); continue
        vol.commit()
        out.append((repo, branch, got, "ok"))
    return out


@app.local_entrypoint()
def fetch():
    for res in fetch_repo.map(list(ITEMS)):
        for r in res: print(r)


@app.function(timeout=1800, volumes={CACHE: vol}, cpu=2.0, memory=4096)
def verify_remote():
    import os
    from huggingface_hub import HfApi
    from transformers import AutoConfig
    vol.reload()
    api, rows = HfApi(), []
    for repo, revs in ITEMS.items():
        refs = {b.name: b.target_commit for b in api.list_repo_refs(repo).branches}
        for branch, sha in revs:
            cur = refs.get(branch, "")
            d = f"{HUB}/models--{repo.replace('/', '--')}/snapshots/{cur}"
            ok = os.path.isdir(d)
            expected = {s.rfilename: s.size for s in api.model_info(repo, revision=cur, files_metadata=True).siblings
                        if _wanted(s.rfilename)} if ok else {}
            have, nbytes, missing, badsize = 0, 0, [], []
            for f, sz in expected.items():
                fp = os.path.join(d, f)
                if not os.path.exists(fp): missing.append(f); continue
                n = os.path.getsize(fp); have += 1; nbytes += n
                if sz is not None and n != sz: badsize.append(f)
            ref = open(f"{HUB}/models--{repo.replace('/', '--')}/refs/{branch}").read().strip() \
                if os.path.exists(f"{HUB}/models--{repo.replace('/', '--')}/refs/{branch}") else None
            try:
                c = AutoConfig.from_pretrained(d); cfg = f"{type(c).__name__}/{c.model_type}"
            except Exception as e:
                cfg = f"FAIL {type(e).__name__}: {str(e)[:120]}"
            rows.append((repo, branch, cur, nbytes, have, len(expected), missing, badsize, ref == cur, cfg))
    return rows


@app.local_entrypoint()
def verify():
    tot = 0
    for r in verify_remote.remote():
        tot += r[3]
        print(f"{r[0]:40s} {r[1]:20s} {r[2]} {r[3]:>14,d} B {r[4]}/{r[5]} files missing={r[6]} badsize={r[7]} "
              f"ref_ok={r[8]} {r[9]}")
    print(f"TOTAL {tot:,d} B = {tot/1e9:.2f} GB")
