#!/usr/bin/env python3
"""PreToolUse guard for the anatta repo: refuses tool calls that break the project's hard rules (CLAUDE.md).

Claude Code passes the call as JSON on stdin; exit 2 blocks it and shows the reason to Claude. Everything else passes
(exit 0). It reads only the call, never files or the environment. A relative file path is resolved against the call's
`cwd` and normalized, so `..` cannot step around a rule. Claude (Opus 5.5) wrote the live copy on 2 Nov 2026 and this
repo copy on 3 Nov 2026; every rule of the live copy is kept (tests/test_guard.py). Strengthened the same day (Fable's
cold check): `git add -- .` and its spellings, `git push --tags`, `find -delete` and `mv` on data, and Bash writes into
anatta-dev (the live copy's proposal: lab/plans/live-guard-proposal.diff).

Run by .claude/settings.json as: python3 -I "$CLAUDE_PROJECT_DIR/tools/guard.py"
"""

import json
import os
import re
import sys

BASH_RULES = [
    (r"\bgit\b[^\n;&|]*\badd\b[^\n;&|]*\s(-A\b|--all\b|([\"']?)\.\/?\2(?=\s|$|[;&|]))",
     "never `git add -A` / `git add .`: commit with a pathspec (git commit -F msg -- <paths>)"),
    (r"--no-verify\b", "never --no-verify (the pre-push hook guards the public repo)"),
    (r"\bgit\b[^\n;&|]*\bpush\b[^\n;&|]*\borigin\b[^\n;&|]*\b(lab|calib/|dev/|consolidate/|consolidation-before-|lib/)",
     "never push lab, calib/*, dev/*, consolidate/* branches or private tags to origin (public)"),
    (r"\bgit\b[^\n;&|]*\bpush\b[^\n;&|]*(--force\b|\s-f\b)", "no force pushes without Eugene's explicit OK"),
    (r"\bgit\b[^\n;&|]*\bpush\b[^\n;&|]*\s--(follow-)?tags\b",
     "never push --tags / --follow-tags (private lib/ and consolidation tags would go public): push a named ref"),
    (r"\bgit\s+stash(\s+(push|save))?\s*($|[;&|])", "stash only with a unique tag: git stash push -u -m <tag>"),
    (r"\bgit\s+stash\s+pop\b", "never `git stash pop` (shared stash): apply by sha, then drop by tag"),
    (r"\b(printenv|env)\s*($|[;&|>])", "don't dump the environment (keys live there)"),
    (r"(echo|printf|cat)[^\n;&|]*\$\{?[A-Z_]*(KEY|TOKEN|SECRET|PASSWORD)\b", "never print a key or token"),
    (r"cat\s+[^\n;&|]*(\.modal\.toml|\.netrc|hf_token|/\.huggingface/token|\.env\b)", "never print credential files"),
    (r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f?|-[a-zA-Z]*f[a-zA-Z]*r)[^\n;&|]*"
     r"(results|data|archive-2026-11|explore-results|calib-results|anatta-lab|anatta-calib|ledger)",
     "never permanently delete data: archive it instead, and ask Eugene"),
    (r"\bfind\b(?=[^\n;&|]*(results|data|archive-2026-11|anatta-lab|anatta-calib|ledger))[^\n;&|]*\s-delete\b",
     "never permanently delete data (find -delete): archive it instead, and ask Eugene"),
    (r"\bmv\b[^\n;&|]*(^|[\s/'\"])([\w.-]*results|data|ledger)(/|[\s'\"]|$)",
     "don't move results, data or ledger files from Bash: ask Eugene (archiving is a deliberate, logged step)"),
    # Bash writes into the pinned library worktree (Edit/Write are refused by PROTECTED_EDITS)
    (r"(>>?|\btee\b(\s+-\S+)*)\s*[\"']?[^\s;&|]*anatta-dev(/|[\s\"']|$)",
     "anatta-dev is the pinned library (read-only); moving the pin needs a new lib/ tag"),
    (r"\bsed\b[^\n;&|]*\s(-[a-zA-Z]*i|--in-place)[^\n;&|]*anatta-dev(/|\s|$)",
     "anatta-dev is the pinned library (read-only); moving the pin needs a new lib/ tag"),
    (r"\b(cp|mv|rsync|install)\b[^\n;&|]*\s[\"']?[^\s;&|]*anatta-dev(/[^\s;&|]*)?[\"']?\s*($|[;&|])",
     "anatta-dev is the pinned library (read-only); moving the pin needs a new lib/ tag"),
    (r"\bmodal\b[^\n;&|]*\b(app\s+stop|volume\s+(rm|delete)|volume\s+put|deploy|run)\b[^\n]*\bmatlab",
     "never touch the matlab-* Modal apps or volumes"),
    (r"\bgh\s+repo\s+edit\b[^\n]*--visibility", "repo visibility changes need Eugene"),
    (r"\bgh\s+pr\s+merge\b", "never merge your own PRs"),
]
PROTECTED_EDITS = [
    (r"(^|/)probe\.py$", "don't edit probe.py"),
    # the pinned library worktree (a sibling of this repo) is read-only while it exists
    (r"(^|/)anatta-dev(/|$)", "anatta-dev is the pinned library (read-only); moving the pin needs a new lib/ tag"),
    (r"(^|/)archive-2026-11/(premove|worktrees)/", "archived copies are read-only"),
]


def block(why: str) -> None:
    print(f"Blocked by the anatta guard (CLAUDE.md): {why}", file=sys.stderr)
    sys.exit(2)


def resolve(path: str, cwd: str) -> str:
    """The path as given, plus (for a relative path) resolved against the call's cwd; both normalized."""
    if not path:
        return path
    full = path if os.path.isabs(path) or not cwd else os.path.join(cwd, path)
    return os.path.normpath(full)


def main() -> None:
    try:
        call = json.load(sys.stdin)
    except Exception:
        sys.exit(0)
    if not isinstance(call, dict):
        sys.exit(0)
    tool = call.get("tool_name", "")
    inp = call.get("tool_input") or {}
    if tool == "Bash":
        cmd = inp.get("command", "")
        for rx, why in BASH_RULES:
            if re.search(rx, cmd):
                block(why)
    elif tool in ("Edit", "Write", "NotebookEdit", "MultiEdit"):
        raw = inp.get("file_path") or inp.get("notebook_path") or ""
        paths = {raw, resolve(raw, call.get("cwd") or "")}
        for path in paths:
            for rx, why in PROTECTED_EDITS:
                if re.search(rx, path):
                    block(why)
            if tool == "Write" and (path.endswith("/lab/ledger/NOTES.md") or path == "lab/ledger/NOTES.md"):
                block("NOTES.md is append-only: append with Edit or `cat >> <<'EOF'`, never rewrite it")
    sys.exit(0)


if __name__ == "__main__":
    main()
