"""tools/guard.py, run as Claude Code runs it: a subprocess with the call as JSON on stdin; exit 2 blocks. Every case the
live session guard blocks is still blocked. Claude (Opus 5.5) wrote this file, 3 Nov 2026."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

GUARD = Path(__file__).resolve().parents[1] / "tools" / "guard.py"
LIVE = Path("/Users/eugene/prj/align/.claude/hooks/guard.py")  # the session copy outside the repo, if present
CWD = "/Users/eugene/prj/align/anatta"


def run(guard: Path, call: dict) -> int:
    r = subprocess.run([sys.executable, "-I", str(guard)], input=json.dumps(call), capture_output=True, text=True)
    return r.returncode


def bash(cmd: str) -> dict:
    return {"tool_name": "Bash", "tool_input": {"command": cmd}, "cwd": CWD}


def edit(path: str, tool: str = "Edit") -> dict:
    key = "notebook_path" if tool == "NotebookEdit" else "file_path"
    return {"tool_name": tool, "tool_input": {key: path}, "cwd": CWD}


BLOCKED = [
    bash("git add -A"),
    bash("git add ."),
    bash("git -C anatta add --all"),
    bash("git push --no-verify origin main"),
    bash("git commit --no-verify -m x"),
    bash("git push origin lab"),
    bash("git push origin calib/K1-graded"),
    bash("git push origin dev/x"),
    bash("git push origin consolidate/step3"),
    bash("git push origin consolidation-before-2026-11"),
    bash("git push origin lib/ministral-ee89587"),
    bash("git push --force origin main"),
    bash("git push -f origin main"),
    bash("git stash"),
    bash("git stash push"),
    bash("git stash pop"),
    bash("printenv"),
    bash("env | grep X"),
    bash("echo $OPENROUTER_API_KEY"),
    bash("printf '%s' ${HF_TOKEN}"),
    bash("cat ~/.modal.toml"),
    bash("cat ~/.netrc"),
    bash("cat .env"),
    bash("rm -rf /Users/eugene/prj/align/results/calib"),
    bash("rm -fr lab/ledger"),
    bash("rm -rf data/k6"),
    bash("modal app stop matlab-train"),
    bash("modal volume rm matlab-cache"),
    bash("gh repo edit --visibility private"),
    bash("gh pr merge 12"),
    # bypasses closed on 3 Nov (Fable's cold check, item 7)
    bash("git add -- ."),
    bash("git add ./"),
    bash('git add "."'),
    bash("git add -v -- './'"),
    bash("git push --tags"),
    bash("git push origin --tags"),
    bash("git push --follow-tags origin main"),
    bash("find /Users/eugene/prj/align/results/explore -name '*.json' -delete"),
    bash("find lab/ledger -delete"),
    bash("find data -type f -delete"),
    bash("mv /Users/eugene/prj/align/results/calib /tmp/x"),
    bash("mv lab/ledger/NOTES.md /tmp/"),
    bash("mv data/k6 data/k6-old"),
    bash("echo x | tee /Users/eugene/prj/align/anatta-dev/src/anatta/dial.py"),
    bash("echo x | tee -a ../anatta-dev/x.py"),
    bash("sed -i '' 's/a/b/' ../anatta-dev/src/anatta/models.py"),
    bash("cp src/anatta/dial.py ../anatta-dev/src/anatta/dial.py"),
    bash("echo x > ../anatta-dev/src/anatta/dial.py"),
    bash("echo x >> /Users/eugene/prj/align/anatta-dev/README.md"),
    bash("mv /tmp/dial.py /Users/eugene/prj/align/anatta-dev/src/anatta/"),
    edit("/Users/eugene/prj/align/anatta/src/anatta/probe.py"),
    edit("/Users/eugene/prj/align/anatta-dev/src/anatta/dial.py"),
    edit("/Users/eugene/prj/align/anatta-dev/src/anatta/dial.py", "Write"),
    edit("/Users/eugene/prj/align/archive-2026-11/premove/x.py"),
    edit("/Users/eugene/prj/align/archive-2026-11/worktrees/anatta-lab/y.md", "Write"),
    edit("/Users/eugene/prj/align/anatta/lab/ledger/NOTES.md", "Write"),
    # relative and `..` paths cannot step around a rule
    edit("lab/ledger/NOTES.md", "Write"),
    edit("../anatta-dev/src/anatta/models.py"),
    edit("src/anatta/probe.py", "MultiEdit"),
    edit("/Users/eugene/prj/align/anatta/../anatta-dev/x.ipynb", "NotebookEdit"),
]

ALLOWED = [
    bash("git status --short"),
    bash("git commit -q -F msg.txt -- lab/ledger/NOTES.md"),
    bash("git push -q origin main"),
    bash("git stash push -u -m clean-lab-step1"),
    bash("date -u"),
    bash("git add lab/explore/p1/p1_spend.py ./tools/guard.py"),
    bash("find lab/explore -name '*.pyc'"),
    bash("find . -name __pycache__ -prune -o -name '*.tmp' -delete"),
    bash("mv lab/plans/a.md lab/plans/b.md"),
    bash("cat ../anatta-dev/src/anatta/dial.py"),
    bash("cp ../anatta-dev/src/anatta/dial.py /tmp/dial-pin.py"),
    bash("git -C ../anatta-dev log --oneline > /tmp/pin-log.txt"),
    bash("sed -n 1,20p ../anatta-dev/src/anatta/models.py"),
    edit("/Users/eugene/prj/align/anatta/lab/ledger/NOTES.md"),  # Edit appends; only Write is refused
    edit("/Users/eugene/prj/align/anatta/src/anatta/spend.py", "Write"),
    edit("/Users/eugene/prj/align/anatta/tools/doctor.py"),
]


@pytest.mark.parametrize("call", BLOCKED, ids=lambda c: json.dumps(c["tool_input"])[:60])
def test_blocked(call):
    assert run(GUARD, call) == 2


@pytest.mark.parametrize("call", ALLOWED, ids=lambda c: json.dumps(c["tool_input"])[:60])
def test_allowed(call):
    assert run(GUARD, call) == 0


def test_bad_input_passes():
    r = subprocess.run([sys.executable, "-I", str(GUARD)], input="not json", capture_output=True, text=True)
    assert r.returncode == 0


@pytest.mark.skipif(not LIVE.exists(), reason="the live session guard is not on this machine")
@pytest.mark.parametrize("call", BLOCKED + ALLOWED, ids=lambda c: json.dumps(c["tool_input"])[:60])
def test_blocks_at_least_what_the_live_guard_blocks(call):
    if run(LIVE, call) == 2:
        assert run(GUARD, call) == 2
