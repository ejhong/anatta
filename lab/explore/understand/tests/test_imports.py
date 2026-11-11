"""Importing step 3's modules makes no network call and loads neither modal nor torch; the two dry runs make no
network call either. Each check runs in a fresh interpreter whose sockets and urlopen raise. Claude (Fable 5.2, an AI
model) wrote this file on 10 Nov 2026. Explore speed."""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]

BLOCK = """
import socket, sys, urllib.request
def refuse(*a, **k):
    raise RuntimeError("a network call")
socket.socket.connect = refuse
socket.socket.connect_ex = refuse
socket.create_connection = refuse
socket.getaddrinfo = refuse
urllib.request.urlopen = refuse
sys.path.insert(0, {here!r})
"""


def run(code: str, tmp_path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", BLOCK.format(here=str(HERE)) + code], capture_output=True, text=True,
                          cwd=tmp_path, timeout=120)


def test_importing_the_modules_calls_no_network_and_loads_no_modal_or_torch(tmp_path):
    p = run("""
import u1_core
assert "torch" not in sys.modules and "modal" not in sys.modules, "u1_core is the pure side"
import run_u1, compare_u1, gen_docs
assert "modal" not in sys.modules and "torch" not in sys.modules
assert "gen_declarations" not in sys.modules, "the pilot's client is imported on first use"
print("imported")
""", tmp_path)
    assert p.returncode == 0 and p.stdout.strip() == "imported", p.stderr[-2000:]


def test_the_dry_runs_call_no_network(tmp_path):
    p = run("""
import gen_docs, run_u1
assert gen_docs.main(["--dry"]) == 0 and run_u1.main(["--dry"]) == 0
assert "modal" not in sys.modules
""", tmp_path)
    assert p.returncode == 0, p.stderr[-2000:]
    assert "Dry run: nothing called, nothing spent." in p.stdout and "Dry run: nothing launched, nothing spent." in p.stdout


def test_a_live_call_with_sockets_shut_fails_before_it_can_spend(tmp_path):
    """The guard that keeps tests offline is real: with the network shut, the pilot's client cannot reach anything."""
    p = run("""
import os
os.environ.pop("OPENROUTER_API_KEY", None)
import gen_docs
try:
    gen_docs.pilot().chat("m", [], 0.0, 1, retries=1)
except SystemExit as e:
    print("refused:", "OPENROUTER_API_KEY" in str(e))
""", tmp_path)
    assert p.returncode == 0 and p.stdout.strip() == "refused: True", p.stderr[-2000:]
