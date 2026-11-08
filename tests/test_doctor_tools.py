"""tools/doctor.py's machine-tools check on a temporary Brewfile (no install; brew may or may not be on PATH). Claude
(Opus 5.5) wrote this file, 8 Nov 2026."""

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("doctor", Path(__file__).resolve().parents[1] / "tools" / "doctor.py")
doctor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(doctor)


def test_a_missing_brewfile_warns(tmp_path):
    status, detail = doctor.check_tools(tmp_path / "Brewfile")
    assert status == "WARN" and "missing" in detail


def test_a_formula_that_is_not_installed_warns(tmp_path):
    f = tmp_path / "Brewfile"
    f.write_text('# a comment\nbrew "some/tap/no-such-formula-xyz"  # trailing comment\n')
    status, detail = doctor.check_tools(f)
    assert status == "WARN" and ("no-such-formula-xyz" in detail or "no brew" in detail)
