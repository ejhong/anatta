"""The harness's test setup: lab/ on sys.path so `import harness` works, and synthetic helpers. No network, GPU, Modal
or real model; the noself3 fixture files are summary numbers only. Claude (Opus 5.5) wrote this file, 2 Nov 2026."""

import sys
from pathlib import Path

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[2]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))


def binary_units(p: np.ndarray, samples: int, rng) -> dict:
    """Per-item means of `samples` Bernoulli(p_i) draws."""
    return {f"i{i:03d}": float(rng.binomial(samples, pi) / samples) for i, pi in enumerate(p)}


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
