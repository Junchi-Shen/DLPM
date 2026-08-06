"""Small deterministic checks for the public payoff module."""
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dlpm"))

from Game.option_payoffs import calculate_asian_payoff, calculate_european_payoff


def test_vanilla_call_intrinsic_value():
    paths = np.array([[100.0, 110.0], [100.0, 90.0]])
    assert np.allclose(calculate_european_payoff(paths, 1, 100.0), [10.0, 0.0])


def test_asian_call_nonnegative():
    paths = np.array([[100.0, 120.0], [100.0, 80.0]])
    assert np.all(calculate_asian_payoff(paths, 1, 100.0) >= 0.0)
