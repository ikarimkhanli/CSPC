import pytest
import numpy as np
from decay import simulate

def test_starts_at_N0():
    assert simulate(1000, 0.4)[0] == 1000

def test_negative_rate_raises_valueerror():
    with pytest.raises(ValueError):
        simulate(1000, -0.4)

def test_average_decay_matches_theory():
    N0 = 1000
    r = 0.4
    runs = [simulate(N0, r)[-1] for _ in range(1000)]
    expected = N0 * np.exp(-r)
    assert np.isclose(np.mean(runs), expected, atol=80)
