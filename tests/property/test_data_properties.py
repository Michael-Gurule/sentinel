"""Property-based tests for dataset configuration."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from sentinel.data.config import Range


@given(
    st.floats(1e-3, 1e3), st.floats(1.0, 1e3), st.booleans(), st.integers(0, 2**32 - 1)
)
def test_range_samples_stay_in_bounds(low, span, log, seed):
    r = Range(low=low, high=low * span if log else low + span, log=log)
    value = r.sample(np.random.default_rng(seed))
    assert r.low <= value <= r.high * (1 + 1e-12)
