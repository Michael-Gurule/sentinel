import pytest

from sentinel.core import chi2_gate, mean_nees_bounds


def test_chi2_gate_known_values():
    assert chi2_gate(1, 0.95) == pytest.approx(3.841, abs=1e-3)
    assert chi2_gate(3, 0.99) == pytest.approx(11.345, abs=1e-3)


@pytest.mark.parametrize(("dof", "probability"), [(0, 0.9), (3, 0.0), (3, 1.0)])
def test_chi2_gate_rejects_invalid_arguments(dof, probability):
    with pytest.raises(ValueError, match="must be"):
        chi2_gate(dof, probability)


def test_mean_nees_bounds_bracket_dimension_and_narrow_with_samples():
    low_10, high_10 = mean_nees_bounds(3, 10)
    low_1000, high_1000 = mean_nees_bounds(3, 1000)
    assert low_10 < low_1000 < 3.0 < high_1000 < high_10


def test_mean_nees_bounds_validates_arguments():
    with pytest.raises(ValueError, match=">= 1"):
        mean_nees_bounds(0, 10)
