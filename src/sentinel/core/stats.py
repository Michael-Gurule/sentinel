"""Chi-square gates and consistency bounds."""

from scipy.stats import chi2


def chi2_gate(dof: int, probability: float = 0.99) -> float:
    """Gate threshold on a squared Mahalanobis distance.

    A measurement whose innovation has ``dof`` dimensions falls inside the gate
    with the given ``probability`` when the filter is consistent.
    """
    if dof < 1:
        raise ValueError(f"dof must be >= 1, got {dof}")
    if not 0.0 < probability < 1.0:
        raise ValueError(f"probability must be in (0, 1), got {probability}")
    return float(chi2.ppf(probability, dof))


def mean_nees_bounds(
    dim: int, num_samples: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Two-sided confidence interval for the *mean* of ``num_samples`` NEES values.

    The sum of N independent χ²(dim) variables is χ²(N·dim), so the mean lies in
    [χ²⁻¹(α/2; N·dim), χ²⁻¹(1-α/2; N·dim)] / N with the given confidence
    (Bar-Shalom, Li & Kirubarajan, *Estimation with Applications to Tracking
    and Navigation*, §5.4).
    """
    if dim < 1 or num_samples < 1:
        raise ValueError("dim and num_samples must be >= 1")
    alpha = 1.0 - confidence
    total_dof = dim * num_samples
    low = float(chi2.ppf(alpha / 2.0, total_dof)) / num_samples
    high = float(chi2.ppf(1.0 - alpha / 2.0, total_dof)) / num_samples
    return low, high
