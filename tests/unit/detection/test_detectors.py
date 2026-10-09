import numpy as np
import pytest
from scipy.special import ndtri

from sentinel.detection import (
    CFARDetector,
    CUSUMDetector,
    StepGLRTDetector,
    calibrate_threshold,
    noise_sigma_from_differences,
    siegmund_arl0,
)

FS = 10.0


def step(rng, n=4, length=640, onset=300, amplitude=8.0, sigma=1.0):
    x = rng.normal(0.0, sigma, (n, length))
    x[:, onset:] += amplitude
    return x


class TestCFAR:
    def test_standardized_residual_is_unit_normal_on_white_noise(self, rng):
        z = CFARDetector().standardized(rng.normal(5.0, 3.0, (200, 640)), FS)
        valid = z[:, ~np.isnan(z[0])]
        assert valid.mean() == pytest.approx(0.0, abs=0.02)
        assert valid.std() == pytest.approx(1.0, abs=0.05)
        assert np.isnan(z[:, :20]).all()  # less than min_reference history

    def test_detects_step_at_its_onset(self, rng):
        scores = CFARDetector().score(step(rng), FS)
        assert np.all(scores.score > 6.0)
        # The peak can trail the onset by up to the guard interval (10 frames).
        assert np.all((scores.onset_index >= 300) & (scores.onset_index <= 310))

    def test_constant_signal_scores_zero(self):
        scores = CFARDetector().score(np.full(640, 280.0), FS)
        assert scores.score[0] == pytest.approx(0.0)

    def test_analytic_threshold_formula(self):
        detector = CFARDetector()
        m = detector.tested_frames(640, FS)
        eta = detector.analytic_threshold(1e-2, 640, FS)
        assert eta == pytest.approx(ndtri(0.99 ** (1 / m)))


class TestCUSUM:
    def test_siegmund_arl_increases_with_threshold(self):
        arls = [siegmund_arl0(h, 0.5) for h in (2.0, 4.0, 8.0)]
        assert arls == sorted(arls)
        assert CUSUMDetector().analytic_threshold(
            1e-3, 640, FS
        ) > CUSUMDetector().analytic_threshold(1e-2, 640, FS)

    def test_accumulates_a_slow_ramp_that_cfar_misses(self, rng):
        x = rng.normal(0.0, 1.0, (50, 640))
        x[:, 200:] += np.linspace(0.0, 3.0, 440)  # slow rise, at most 3 sigma
        cusum = CUSUMDetector().score(x, FS).score
        cfar = CFARDetector().score(x, FS).score
        assert np.median(cusum) > CUSUMDetector().analytic_threshold(1e-2, 640, FS)
        assert np.median(cfar) < CFARDetector().analytic_threshold(1e-2, 640, FS)

    def test_onset_estimate_precedes_the_peak(self, rng):
        scores = CUSUMDetector().score(step(rng, amplitude=3.0), FS)
        assert np.all(np.abs(scores.onset_index - 300) <= 10)


class TestGLRT:
    def test_noise_sigma_is_robust_to_steps(self, rng):
        sigma = noise_sigma_from_differences(
            step(rng, n=50, amplitude=500.0, sigma=2.0)
        )
        assert np.median(sigma) == pytest.approx(2.0, rel=0.05)

    def test_locates_step(self, rng):
        scores = StepGLRTDetector().score(step(rng, amplitude=2.0), FS)
        assert np.all(np.abs(scores.onset_index - 300) <= 10)
        assert np.all(
            scores.score > StepGLRTDetector().analytic_threshold(1e-3, 640, FS)
        )

    def test_bonferroni_threshold_is_conservative_on_white_noise(self, rng):
        detector = StepGLRTDetector()
        scores = detector.score(rng.normal(size=(2_000, 640)), FS).score
        assert np.mean(scores > detector.analytic_threshold(1e-2, 640, FS)) <= 0.01


class TestCalibration:
    def test_calibrated_threshold_achieves_rate(self, rng):
        background = rng.normal(size=100_000)
        threshold = calibrate_threshold(background, 1e-2)
        assert np.mean(background > threshold) <= 1e-2
        assert np.mean(background >= np.nextafter(threshold, -np.inf)) >= 1e-2 - 1e-5

    def test_rejects_too_few_windows_and_bad_rates(self):
        with pytest.raises(ValueError, match="cannot resolve"):
            calibrate_threshold(np.ones(50), 1e-3)
        with pytest.raises(ValueError, match="pfa"):
            calibrate_threshold(np.ones(50), 1.5)

    def test_batch_shape_validation(self):
        with pytest.raises(ValueError, match="shape"):
            CFARDetector().score(np.ones((2, 2, 2)), FS)
