import numpy as np
import pytest

from locant.core import is_psd
from locant.tracking import ConstantVelocity, Gaussian, innovation, predict, update
from locant.tracking.kalman import batch_nis

H = np.hstack([np.eye(3), np.zeros((3, 3))])


class TestConstantVelocity:
    def test_matrices_match_cwna_closed_form(self):
        model = ConstantVelocity(noise_intensity=3.0)
        dt = 2.0
        f = model.transition(dt)
        np.testing.assert_allclose(f[:3, 3:], dt * np.eye(3))
        np.testing.assert_allclose(f[:3, :3], np.eye(3))
        q = model.process_noise(dt)
        np.testing.assert_allclose(q[0, 0], 3.0 * dt**3 / 3)
        np.testing.assert_allclose(q[0, 3], 3.0 * dt**2 / 2)
        np.testing.assert_allclose(q[3, 3], 3.0 * dt)
        assert q[0, 1] == 0.0

    def test_transition_composes(self):
        model = ConstantVelocity(1.0)
        np.testing.assert_allclose(
            model.transition(0.7) @ model.transition(1.8), model.transition(2.5)
        )

    def test_zero_dt_is_identity_and_noise_free(self):
        model = ConstantVelocity(5.0)
        np.testing.assert_allclose(model.transition(0.0), np.eye(6))
        np.testing.assert_allclose(model.process_noise(0.0), np.zeros((6, 6)))

    def test_rejects_invalid_parameters(self):
        with pytest.raises(ValueError, match="non-negative"):
            ConstantVelocity(-1.0)
        with pytest.raises(ValueError, match="dim"):
            ConstantVelocity(1.0, dim=0)
        with pytest.raises(ValueError, match="non-negative"):
            ConstantVelocity(1.0).transition(-0.1)


class TestKalman:
    def test_gaussian_validates_shapes(self):
        with pytest.raises(ValueError, match="match"):
            Gaussian(np.zeros(6), np.eye(3))

    def test_predict_propagates_mean_and_covariance(self):
        model = ConstantVelocity(2.0)
        state = Gaussian(np.array([0, 0, 0, 1.0, 2.0, 3.0]), np.eye(6))
        out = predict(state, model, 0.5)
        f = model.transition(0.5)
        np.testing.assert_allclose(out.mean, [0.5, 1.0, 1.5, 1, 2, 3])
        np.testing.assert_allclose(
            out.covariance, f @ np.eye(6) @ f.T + model.process_noise(0.5)
        )
        assert predict(state, model, 0.0) is state

    def test_update_matches_textbook_gain(self):
        p = np.diag([4.0, 4.0, 4.0, 1.0, 1.0, 1.0])
        r = np.eye(3)
        state = Gaussian(np.zeros(6), p)
        z = np.array([1.0, -2.0, 0.5])
        posterior, innov = update(state, z, H, r)
        # Decoupled axes: gain on position is 4 / (4 + 1).
        np.testing.assert_allclose(posterior.mean[:3], 0.8 * z)
        np.testing.assert_allclose(np.diag(posterior.covariance)[:3], 0.8)
        np.testing.assert_allclose(innov.covariance, 5.0 * np.eye(3))
        assert innov.nis == pytest.approx(z @ z / 5.0)

    def test_joseph_form_equals_short_form_in_exact_arithmetic(self, rng):
        a = rng.normal(size=(6, 6))
        p = a @ a.T + np.eye(6)
        r = np.diag([2.0, 3.0, 4.0])
        state = Gaussian(rng.normal(size=6), p)
        posterior, _ = update(state, rng.normal(size=3), H, r)
        gain = p @ H.T @ np.linalg.inv(H @ p @ H.T + r)
        np.testing.assert_allclose(
            posterior.covariance, (np.eye(6) - gain @ H) @ p, atol=1e-10
        )
        assert is_psd(posterior.covariance)

    def test_innovation_of_perfect_prediction_is_zero(self):
        state = Gaussian(np.array([1.0, 2, 3, 0, 0, 0]), np.eye(6))
        innov = innovation(state, np.array([1.0, 2, 3]), H, np.eye(3))
        assert innov.nis == 0.0


def test_batch_nis_matches_single_innovations(rng):
    matrix = np.hstack([np.eye(3), np.zeros((3, 3))])
    noise = np.eye(3) * 4.0
    z = rng.normal(0, 10, 3)
    states = [
        Gaussian(rng.normal(0, 10, 6), np.diag(rng.uniform(1, 50, 6))) for _ in range(5)
    ]
    expected = [innovation(s, z, matrix, noise).nis for s in states]
    got = batch_nis(
        z,
        np.array([matrix @ s.mean for s in states]),
        np.broadcast_to(matrix, (5, 3, 6)),
        np.array([s.covariance for s in states]),
        noise,
    )
    np.testing.assert_allclose(got, expected, rtol=1e-10)
