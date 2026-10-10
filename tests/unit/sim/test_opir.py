import numpy as np
import pytest

from locant.sim.geometry import GeostationaryPlatform, LocalFrame
from locant.sim.opir.sensor import (
    PW_PER_W,
    OPIRSensor,
    SceneConditions,
    cloud_mask,
    ensquared_energy,
    frame_times,
    glint_series,
    observe,
)
from locant.sim.opir.signatures import (
    AircraftSignature,
    ExplosionSignature,
    FireSignature,
    LaunchSignature,
    ar1,
)
from locant.sim.trajectories import BallisticBoost, ConstantVelocity, Stationary

FRAME = LocalFrame.from_degrees(30.0, -100.0)
SENSOR = OPIRSensor(GeostationaryPlatform(np.radians(-100.0)))
T = frame_times(60.0, 10.0)


class TestSignatures:
    @pytest.mark.parametrize(
        "signature",
        [
            LaunchSignature(1e5, 2.0, 100.0),
            ExplosionSignature(1e6),
            FireSignature(1e4),
            AircraftSignature(1e3),
        ],
    )
    def test_zero_before_onset_and_nonnegative(self, signature, rng):
        t = np.linspace(-10.0, 50.0, 601)
        intensity = signature.intensity(t, rng)
        assert np.all(intensity[t < 0] == 0.0)
        assert np.all(intensity >= 0.0)
        assert intensity.max() > 0.0

    def test_launch_rise_time_is_10_to_90_percent(self, rng):
        sig = LaunchSignature(1.0, rise_time=4.0, burn_time=100.0, flicker=0.0)
        t = np.linspace(0.0, 20.0, 20_001)
        i = sig.intensity(t, rng)
        t10, t90 = t[np.argmax(i >= 0.1)], t[np.argmax(i >= 0.9)]
        assert t90 - t10 == pytest.approx(4.0, abs=1e-2)

    def test_launch_staging_and_burnout(self, rng):
        sig = LaunchSignature(1.0, 1.0, burn_time=60.0, staging_time=30.0, flicker=0.0)
        i = sig.intensity(np.array([20.0, 30.5, 40.0, 70.0]), rng)
        assert i[1] < 0.05 * i[0]  # coast between stages
        assert i[2] == pytest.approx(0.4 * i[0], rel=1e-3)
        assert i[3] < 0.01 * i[2]  # burned out

    def test_explosion_decays_and_aircraft_afterburner_scales(self, rng):
        e = ExplosionSignature(1.0).intensity(np.array([0.0, 1.0, 10.0]), rng)
        assert e[0] == pytest.approx(1.0)
        assert e[0] > e[1] > e[2]
        a = AircraftSignature(
            1.0, aspect_modulation=0.0, afterburner_start=5.0, afterburner_factor=4.0
        ).intensity(np.array([1.0, 6.0]), rng)
        assert a[1] == pytest.approx(4.0 * a[0])

    def test_ar1_has_requested_std_and_correlation(self, rng):
        x = ar1(200_000, 2.0, 10.0, rng)
        assert x.std() == pytest.approx(2.0, rel=0.03)
        lag1 = np.corrcoef(x[:-1], x[1:])[0, 1]
        assert lag1 == pytest.approx(np.exp(-0.1), abs=0.01)
        assert ar1(0, 1.0, 1.0, rng).size == 0
        assert not ar1(5, 0.0, 1.0, rng).any()


class TestSensorComponents:
    def test_ensquared_energy_bounds_and_tiling(self):
        assert ensquared_energy(np.zeros(2), 0.05) == pytest.approx(1.0)
        offsets = np.array([0.3, -0.2])
        total = sum(
            ensquared_energy(offsets + np.array([dx, dy]), 0.4)
            for dx in range(-4, 5)
            for dy in range(-4, 5)
        )
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_cloud_mask(self, rng):
        never = SceneConditions()
        assert not cloud_mask(T, never, rng).any()
        cloudy = SceneConditions(mean_clear_s=20.0, mean_cloudy_s=20.0)
        long_t = frame_times(20_000.0, 1.0)
        assert cloud_mask(long_t, cloudy, rng).mean() == pytest.approx(0.5, abs=0.05)

    def test_glints(self, rng):
        series, count = glint_series(T, SceneConditions(), rng)
        assert count == 0
        assert not series.any()
        scene = SceneConditions(glint_rate_hz=0.5, glint_amplitude=(10.0, 20.0))
        series, count = glint_series(frame_times(600.0, 10.0), scene, rng)
        assert count > 100
        assert 0.0 < series.max() <= 20.0 * 3  # overlapping pulses can add


class TestObserve:
    def test_signal_chain_is_consistent(self, rng):
        obs = observe(
            SENSOR,
            FRAME,
            Stationary(np.zeros(3)),
            ExplosionSignature(1e6),
            T,
            10.0,
            SceneConditions(),
            rng,
        )
        intensity = ExplosionSignature(1e6).intensity(T - 10.0, rng)
        expected = (
            obs.ensquared_energy
            * obs.transmittance
            * intensity
            / obs.range_m**2
            * PW_PER_W
        )
        np.testing.assert_allclose(obs.signal, expected)
        assert np.all(obs.signal[T < 10.0] == 0.0)
        assert obs.peak_snr == pytest.approx(obs.signal.max() / np.hypot(1.0, 1.0))

    def test_noise_has_declared_level(self, rng):
        sensor = OPIRSensor(GeostationaryPlatform(np.radians(-100.0)), nei=3.0)
        t = frame_times(600.0, 10.0)
        scene = SceneConditions(clutter_std=0.0)
        obs = observe(
            sensor,
            FRAME,
            Stationary(np.zeros(3)),
            FireSignature(1.0),
            t,
            0.0,
            scene,
            rng,
        )
        residual = obs.measured - obs.background - obs.glint - obs.signal
        assert residual.std() == pytest.approx(3.0, rel=0.05)

    def test_launch_brightens_as_it_climbs(self, rng):
        boost = BallisticBoost(np.zeros(3), 0.0, 30.0, 120.0)
        obs = observe(
            SENSOR,
            FRAME,
            boost,
            LaunchSignature(1e5, 2.0, 120.0),
            frame_times(100.0, 10.0),
            0.0,
            SceneConditions(zenith_optical_depth=1.0),
            rng,
        )
        assert obs.transmittance[-1] > obs.transmittance[100] > 0.0
        assert obs.target_enu[-1, 2] > 50_000.0

    def test_clouds_hide_low_targets_only(self, rng):
        scene = SceneConditions(
            mean_clear_s=1e-3,
            mean_cloudy_s=1e6,
            cloud_top_m=5_000.0,
            cloud_transmittance=0.1,
        )
        low = observe(
            SENSOR,
            FRAME,
            Stationary(np.zeros(3)),
            FireSignature(1.0),
            T,
            0.0,
            scene,
            rng,
        )
        high = observe(
            SENSOR,
            FRAME,
            ConstantVelocity(np.array([0, 0, 9_000.0]), np.zeros(3)),
            AircraftSignature(1.0),
            T,
            0.0,
            scene,
            rng,
        )
        assert low.occluded.all()
        assert not high.occluded.any()

    def test_line_of_sight_measurement_noise(self, rng):
        sensor = OPIRSensor(
            GeostationaryPlatform(np.radians(-100.0)), los_noise_rad=20e-6
        )
        t = frame_times(500.0, 10.0)
        obs = observe(
            sensor,
            FRAME,
            Stationary(np.zeros(3)),
            FireSignature(1.0),
            t,
            0.0,
            SceneConditions(),
            rng,
        )
        angles = np.arccos(
            np.clip(np.sum(obs.los_true * obs.los_measured, axis=1), -1, 1)
        )
        # Two independent axes: the angular error is Rayleigh with RMS sqrt(2)·sigma.
        assert np.sqrt(np.mean(angles**2)) == pytest.approx(
            np.sqrt(2) * 20e-6, rel=0.05
        )
