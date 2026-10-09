import numpy as np
import pytest

from locant.core import mean_nees_bounds, nees
from locant.core.constants import SPEED_OF_LIGHT
from locant.geolocation import solve_tdoa
from locant.sim.rf.network import ReceiverModel, RFNetwork
from locant.sim.rf.waveforms import (
    EMITTER_PROFILES,
    EmitterProfile,
    add_awgn,
    apply_delay,
    apply_doppler,
    fm_tone,
    generate_waveform,
    lfm_pulse_train,
    psk,
    pulse_train,
    qam,
)
from locant.sim.trajectories import ConstantVelocity, Stationary

FS = 10e6


class TestWaveforms:
    def test_pulse_train_duty_cycle(self):
        w = pulse_train(prf_hz=1e3, pulse_width_s=50e-6, duration_s=10e-3, fs=FS)
        assert np.mean(np.abs(w)) == pytest.approx(0.05, abs=1e-3)

    def test_lfm_sweeps_its_bandwidth(self):
        w = lfm_pulse_train(1e3, 100e-6, 2e6, 1e-3, FS)
        pulse = w[: int(100e-6 * FS)]
        inst_freq = np.diff(np.unwrap(np.angle(pulse))) * FS / (2 * np.pi)
        assert inst_freq[0] == pytest.approx(-1e6, rel=0.01)
        assert inst_freq[-1] == pytest.approx(1e6, rel=0.01)

    def test_constant_envelope_and_constellations(self, rng):
        np.testing.assert_allclose(np.abs(fm_tone(1e3, 5e3, 1e-3, FS)), 1.0)
        p = psk(1e6, 4, 1e-3, FS, rng)
        np.testing.assert_allclose(np.abs(p), 1.0)
        phases = np.unique(np.round(np.angle(p) / (np.pi / 2)) % 4)
        assert set(phases) <= {0.0, 1.0, 2.0, 3.0}
        q = qam(1e6, 16, 20e-3, FS, rng)
        assert np.mean(np.abs(q) ** 2) == pytest.approx(1.0, rel=0.05)
        with pytest.raises(ValueError, match="perfect square"):
            qam(1e6, 8, 1e-3, FS, rng)

    def test_awgn_hits_requested_snr(self, rng):
        clean = np.exp(1j * rng.uniform(0, 2 * np.pi, 200_000))
        noisy = add_awgn(clean, 10.0, rng)
        measured = 10 * np.log10(1.0 / np.mean(np.abs(noisy - clean) ** 2))
        assert measured == pytest.approx(10.0, abs=0.1)

    def test_delay_and_doppler_are_recoverable(self, rng):
        w = generate_waveform(EMITTER_PROFILES["satellite_uplink"], 1e-3, FS, rng)
        delayed = apply_delay(w, 37 / FS, FS)
        xcorr = np.fft.ifft(np.fft.fft(delayed) * np.conj(np.fft.fft(w)))
        assert int(np.argmax(np.abs(xcorr))) == 37
        tone = np.ones(10_000, dtype=complex)
        shifted = apply_doppler(tone, 2_000.0, FS)
        spectrum = np.abs(np.fft.fft(shifted))
        freqs = np.fft.fftfreq(tone.size, 1 / FS)
        assert freqs[np.argmax(spectrum)] == pytest.approx(2_000.0, abs=FS / tone.size)

    @pytest.mark.parametrize("name", sorted(EMITTER_PROFILES))
    def test_every_profile_generates(self, name, rng):
        w = generate_waveform(EMITTER_PROFILES[name], 1e-3, 20e6, rng)
        assert w.dtype == np.complex128
        assert w.size == 20_000
        assert np.all(np.isfinite(w))

    def test_invalid_arguments(self, rng):
        with pytest.raises(ValueError, match="positive"):
            pulse_train(1e3, 1e-6, 0.0, FS)
        with pytest.raises(ValueError, match="pulse width"):
            pulse_train(1e3, 2e-3, 1e-2, FS)
        with pytest.raises(ValueError, match="bandwidth"):
            lfm_pulse_train(1e3, 1e-4, 2 * FS, 1e-3, FS)
        with pytest.raises(ValueError, match="modulation"):
            generate_waveform(EmitterProfile("x", 1e9, "ofdm", 1e6), 1e-3, FS, rng)


POSITIONS = [
    [0, 0, 500],
    [10e3, 0, 1500],
    [10e3, 10e3, 1000],
    [0, 10e3, 2000],
    [5e3, -4e3, 6e3],
]
EMITTER = np.array([5e3, 5e3, 500.0])


def network(rng, **kwargs):
    models = [
        ReceiverModel(i, Stationary(np.array(p, dtype=float)), **kwargs)
        for i, p in enumerate(POSITIONS)
    ]
    return RFNetwork(models, rng)


class TestNetwork:
    def test_ideal_network_is_consistent(self, rng):
        net = network(rng)
        runs, values = 300, []
        for _ in range(runs):
            scan = net.scan(0.0, EMITTER, np.zeros(3), rng)
            result = solve_tdoa(scan.receivers, scan.tdoa)
            values.append(nees(result.position - EMITTER, result.position_covariance))
        low, high = mean_nees_bounds(3, runs, confidence=0.99)
        assert low <= np.mean(values) <= high

    def test_clock_biases_shift_tdoas(self, rng):
        net = network(rng, toa_std=1e-15, clock_bias_std=20e-9)
        scan = net.scan(0.0, EMITTER, np.zeros(3), rng)
        ideal = network(rng, toa_std=1e-15).scan(0.0, EMITTER, np.zeros(3), rng)
        expected = net.clock_biases[1:] - net.clock_biases[0]
        np.testing.assert_allclose(
            scan.tdoa.values - ideal.tdoa.values, expected, atol=1e-13
        )

    def test_lo_offsets_bias_fdoa(self, rng):
        net = network(rng, frequency_std=1e-9, lo_offset_std=5.0)
        scan = net.scan(
            0.0, EMITTER, np.array([100.0, 0, 0]), rng, carrier_frequency=1e9
        )
        ideal = network(rng, frequency_std=1e-9).scan(
            0.0, EMITTER, np.array([100.0, 0, 0]), rng, carrier_frequency=1e9
        )
        assert scan.fdoa is not None
        assert ideal.fdoa is not None
        expected = net.lo_offsets[1:] - net.lo_offsets[0]
        np.testing.assert_allclose(
            scan.fdoa.values - ideal.fdoa.values, expected, atol=1e-6
        )

    def test_reported_positions_carry_survey_error(self, rng):
        net = network(rng, position_error_std=10.0)
        scan = net.scan(0.0, EMITTER, np.zeros(3), rng)
        reported = np.array([r.position for r in scan.receivers])
        true = np.array([r.position for r in scan.true_receivers])
        np.testing.assert_allclose(reported - true, net.position_errors)
        assert np.abs(net.position_errors).max() > 0.0

    def test_moving_receivers_and_tdoa_only_scan(self, rng):
        models = [
            ReceiverModel(0, ConstantVelocity(np.zeros(3), np.array([100.0, 0, 0]))),
            ReceiverModel(1, Stationary(np.array([1e4, 0, 0]))),
        ]
        net = RFNetwork(models, rng)
        scan = net.scan(10.0, EMITTER, np.zeros(3), rng)
        np.testing.assert_allclose(scan.receivers[0].position, [1_000.0, 0, 0])
        assert scan.fdoa is None
        expected = (
            np.linalg.norm(EMITTER - [1e4, 0, 0])
            - np.linalg.norm(EMITTER - [1e3, 0, 0])
        ) / SPEED_OF_LIGHT
        assert scan.tdoa.values[0] == pytest.approx(expected, abs=1e-7)

    def test_requires_two_receivers(self, rng):
        with pytest.raises(ValueError, match="two receivers"):
            RFNetwork([ReceiverModel(0, Stationary(np.zeros(3)))], rng)
