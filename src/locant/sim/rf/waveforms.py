"""Complex-baseband RF waveforms and channel operators.

Signals are complex baseband sampled at ``fs``. The carrier frequency is
metadata only (it sets Doppler scale), never synthesized: modulating a
10 GHz carrier at a 100 MHz sample rate would alias (v1 defect H9).

The delay and Doppler operators connect waveforms to geolocation: a
receiver's copy of an emission is the waveform delayed by r/c and shifted by
the Doppler frequency, which is what TDOA/FDOA estimators measure.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]


def _num_samples(duration_s: float, fs: float) -> int:
    if duration_s <= 0 or fs <= 0:
        raise ValueError("duration and sample rate must be positive")
    return round(duration_s * fs)


def pulse_train(
    prf_hz: float, pulse_width_s: float, duration_s: float, fs: float
) -> ComplexArray:
    """Unmodulated rectangular pulses (unit amplitude) at the given PRF."""
    n = _num_samples(duration_s, fs)
    if pulse_width_s * prf_hz >= 1.0:
        raise ValueError("pulse width must be shorter than the pulse period")
    t = np.arange(n) / fs
    on = (t % (1.0 / prf_hz)) < pulse_width_s
    return on.astype(np.complex128)


def lfm_pulse_train(
    prf_hz: float,
    pulse_width_s: float,
    bandwidth_hz: float,
    duration_s: float,
    fs: float,
) -> ComplexArray:
    """Linear-FM (chirp) pulses sweeping -B/2 to +B/2 within each pulse."""
    if bandwidth_hz >= fs:
        raise ValueError("chirp bandwidth must be below the sample rate")
    envelope = pulse_train(prf_hz, pulse_width_s, duration_s, fs)
    t = np.arange(envelope.size) / fs
    tau = t % (1.0 / prf_hz)
    rate = bandwidth_hz / pulse_width_s
    phase = 2.0 * np.pi * (-0.5 * bandwidth_hz * tau + 0.5 * rate * tau**2)
    return envelope * np.exp(1j * phase)


def fm_tone(
    message_hz: float, deviation_hz: float, duration_s: float, fs: float
) -> ComplexArray:
    """Analog FM carrying a single tone (e.g. a voice-band radio)."""
    n = _num_samples(duration_s, fs)
    t = np.arange(n) / fs
    beta = deviation_hz / message_hz
    return np.exp(1j * beta * np.sin(2.0 * np.pi * message_hz * t))


def _symbols_to_samples(
    symbols: ComplexArray, symbol_rate: float, n: int, fs: float
) -> ComplexArray:
    index = np.minimum((np.arange(n) / fs * symbol_rate).astype(int), symbols.size - 1)
    return np.asarray(symbols[index], dtype=np.complex128)


def psk(
    symbol_rate: float,
    order: int,
    duration_s: float,
    fs: float,
    rng: np.random.Generator,
) -> ComplexArray:
    """M-ary PSK with rectangular pulses and random symbols."""
    n = _num_samples(duration_s, fs)
    count = int(np.ceil(duration_s * symbol_rate))
    symbols = np.exp(2j * np.pi * rng.integers(0, order, count) / order)
    return _symbols_to_samples(symbols, symbol_rate, n, fs)


def qam(
    symbol_rate: float,
    order: int,
    duration_s: float,
    fs: float,
    rng: np.random.Generator,
) -> ComplexArray:
    """Square M-QAM (unit average power) with rectangular pulses."""
    side = round(np.sqrt(order))
    if side * side != order:
        raise ValueError("QAM order must be a perfect square")
    n = _num_samples(duration_s, fs)
    count = int(np.ceil(duration_s * symbol_rate))
    levels = 2.0 * np.arange(side) - (side - 1)
    symbols = rng.choice(levels, count) + 1j * rng.choice(levels, count)
    symbols /= np.sqrt(np.mean(np.abs(levels[:, None] + 1j * levels[None, :]) ** 2))
    return _symbols_to_samples(symbols, symbol_rate, n, fs)


def add_awgn(
    signal: ComplexArray, snr_db: float, rng: np.random.Generator
) -> ComplexArray:
    """Add circular complex white Gaussian noise at an SNR relative to mean signal power."""
    power = float(np.mean(np.abs(signal) ** 2))
    noise_power = power / 10.0 ** (snr_db / 10.0)
    noise = rng.normal(size=signal.size) + 1j * rng.normal(size=signal.size)
    return np.asarray(signal + np.sqrt(noise_power / 2.0) * noise, dtype=np.complex128)


def apply_delay(signal: ComplexArray, delay_s: float, fs: float) -> ComplexArray:
    """Delay by a (fractional) number of samples via an FFT phase ramp (circular)."""
    freqs = np.fft.fftfreq(signal.size, d=1.0 / fs)
    return np.fft.ifft(np.fft.fft(signal) * np.exp(-2j * np.pi * freqs * delay_s))


def apply_doppler(signal: ComplexArray, doppler_hz: float, fs: float) -> ComplexArray:
    """Shift the spectrum by ``doppler_hz``."""
    t = np.arange(signal.size) / fs
    return signal * np.exp(2j * np.pi * doppler_hz * t)


@dataclass(frozen=True)
class EmitterProfile:
    """Parameters of a representative emitter class (illustrative values)."""

    name: str
    carrier_hz: float
    modulation: str  # "pulse", "lfm", "fm", "psk", "qam"
    bandwidth_hz: float
    prf_hz: float = 0.0
    pulse_width_s: float = 0.0
    symbol_rate: float = 0.0
    order: int = 0


EMITTER_PROFILES: dict[str, EmitterProfile] = {
    "early_warning_radar": EmitterProfile(
        "early_warning_radar", 600e6, "pulse", 1e6, prf_hz=300.0, pulse_width_s=30e-6
    ),
    "fire_control_radar": EmitterProfile(
        "fire_control_radar", 9.5e9, "lfm", 5e6, prf_hz=5e3, pulse_width_s=20e-6
    ),
    "tactical_radio": EmitterProfile("tactical_radio", 60e6, "fm", 25e3),
    "satellite_uplink": EmitterProfile(
        "satellite_uplink", 14.2e9, "psk", 5e6, symbol_rate=2e6, order=4
    ),
    "datalink": EmitterProfile(
        "datalink", 1.2e9, "qam", 2e6, symbol_rate=1e6, order=16
    ),
}


def generate_waveform(
    profile: EmitterProfile, duration_s: float, fs: float, rng: np.random.Generator
) -> ComplexArray:
    """Baseband waveform of an emitter profile."""
    if profile.modulation == "pulse":
        return pulse_train(profile.prf_hz, profile.pulse_width_s, duration_s, fs)
    if profile.modulation == "lfm":
        return lfm_pulse_train(
            profile.prf_hz, profile.pulse_width_s, profile.bandwidth_hz, duration_s, fs
        )
    if profile.modulation == "fm":
        return fm_tone(1e3, profile.bandwidth_hz / 2.0, duration_s, fs)
    if profile.modulation == "psk":
        return psk(profile.symbol_rate, profile.order, duration_s, fs, rng)
    if profile.modulation == "qam":
        return qam(profile.symbol_rate, profile.order, duration_s, fs, rng)
    raise ValueError(f"unknown modulation {profile.modulation!r}")
