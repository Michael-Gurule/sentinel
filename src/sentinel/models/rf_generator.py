"""
RF Signal Generator

Generates synthetic radar and communication signals for training and testing.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.signal as signal


@dataclass
class RFEmitter:
    emitter_type: str
    timestamp: float
    latitude: float
    longitude: float
    carrier_freq_hz: float
    power_dbm: float
    modulation: str
    bandwidth_hz: float


class RadarSignalGenerator:
    def __init__(self, sample_rate_hz=100e6):

        self.fs = sample_rate_hz

    def generate_pulse_train(
        self,
        carrier_freq_hz: float,
        prf_hz: float,
        pulse_width_s: float,
        duration_s: float,
        power_dbm: float,
        scan_rate_deg_s: float = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate radar pulse train

        Args:
            carrier_freq_hz: Carrier frequency (e.g., 10e9 for 10 GHz X-band)
            prf_hz: Pulse Repetition Frequency
            pulse_width_s: Pulse width in seconds
            duration_s: Total duration
            power_dbm: Transmit power
            scan_rate_deg_s: Scanning rate (for rotating radars)

        Returns:
            signal, time_vector
        """
        t = np.arange(0, duration_s, 1 / self.fs)

        # Generate carrier wave (complex baseband)
        carrier = np.exp(2j * np.pi * carrier_freq_hz * t)

        # Generate pulse train envelope
        pulse_period_s = 1 / prf_hz
        pulse_samples = int(pulse_width_s * self.fs)
        period_samples = int(pulse_period_s * self.fs)

        envelope = np.zeros(len(t))
        for i in range(0, len(t), period_samples):
            if i + pulse_samples < len(t):
                # Rectangular pulse with rise/fall time
                rise_fall_samples = int(pulse_samples * 0.1)

                # Rise
                envelope[i : i + rise_fall_samples] = np.linspace(
                    0, 1, rise_fall_samples
                )
                # Flat top
                envelope[
                    i + rise_fall_samples : i + pulse_samples - rise_fall_samples
                ] = 1
                # Fall
                envelope[i + pulse_samples - rise_fall_samples : i + pulse_samples] = (
                    np.linspace(1, 0, rise_fall_samples)
                )

        # Apply scanning modulation if rotating
        if scan_rate_deg_s is not None:
            # Simulate antenna gain pattern (sinc function)
            beam_angle_deg = scan_rate_deg_s * t
            beam_pattern = np.sinc(beam_angle_deg / 3) ** 2  # 3 degree beamwidth
            envelope *= beam_pattern

        # Modulate carrier
        radar_signal = carrier * envelope

        # Apply power scaling
        power_linear = 10 ** (power_dbm / 10) / 1000  # dBm to Watts
        radar_signal *= np.sqrt(power_linear)

        # Add noise (thermal noise floor)
        noise_power_dbm = -100  # Typical noise floor
        noise_power_linear = 10 ** (noise_power_dbm / 10) / 1000
        noise = np.sqrt(noise_power_linear / 2) * (
            np.random.randn(len(t)) + 1j * np.random.randn(len(t))
        )

        return radar_signal + noise, t

    def add_doppler_shift(
        self, signal: np.ndarray, velocity_mps: float, carrier_freq_hz: float
    ) -> np.ndarray:
        """
        Add Doppler shift for moving target/platform

        Args:
            signal: Input signal
            velocity_mps: Radial velocity (positive = approaching)
            carrier_freq_hz: Carrier frequency

        Returns:
            Doppler-shifted signal
        """
        c = 3e8  # Speed of light
        doppler_shift_hz = (velocity_mps / c) * carrier_freq_hz

        t = np.arange(len(signal)) / self.fs
        phase_shift = np.exp(2j * np.pi * doppler_shift_hz * t)

        return signal * phase_shift

    def generate_chirp_pulse(
        self,
        carrier_freq_hz: float,
        pulse_width_s: float,
        bandwidth_hz: float,
        prf_hz: float,
        duration_s: float,
        power_dbm: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate chirped (LFM) radar pulse train

        Linear Frequency Modulation provides pulse compression
        Used in modern radars for better range resolution
        """
        t = np.arange(0, duration_s, 1 / self.fs)

        # Generate chirp pulse
        chirp_rate = bandwidth_hz / pulse_width_s  # Hz/s

        pulse_period_s = 1 / prf_hz
        pulse_samples = int(pulse_width_s * self.fs)
        period_samples = int(pulse_period_s * self.fs)

        signal_out = np.zeros(len(t), dtype=complex)

        for i in range(0, len(t), period_samples):
            if i + pulse_samples < len(t):
                # Time vector for this pulse
                t_pulse = np.arange(pulse_samples) / self.fs

                # Chirp signal: f(t) = f_c + chirp_rate * t
                instantaneous_freq = carrier_freq_hz + chirp_rate * t_pulse
                phase = (
                    2
                    * np.pi
                    * (carrier_freq_hz * t_pulse + 0.5 * chirp_rate * t_pulse**2)
                )

                # Apply window (Hamming) to reduce sidelobes
                window = np.hamming(pulse_samples)
                chirp = window * np.exp(1j * phase)

                signal_out[i : i + pulse_samples] = chirp

        # Power scaling
        power_linear = 10 ** (power_dbm / 10) / 1000
        signal_out *= np.sqrt(power_linear)

        # Add noise
        noise_power_linear = 10 ** (-100 / 10) / 1000
        noise = np.sqrt(noise_power_linear / 2) * (
            np.random.randn(len(t)) + 1j * np.random.randn(len(t))
        )

        return signal_out + noise, t


class CommunicationSignalGenerator:
    """Generate communication signals with various modulations"""

    def __init__(self, sample_rate_hz=10e6):
        """
        Args:
            sample_rate_hz: Sample rate (10 MHz typical for comms)
        """
        self.fs = sample_rate_hz

    def generate_fm_signal(
        self,
        carrier_freq_hz: float,
        message_freq_hz: float,
        duration_s: float,
        modulation_index: float,
        power_dbm: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate FM modulated signal

        Used in: Tactical radios, analog communications
        """
        t = np.arange(0, duration_s, 1 / self.fs)

        # Message signal (simulate voice/data)
        message = np.sin(2 * np.pi * message_freq_hz * t)

        # Add some harmonics for realism
        message += 0.3 * np.sin(2 * np.pi * 2 * message_freq_hz * t)
        message += 0.1 * np.sin(2 * np.pi * 3 * message_freq_hz * t)

        # FM modulation: s(t) = cos(2πf_c*t + β*sin(2πf_m*t))
        fm_signal = np.cos(2 * np.pi * carrier_freq_hz * t + modulation_index * message)

        # Power scaling
        power_linear = 10 ** (power_dbm / 10) / 1000
        fm_signal *= np.sqrt(power_linear)

        # Add noise
        noise_power_linear = 10 ** (-90 / 10) / 1000
        noise = np.sqrt(noise_power_linear) * np.random.randn(len(t))

        return fm_signal + noise, t

    def generate_psk_signal(
        self,
        carrier_freq_hz: float,
        symbol_rate_hz: float,
        duration_s: float,
        modulation_order: int,
        power_dbm: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate PSK modulated signal

        Args:
            modulation_order: 2 (BPSK), 4 (QPSK), 8 (8PSK)

        Used in: Satellite communications, military datalinks
        """
        t = np.arange(0, duration_s, 1 / self.fs)

        # Generate random data bits
        num_symbols = int(duration_s * symbol_rate_hz)
        bits_per_symbol = int(np.log2(modulation_order))
        symbols = np.random.randint(0, modulation_order, num_symbols)

        # Map symbols to phases
        phases = 2 * np.pi * symbols / modulation_order

        # Upsample to match sample rate
        samples_per_symbol = int(self.fs / symbol_rate_hz)
        phase_train = np.repeat(phases, samples_per_symbol)

        # Truncate or pad to match duration
        if len(phase_train) > len(t):
            phase_train = phase_train[: len(t)]
        else:
            phase_train = np.pad(
                phase_train, (0, len(t) - len(phase_train)), mode="edge"
            )

        # Generate carrier
        carrier = np.exp(2j * np.pi * carrier_freq_hz * t)

        # Apply phase modulation with pulse shaping (root raised cosine)
        psk_signal = np.exp(1j * phase_train) * carrier

        # Apply root raised cosine filter for bandwidth control
        # (simplified - just apply a low-pass filter)

        # Cap filter cutoff at 0.9 * Nyquist to avoid aliasing
        nyquist = self.fs / 2
        filter_cutoff = min(symbol_rate_hz * 1.5, nyquist * 0.9)
        sos = signal.butter(4, filter_cutoff, "low", fs=self.fs, output="sos")
        psk_signal_real = signal.sosfilt(sos, psk_signal.real)
        psk_signal_imag = signal.sosfilt(sos, psk_signal.imag)
        psk_signal = psk_signal_real + 1j * psk_signal_imag

        # Power scaling
        power_linear = 10 ** (power_dbm / 10) / 1000
        psk_signal *= np.sqrt(power_linear / np.mean(np.abs(psk_signal) ** 2))

        # Add noise
        noise_power_linear = 10 ** (-90 / 10) / 1000
        noise = np.sqrt(noise_power_linear / 2) * (
            np.random.randn(len(t)) + 1j * np.random.randn(len(t))
        )

        return psk_signal + noise, t

    def generate_qam_signal(
        self,
        carrier_freq_hz: float,
        symbol_rate_hz: float,
        duration_s: float,
        modulation_order: int,
        power_dbm: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate QAM modulated signal

        Args:
            modulation_order: 16, 64, 256 (QAM-16, QAM-64, etc.)

        Used in: High-speed data links, satellite communications
        """
        t = np.arange(0, duration_s, 1 / self.fs)

        # Generate random symbols
        num_symbols = int(duration_s * symbol_rate_hz)
        symbols = np.random.randint(0, modulation_order, num_symbols)

        # QAM constellation mapping
        # For simplicity, use square QAM
        constellation_size = int(np.sqrt(modulation_order))

        # Map symbols to I and Q components
        i_indices = symbols % constellation_size
        q_indices = symbols // constellation_size

        # Normalize to [-1, 1] range
        i_symbols = 2 * (i_indices / (constellation_size - 1)) - 1
        q_symbols = 2 * (q_indices / (constellation_size - 1)) - 1

        # Upsample
        samples_per_symbol = int(self.fs / symbol_rate_hz)
        i_train = np.repeat(i_symbols, samples_per_symbol)
        q_train = np.repeat(q_symbols, samples_per_symbol)

        # Truncate to match duration
        if len(i_train) > len(t):
            i_train = i_train[: len(t)]
            q_train = q_train[: len(t)]
        else:
            i_train = np.pad(i_train, (0, len(t) - len(i_train)), mode="edge")
            q_train = np.pad(q_train, (0, len(t) - len(q_train)), mode="edge")

        # Generate carrier
        carrier_i = np.cos(2 * np.pi * carrier_freq_hz * t)
        carrier_q = -np.sin(2 * np.pi * carrier_freq_hz * t)

        # Modulate
        qam_signal = i_train * carrier_i + q_train * carrier_q

        # Apply pulse shaping filter
        # Cap filter cutoff at 0.9 * Nyquist to avoid aliasing
        nyquist = self.fs / 2
        filter_cutoff = min(symbol_rate_hz * 1.5, nyquist * 0.9)
        sos = signal.butter(4, filter_cutoff, "low", fs=self.fs, output="sos")
        qam_signal = signal.sosfilt(sos, qam_signal)

        # Power scaling
        power_linear = 10 ** (power_dbm / 10) / 1000
        qam_signal *= np.sqrt(power_linear / np.mean(qam_signal**2))

        # Add noise
        noise_power_linear = 10 ** (-90 / 10) / 1000
        noise = np.sqrt(noise_power_linear) * np.random.randn(len(t))

        return qam_signal + noise, t


class RFScenarioGenerator:
    """Generate complete RF scenarios with multiple emitters"""

    def __init__(self):
        self.radar_gen = RadarSignalGenerator()
        self.comm_gen = CommunicationSignalGenerator()

    def generate_early_warning_radar(
        self, start_time: float, duration_s: float, lat: float, lon: float
    ) -> Tuple[np.ndarray, np.ndarray, RFEmitter]:
        """Generate early warning radar signal"""

        carrier_freq = np.random.uniform(400e6, 1000e6)  # UHF/L-band
        prf = np.random.uniform(200, 500)
        pulse_width = np.random.uniform(10e-6, 50e-6)
        power_dbm = np.random.uniform(80, 100)
        scan_rate = 6  # 6 deg/s = 10 RPM typical

        signal_data, t = self.radar_gen.generate_pulse_train(
            carrier_freq, prf, pulse_width, duration_s, power_dbm, scan_rate
        )

        emitter = RFEmitter(
            emitter_type="early_warning_radar",
            timestamp=start_time,
            latitude=lat,
            longitude=lon,
            carrier_freq_hz=carrier_freq,
            power_dbm=power_dbm,
            modulation="pulse",
            bandwidth_hz=1 / pulse_width,
        )

        return signal_data, t, emitter

    def generate_fire_control_radar(
        self,
        start_time: float,
        duration_s: float,
        lat: float,
        lon: float,
        target_velocity_mps: float = 200,
    ) -> Tuple[np.ndarray, np.ndarray, RFEmitter]:
        """Generate fire control (tracking) radar with Doppler"""

        carrier_freq = np.random.uniform(8e9, 12e9)  # X-band
        prf = np.random.uniform(1000, 3000)
        pulse_width = np.random.uniform(0.5e-6, 2e-6)
        power_dbm = np.random.uniform(70, 90)

        # Use chirp for better resolution
        bandwidth = 100e6  # 100 MHz chirp bandwidth

        signal_data, t = self.radar_gen.generate_chirp_pulse(
            carrier_freq, pulse_width, bandwidth, prf, duration_s, power_dbm
        )

        # Add Doppler shift from target
        signal_data = self.radar_gen.add_doppler_shift(
            signal_data, target_velocity_mps, carrier_freq
        )

        emitter = RFEmitter(
            emitter_type="fire_control_radar",
            timestamp=start_time,
            latitude=lat,
            longitude=lon,
            carrier_freq_hz=carrier_freq,
            power_dbm=power_dbm,
            modulation="chirp",
            bandwidth_hz=bandwidth,
        )

        return signal_data, t, emitter

    def generate_tactical_radio(
        self, start_time: float, duration_s: float, lat: float, lon: float
    ) -> Tuple[np.ndarray, np.ndarray, RFEmitter]:
        """Generate tactical FM radio transmission"""

        carrier_freq = np.random.uniform(30e6, 90e6)  # VHF
        message_freq = 1000  # 1 kHz tone
        mod_index = 5
        power_dbm = np.random.uniform(30, 50)

        signal_data, t = self.comm_gen.generate_fm_signal(
            carrier_freq, message_freq, duration_s, mod_index, power_dbm
        )

        emitter = RFEmitter(
            emitter_type="tactical_radio",
            timestamp=start_time,
            latitude=lat,
            longitude=lon,
            carrier_freq_hz=carrier_freq,
            power_dbm=power_dbm,
            modulation="FM",
            bandwidth_hz=25e3,  # 25 kHz channel
        )

        return signal_data, t, emitter

    def generate_satellite_uplink(
        self, start_time: float, duration_s: float, lat: float, lon: float
    ) -> Tuple[np.ndarray, np.ndarray, RFEmitter]:
        """Generate satellite uplink (QPSK modulation)"""

        carrier_freq = np.random.uniform(14e9, 14.5e9)  # Ku-band
        symbol_rate = 5e6  # 5 Msps
        power_dbm = np.random.uniform(60, 80)

        signal_data, t = self.comm_gen.generate_psk_signal(
            carrier_freq,
            symbol_rate,
            duration_s,
            modulation_order=4,
            power_dbm=power_dbm,
        )

        emitter = RFEmitter(
            emitter_type="satellite_uplink",
            timestamp=start_time,
            latitude=lat,
            longitude=lon,
            carrier_freq_hz=carrier_freq,
            power_dbm=power_dbm,
            modulation="QPSK",
            bandwidth_hz=symbol_rate * 1.2,  # Symbol rate * roll-off
        )

        return signal_data, t, emitter


if __name__ == "__main__":
    # Test the RF signal generator
    import matplotlib.pyplot as plt

    print("Testing RF Signal Generator...")

    scenario_gen = RFScenarioGenerator()

    # Generate different signal types
    print("\n1. Generating Early Warning Radar...")
    radar_sig, t_radar, radar_emitter = scenario_gen.generate_early_warning_radar(
        start_time=0, duration_s=0.01, lat=40.0, lon=-100.0
    )
    print(f"   Carrier: {radar_emitter.carrier_freq_hz / 1e6:.1f} MHz")
    print(f"   Power: {radar_emitter.power_dbm:.1f} dBm")

    print("\n2. Generating Fire Control Radar...")
    fc_sig, t_fc, fc_emitter = scenario_gen.generate_fire_control_radar(
        start_time=0, duration_s=0.001, lat=40.0, lon=-100.0
    )
    print(f"   Carrier: {fc_emitter.carrier_freq_hz / 1e9:.2f} GHz")
    print(f"   Bandwidth: {fc_emitter.bandwidth_hz / 1e6:.1f} MHz")

    print("\n3. Generating Tactical Radio...")
    radio_sig, t_radio, radio_emitter = scenario_gen.generate_tactical_radio(
        start_time=0, duration_s=0.01, lat=40.0, lon=-100.0
    )
    print(f"   Carrier: {radio_emitter.carrier_freq_hz / 1e6:.1f} MHz")
    print(f"   Modulation: {radio_emitter.modulation}")

    # Visualize
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))

    # Early warning radar
    axes[0].plot(t_radar[:1000], np.abs(radar_sig[:1000]))
    axes[0].set_title("Early Warning Radar (Pulse Train)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)

    # Fire control radar
    axes[1].plot(t_fc[:1000], np.abs(fc_sig[:1000]))
    axes[1].set_title("Fire Control Radar (Chirped Pulse)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True)

    # Tactical radio
    axes[2].plot(t_radio[:1000], radio_sig[:1000].real)
    axes[2].set_title("Tactical Radio (FM)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Amplitude")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig("rf_signals_test.png", dpi=150)
    print("\n✓ RF signal generator test complete")
    print("✓ Saved visualization to rf_signals_test.png")
