"""
OPIR Synthetic Signal Generator

Generates realistic thermal event signatures for training and testing.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class ThermalEvent:
    event_type: str
    timestamp: float
    latitude: float
    longitude: float
    peak_temperature: float
    rise_time: float
    duration: float
    spatial_extent: float


class OPIRSignalGenerator:
    def __init__(self, duration=10.0, sampling_rate=100, sample_rate=None):
        self.duration = duration
        self.sampling_rate = sampling_rate  # samples per second
        self.fs = sampling_rate  # alias for sampling_rate
        self.num_samples = int(duration * sampling_rate)

        if sample_rate is not None:
            sampling_rate = sample_rate

    def generate_launch_signature(
        self,
        start_time: float,
        peak_temp: float = 3500,
        rise_time: float = 3,
        sustain_duration: float = 30,
        decay_time: float = 40,
    ) -> np.ndarray:
        """
        Generate missile launch thermal signature

        Characteristics:
        - Rapid rise (1-5 seconds)
        - High peak temperature (3000-4000K)
        - Sustained burn (30-120 seconds)
        - Gradual decay
        """
        signature = np.zeros(self.num_samples)

        # Find indices for each phase
        start_idx = int(start_time * self.fs)
        rise_samples = int(rise_time * self.fs)
        sustain_samples = int(sustain_duration * self.fs)
        decay_samples = int(decay_time * self.fs)

        # Rise phase (sigmoid)
        if start_idx + rise_samples < len(signature):
            t_rise = np.linspace(-3, 3, rise_samples)
            rise_curve = peak_temp / (1 + np.exp(-2 * t_rise))
            signature[start_idx : start_idx + rise_samples] = rise_curve

        # Sustain phase
        sustain_end = start_idx + rise_samples + sustain_samples
        if sustain_end < len(signature):
            signature[start_idx + rise_samples : sustain_end] = peak_temp * 0.95

        # Decay phase (exponential)
        decay_end = min(sustain_end + decay_samples, len(signature))
        if decay_end > sustain_end:
            t_decay = np.linspace(0, 5, decay_end - sustain_end)
            decay_curve = peak_temp * 0.95 * np.exp(-t_decay / 2)
            signature[sustain_end:decay_end] = decay_curve

        # Add noise
        noise = np.random.normal(0, peak_temp * 0.02, len(signature))
        signature = signature + noise

        return np.maximum(signature, 0)  # No negative temperatures

    def generate_explosion_signature(
        self,
        start_time: float,
        peak_temp: float = 5000,
        flash_duration: float = 2,
        decay_time: float = 10,
    ) -> np.ndarray:
        """
        Generate explosion thermal signature

        Characteristics:
        - Instantaneous rise (<1 second)
        - Very high peak (4000-6000K)
        - Rapid decay (1-10 seconds)
        """
        signature = np.zeros(self.num_samples)

        start_idx = int(start_time * self.fs)
        flash_samples = int(flash_duration * self.fs)
        decay_samples = int(decay_time * self.fs)

        # Flash phase (instant rise)
        if start_idx < len(signature):
            signature[start_idx] = peak_temp

        # Initial high temperature
        flash_end = min(start_idx + flash_samples, len(signature))
        if flash_end > start_idx:
            signature[start_idx:flash_end] = peak_temp * np.exp(
                -np.linspace(0, 2, flash_end - start_idx)
            )

        # Decay phase
        decay_end = min(flash_end + decay_samples, len(signature))
        if decay_end > flash_end:
            t_decay = np.linspace(0, 5, decay_end - flash_end)
            signature[flash_end:decay_end] = peak_temp * 0.2 * np.exp(-t_decay)

        # Add noise
        noise = np.random.normal(0, peak_temp * 0.05, len(signature))
        signature = signature + noise

        return np.maximum(signature, 0)

    def generate_fire_signature(
        self,
        start_time: float,
        peak_temp: float = 1200,
        growth_time: float = 60,
        sustain_duration: float = 180,
        decay_time: float = 60,
    ) -> np.ndarray:
        """
        Generate wildfire thermal signature

        Characteristics:
        - Slow rise (minutes)
        - Moderate temperature (800-1500K)
        - Long duration (hours)
        - Stationary
        """
        signature = np.zeros(self.num_samples)

        start_idx = int(start_time * self.fs)
        growth_samples = int(growth_time * self.fs)
        sustain_samples = int(sustain_duration * self.fs)
        decay_samples = int(decay_time * self.fs)

        # Define base_temp at the start to ensure it's always defined
        base_temp = peak_temp * 0.9

        # Growth phase (logarithmic)
        growth_end = min(start_idx + growth_samples, len(signature))
        if growth_end > start_idx:
            t_growth = np.linspace(0.1, 5, growth_end - start_idx)
            growth_curve = peak_temp * np.log(t_growth + 1) / np.log(6)
            signature[start_idx:growth_end] = growth_curve

        # Sustain phase (with fluctuation)
        sustain_end = min(growth_end + sustain_samples, len(signature))
        if sustain_end > growth_end:
            base_temp = peak_temp * 0.9
            fluctuation = (
                peak_temp
                * 0.1
                * np.sin(np.linspace(0, 10 * np.pi, sustain_end - growth_end))
            )
            signature[growth_end:sustain_end] = base_temp + fluctuation

        # Decay phase
        decay_end = min(sustain_end + decay_samples, len(signature))
        if decay_end > sustain_end:
            t_decay = np.linspace(0, 5, decay_end - sustain_end)
            decay_curve = base_temp * np.exp(-t_decay / 3)
            signature[sustain_end:decay_end] = decay_curve

        # Add noise (fires are noisy)
        noise = np.random.normal(0, peak_temp * 0.1, len(signature))
        signature = signature + noise

        return np.maximum(signature, 0)

    def generate_aircraft_signature(
        self,
        start_time: float,
        peak_temp: float = 1000,
        transit_duration: float = 30,
        velocity_mps: float = 250,
    ) -> np.ndarray:
        """
        Generate aircraft exhaust signature

        Characteristics:
        - Constant temperature (800-1200K)
        - Moving source (Gaussian envelope)
        - Moderate duration (seconds to minutes)
        """
        signature = np.zeros(self.num_samples)

        start_idx = int(start_time * self.fs)
        transit_samples = int(transit_duration * self.fs)

        # Gaussian envelope (aircraft passing through FOV)
        transit_end = min(start_idx + transit_samples, len(signature))
        if transit_end > start_idx:
            t_transit = np.linspace(-3, 3, transit_end - start_idx)
            envelope = np.exp(-(t_transit**2) / 2)
            signature[start_idx:transit_end] = peak_temp * envelope

        # Add noise
        noise = np.random.normal(0, peak_temp * 0.05, len(signature))
        signature = signature + noise

        return np.maximum(signature, 0)

    def generate_background(
        self,
        base_temp: float = 280,
        diurnal_amplitude: float = 15,
        noise_level: float = 5,
    ) -> np.ndarray:
        """
        Generate Earth background thermal signature

        Includes:
        - Diurnal (day/night) variation
        - Random noise
        - Seasonal variation (simplified)
        """
        # Diurnal cycle (24-hour period)
        diurnal_freq = 2 * np.pi / (24 * 3600)  # rad/s
        diurnal_component = diurnal_amplitude * np.sin(diurnal_freq * self.t)

        # Base temperature + diurnal + noise
        background = (
            base_temp
            + diurnal_component
            + np.random.normal(0, noise_level, self.num_samples)
        )

        return background

    def generate_scenario(
        self, events: list[Dict]
    ) -> Tuple[np.ndarray, list[ThermalEvent]]:
        """
        Generate complete scenario with multiple events

        Args:
            events: List of event dictionaries with parameters

        Returns:
            Combined thermal signature and list of ThermalEvent objects
        """
        # Start with background
        scenario = self.generate_background()

        event_records = []

        for event in events:
            event_type = event["type"]
            start_time = event["start_time"]

            if event_type == "launch":
                signature = self.generate_launch_signature(
                    start_time,
                    **{
                        k: v
                        for k, v in event.items()
                        if k not in ["type", "start_time", "lat", "lon"]
                    },
                )
                peak_temp = event.get("peak_temp", 3500)
                rise_time = event.get("rise_time", 3)
                duration = event.get("sustain_duration", 30) + event.get(
                    "decay_time", 40
                )

            elif event_type == "explosion":
                signature = self.generate_explosion_signature(
                    start_time,
                    **{
                        k: v
                        for k, v in event.items()
                        if k not in ["type", "start_time", "lat", "lon"]
                    },
                )
                peak_temp = event.get("peak_temp", 5000)
                rise_time = 0.5
                duration = event.get("flash_duration", 2) + event.get("decay_time", 10)

            elif event_type == "fire":
                signature = self.generate_fire_signature(
                    start_time,
                    **{
                        k: v
                        for k, v in event.items()
                        if k not in ["type", "start_time", "lat", "lon"]
                    },
                )
                peak_temp = event.get("peak_temp", 1200)
                rise_time = event.get("growth_time", 60)
                duration = event.get("sustain_duration", 180)

            elif event_type == "aircraft":
                signature = self.generate_aircraft_signature(
                    start_time,
                    **{
                        k: v
                        for k, v in event.items()
                        if k not in ["type", "start_time", "lat", "lon"]
                    },
                )
                peak_temp = event.get("peak_temp", 1000)
                rise_time = 1
                duration = event.get("transit_duration", 30)

            # Add to scenario
            scenario = scenario + signature

            # Record event
            thermal_event = ThermalEvent(
                event_type=event_type,
                timestamp=start_time,
                latitude=event.get("lat", 0),
                longitude=event.get("lon", 0),
                peak_temperature=peak_temp,
                rise_time=rise_time,
                duration=duration,
                spatial_extent=event.get("spatial_extent", 50),
            )
            event_records.append(thermal_event)

        return scenario, event_records


if __name__ == "__main__":
    # Test the generator
    import matplotlib.pyplot as plt

    generator = OPIRSignalGenerator(sample_rate_hz=1.0, duration_s=300)

    # Define scenario
    events = [
        {"type": "launch", "start_time": 50, "lat": 40.0, "lon": -100.0},
        {"type": "explosion", "start_time": 150, "lat": 40.5, "lon": -100.5},
        {"type": "aircraft", "start_time": 200, "lat": 41.0, "lon": -101.0},
    ]

    scenario, event_records = generator.generate_scenario(events)

    # Plot
    plt.figure(figsize=(15, 6))
    plt.plot(generator.t, scenario)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Temperature (K)")
    plt.title("OPIR Thermal Scenario")
    plt.grid(True)

    # Mark events
    for event in event_records:
        plt.axvline(event.timestamp, color="r", linestyle="--", alpha=0.5)
        plt.text(event.timestamp, plt.ylim()[1] * 0.9, event.event_type, rotation=90)

    plt.tight_layout()
    plt.savefig("opir_scenario_test.png")
    print("✓ OPIR signal generator test complete")
