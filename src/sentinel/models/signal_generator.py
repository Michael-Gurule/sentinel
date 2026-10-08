"""
OPIR Synthetic Signal Generator

Generates thermal event signatures (temperature in K vs. time) for training
and testing. All randomness comes from an injected ``numpy.random.Generator``
so scenarios are reproducible from a seed.

The physics-informed scenario simulator in Phase 2 replaces this module.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from sentinel.core.linalg import FloatArray

EVENT_TYPES = ("launch", "explosion", "fire", "aircraft")


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
    """Synthetic single-pixel OPIR intensity time series."""

    def __init__(
        self,
        duration_s: float = 10.0,
        sample_rate_hz: float = 100.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        """
        Args:
            duration_s: Length of each generated series in seconds.
            sample_rate_hz: Samples per second.
            rng: Random generator; a fresh unseeded one if omitted.
        """
        if duration_s <= 0 or sample_rate_hz <= 0:
            raise ValueError("duration_s and sample_rate_hz must be positive")
        self.duration_s = duration_s
        self.sample_rate_hz = sample_rate_hz
        self.num_samples = round(duration_s * sample_rate_hz)
        self.t: FloatArray = np.arange(self.num_samples) / sample_rate_hz
        self.rng = rng if rng is not None else np.random.default_rng()

    @property
    def sampling_rate(self) -> float:
        """Alias of ``sample_rate_hz`` used by detectors and pipelines."""
        return self.sample_rate_hz

    @property
    def fs(self) -> float:
        return self.sample_rate_hz

    def generate_launch_signature(
        self,
        start_time: float,
        peak_temp: float = 3500,
        rise_time: float = 3,
        sustain_duration: float = 30,
        decay_time: float = 40,
    ) -> FloatArray:
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
        noise = self.rng.normal(0, peak_temp * 0.02, len(signature))
        signature = signature + noise

        return np.maximum(signature, 0)  # No negative temperatures

    def generate_explosion_signature(
        self,
        start_time: float,
        peak_temp: float = 5000,
        flash_duration: float = 2,
        decay_time: float = 10,
    ) -> FloatArray:
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
        noise = self.rng.normal(0, peak_temp * 0.05, len(signature))
        signature = signature + noise

        return np.maximum(signature, 0)

    def generate_fire_signature(
        self,
        start_time: float,
        peak_temp: float = 1200,
        growth_time: float = 60,
        sustain_duration: float = 180,
        decay_time: float = 60,
    ) -> FloatArray:
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
        noise = self.rng.normal(0, peak_temp * 0.1, len(signature))
        signature = signature + noise

        return np.maximum(signature, 0)

    def generate_aircraft_signature(
        self,
        start_time: float,
        peak_temp: float = 1000,
        transit_duration: float = 30,
        velocity_mps: float = 250,
    ) -> FloatArray:
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
        noise = self.rng.normal(0, peak_temp * 0.05, len(signature))
        signature = signature + noise

        return np.maximum(signature, 0)

    def generate_background(
        self,
        base_temp: float = 280,
        diurnal_amplitude: float = 15,
        noise_level: float = 5,
    ) -> FloatArray:
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
            + self.rng.normal(0, noise_level, self.num_samples)
        )

        return background

    def generate_scenario(
        self, events: list[dict[str, Any]]
    ) -> tuple[FloatArray, list[ThermalEvent]]:
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

            else:
                raise ValueError(
                    f"unknown event type {event_type!r}; expected one of {EVENT_TYPES}"
                )

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
