"""In-band radiant intensity (W/sr) of thermal events versus time since onset.

The shapes follow the v1 generator's event phenomenology (sigmoid launch rise
and sustained burn, explosive flash with fireball decay, slow fire growth,
steady aircraft exhaust), recast as source radiant intensity so the sensor
model can apply range, atmosphere, and optics. Magnitudes are order-of-
magnitude and illustrative only; they do not describe any real system.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from sentinel.core.linalg import FloatArray


class Signature(Protocol):
    def intensity(self, t: FloatArray, rng: np.random.Generator) -> FloatArray:
        """Radiant intensity (W/sr) at times ``t`` since onset; zero for t < 0."""
        ...


def ar1(
    n: int, sigma: float, correlation_steps: float, rng: np.random.Generator
) -> FloatArray:
    """Stationary AR(1) series with standard deviation ``sigma``.

    ``correlation_steps`` is the e-folding correlation length in samples.
    """
    if n == 0:
        return np.zeros(0)
    if sigma == 0.0:
        return np.zeros(n)
    phi = float(np.exp(-1.0 / max(correlation_steps, 1e-9)))
    innovations = rng.normal(0.0, sigma * np.sqrt(1.0 - phi**2), n)
    out = np.empty(n)
    out[0] = rng.normal(0.0, sigma)
    for k in range(1, n):
        out[k] = phi * out[k - 1] + innovations[k]
    return out


def _sample_spacing(t: FloatArray) -> float:
    return float(np.median(np.diff(t))) if t.size > 1 else 1.0


@dataclass(frozen=True)
class LaunchSignature:
    """Rocket plume: sigmoid ignition rise, flickering burn, optional staging, burnout.

    Attributes:
        peak_intensity: First-stage plume intensity, W/sr.
        rise_time: 10–90% ignition rise, s.
        burn_time: Total powered flight, s (matches the trajectory).
        staging_time: Stage separation time, s, or ``None`` for one stage.
        stage2_ratio: Second-stage intensity relative to the first.
        flicker: Fractional RMS plume flicker.
        burnout_decay: e-folding decay of the plume after burnout, s.
    """

    peak_intensity: float
    rise_time: float
    burn_time: float
    staging_time: float | None = None
    stage2_ratio: float = 0.4
    flicker: float = 0.03
    burnout_decay: float = 2.0

    def intensity(self, t: FloatArray, rng: np.random.Generator) -> FloatArray:
        t = np.asarray(t, dtype=np.float64)
        # Logistic rise: 10%→90% spans 2·ln 9 / k seconds.
        k = 2.0 * np.log(9.0) / self.rise_time
        profile = 1.0 / (1.0 + np.exp(-k * (t - self.rise_time)))
        if self.staging_time is not None:
            gap = 1.0  # coast between stages, s
            dip = (t >= self.staging_time) & (t < self.staging_time + gap)
            profile = np.where(
                t >= self.staging_time, self.stage2_ratio * profile, profile
            )
            profile = np.where(dip, 0.02 * profile, profile)
        after = t > self.burn_time
        profile = np.where(
            after,
            profile * np.exp(-(t - self.burn_time) / self.burnout_decay),
            profile,
        )
        flicker = 1.0 + ar1(t.size, self.flicker, 0.2 / _sample_spacing(t), rng)
        return np.where(
            t >= 0.0, self.peak_intensity * profile * np.clip(flicker, 0, None), 0.0
        )


@dataclass(frozen=True)
class ExplosionSignature:
    """Explosive flash (fast decay) followed by a cooling fireball (slow decay)."""

    peak_intensity: float
    flash_decay: float = 0.2
    fireball_ratio: float = 0.15
    fireball_decay: float = 3.0

    def intensity(self, t: FloatArray, rng: np.random.Generator) -> FloatArray:
        t = np.asarray(t, dtype=np.float64)
        tau = np.clip(t, 0.0, None)
        profile = (1.0 - self.fireball_ratio) * np.exp(-tau / self.flash_decay)
        profile += self.fireball_ratio * np.exp(-tau / self.fireball_decay)
        del rng  # deterministic shape
        return np.where(t >= 0.0, self.peak_intensity * profile, 0.0)


@dataclass(frozen=True)
class FireSignature:
    """Wildfire or industrial fire: logistic growth, then a fluctuating plateau."""

    peak_intensity: float
    growth_time: float = 120.0
    fluctuation: float = 0.15
    correlation_time: float = 8.0

    def intensity(self, t: FloatArray, rng: np.random.Generator) -> FloatArray:
        t = np.asarray(t, dtype=np.float64)
        k = 2.0 * np.log(9.0) / self.growth_time
        profile = 1.0 / (1.0 + np.exp(-k * (t - self.growth_time)))
        steps = self.correlation_time / _sample_spacing(t)
        modulation = 1.0 + ar1(t.size, self.fluctuation, steps, rng)
        return np.where(
            t >= 0.0, self.peak_intensity * profile * np.clip(modulation, 0, None), 0.0
        )


@dataclass(frozen=True)
class AircraftSignature:
    """Jet exhaust: steady with slow aspect modulation and optional afterburner."""

    base_intensity: float
    aspect_modulation: float = 0.2
    aspect_period: float = 40.0
    afterburner_start: float | None = None
    afterburner_duration: float = 20.0
    afterburner_factor: float = 4.0

    def intensity(self, t: FloatArray, rng: np.random.Generator) -> FloatArray:
        t = np.asarray(t, dtype=np.float64)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        profile = 1.0 + self.aspect_modulation * np.sin(
            2.0 * np.pi * t / self.aspect_period + phase
        )
        if self.afterburner_start is not None:
            on = (t >= self.afterburner_start) & (
                t < self.afterburner_start + self.afterburner_duration
            )
            profile = np.where(on, profile * self.afterburner_factor, profile)
        return np.where(t >= 0.0, self.base_intensity * profile, 0.0)
