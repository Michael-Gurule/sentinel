"""Dataset configuration: parameter priors, splits, and domain-shift overrides.

Every continuous parameter is drawn from a :class:`Range` (uniform, or
log-uniform for quantities spanning orders of magnitude). A split may override
any prior with a nested mapping, which is how the domain-shift test sets are
defined: same generative model, deliberately shifted parameter ranges.
"""

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Range(_Model):
    """Closed interval sampled uniformly, or log-uniformly when ``log`` is set."""

    low: float
    high: float
    log: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> "Range":
        if self.high < self.low:
            raise ValueError("high must be >= low")
        if self.log and self.low <= 0:
            raise ValueError("log-uniform ranges must be positive")
        return self

    def sample(self, rng: np.random.Generator) -> float:
        if self.log:
            return float(np.exp(rng.uniform(np.log(self.low), np.log(self.high))))
        return float(rng.uniform(self.low, self.high))


def _r(low: float, high: float, log: bool = False) -> Range:
    return Range(low=low, high=high, log=log)


class LaunchPrior(_Model):
    onset_s: Range = _r(2.0, 40.0)
    peak_intensity: Range = _r(5e4, 1e6, log=True)
    rise_time: Range = _r(1.0, 5.0)
    burn_time: Range = _r(60.0, 180.0)
    thrust_acceleration: Range = _r(20.0, 40.0)
    pitch_elevation_deg: Range = _r(30.0, 70.0)
    staging_probability: float = Field(0.5, ge=0, le=1)
    staging_fraction: Range = _r(0.4, 0.6)
    flicker: Range = _r(0.01, 0.06)


class ExplosionPrior(_Model):
    onset_s: Range = _r(2.0, 40.0)
    peak_intensity: Range = _r(2e5, 2e7, log=True)
    flash_decay: Range = _r(0.05, 0.5)
    fireball_ratio: Range = _r(0.05, 0.3)
    fireball_decay: Range = _r(1.0, 6.0)


class FirePrior(_Model):
    onset_s: Range = _r(2.0, 40.0)
    peak_intensity: Range = _r(1e5, 3e6, log=True)
    growth_time: Range = _r(20.0, 200.0)
    fluctuation: Range = _r(0.05, 0.3)
    correlation_time: Range = _r(3.0, 15.0)


class AircraftPrior(_Model):
    onset_s: Range = _r(-60.0, 30.0)
    base_intensity: Range = _r(5e3, 6e4, log=True)
    altitude_m: Range = _r(3_000.0, 13_000.0)
    speed_mps: Range = _r(150.0, 300.0)
    aspect_modulation: Range = _r(0.05, 0.3)
    aspect_period: Range = _r(20.0, 80.0)
    afterburner_probability: float = Field(0.2, ge=0, le=1)
    afterburner_factor: Range = _r(2.0, 6.0)


class ScenePrior(_Model):
    background: Range = _r(100.0, 400.0)
    clutter_std: Range = _r(0.3, 3.0, log=True)
    clutter_correlation_s: Range = _r(3.0, 30.0)
    zenith_optical_depth: Range = _r(0.2, 0.8)
    cloud_probability: float = Field(0.4, ge=0, le=1)
    mean_clear_s: Range = _r(30.0, 120.0)
    mean_cloudy_s: Range = _r(10.0, 60.0)
    cloud_top_m: Range = _r(2_000.0, 10_000.0)
    cloud_transmittance: Range = _r(0.02, 0.3)
    glint_probability: float = Field(0.3, ge=0, le=1)
    glint_rate_hz: Range = _r(1 / 120, 1 / 20)
    glint_amplitude: tuple[float, float] = (5.0, 150.0)


class SensorPrior(_Model):
    nei: Range = _r(0.5, 2.0, log=True)
    frame_rate_hz: float = Field(10.0, gt=0)
    ifov_urad: float = Field(30.0, gt=0)
    psf_sigma_px: Range = _r(0.3, 0.6)
    site_lat_deg: Range = _r(0.0, 60.0)
    site_lon_offset_deg: Range = _r(-40.0, 40.0)
    """Site longitude relative to the GEO sub-satellite point."""


class Priors(_Model):
    launch: LaunchPrior = LaunchPrior()
    explosion: ExplosionPrior = ExplosionPrior()
    fire: FirePrior = FirePrior()
    aircraft: AircraftPrior = AircraftPrior()
    scene: ScenePrior = ScenePrior()
    sensor: SensorPrior = SensorPrior()


class SplitConfig(_Model):
    name: str
    samples_per_class: int = Field(gt=0)
    overrides: dict[str, Any] = {}
    """Nested mapping deep-merged into the base priors for this split."""
    description: str = ""


class DatasetConfig(_Model):
    name: str
    version: str
    root_seed: int = Field(ge=0)
    window_s: float = Field(64.0, gt=0)
    priors: Priors = Priors()
    splits: list[SplitConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_splits(self) -> "DatasetConfig":
        names = [s.name for s in self.splits]
        if len(set(names)) != len(names):
            raise ValueError("split names must be unique")
        for split in self.splits:
            priors = self.priors_for(split)  # validates overrides eagerly
            for name in ("launch", "explosion", "fire", "aircraft"):
                onset = getattr(priors, name).onset_s
                if onset.high >= self.window_s:
                    raise ValueError(
                        f"split {split.name!r}: {name} onset_s.high ({onset.high}) must be "
                        f"inside the {self.window_s} s window"
                    )
        return self

    def priors_for(self, split: SplitConfig) -> Priors:
        """Base priors with the split's overrides applied and validated."""
        merged = deep_merge(self.priors.model_dump(), split.overrides)
        return Priors.model_validate(merged)

    def scaled(self, factor: float) -> "DatasetConfig":
        """Same config with every split's size multiplied (at least 1 per class)."""
        splits = [
            s.model_copy(
                update={
                    "samples_per_class": max(1, round(s.samples_per_class * factor))
                }
            )
            for s in self.splits
        ]
        return self.model_copy(update={"splits": splits})


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overrides`` into a copy of ``base``."""
    out = dict(base)
    for key, value in overrides.items():
        if key not in out:
            raise KeyError(f"unknown override key {key!r}")
        if isinstance(value, dict) and isinstance(out[key], dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_dataset_config(path: str | Path) -> DatasetConfig:
    return DatasetConfig.model_validate(yaml.safe_load(Path(path).read_text()))
