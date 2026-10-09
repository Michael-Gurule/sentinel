"""Draw one labeled OPIR pixel time series from the dataset priors.

Event parameters are sampled from :mod:`sentinel.data.config` priors and
turned into the same event models the scenario simulator uses, so the dataset
and full scenarios share one generative model.
"""

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from sentinel.core.linalg import FloatArray
from sentinel.data.config import Priors
from sentinel.sim.geometry import GeostationaryPlatform, LocalFrame
from sentinel.sim.opir.sensor import OPIRSensor, SceneConditions, frame_times, observe
from sentinel.sim.scenario import (
    AircraftEvent,
    ExplosionEvent,
    FireEvent,
    LaunchEvent,
    SceneConfig,
)
from sentinel.sim.trajectories import Stationary
from sentinel.taxonomy import EVENT_CLASSES

METADATA_FIELDS = (
    "onset_s",
    "peak_snr",
    "occluded_fraction",
    "glint_count",
    "nei",
    "clutter_std",
    "range_km",
    "site_lat_deg",
)
"""Numeric per-sample metadata stored alongside each signal."""


@dataclass(frozen=True, eq=False)
class Sample:
    signal: FloatArray
    label: int
    metadata: dict[str, float]
    params: dict[str, Any]


class _NoSource:
    """Background-only samples: no target in the pixel."""

    def intensity(self, t: FloatArray, rng: np.random.Generator) -> FloatArray:
        del rng
        return np.zeros(np.asarray(t).size)


def _event(
    label: str, priors: Priors, rng: np.random.Generator
) -> LaunchEvent | ExplosionEvent | FireEvent | AircraftEvent:
    if label == "launch":
        p = priors.launch
        burn = p.burn_time.sample(rng)
        staged = rng.random() < p.staging_probability
        return LaunchEvent(
            id="event",
            onset_s=p.onset_s.sample(rng),
            azimuth_deg=rng.uniform(0.0, 360.0),
            thrust_acceleration=p.thrust_acceleration.sample(rng),
            burn_time=burn,
            pitch_elevation_deg=p.pitch_elevation_deg.sample(rng),
            peak_intensity=p.peak_intensity.sample(rng),
            rise_time=p.rise_time.sample(rng),
            staging_time=burn * p.staging_fraction.sample(rng) if staged else None,
            flicker=p.flicker.sample(rng),
        )
    if label == "explosion":
        q = priors.explosion
        return ExplosionEvent(
            id="event",
            onset_s=q.onset_s.sample(rng),
            peak_intensity=q.peak_intensity.sample(rng),
            flash_decay=q.flash_decay.sample(rng),
            fireball_ratio=q.fireball_ratio.sample(rng),
            fireball_decay=q.fireball_decay.sample(rng),
        )
    if label == "fire":
        f = priors.fire
        return FireEvent(
            id="event",
            onset_s=f.onset_s.sample(rng),
            peak_intensity=f.peak_intensity.sample(rng),
            growth_time=f.growth_time.sample(rng),
            fluctuation=f.fluctuation.sample(rng),
            correlation_time=f.correlation_time.sample(rng),
        )
    if label == "aircraft":
        a = priors.aircraft
        heading = rng.uniform(0.0, 2.0 * np.pi)
        speed = a.speed_mps.sample(rng)
        afterburner = rng.random() < a.afterburner_probability
        return AircraftEvent(
            id="event",
            onset_s=a.onset_s.sample(rng),
            position_m=(0.0, 0.0, a.altitude_m.sample(rng)),
            velocity_mps=(speed * np.sin(heading), speed * np.cos(heading), 0.0),
            base_intensity=a.base_intensity.sample(rng),
            aspect_modulation=a.aspect_modulation.sample(rng),
            aspect_period=a.aspect_period.sample(rng),
            afterburner_start=rng.uniform(0.0, 60.0) if afterburner else None,
            afterburner_factor=a.afterburner_factor.sample(rng),
        )
    raise ValueError(f"no event model for label {label!r}")


def _scene(priors: Priors, rng: np.random.Generator) -> SceneConfig:
    s = priors.scene
    cloudy = rng.random() < s.cloud_probability
    glinty = rng.random() < s.glint_probability
    return SceneConfig(
        background=s.background.sample(rng),
        clutter_std=s.clutter_std.sample(rng),
        clutter_correlation_s=s.clutter_correlation_s.sample(rng),
        zenith_optical_depth=s.zenith_optical_depth.sample(rng),
        cloud_top_m=s.cloud_top_m.sample(rng),
        cloud_transmittance=s.cloud_transmittance.sample(rng),
        mean_clear_s=s.mean_clear_s.sample(rng) if cloudy else float("inf"),
        mean_cloudy_s=s.mean_cloudy_s.sample(rng),
        glint_rate_hz=s.glint_rate_hz.sample(rng) if glinty else 0.0,
        glint_amplitude=s.glint_amplitude,
    )


def generate_sample(
    label: str, priors: Priors, window_s: float, rng: np.random.Generator
) -> Sample:
    """Simulate one sample of class ``label`` (one of ``EVENT_CLASSES``)."""
    if label not in EVENT_CLASSES:
        raise ValueError(f"unknown label {label!r}")
    sp = priors.sensor
    lat = sp.site_lat_deg.sample(rng)
    lon_offset = sp.site_lon_offset_deg.sample(rng)
    sat_lon = rng.uniform(-180.0, 180.0)
    frame = LocalFrame.from_degrees(lat, (sat_lon + lon_offset + 180.0) % 360.0 - 180.0)
    sensor = OPIRSensor(
        platform=GeostationaryPlatform(np.radians(sat_lon)),
        frame_rate_hz=sp.frame_rate_hz,
        ifov_rad=sp.ifov_urad * 1e-6,
        psf_sigma_px=sp.psf_sigma_px.sample(rng),
        nei=sp.nei.sample(rng),
    )
    scene_config = _scene(priors, rng)
    scene: SceneConditions = scene_config.build()
    times = frame_times(window_s, sensor.frame_rate_hz)

    if label == "background":
        event_params: dict[str, Any] = {}
        observation = observe(
            sensor, frame, Stationary(np.zeros(3)), _NoSource(), times, 0.0, scene, rng
        )
        onset = float("nan")
    else:
        event = _event(label, priors, rng)
        trajectory, signature = event.build()
        observation = observe(
            sensor, frame, trajectory, signature, times, event.onset_s, scene, rng
        )
        event_params = event.model_dump(exclude={"id"})
        onset = event.onset_s

    metadata = {
        "onset_s": onset,
        "peak_snr": observation.peak_snr,
        "occluded_fraction": float(observation.occluded.mean()),
        "glint_count": float(observation.glint_count),
        "nei": sensor.nei,
        "clutter_std": scene.clutter_std,
        "range_km": float(observation.range_m.mean() / 1e3),
        "site_lat_deg": lat,
    }
    params = {
        "event": event_params,
        "scene": scene_config.model_dump(),
        "sensor": {"nei": sensor.nei, "psf_sigma_px": sensor.psf_sigma_px},
        "site": {"lat_deg": lat, "lon_offset_deg": lon_offset, "sat_lon_deg": sat_lon},
    }
    return Sample(
        signal=observation.measured.astype(np.float32),
        label=EVENT_CLASSES.index(label),
        metadata=metadata,
        params=params,
    )


def params_to_json(params: dict[str, Any]) -> str:
    """Canonical JSON (sorted keys, ``inf``/``nan`` as strings) for storage."""

    def fix(value: Any) -> Any:
        if isinstance(value, float) and not np.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {k: fix(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [fix(v) for v in value]
        return value

    return json.dumps(fix(params), sort_keys=True)
