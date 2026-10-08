"""Staring OPIR sensor model: from source radiant intensity to a pixel time series.

For each frame the model computes, in order:

1. target position (trajectory) and sensor position (platform) in ECEF;
2. range R and line of sight; atmospheric transmittance from the target's
   altitude and the sensor's elevation as seen from the target;
3. cloud occlusion (two-state Markov process) for targets below the cloud top;
4. irradiance at the aperture, E = τ · I / R²;
5. the fraction of energy landing in the tracked pixel (Gaussian PSF with
   sub-pixel phasing that changes as the target moves);
6. pixel background (Earth scene + correlated clutter), sun glints, and
   noise-equivalent irradiance (NEI) noise.

It also produces a noisy line-of-sight measurement for each frame, which
Phase 5 uses to geolocate OPIR events.

Irradiance is reported in pW/m² (1e-12 W/m²).
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from sentinel.core.linalg import FloatArray
from sentinel.sim.geometry import LocalFrame, Platform, elevation_angle, line_of_sight
from sentinel.sim.opir.signatures import Signature, ar1
from sentinel.sim.trajectories import Trajectory

PW_PER_W = 1e12
SCALE_HEIGHT_M = 8_000.0


@dataclass(frozen=True)
class OPIRSensor:
    """Staring focal-plane sensor on an orbital platform.

    Attributes:
        platform: Satellite platform.
        frame_rate_hz: Frames per second.
        ifov_rad: Instantaneous field of view of one pixel, rad.
        psf_sigma_px: Gaussian PSF standard deviation, in pixels.
        nei: Noise-equivalent irradiance per frame, pW/m².
        los_noise_rad: Per-axis line-of-sight measurement noise, rad.
    """

    platform: Platform
    frame_rate_hz: float = 10.0
    ifov_rad: float = 30e-6
    psf_sigma_px: float = 0.4
    nei: float = 1.0
    los_noise_rad: float = 10e-6


@dataclass(frozen=True)
class SceneConditions:
    """Background, atmosphere, cloud, and glint conditions for one pixel.

    Attributes:
        background: Mean Earth-scene irradiance in the pixel, pW/m².
        clutter_std: RMS of temporally correlated background clutter, pW/m².
        clutter_correlation_s: Clutter correlation time, s.
        zenith_optical_depth: Sea-level vertical optical depth in band.
        cloud_top_m: Cloud-top altitude; targets below it can be occluded.
        cloud_transmittance: Fraction of signal passing a cloud.
        mean_clear_s: Mean duration of clear intervals, s (inf: never cloudy).
        mean_cloudy_s: Mean duration of cloudy intervals, s.
        glint_rate_hz: Poisson rate of sun glints in the pixel.
        glint_amplitude: (min, max) glint peak irradiance, pW/m², log-uniform.
        glint_duration_s: (min, max) glint duration, s, uniform.
    """

    background: float = 200.0
    clutter_std: float = 1.0
    clutter_correlation_s: float = 10.0
    zenith_optical_depth: float = 0.5
    cloud_top_m: float = 6_000.0
    cloud_transmittance: float = 0.1
    mean_clear_s: float = float("inf")
    mean_cloudy_s: float = 30.0
    glint_rate_hz: float = 0.0
    glint_amplitude: tuple[float, float] = (5.0, 100.0)
    glint_duration_s: tuple[float, float] = (0.2, 2.0)


@dataclass(frozen=True, eq=False)
class PixelObservation:
    """One pixel's time series plus the truth behind it.

    ``measured`` = ``background`` + ``glint`` + ``signal`` + NEI noise.
    """

    times: FloatArray
    measured: FloatArray
    signal: FloatArray
    background: FloatArray
    glint: FloatArray
    occluded: np.ndarray
    transmittance: FloatArray
    ensquared_energy: FloatArray
    range_m: FloatArray
    los_true: FloatArray
    los_measured: FloatArray
    target_enu: FloatArray
    sensor_ecef: FloatArray
    noise_std: float
    glint_count: int = 0

    @property
    def peak_snr(self) -> float:
        """Peak signal over the combined clutter + NEI standard deviation."""
        return float(self.signal.max() / self.noise_std)


def frame_times(
    duration_s: float, frame_rate_hz: float, start: float = 0.0
) -> FloatArray:
    n = round(duration_s * frame_rate_hz)
    return np.asarray(start + np.arange(n) / frame_rate_hz)


def cloud_mask(
    times: FloatArray, scene: SceneConditions, rng: np.random.Generator
) -> np.ndarray:
    """Cloudy/clear state per time from an alternating renewal process."""
    if not np.isfinite(scene.mean_clear_s) or times.size == 0:
        return np.zeros(times.size, dtype=bool)
    p_cloudy = scene.mean_cloudy_s / (scene.mean_clear_s + scene.mean_cloudy_s)
    cloudy = bool(rng.random() < p_cloudy)
    edges, states = [times[0]], [cloudy]
    t = times[0]
    while t < times[-1]:
        mean = scene.mean_cloudy_s if cloudy else scene.mean_clear_s
        t += rng.exponential(mean)
        cloudy = not cloudy
        edges.append(t)
        states.append(cloudy)
    index = np.searchsorted(np.asarray(edges), times, side="right") - 1
    return np.asarray(states, dtype=bool)[index]


def glint_series(
    times: FloatArray, scene: SceneConditions, rng: np.random.Generator
) -> tuple[FloatArray, int]:
    """Sun-glint irradiance per frame (half-sine pulses) and the glint count."""
    out = np.zeros(times.size)
    if scene.glint_rate_hz <= 0.0 or times.size == 0:
        return out, 0
    span = float(times[-1] - times[0])
    count = int(rng.poisson(scene.glint_rate_hz * span))
    low, high = scene.glint_amplitude
    for _ in range(count):
        start = rng.uniform(times[0], times[-1])
        duration = rng.uniform(*scene.glint_duration_s)
        amplitude = float(np.exp(rng.uniform(np.log(low), np.log(high))))
        phase = (times - start) / duration
        inside = (phase >= 0.0) & (phase <= 1.0)
        out[inside] += amplitude * np.sin(np.pi * phase[inside])
    return out, count


def ensquared_energy(offset_px: FloatArray, sigma_px: float) -> FloatArray:
    """Fraction of a Gaussian PSF inside a unit pixel, per (…, 2) sub-pixel offset."""
    u = np.asarray(offset_px)
    per_axis = ndtr((0.5 - u) / sigma_px) - ndtr((-0.5 - u) / sigma_px)
    return np.asarray(np.prod(per_axis, axis=-1))


def _perpendicular_basis(direction: FloatArray) -> tuple[FloatArray, FloatArray]:
    helper = (
        np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.9 else np.array([1.0, 0, 0])
    )
    e1 = np.cross(direction, helper)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(direction, e1)


def observe(
    sensor: OPIRSensor,
    frame: LocalFrame,
    trajectory: Trajectory,
    signature: Signature,
    times: FloatArray,
    onset: float,
    scene: SceneConditions,
    rng: np.random.Generator,
) -> PixelObservation:
    """Simulate the tracked pixel for an event with onset at ``onset`` (s).

    ``times`` are absolute scenario times; the trajectory and signature are
    evaluated at ``times - onset``.
    """
    times = np.asarray(times, dtype=np.float64)
    local_t = times - onset
    truth = trajectory.sample(local_t)
    target_ecef = frame.to_ecef(truth.position)
    sensor_ecef = sensor.platform.position_ecef(times)
    los, ranges = line_of_sight(sensor_ecef, target_ecef)

    altitude = truth.position[:, 2] + frame.alt
    elevation = elevation_angle(target_ecef, sensor_ecef)
    air_mass = np.exp(-np.clip(altitude, 0.0, None) / SCALE_HEIGHT_M) / np.sin(
        np.clip(elevation, np.radians(3.0), None)
    )
    transmittance = np.exp(-scene.zenith_optical_depth * air_mass)
    occluded = cloud_mask(times, scene, rng) & (altitude < scene.cloud_top_m)
    transmittance = np.where(
        occluded, transmittance * scene.cloud_transmittance, transmittance
    )

    irradiance = (
        transmittance * signature.intensity(local_t, rng) / ranges**2 * PW_PER_W
    )

    e1, e2 = _perpendicular_basis(los[0])
    angles = np.column_stack([(los - los[0]) @ e1, (los - los[0]) @ e2])
    offset = rng.uniform(-0.5, 0.5, 2) + angles / sensor.ifov_rad
    offset = (offset + 0.5) % 1.0 - 0.5  # sensor follows the brightest pixel
    fraction = ensquared_energy(offset, sensor.psf_sigma_px)
    signal = fraction * irradiance

    steps = scene.clutter_correlation_s * sensor.frame_rate_hz
    background = scene.background + ar1(times.size, scene.clutter_std, steps, rng)
    glint, glint_count = glint_series(times, scene, rng)
    measured = background + glint + signal + rng.normal(0.0, sensor.nei, times.size)

    los_noise = rng.normal(0.0, sensor.los_noise_rad, (times.size, 2))
    los_measured = los + los_noise[:, :1] * e1 + los_noise[:, 1:] * e2
    los_measured /= np.linalg.norm(los_measured, axis=1, keepdims=True)

    return PixelObservation(
        times=times,
        measured=measured,
        signal=signal,
        background=background,
        glint=glint,
        occluded=occluded,
        transmittance=transmittance,
        ensquared_energy=fraction,
        range_m=ranges,
        los_true=los,
        los_measured=los_measured,
        target_enu=truth.position,
        sensor_ecef=sensor_ecef,
        noise_std=float(np.hypot(sensor.nei, scene.clutter_std)),
        glint_count=glint_count,
    )
