"""Scenario configuration (pydantic + YAML) and end-to-end simulation.

A scenario places thermal events and RF emitters around a geodetic origin,
observes the events with an OPIR sensor, and measures the emitters with an RF
receiver network. Every random stream is derived from one integer seed with
``numpy.random.SeedSequence``, keyed by component, so results are identical
for the same seed regardless of iteration order.
"""

from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from locant.core.linalg import FloatArray
from locant.sim.geometry import (
    GeostationaryPlatform,
    KeplerianPlatform,
    LocalFrame,
    Platform,
)
from locant.sim.opir.sensor import (
    OPIRSensor,
    PixelObservation,
    SceneConditions,
    frame_times,
    observe,
)
from locant.sim.opir.signatures import (
    AircraftSignature,
    ExplosionSignature,
    FireSignature,
    LaunchSignature,
    Signature,
)
from locant.sim.rf.network import ReceiverModel, RFNetwork, RFScan
from locant.sim.rf.waveforms import EMITTER_PROFILES
from locant.sim.trajectories import (
    BallisticBoost,
    ConstantVelocity,
    Stationary,
    Trajectory,
    TrajectorySample,
)

Vector3 = tuple[float, float, float]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OriginConfig(_Model):
    lat_deg: float = Field(ge=-90, le=90)
    lon_deg: float = Field(ge=-180, le=180)
    alt_m: float = 0.0

    def frame(self) -> LocalFrame:
        return LocalFrame.from_degrees(self.lat_deg, self.lon_deg, self.alt_m)


class PlatformConfig(_Model):
    kind: Literal["geo", "molniya"] = "geo"
    longitude_deg: float = 0.0
    """GEO sub-satellite longitude."""
    raan_deg: float = 0.0
    mean_anomaly_deg: float = 180.0
    """Molniya mean anomaly at the epoch (180° = apogee)."""

    def build(self) -> Platform:
        if self.kind == "geo":
            return GeostationaryPlatform(np.radians(self.longitude_deg))
        return KeplerianPlatform.molniya(
            raan=np.radians(self.raan_deg),
            mean_anomaly=np.radians(self.mean_anomaly_deg),
        )


class SensorConfig(_Model):
    platform: PlatformConfig = PlatformConfig()
    frame_rate_hz: float = Field(10.0, gt=0)
    ifov_urad: float = Field(30.0, gt=0)
    psf_sigma_px: float = Field(0.4, gt=0)
    nei: float = Field(1.0, gt=0, description="Noise-equivalent irradiance, pW/m²")
    los_noise_urad: float = Field(10.0, ge=0)

    def build(self) -> OPIRSensor:
        return OPIRSensor(
            platform=self.platform.build(),
            frame_rate_hz=self.frame_rate_hz,
            ifov_rad=self.ifov_urad * 1e-6,
            psf_sigma_px=self.psf_sigma_px,
            nei=self.nei,
            los_noise_rad=self.los_noise_urad * 1e-6,
        )


class SceneConfig(_Model):
    background: float = Field(200.0, ge=0)
    clutter_std: float = Field(1.0, ge=0)
    clutter_correlation_s: float = Field(10.0, gt=0)
    zenith_optical_depth: float = Field(0.5, ge=0)
    cloud_top_m: float = 6_000.0
    cloud_transmittance: float = Field(0.1, ge=0, le=1)
    mean_clear_s: float = Field(float("inf"), gt=0)
    mean_cloudy_s: float = Field(30.0, gt=0)
    glint_rate_hz: float = Field(0.0, ge=0)
    glint_amplitude: tuple[float, float] = (5.0, 100.0)
    glint_duration_s: tuple[float, float] = (0.2, 2.0)

    def build(self) -> SceneConditions:
        return SceneConditions(**self.model_dump())


class _EventBase(_Model):
    id: str
    onset_s: float = 0.0
    position_m: Vector3 = (0.0, 0.0, 0.0)
    """Initial position in the scenario ENU frame."""


class LaunchEvent(_EventBase):
    kind: Literal["launch"] = "launch"
    azimuth_deg: float = 0.0
    thrust_acceleration: float = Field(30.0, gt=9.81)
    burn_time: float = Field(120.0, gt=0)
    vertical_rise_s: float = Field(10.0, gt=0)
    pitch_elevation_deg: float = Field(45.0, gt=0, le=90)
    peak_intensity: float = Field(3e5, gt=0)
    rise_time: float = Field(3.0, gt=0)
    staging_time: float | None = None
    stage2_ratio: float = Field(0.4, gt=0)
    flicker: float = Field(0.03, ge=0)
    burnout_decay: float = Field(2.0, gt=0)

    def build(self) -> tuple[Trajectory, Signature]:
        trajectory = BallisticBoost(
            position=np.array(self.position_m),
            azimuth=np.radians(self.azimuth_deg),
            thrust_acceleration=self.thrust_acceleration,
            burn_time=self.burn_time,
            vertical_rise_s=self.vertical_rise_s,
            pitch_elevation=np.radians(self.pitch_elevation_deg),
        )
        signature = LaunchSignature(
            peak_intensity=self.peak_intensity,
            rise_time=self.rise_time,
            burn_time=self.burn_time,
            staging_time=self.staging_time,
            stage2_ratio=self.stage2_ratio,
            flicker=self.flicker,
            burnout_decay=self.burnout_decay,
        )
        return trajectory, signature


class ExplosionEvent(_EventBase):
    kind: Literal["explosion"] = "explosion"
    peak_intensity: float = Field(2e6, gt=0)
    flash_decay: float = Field(0.2, gt=0)
    fireball_ratio: float = Field(0.15, ge=0, le=1)
    fireball_decay: float = Field(3.0, gt=0)

    def build(self) -> tuple[Trajectory, Signature]:
        return Stationary(np.array(self.position_m)), ExplosionSignature(
            self.peak_intensity,
            self.flash_decay,
            self.fireball_ratio,
            self.fireball_decay,
        )


class FireEvent(_EventBase):
    kind: Literal["fire"] = "fire"
    peak_intensity: float = Field(5e4, gt=0)
    growth_time: float = Field(120.0, gt=0)
    fluctuation: float = Field(0.15, ge=0)
    correlation_time: float = Field(8.0, gt=0)

    def build(self) -> tuple[Trajectory, Signature]:
        return Stationary(np.array(self.position_m)), FireSignature(
            self.peak_intensity,
            self.growth_time,
            self.fluctuation,
            self.correlation_time,
        )


class AircraftEvent(_EventBase):
    kind: Literal["aircraft"] = "aircraft"
    velocity_mps: Vector3 = (200.0, 0.0, 0.0)
    base_intensity: float = Field(5e3, gt=0)
    aspect_modulation: float = Field(0.2, ge=0, lt=1)
    aspect_period: float = Field(40.0, gt=0)
    afterburner_start: float | None = None
    afterburner_duration: float = Field(20.0, gt=0)
    afterburner_factor: float = Field(4.0, ge=1)

    def build(self) -> tuple[Trajectory, Signature]:
        trajectory = ConstantVelocity(
            np.array(self.position_m), np.array(self.velocity_mps)
        )
        signature = AircraftSignature(
            self.base_intensity,
            self.aspect_modulation,
            self.aspect_period,
            self.afterburner_start,
            self.afterburner_duration,
            self.afterburner_factor,
        )
        return trajectory, signature


EventConfig = Annotated[
    LaunchEvent | ExplosionEvent | FireEvent | AircraftEvent,
    Field(discriminator="kind"),
]


class ReceiverConfig(_Model):
    id: int
    position_m: Vector3
    velocity_mps: Vector3 = (0.0, 0.0, 0.0)
    toa_std_ns: float = Field(10.0, gt=0)
    frequency_std_hz: float = Field(1.0, gt=0)
    position_error_std_m: float = Field(0.0, ge=0)
    clock_bias_std_ns: float = Field(0.0, ge=0)
    lo_offset_std_hz: float = Field(0.0, ge=0)

    def build(self) -> ReceiverModel:
        moving = any(v != 0.0 for v in self.velocity_mps)
        trajectory: Trajectory = (
            ConstantVelocity(np.array(self.position_m), np.array(self.velocity_mps))
            if moving
            else Stationary(np.array(self.position_m))
        )
        return ReceiverModel(
            id=self.id,
            trajectory=trajectory,
            toa_std=self.toa_std_ns * 1e-9,
            frequency_std=self.frequency_std_hz,
            position_error_std=self.position_error_std_m,
            clock_bias_std=self.clock_bias_std_ns * 1e-9,
            lo_offset_std=self.lo_offset_std_hz,
        )


class EmitterConfig(_Model):
    id: str
    attached_to: str
    """Id of the event whose trajectory carries the emitter."""
    profile: str
    scan_period_s: float = Field(1.0, gt=0)
    active_intervals_s: list[tuple[float, float]] | None = None
    """Emission windows (absolute time); ``None`` means always on."""
    measure_fdoa: bool = False

    @model_validator(mode="after")
    def _known_profile(self) -> "EmitterConfig":
        if self.profile not in EMITTER_PROFILES:
            raise ValueError(f"unknown emitter profile {self.profile!r}")
        return self


class RFConfig(_Model):
    receivers: list[ReceiverConfig] = Field(min_length=2)
    emitters: list[EmitterConfig] = []


class ScenarioConfig(_Model):
    name: str
    origin: OriginConfig
    duration_s: float = Field(gt=0)
    sensor: SensorConfig = SensorConfig()
    extra_sensors: list[SensorConfig] = []
    """Additional OPIR platforms (e.g. an HEO satellite for stereo); all share
    ``sensor``'s frame rate."""
    scene: SceneConfig = SceneConfig()
    events: list[EventConfig] = []
    rf: RFConfig | None = None

    @model_validator(mode="after")
    def _consistent_ids(self) -> "ScenarioConfig":
        ids = [e.id for e in self.events]
        if len(set(ids)) != len(ids):
            raise ValueError("event ids must be unique")
        if self.rf is not None:
            for emitter in self.rf.emitters:
                if emitter.attached_to not in ids:
                    raise ValueError(
                        f"emitter {emitter.id!r} attached to unknown event"
                    )
        return self


def load_scenario(path: str | Path) -> ScenarioConfig:
    """Read a scenario from YAML."""
    return ScenarioConfig.model_validate(yaml.safe_load(Path(path).read_text()))


class ScenarioResult(BaseModel):
    """Truth and observations of one simulated scenario."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    config: ScenarioConfig
    seed: int
    times: FloatArray
    truth: dict[str, TrajectorySample]
    opir_by_sensor: list[dict[str, PixelObservation]]
    """Per OPIR sensor (index 0 = ``config.sensor``), per event."""
    rf_scans: list[tuple[str, RFScan]]

    @property
    def opir(self) -> dict[str, PixelObservation]:
        """Observations from the primary OPIR sensor."""
        return self.opir_by_sensor[0]


def _rng(seed: int, *key: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=key))


_OPIR_STREAM, _RF_NETWORK_STREAM, _RF_SCAN_STREAM = 0, 1, 2


def simulate_scenario(config: ScenarioConfig, seed: int) -> ScenarioResult:
    """Simulate all events and emitters of ``config`` deterministically from ``seed``."""
    frame = config.origin.frame()
    sensors = [config.sensor.build()] + [
        extra.model_copy(update={"frame_rate_hz": config.sensor.frame_rate_hz}).build()
        for extra in config.extra_sensors
    ]
    scene = config.scene.build()
    times = frame_times(config.duration_s, sensors[0].frame_rate_hz)

    trajectories: dict[str, tuple[Trajectory, float]] = {}
    truth: dict[str, TrajectorySample] = {}
    opir_by_sensor: list[dict[str, PixelObservation]] = [{} for _ in sensors]
    for index, event in enumerate(config.events):
        trajectory, signature = event.build()
        trajectories[event.id] = (trajectory, event.onset_s)
        truth[event.id] = trajectory.sample(times - event.onset_s)
        for sensor_index, sensor in enumerate(sensors):
            # Sensor 0 keeps the original stream key so single-sensor results
            # are unchanged; extra sensors get their own streams.
            key = (
                (_OPIR_STREAM, index)
                if sensor_index == 0
                else (_OPIR_STREAM, index, sensor_index)
            )
            opir_by_sensor[sensor_index][event.id] = observe(
                sensor,
                frame,
                trajectory,
                signature,
                times,
                event.onset_s,
                scene,
                _rng(seed, *key),
            )

    scans: list[tuple[str, RFScan]] = []
    if config.rf is not None:
        network = RFNetwork(
            [r.build() for r in config.rf.receivers], _rng(seed, _RF_NETWORK_STREAM)
        )
        for index, emitter in enumerate(config.rf.emitters):
            rng = _rng(seed, _RF_SCAN_STREAM, index)
            trajectory, onset = trajectories[emitter.attached_to]
            carrier = EMITTER_PROFILES[emitter.profile].carrier_hz
            for t in np.arange(0.0, config.duration_s, emitter.scan_period_s):
                if emitter.active_intervals_s is not None and not any(
                    a <= t < b for a, b in emitter.active_intervals_s
                ):
                    continue
                state = trajectory.sample(np.array([t - onset]))
                scan = network.scan(
                    float(t),
                    state.position[0],
                    state.velocity[0],
                    rng,
                    carrier_frequency=carrier if emitter.measure_fdoa else None,
                )
                scans.append((emitter.id, scan))

    return ScenarioResult(
        config=config,
        seed=seed,
        times=times,
        truth=truth,
        opir_by_sensor=opir_by_sensor,
        rf_scans=scans,
    )
