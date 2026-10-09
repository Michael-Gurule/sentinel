"""Shared scenario generator and fusion runner for E5–E7.

A scenario has launches (OPIR only), aircraft carrying datalinks (OPIR + RF),
and a ground fire (OPIR only). Two OPIR satellites (GEO + Molniya HEO) give
stereo, and a 40 km RF network with clock bias and survey error geolocates
the datalinks.

``run_fusion`` replays a simulated scenario scan by scan through one of five
architectures and returns per-scan snapshots for :mod:`sentinel.eval.tracking`.
The truth set at each scan contains every target observable by *any*
modality, so a single-sensor architecture is charged for targets only the
other modality could see. That is the coverage question fusion should answer.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from sentinel.classification import EventClassifier
from sentinel.core import GeometryError, InsufficientMeasurementsError, nees
from sentinel.eval import Snapshot, evaluate_tracking, gospa, seed_summary
from sentinel.fusion import (
    FusionEngine,
    accept_fix,
    extrapolate,
    fuse_track_lists,
    measurements_from_reports,
    rf_measurement,
)
from sentinel.geolocation import SystematicErrors, solve_tdoa_fdoa
from sentinel.sim import ScenarioConfig, ScenarioResult, simulate_scenario
from sentinel.sim.geometry import LocalFrame
from sentinel.sim.opir.reports import (
    OPIRReport,
    clutter_reports,
    event_reports,
    to_local,
)
from sentinel.sim.scenario import (
    AircraftEvent,
    EmitterConfig,
    FireEvent,
    LaunchEvent,
    OriginConfig,
    PlatformConfig,
    ReceiverConfig,
    RFConfig,
    SceneConfig,
    SensorConfig,
)
from sentinel.taxonomy import EVENT_CLASSES
from sentinel.tracking import ConstantVelocity, LinearMeasurement, Measurement, predict

Mode = Literal["rf", "opir", "centralized", "t2t_naive", "t2t_ci"]
MODES: tuple[Mode, ...] = ("rf", "opir", "centralized", "t2t_naive", "t2t_ci")
ANGLE_STD = 10e-6
CLOCK_BIAS_NS = 5.0
SURVEY_M = 2.0
WINDOW = 640


def make_scenario(
    seed: int,
    n_launch: int = 2,
    n_aircraft: int = 3,
    duration_s: float = 180.0,
    clock_bias_ns: float = CLOCK_BIAS_NS,
    survey_m: float = SURVEY_M,
) -> ScenarioConfig:
    """A randomized multi-target scenario (deterministic for a given seed)."""
    rng = np.random.default_rng(seed)
    events: list[LaunchEvent | AircraftEvent | FireEvent] = []
    for i in range(n_launch):
        events.append(
            LaunchEvent(
                id=f"launch-{i}",
                onset_s=float(rng.uniform(10, 60)),
                position_m=(*rng.uniform(-30_000, 30_000, 2), 0.0),
                azimuth_deg=float(rng.uniform(0, 360)),
                burn_time=float(rng.uniform(90, 150)),
                peak_intensity=float(10 ** rng.uniform(5.0, 6.0)),
            )
        )
    for i in range(n_aircraft):
        heading = rng.uniform(0, 2 * np.pi)
        speed = rng.uniform(200, 250)
        events.append(
            AircraftEvent(
                id=f"aircraft-{i}",
                onset_s=-100.0,
                position_m=(
                    *rng.uniform(-40_000, 40_000, 2),
                    float(rng.uniform(8_000, 11_000)),
                ),
                velocity_mps=(speed * np.sin(heading), speed * np.cos(heading), 0.0),
                base_intensity=float(10 ** rng.uniform(4.5, 5.0)),
            )
        )
    events.append(
        FireEvent(
            id="fire-0",
            onset_s=0.0,
            position_m=(*rng.uniform(-30_000, 30_000, 2), 0.0),
            peak_intensity=5e5,
            growth_time=60.0,
        )
    )
    receivers = [
        ReceiverConfig(
            id=i,
            position_m=p,
            toa_std_ns=10.0,
            frequency_std_hz=1.0,
            clock_bias_std_ns=clock_bias_ns,
            position_error_std_m=survey_m,
        )
        for i, p in enumerate(
            [
                (-20_000.0, -20_000.0, 500.0),
                (20_000.0, -20_000.0, 1_500.0),
                (20_000.0, 20_000.0, 1_000.0),
                (-20_000.0, 20_000.0, 2_000.0),
                (0.0, -36_000.0, 8_000.0),
            ]
        )
    ]
    emitters = [
        EmitterConfig(
            id=f"datalink-{i}",
            attached_to=f"aircraft-{i}",
            profile="datalink",
            measure_fdoa=True,
        )
        for i in range(n_aircraft)
    ]
    return ScenarioConfig(
        name=f"fusion-{seed}",
        origin=OriginConfig(lat_deg=40.0, lon_deg=-100.0),
        duration_s=duration_s,
        sensor=SensorConfig(platform=PlatformConfig(kind="geo", longitude_deg=-100.0)),
        extra_sensors=[
            SensorConfig(platform=PlatformConfig(kind="molniya", raan_deg=-160.0))
        ],
        scene=SceneConfig(clutter_std=1.0, glint_rate_hz=0.0),
        events=events,
        rf=RFConfig(receivers=receivers, emitters=emitters),
    )


@dataclass
class FusionOptions:
    clutter_rate: float = 0.0
    """OPIR false reports per sensor per scan."""
    confirm: tuple[int, int] = (3, 5)
    class_weight: float = 0.3
    classifier: EventClassifier | None = None
    use_systematic: bool = True
    """Fold the RF network's clock-bias/survey levels into each fix."""
    disabled_opir_sensors: tuple[int, ...] = ()
    opir_outage: tuple[int, float, float] | None = None
    """(sensor index, start, end): that OPIR sensor is down for start ≤ t < end."""
    rf_outage: tuple[float, float] | None = None
    rf_delay_s: float = 0.0
    delay_strategy: Literal["drop", "buffer", "extrapolate"] = "drop"
    clutter_seed: int = 0
    imm: bool = True
    process_noise: float = 25.0
    """CV process-noise intensity (m²/s³) when ``imm`` is off."""
    reject_ambiguous: bool = True
    """Defer ambiguous stereo pairings to line-of-sight updates."""
    class_age: tuple[float, float] = (24.0, 62.0)
    """Classify a report only while its detection age (seconds since that
    sensor first reported the source, standing in for the focal-plane
    tracker's 2-D track age) is in this range. The classifier was trained on
    64 s windows with the onset 2–40 s in, i.e. windows ending 24–62 s after
    onset; later windows no longer contain the onset and are out of domain."""


@dataclass
class RunOutput:
    snapshots: list[Snapshot]
    class_trace: dict[str, list[tuple[float, np.ndarray | None]]] = field(
        default_factory=dict
    )
    """Truth id → (time, posterior of its matched track) over time."""
    track_ages: list[tuple[float, float, float, float]] = field(default_factory=list)
    """(time, track age, NEES under the filter covariance, NEES under the
    reported covariance) for matched RF-updated aircraft tracks (E7)."""


def _first_reports(
    result: ScenarioResult, frame: LocalFrame
) -> dict[tuple[int, str], float]:
    """Time each sensor first reported each event."""
    scans = np.arange(0.0, result.config.duration_s, 1.0)
    first: dict[tuple[int, str], float] = {}
    for r in event_reports(
        frame, result.opir_by_sensor, result.times, scans, ANGLE_STD
    ):
        assert r.event_id is not None
        first.setdefault((r.sensor_index, r.event_id), r.time)
    return first


def _classify_windows(
    classifier: EventClassifier | None,
    result: ScenarioResult,
    reports: Sequence[OPIRReport],
    first_report: dict[tuple[int, str], float],
    age_range: tuple[float, float],
) -> list[np.ndarray | None]:
    if classifier is None:
        return [None] * len(reports)
    windows, index = [], []
    for i, r in enumerate(reports):
        if r.event_id is None:
            continue
        age = r.time - first_report[(r.sensor_index, r.event_id)]
        if not age_range[0] <= age <= age_range[1]:
            continue
        signal = result.opir_by_sensor[r.sensor_index][r.event_id].measured[
            : r.frame_index + 1
        ]
        window = signal[-WINDOW:]
        if window.size < WINDOW:
            window = np.concatenate(
                [np.full(WINDOW - window.size, np.median(window)), window]
            )
        windows.append(window)
        index.append(i)
    out: list[np.ndarray | None] = [None] * len(reports)
    if windows:
        probs = classifier.predict(np.stack(windows)).probabilities
        for i, p in zip(index, probs, strict=True):
            out[i] = p
    return out


def _rf_measurements(
    result: ScenarioResult, t: float, systematic: SystematicErrors | None
) -> list[LinearMeasurement]:
    out: list[LinearMeasurement] = []
    for _, scan in result.rf_scans:
        if abs(scan.t - t) > 1e-9:
            continue
        try:
            fix = solve_tdoa_fdoa(
                scan.receivers, scan.tdoa, scan.fdoa, systematic=systematic
            )
        except (GeometryError, InsufficientMeasurementsError):
            continue
        if accept_fix(fix):
            out.append(rf_measurement(fix))
    return out


def _opir_active(opts: FusionOptions, sensor: int, t: float) -> bool:
    if sensor in opts.disabled_opir_sensors:
        return False
    if opts.opir_outage is not None:
        down, start, end = opts.opir_outage
        return not (sensor == down and start <= t < end)
    return True


def _rf_active(opts: FusionOptions, t: float) -> bool:
    return opts.rf_outage is None or not (opts.rf_outage[0] <= t < opts.rf_outage[1])


def _measurements_at(
    result: ScenarioResult,
    t: float,
    mode: Mode,
    opts: FusionOptions,
    rng: np.random.Generator,
    sensor_positions: list[np.ndarray],
    systematic: SystematicErrors | None,
    first_report: dict[tuple[int, str], float],
) -> tuple[list[Measurement], list[Measurement]]:
    """OPIR and RF measurements taken at time ``t``."""
    frame = result.config.origin.frame()
    reports = event_reports(
        frame, result.opir_by_sensor, result.times, np.array([t]), ANGLE_STD
    )
    reports = [r for r in reports if _opir_active(opts, r.sensor_index, t)]
    rf_meas: list[Measurement] = []
    if _rf_active(opts, t) and mode != "opir":
        rf_meas = _rf_measurements(result, t, systematic)
    opir_meas: list[Measurement] = []
    if mode != "rf":
        clutter = clutter_reports(
            [p for i, p in enumerate(sensor_positions) if _opir_active(opts, i, t)],
            np.array([t]),
            opts.clutter_rate,
            60_000.0,
            ANGLE_STD,
            rng,
        )
        all_reports = [*reports, *clutter]
        opir_meas = list(
            measurements_from_reports(
                all_reports,
                _classify_windows(
                    opts.classifier, result, all_reports, first_report, opts.class_age
                ),
                reject_ambiguous=opts.reject_ambiguous,
            )
        )
    return opir_meas, rf_meas


def run_fusion(
    result: ScenarioResult, mode: Mode, options: FusionOptions | None = None
) -> RunOutput:
    """Replay ``result`` through one fusion architecture at one scan per second.

    RF delay handling (``rf_delay_s`` = L > 0):

    * ``drop``: RF fixes arrive L s late, after OPIR has moved the tracker
      past their timestamp, so they are discarded (out of sequence).
    * ``buffer``: every measurement is processed L s late, in time order,
      and the reported estimate is predicted forward L s to the current time.
    * ``extrapolate``: each late RF fix is propagated L s forward with its own
      FDOA velocity (z' = F z, R' = F R Fᵀ + Q) and fused as current. This
      ignores the correlation between Q in R' and the track's own process
      noise over the same interval, a small optimism for L of a few seconds.
    """
    opts = options or FusionOptions()
    frame = result.config.origin.frame()
    levels = _systematic_levels(result)
    systematic = levels if opts.use_systematic else None
    rng = np.random.default_rng(opts.clutter_seed)
    first_report = _first_reports(result, frame) if opts.classifier else {}
    sensor_positions = [
        to_local(frame, next(iter(obs.values())), 0)[0] for obs in result.opir_by_sensor
    ]

    def engine() -> FusionEngine:
        return FusionEngine(
            ConstantVelocity(noise_intensity=opts.process_noise),
            confirm_hits=opts.confirm[0],
            confirm_window=opts.confirm[1],
            class_weight=opts.class_weight,
            imm=opts.imm,
        )

    central, rf_engine, opir_engine = engine(), engine(), engine()
    lag = (
        opts.rf_delay_s
        if (opts.delay_strategy == "buffer" and opts.rf_delay_s > 0)
        else 0.0
    )
    out = RunOutput(snapshots=[])
    t2t_ids: dict[tuple[str, int], int] = {}
    for t in np.arange(0.0, result.config.duration_s, 1.0):
        process_t = float(t - lag)
        if process_t < 0:
            continue
        opir_meas, rf_meas = _measurements_at(
            result,
            process_t,
            mode,
            opts,
            rng,
            sensor_positions,
            systematic,
            first_report,
        )
        if opts.rf_delay_s > 0 and opts.delay_strategy == "drop":
            rf_meas = []  # always older than the OPIR scan that preceded them
        elif opts.rf_delay_s > 0 and opts.delay_strategy == "extrapolate":
            taken = float(t - opts.rf_delay_s)
            rf_meas = (
                [
                    extrapolate(m, opts.rf_delay_s, central.tracker.motion_model)
                    for m in _rf_measurements(result, taken, systematic)
                ]
                if taken >= 0 and mode != "opir" and _rf_active(opts, taken)
                else []
            )
        observable = _observable(result, frame, float(t), opts)

        if mode in ("rf", "opir", "centralized"):
            central.process([*rf_meas, *opir_meas], process_t)
            tracks = central.confirmed_tracks
            ids = tuple(tr.id for tr in tracks)
            states = [tr.reported for tr in tracks]
            posteriors = [tr.class_posterior for tr in tracks]
        else:
            rf_engine.process(rf_meas, process_t)
            opir_engine.process(opir_meas, process_t)
            opir_tracks = opir_engine.confirmed_tracks
            rf_tracks = rf_engine.confirmed_tracks
            fused = fuse_track_lists(
                [tr.reported for tr in opir_tracks],
                [tr.reported for tr in rf_tracks],
                method="naive" if mode == "t2t_naive" else "ci",
                labels=("opir", "rf"),
            )
            # Fused identity follows the OPIR track when there is one (OPIR
            # sees every target), otherwise the RF track.
            keys = [
                ("opir", opir_tracks[i].id)
                if i is not None
                else ("rf", rf_tracks[j].id)  # type: ignore[index]
                for i, j in (f.indices for f in fused)
            ]
            ids = tuple(t2t_ids.setdefault(key, len(t2t_ids)) for key in keys)
            states = [f.state for f in fused]
            posteriors = [
                opir_tracks[i].class_posterior if i is not None else None
                for i, _ in (f.indices for f in fused)
            ]
        if lag:
            states = [predict(s, central.tracker.motion_model, lag) for s in states]

        k = int(np.argmin(np.abs(result.times - t)))
        truth = {eid: result.truth[eid].position[k] for eid in observable}
        snapshot = Snapshot(
            time=float(t),
            truth=truth,
            track_ids=ids,
            positions=np.array([s.mean[:3] for s in states]).reshape(-1, 3),
            covariances=np.array([s.covariance[:3, :3] for s in states]).reshape(
                -1, 3, 3
            ),
        )
        out.snapshots.append(snapshot)
        _record_classes(out, snapshot, posteriors, cutoff=2_000.0)
        if mode in ("rf", "centralized") and not lag:
            _record_ages(out, snapshot, central, truth)
    return out


def _systematic_levels(result: ScenarioResult) -> SystematicErrors:
    """The network's systematic-error levels (identical for every receiver)."""
    assert result.config.rf is not None
    receiver = result.config.rf.receivers[0]
    return SystematicErrors(
        receiver.position_error_std_m,
        receiver.clock_bias_std_ns * 1e-9,
        receiver.lo_offset_std_hz,
    )


def _observable(
    result: ScenarioResult, frame: LocalFrame, t: float, opts: FusionOptions
) -> set[str]:
    """Targets observable by any modality at ``t`` (the truth set to be tracked)."""
    reports = event_reports(
        frame, result.opir_by_sensor, result.times, np.array([t]), ANGLE_STD
    )
    observable = {
        r.event_id
        for r in reports
        if r.event_id is not None and _opir_active(opts, r.sensor_index, t)
    }
    if _rf_active(opts, t) and result.config.rf is not None:
        emitters = {e.id: e.attached_to for e in result.config.rf.emitters}
        observable |= {
            emitters[eid] for eid, scan in result.rf_scans if abs(scan.t - t) < 1e-9
        }
    return observable


def _record_classes(
    out: RunOutput,
    snapshot: Snapshot,
    posteriors: Sequence[np.ndarray | None],
    cutoff: float,
) -> None:
    truth_ids = list(snapshot.truth)
    truth_pos = np.array([snapshot.truth[t] for t in truth_ids]).reshape(-1, 3)
    match = gospa(truth_pos, snapshot.positions, cutoff)
    for r, c in match.assignments:
        out.class_trace.setdefault(truth_ids[r], []).append(
            (snapshot.time, posteriors[c])
        )


def _record_ages(
    out: RunOutput,
    snapshot: Snapshot,
    engine: FusionEngine,
    truth: dict[str, np.ndarray],
) -> None:
    """Record the NEES of RF-updated aircraft tracks against track age, under
    the filter covariance and under the reported covariance (filter + bias
    floor; the floor exists only when fixes were solved with the network's
    systematic levels)."""
    tracks = engine.confirmed_tracks
    truth_ids = [t for t in truth if t.startswith("aircraft")]
    if not tracks or not truth_ids:
        return
    positions = np.array([tr.position for tr in tracks])
    match = gospa(np.array([truth[t] for t in truth_ids]), positions, 2_000.0)
    for r, c in match.assignments:
        tr = tracks[c]
        if tr.hits_by_source.get("rf", 0) == 0:
            continue
        error = tr.position - truth[truth_ids[r]]
        out.track_ages.append(
            (
                snapshot.time,
                snapshot.time - tr.created,
                nees(error, tr.state.covariance[:3, :3]),
                nees(error, tr.position_covariance),
            )
        )


def true_class(event_id: str) -> int:
    kind = event_id.split("-")[0]
    return EVENT_CLASSES.index(kind)


def target_type(event_id: str) -> str:
    return event_id.split("-")[0]


def simulate(seed: int, **kwargs: float) -> ScenarioResult:
    return simulate_scenario(make_scenario(seed, **kwargs), seed)  # type: ignore[arg-type]


# --- scoring ---------------------------------------------------------------

CUTOFF = 2_000.0
"""GOSPA cutoff c (m): an estimate farther than this from a target is a false
track plus a missed target. It is ~5× the OPIR stereo error, so it separates
gross errors from localization error."""
TARGET_TYPES = ("aircraft", "launch", "fire")


def score(snapshots: Sequence[Snapshot]) -> dict[str, Any]:
    """Tracking metrics of one run, overall and per target type."""
    evaluation = evaluate_tracking(snapshots, CUTOFF)
    sq_errors: dict[str, list[float]] = {k: [] for k in TARGET_TYPES}
    present = dict.fromkeys(TARGET_TYPES, 0)
    matched = dict.fromkeys(TARGET_TYPES, 0)
    for snap in snapshots:
        truth_ids = list(snap.truth)
        truth_pos = np.array([snap.truth[t] for t in truth_ids]).reshape(-1, 3)
        pairs = dict(gospa(truth_pos, snap.positions, CUTOFF).assignments)
        for r, tid in enumerate(truth_ids):
            kind = target_type(tid)
            present[kind] += 1
            if r in pairs:
                matched[kind] += 1
                error = snap.positions[pairs[r]] - snap.truth[tid]
                sq_errors[kind].append(float(error @ error))
    latency = {
        kind: [v for t, v in evaluation.latency.items() if target_type(t) == kind]
        for kind in TARGET_TYPES
    }
    return {
        "gospa": evaluation.gospa_mean,
        "ospa": evaluation.ospa_mean,
        "false_per_scan": evaluation.false_mean,
        "missed_per_scan": evaluation.missed_mean,
        "rmse": evaluation.localization_rmse,
        "nees": evaluation.nees_mean,
        "purity": evaluation.purity,
        "fragmentations": evaluation.fragmentations,
        "id_switches": evaluation.id_switches,
        "rmse_by_type": {
            k: float(np.sqrt(np.mean(v))) if v else float("nan")
            for k, v in sq_errors.items()
        },
        "coverage_by_type": {
            k: matched[k] / present[k] if present[k] else float("nan")
            for k in TARGET_TYPES
        },
        "latency_by_type": {
            k: float(np.median(v)) if v and np.isfinite(v).any() else float("nan")
            for k, v in latency.items()
        },
        "never_tracked": int(
            sum(not np.isfinite(v) for v in evaluation.latency.values())
        ),
    }


def window(snapshots: Sequence[Snapshot], start: float, end: float) -> list[Snapshot]:
    return [s for s in snapshots if start <= s.time < end]


def gospa_series(snapshots: Sequence[Snapshot]) -> np.ndarray:
    """Per-scan GOSPA distance."""
    return np.array(
        [
            gospa(
                np.array(list(s.truth.values())).reshape(-1, 3), s.positions, CUTOFF
            ).distance
            for s in snapshots
        ]
    )


def aggregate(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Mean and 95% Student-t interval across seeds, key by key (NaNs, e.g. a
    target type a mode never tracked, are dropped)."""
    out: dict[str, Any] = {}
    for key, first in runs[0].items():
        if isinstance(first, dict):
            out[key] = aggregate([r[key] for r in runs])
            continue
        values = [float(r[key]) for r in runs if np.isfinite(r[key])]
        out[key] = (
            seed_summary(values).as_dict() | {"runs": len(values)}
            if values
            else {"value": float("nan"), "runs": 0}
        )
    return out
