"""Run the pipeline on a simulated scenario and score it against truth."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from sentinel.eval.tracking import Snapshot, TrackingEvaluation, evaluate_tracking
from sentinel.pipeline.runner import (
    FrameResult,
    OPIRObservation,
    SensorFrame,
    SentinelPipeline,
)
from sentinel.pipeline.stages import RFObservation
from sentinel.sim.opir.reports import event_reports, to_local
from sentinel.sim.scenario import ScenarioResult

GOSPA_CUTOFF_M = 2_000.0


def frames_from_scenario(
    result: ScenarioResult, scan_period_s: float = 1.0, window_s: float = 64.0
) -> Iterator[SensorFrame]:
    """One :class:`SensorFrame` per scan: every event pixel from every OPIR
    sensor (the window ending at the scan, with its measured line of sight)
    and every RF scan taken at that time."""
    frame = result.config.origin.frame()
    fs = result.config.sensor.frame_rate_hz
    window = round(window_s * fs)
    rf_by_time: dict[float, list[RFObservation]] = {}
    for _, scan in result.rf_scans:
        rf_by_time.setdefault(round(scan.t, 6), []).append(
            RFObservation(scan.tdoa, scan.fdoa, scan.receivers, scan.t)
        )
    for t in np.arange(scan_period_s, result.config.duration_s, scan_period_s):
        k = int(np.argmin(np.abs(result.times - t)))
        opir = []
        for sensor_index, observations in enumerate(result.opir_by_sensor):
            for pixel in observations.values():
                position, los = to_local(frame, pixel, k)
                opir.append(
                    OPIRObservation(
                        pixel.measured[max(0, k + 1 - window) : k + 1],
                        sensor_index=sensor_index,
                        sensor_position=position,
                        line_of_sight=los,
                    )
                )
        yield SensorFrame(float(t), opir, rf_by_time.get(round(float(t), 6), []), fs)


@dataclass(frozen=True, eq=False)
class ScenarioRun:
    frames: list[FrameResult]
    snapshots: list[Snapshot]
    evaluation: TrackingEvaluation

    def metrics(self) -> dict[str, Any]:
        e = self.evaluation
        return {
            "gospa_mean_m": e.gospa_mean,
            "localization_rmse_m": e.localization_rmse,
            "false_tracks_per_scan": e.false_mean,
            "missed_targets_per_scan": e.missed_mean,
            "nees_mean": e.nees_mean,
            "fragmentations": e.fragmentations,
            "id_switches": e.id_switches,
            "confirmation_latency_s": e.latency,
            "scans": e.scans,
        }


def _observable(result: ScenarioResult, t: float, angle_std: float) -> set[str]:
    """Targets any sensor reports at ``t``: an OPIR pixel above SNR 5 on any
    satellite, or an emitter the RF network measured."""
    frame = result.config.origin.frame()
    ids = {
        r.event_id
        for r in event_reports(
            frame, result.opir_by_sensor, result.times, np.array([t]), angle_std
        )
        if r.event_id is not None
    }
    if result.config.rf is not None:
        carriers = {e.id: e.attached_to for e in result.config.rf.emitters}
        ids |= {carriers[e] for e, scan in result.rf_scans if abs(scan.t - t) < 1e-6}
    return ids


def run_scenario(
    pipeline: SentinelPipeline, result: ScenarioResult, scan_period_s: float = 1.0
) -> ScenarioRun:
    """Process every frame of ``result`` and evaluate the confirmed tracks
    against the targets observable at each scan."""
    frames, snapshots = [], []
    window_s = pipeline.config.detection.window_s
    angle_std = pipeline.config.opir.angle_std_rad
    for frame in frames_from_scenario(result, scan_period_s, window_s):
        outcome = pipeline.process_frame(frame)
        frames.append(outcome)
        k = int(np.argmin(np.abs(result.times - frame.time)))
        truth = {
            eid: result.truth[eid].position[k]
            for eid in _observable(result, frame.time, angle_std)
        }
        tracks = outcome.tracks
        snapshots.append(
            Snapshot(
                time=frame.time,
                truth=truth,
                track_ids=tuple(t.id for t in tracks),
                positions=np.array([t.position for t in tracks]).reshape(-1, 3),
                covariances=np.array([t.position_covariance for t in tracks]).reshape(
                    -1, 3, 3
                ),
            )
        )
    return ScenarioRun(frames, snapshots, evaluate_tracking(snapshots, GOSPA_CUTOFF_M))
