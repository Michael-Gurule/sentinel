"""Frame-level OPIR detection reports for tracking experiments.

At each scan time, an event's pixel produces a report when its in-pixel
signal exceeds ``snr_threshold`` times the pixel noise. The report carries the
sensor position and the *measured* line of sight, both in the scenario ENU
frame. Clutter reports (false detections) arrive as a Poisson process per
sensor, with lines of sight toward random ground points in the surveillance
area.
"""

from dataclasses import dataclass

import numpy as np

from locant.core.linalg import FloatArray
from locant.sim.geometry import LocalFrame
from locant.sim.opir.sensor import PixelObservation


@dataclass(frozen=True, eq=False)
class OPIRReport:
    """One OPIR detection: where the sensor was and where it looked.

    Attributes:
        time: Scan time, s.
        sensor_index: Which OPIR platform reported it.
        sensor_position: Sensor position in the scenario ENU frame, m.
        line_of_sight: Measured unit line of sight (ENU), sensor → target.
        angle_std: Per-axis line-of-sight noise, rad.
        event_id: Truth event that produced it, or ``None`` for clutter.
        snr: In-pixel SNR at the report frame (0 for clutter).
        frame_index: Frame of the report in the scenario time grid (-1 for clutter).
    """

    time: float
    sensor_index: int
    sensor_position: FloatArray
    line_of_sight: FloatArray
    angle_std: float
    event_id: str | None
    snr: float
    frame_index: int = -1


def to_local(
    frame: LocalFrame, observation: PixelObservation, k: int
) -> tuple[FloatArray, FloatArray]:
    """Sensor position (ENU) and measured LOS (ENU) at frame ``k``."""
    position = frame.from_ecef(observation.sensor_ecef[k])
    los = frame.rotation @ observation.los_measured[k]
    return np.asarray(position), np.asarray(los / np.linalg.norm(los))


def event_reports(
    frame: LocalFrame,
    opir_by_sensor: list[dict[str, PixelObservation]],
    times: FloatArray,
    scan_times: FloatArray,
    angle_std: float,
    snr_threshold: float = 5.0,
) -> list[OPIRReport]:
    """Detections of real events at each scan time, per sensor."""
    reports = []
    for scan in scan_times:
        k = int(np.argmin(np.abs(times - scan)))
        for sensor_index, observations in enumerate(opir_by_sensor):
            for event_id, obs in observations.items():
                snr = float(obs.signal[k] / obs.noise_std)
                if snr < snr_threshold:
                    continue
                position, los = to_local(frame, obs, k)
                reports.append(
                    OPIRReport(
                        float(scan),
                        sensor_index,
                        position,
                        los,
                        angle_std,
                        event_id,
                        snr,
                        k,
                    )
                )
    return reports


def clutter_reports(
    sensor_positions: list[FloatArray],
    scan_times: FloatArray,
    rate_per_scan: float,
    area_half_width_m: float,
    angle_std: float,
    rng: np.random.Generator,
) -> list[OPIRReport]:
    """False detections: Poisson count per sensor and scan, toward random ground points."""
    reports = []
    for scan in scan_times:
        for sensor_index, sensor in enumerate(sensor_positions):
            for _ in range(int(rng.poisson(rate_per_scan))):
                point = np.array(
                    [*rng.uniform(-area_half_width_m, area_half_width_m, 2), 0.0]
                )
                los = point - sensor
                los /= np.linalg.norm(los)
                reports.append(
                    OPIRReport(
                        float(scan), sensor_index, sensor, los, angle_std, None, 0.0
                    )
                )
    return reports
