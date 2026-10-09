"""Multi-INT fusion engine.

Centralized measurement-level fusion: OPIR and RF measurements update shared
tracks sequentially, each with its own noise covariance. Under independent
sensor errors this is the exact Bayesian update; it replaces v1's
pre-fuse-then-update scheme, which counted information twice.
"""

from collections import Counter
from collections.abc import Sequence

import numpy as np

from sentinel.tracking.models import ConstantVelocity
from sentinel.tracking.tracker import Measurement, MultiTargetTracker, Track


class FusionEngine:
    """Maintains fused tracks from OPIR and RF measurements."""

    def __init__(
        self,
        motion_model: ConstantVelocity | None = None,
        gate_probability: float = 0.99,
        max_coast_time: float = 10.0,
        initial_velocity_std: float = 300.0,
        confirm_hits: int = 3,
        confirm_window: int = 5,
        class_weight: float = 0.3,
        merge_probability: float | None = 0.9,
        imm: bool = True,
    ) -> None:
        """
        Args:
            motion_model: Target dynamics; defaults to constant velocity with
                q = 25 m²/s³ (≈5 m/s² RMS acceleration over 1 s updates).
            gate_probability: χ² gate probability for association.
            max_coast_time: Seconds without an update before a track is dropped.
            initial_velocity_std: Velocity prior (m/s) for position-only births.
            confirm_hits, confirm_window: M-of-N track confirmation.
            class_weight: Tempering weight for pooling class evidence on tracks.
            imm: Track with an IMM of a quiet (q = 25) and a maneuvering
                (q = 1600, ≈40 m/s² over 1 s) constant-velocity model, so both
                steady targets and boosting launches are followed.
        """
        self.tracker = MultiTargetTracker(
            motion_model or ConstantVelocity(noise_intensity=25.0),
            gate_probability=gate_probability,
            max_coast_time=max_coast_time,
            initial_velocity_std=initial_velocity_std,
            confirm_hits=confirm_hits,
            confirm_window=confirm_window,
            class_weight=class_weight,
            merge_probability=merge_probability,
            imm_models=(
                [
                    ConstantVelocity(noise_intensity=25.0),
                    ConstantVelocity(noise_intensity=1_600.0),
                ]
                if imm
                else None
            ),
        )

    @property
    def time(self) -> float | None:
        return self.tracker.time

    @property
    def tracks(self) -> list[Track]:
        """All tracks, tentative and confirmed."""
        return self.tracker.tracks

    @property
    def confirmed_tracks(self) -> list[Track]:
        return self.tracker.confirmed_tracks

    def process(
        self, measurements: Sequence[Measurement], timestamp: float
    ) -> list[Track]:
        """Fuse one scan of measurements and return the current tracks."""
        return self.tracker.step(measurements, timestamp)

    def in_gate(self, track: Track, measurement: Measurement) -> bool:
        return self.tracker.in_gate(track, measurement)

    def summary(self) -> dict[str, object]:
        """Counts and uncertainty statistics over the current tracks."""
        tracks = self.tracks
        sources: Counter[str] = Counter()
        for track in tracks:
            sources.update(track.hits_by_source.keys())
        labels = Counter(track.label or "unknown" for track in tracks)
        uncertainty = [t.position_rms_uncertainty for t in tracks]
        return {
            "time": self.time,
            "total_tracks": len(tracks),
            "tracks_by_label": dict(labels),
            "tracks_with_source": dict(sources),
            "multi_source_tracks": sum(len(t.hits_by_source) > 1 for t in tracks),
            "mean_position_rms_uncertainty_m": (
                float(np.mean(uncertainty)) if uncertainty else None
            ),
        }
