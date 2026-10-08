"""End-to-end tests of the multi-sensor pipeline on simulated scenarios."""

import json

import numpy as np
import pytest

from sentinel.geolocation import simulate_fdoa, simulate_tdoa
from sentinel.models.signal_generator import OPIRSignalGenerator
from sentinel.pipeline.phase3_pipeline import SENTINELPhase3Pipeline, demo_phase3_system

START = np.array([5_000.0, 5_000.0, 500.0])
VELOCITY = np.array([100.0, 50.0, 0.0])


@pytest.fixture
def pipeline() -> SENTINELPhase3Pipeline:
    return SENTINELPhase3Pipeline()


def run_scenario(pipeline, rng, frames=15, with_fdoa=False):
    generator = OPIRSignalGenerator(rng=rng)
    for frame in range(frames):
        t = float(frame)
        truth = START + VELOCITY * t
        tdoa = simulate_tdoa(truth, pipeline.receivers, 10e-9, rng)
        rf = (
            (tdoa, simulate_fdoa(truth, VELOCITY, pipeline.receivers, 1e9, 1.0, rng))
            if with_fdoa
            else tdoa
        )
        result = pipeline.process_multi_sensor_frame(
            opir_signals=[generator.generate_launch_signature(start_time=2.0)],
            rf_measurements=[rf],
            sampling_rate=generator.sampling_rate,
            timestamp=t,
        )
    return result, truth


@pytest.mark.parametrize("with_fdoa", [False, True])
def test_moving_emitter_is_tracked_within_its_uncertainty(pipeline, rng, with_fdoa):
    result, truth = run_scenario(pipeline, rng, with_fdoa=with_fdoa)
    assert result["fused_tracks"] == 1
    assert result["rf_geolocations"] == 1
    assert result["opir_detections"] == 1
    (track,) = pipeline.fusion_engine.tracks
    assert track.hits_by_source == {"rf": 15}
    error = np.linalg.norm(track.position - truth)
    assert error < 4 * track.position_rms_uncertainty
    np.testing.assert_allclose(track.velocity, VELOCITY, atol=15.0)


def test_rf_failures_are_reported_not_raised(pipeline, rng):
    too_few = simulate_tdoa(START, pipeline.receivers[:3], 10e-9, rng)
    result = pipeline.process_multi_sensor_frame([], [too_few], 100.0, 0.0)
    assert result["rf_failures"] == 1
    assert result["fused_tracks"] == 0


def test_situation_awareness_and_export(pipeline, rng, tmp_path):
    run_scenario(pipeline, rng, frames=5)
    summary = pipeline.get_situation_awareness()
    assert summary["total_tracks"] == 1
    assert summary["tracks_with_source"] == {"rf": 1}
    track_id = pipeline.fusion_engine.tracks[0].id
    assert pipeline.get_track_by_id(track_id) is not None
    assert pipeline.get_track_by_id(track_id + 100) is None
    path = tmp_path / "tracks.json"
    pipeline.export_tracks_to_file(path)
    exported = json.loads(path.read_text())
    assert exported["tracks"][0]["hits_by_source"] == {"rf": 5}


def test_demo_runs(capsys):
    demo_phase3_system(seed=1)
    assert "track 0" in capsys.readouterr().out
