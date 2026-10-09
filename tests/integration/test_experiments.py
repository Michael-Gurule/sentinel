"""Smoke-run the experiments end to end on a tiny dataset / short scenarios."""

import json

import pytest
from experiments import (
    e1_detection,
    e2_classification,
    e3_uncertainty,
    e4_geolocation,
    e5_tracking,
    e6_fusion,
    e7_robustness,
)
from experiments.common import RunOptions

from sentinel.data import build_dataset, load_dataset_config


@pytest.fixture(scope="module")
def options(tmp_path_factory) -> RunOptions:
    root = tmp_path_factory.mktemp("experiments")
    config = load_dataset_config("configs/dataset/opir_v2.yaml").scaled(0.012)
    build_dataset(config, root / "data")
    return RunOptions(
        quick=True,
        data_dir=root / "data",
        report_dir=root / "reports",
        model_dir=root / "models",
        device="cpu",
        workers=1,
    )


def test_experiments_chain(options):
    e1 = e1_detection.run(options)
    assert {"cfar", "cusum", "glrt", "v1_multi_method"} <= set(e1["results"])
    e2 = e2_classification.run(options)
    assert e2["selected_model"] in e2["summary"]
    assert (options.model_dir / "e2" / "cnn_seed0" / "artifact.json").exists()
    e3 = e3_uncertainty.run(options)
    assert e3["architecture"] in ("cnn", "tcn")
    assert "launch" in e3["ood"]
    exported = options.model_dir / "export" / "opir_event_classifier" / "artifact.json"
    artifact = json.loads(exported.read_text())
    assert artifact["conformal"]["alpha"] == 0.1
    assert artifact["temperature"] > 0
    for name in ("e1_detection", "e2_classification", "e3_uncertainty"):
        assert (options.report_dir / f"{name}.json").exists()


def test_e4_geolocation_quick(options):
    report = e4_geolocation.run(options)
    sweep = report["noise_sweep"]["10ns"]
    assert sweep["ml"]["rmse"] < 3 * sweep["crlb_rms"]
    assert set(report["dop_maps"]) == {"mixed_altitude", "ground_only", "compact_2km"}
    assert (options.report_dir / "e4_geolocation.json").exists()


def test_e5_tracking_quick(options):
    report = e5_tracking.run(options)
    assert set(report["motion_model"]) == {"cv_q25", "cv_q400", "imm_q25_q1600"}
    assert "centralized/defer" in report["stereo_ambiguity"]
    pairing = report["stereo_pairing"]
    assert pairing["defer"]["ghost_pairs"] <= pairing["gnn"]["ghost_pairs"]
    assert (options.report_dir / "e5_tracking.json").exists()


def test_e6_fusion_quick(options):
    report = e6_fusion.run(options)
    assert set(report["architectures"]) == {
        "rf",
        "opir",
        "centralized",
        "t2t_naive",
        "t2t_ci",
    }
    (classification,) = report["classification"].values()
    assert 0.0 <= classification["final_accuracy"]["value"] <= 1.0
    assert (options.report_dir / "e6_fusion.json").exists()


def test_e7_robustness_quick(options):
    report = e7_robustness.run(options)
    assert "centralized/rf_down" in report["rf_outage"]
    assert "2s/extrapolate" in report["rf_latency"]
    assert {"5ns_2m/naive", "30ns_10m/consider"} <= set(report["rf_bias"])
    assert (options.report_dir / "e7_robustness.json").exists()
