"""Smoke-run the Phase 3 experiments end to end on a tiny dataset."""

import json

import pytest
from experiments import e1_detection, e2_classification, e3_uncertainty
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
