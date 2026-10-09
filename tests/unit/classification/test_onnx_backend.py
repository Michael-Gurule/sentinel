"""ONNX export reproduces the PyTorch classifier."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from locant.classification import EventClassifier
from locant.classification.onnx_backend import (
    ONNX_MODEL,
    OnnxEventClassifier,
    export_onnx,
)
from locant.data.build import generate_samples
from locant.data.config import Priors
from locant.pipeline import ClassificationConfig, LocantPipeline, PipelineConfig
from locant.taxonomy import EVENT_CLASSES

ARTIFACT = Path(__file__).resolve().parents[3] / "models" / "opir_event_classifier"


@pytest.fixture(scope="module")
def exported(tmp_path_factory) -> Path:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    if not ARTIFACT.exists():
        pytest.skip("exported classifier not present")
    root = tmp_path_factory.mktemp("artifact") / "classifier"
    shutil.copytree(ARTIFACT, root)
    export_onnx(root)
    return root


def test_onnx_matches_torch(exported):
    signals = np.concatenate(
        [
            generate_samples(5, f"onnx_{label}", label, 4, Priors(), 64.0)[0]
            for label in EVENT_CLASSES
        ]
    )
    reference = EventClassifier.load(exported).predict(signals)
    onnx = OnnxEventClassifier.load(exported)
    got = onnx.predict(signals)
    assert onnx.classes == EVENT_CLASSES
    np.testing.assert_allclose(got.probabilities, reference.probabilities, atol=1e-5)
    np.testing.assert_allclose(got.energy, reference.energy, atol=1e-4)
    assert got.labels == reference.labels
    np.testing.assert_array_equal(got.prediction_sets, reference.prediction_sets)
    shorter = onnx.predict(signals[:3, -320:])  # dynamic length
    assert shorter.probabilities.shape == (3, len(EVENT_CLASSES))


def test_pipeline_onnx_backend_and_missing_export(exported, tmp_path):
    config = PipelineConfig(
        classification=ClassificationConfig(artifact=exported, backend="onnx")
    )
    assert isinstance(LocantPipeline(config).classifier, OnnxEventClassifier)
    bare = tmp_path / "bare"
    shutil.copytree(exported, bare)
    (bare / ONNX_MODEL).unlink()
    with pytest.raises(FileNotFoundError, match="export-onnx"):
        OnnxEventClassifier.load(bare)
