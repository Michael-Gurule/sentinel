import numpy as np
import pytest
import torch

from sentinel.inference import OPIRInference
from sentinel.models.cnn_classifier import (
    OPIRClassifier,
    OPIREventCNN,
    select_device,
)
from sentinel.models.taxonomy import EVENT_CLASSES, INPUT_LENGTH, preprocess_signal


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(0)
    model = OPIREventCNN()
    path = tmp_path / "model.pth"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 3}, path)
    return path


def test_preprocess_normalizes_and_resamples():
    x = preprocess_signal(np.linspace(0.0, 10.0, 250))
    assert x.shape == (INPUT_LENGTH,)
    assert x.mean() == pytest.approx(0.0, abs=1e-9)
    assert x.std() == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError, match="1-D"):
        preprocess_signal(np.ones((2, 5)))


def test_cnn_defaults_follow_the_taxonomy():
    model = OPIREventCNN()
    assert model.input_length == INPUT_LENGTH
    assert model(torch.zeros(2, 1, INPUT_LENGTH)).shape == (2, len(EVENT_CLASSES))
    features = model.get_feature_maps(torch.zeros(1, 1, INPUT_LENGTH), layer=3)
    assert features.shape[1] == 128


def test_classify_matches_classify_batch(checkpoint, rng):
    classifier = OPIRClassifier(model_path=checkpoint)
    signals = rng.normal(size=(4, 180))
    batch = classifier.classify_batch(signals)
    single = [classifier.classify(s) for s in signals]
    for a, b in zip(batch, single, strict=True):
        np.testing.assert_allclose(a.probabilities, b.probabilities, rtol=1e-5)
        assert a.class_name == b.class_name
    assert batch[0].probabilities.sum() == pytest.approx(1.0, abs=1e-6)
    assert set(batch[0].to_dict()["probabilities"]) == set(EVENT_CLASSES)


def test_inference_and_pipeline_classifier_agree(checkpoint, rng):
    """Regression for C4: one taxonomy, input length, and preprocessing."""
    signal = rng.normal(size=INPUT_LENGTH)
    pipeline_result = OPIRClassifier(model_path=checkpoint).classify(signal)
    inference = OPIRInference(model_path=checkpoint, device="cpu")
    result = inference.predict(signal)
    assert result["class_name"] == pipeline_result.class_name
    assert result["confidence"] == pytest.approx(pipeline_result.confidence, rel=1e-5)
    assert "probabilities" not in inference.predict(signal, return_probs=False)
    info = inference.get_model_info()
    assert info["training_epoch"] == 3
    assert info["class_names"] == list(EVENT_CLASSES)


def test_predict_batch_preserves_order(checkpoint, rng):
    inference = OPIRInference(model_path=checkpoint, device="cpu")
    signals = rng.normal(size=(5, INPUT_LENGTH))
    batched = inference.predict_batch(signals, batch_size=2)
    singles = [inference.predict(s) for s in signals]
    assert [b["class_name"] for b in batched] == [s["class_name"] for s in singles]


def test_bare_state_dict_checkpoint_loads(tmp_path):
    path = tmp_path / "bare.pth"
    torch.save(OPIREventCNN().state_dict(), path)
    assert OPIRClassifier(model_path=path).checkpoint_info == {}


def test_select_device_prefers_explicit_choice():
    assert select_device("cpu").type == "cpu"
    assert select_device().type in {"cpu", "cuda", "mps"}
