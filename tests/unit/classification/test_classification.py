import numpy as np
import pytest
import torch

from sentinel.classification import (
    EventClassifier,
    FeatureClassifier,
    ModelArtifact,
    PreprocessSpec,
    TrainConfig,
    load_artifact,
    preprocess,
    save_artifact,
    select_device,
    train_model,
)
from sentinel.classification.artifact import ConformalSpec
from sentinel.classification.calibration import (
    energy_score,
    fit_temperature,
    max_softmax,
    nll,
    probabilities,
)
from sentinel.classification.conformal import (
    calibrate,
    conformal_scores,
    coverage,
    prediction_sets,
)
from sentinel.classification.features import FEATURE_NAMES, extract_features
from sentinel.classification.models import CNN1D, TCN, build_model
from sentinel.classification.preprocess import noise_normalize
from sentinel.taxonomy import EVENT_CLASSES


class TestPreprocess:
    def test_noise_units(self, rng):
        x = 50.0 + rng.normal(0.0, 3.0, (20, 2_000))
        z = noise_normalize(x)
        assert np.median(z.std(axis=1)) == pytest.approx(1.0, abs=0.05)

    def test_specs(self, rng):
        x = rng.normal(size=(3, 640))
        default = preprocess(x, PreprocessSpec())
        assert default.dtype == np.float32
        assert default.shape == (3, 640)
        z = preprocess(x, PreprocessSpec(normalization="zscore", length=100))
        assert z.shape == (3, 100)
        np.testing.assert_allclose(
            preprocess(x[0], PreprocessSpec()), default[:1], rtol=1e-6
        )
        with pytest.raises(ValueError, match="2 frames"):
            preprocess(np.ones((2, 1)), PreprocessSpec())

    def test_asinh_compresses_bright_events(self, rng):
        x = rng.normal(0.0, 1.0, (1, 400))
        x[0, 200:210] += 1e4  # a 10,000-sigma flash
        assert preprocess(x, PreprocessSpec()).max() < 15.0


def test_features_are_finite_and_named(toy_windows):
    signals, _ = toy_windows
    features = extract_features(signals)
    assert features.shape == (len(signals), len(FEATURE_NAMES))
    assert np.all(np.isfinite(features))


@pytest.mark.parametrize("kind", ["logistic", "gbm"])
def test_baselines_learn_toy_classes(kind, toy_windows):
    signals, labels = toy_windows
    model = FeatureClassifier(kind).fit(signals, labels)
    probs = model.predict_proba(signals)
    assert probs.shape == (len(labels), len(EVENT_CLASSES))
    assert np.mean(probs.argmax(axis=1) == labels) > 0.9


def test_unknown_baseline():
    with pytest.raises(ValueError, match="unknown baseline"):
        FeatureClassifier("svm")


class TestModels:
    @pytest.mark.parametrize("name", ["cnn", "tcn"])
    @pytest.mark.parametrize("length", [100, 333, 640])
    def test_any_length(self, name, length):
        model = build_model(name, 5).eval()
        assert model(torch.zeros(2, 1, length)).shape == (2, 5)

    def test_tcn_features_are_causal(self):
        torch.manual_seed(0)
        model = TCN(num_classes=5).eval()
        x = torch.randn(1, 1, 200)
        y = x.clone()
        y[..., 150:] += 5.0
        with torch.no_grad():
            fx, fy = model.features(x), model.features(y)
        torch.testing.assert_close(fx[..., :150], fy[..., :150])
        assert not torch.allclose(fx[..., 150:], fy[..., 150:])

    def test_invalid_configuration(self):
        with pytest.raises(ValueError, match="same length"):
            CNN1D(5, channels=(8,), kernel_sizes=(3, 3))
        with pytest.raises(ValueError, match="unknown model"):
            build_model("transformer", 5)


def test_training_is_deterministic_and_learns(toy_windows):
    signals, labels = toy_windows
    config = TrainConfig(
        model="cnn", epochs=8, batch_size=32, learning_rate=5e-3, seed=7
    )
    device = torch.device("cpu")
    a = train_model(config, signals, labels, signals, labels, 5, device)
    b = train_model(config, signals, labels, signals, labels, 5, device)
    for (ka, va), (kb, vb) in zip(
        a.model.state_dict().items(), b.model.state_dict().items(), strict=True
    ):
        assert ka == kb
        torch.testing.assert_close(va, vb)
    assert max(a.history["val_accuracy"]) > 0.9
    assert a.history["val_loss"][a.best_epoch] == min(a.history["val_loss"])


def test_select_device():
    assert select_device("cpu").type == "cpu"
    assert select_device().type in {"cpu", "cuda", "mps"}


class TestCalibration:
    def test_temperature_recovers_known_scale(self, rng):
        true_logits = rng.normal(0.0, 2.0, (4_000, 5))
        labels = np.array([rng.choice(5, p=p) for p in probabilities(true_logits)])
        overconfident = 3.0 * true_logits
        t = fit_temperature(overconfident, labels)
        assert t == pytest.approx(3.0, rel=0.1)
        assert nll(overconfident, labels, t) < nll(overconfident, labels)

    def test_scores(self):
        logits = np.array([[10.0, 0.0], [0.1, 0.0]])
        assert max_softmax(logits)[0] > max_softmax(logits)[1]
        assert energy_score(logits)[0] > energy_score(logits)[1]


class TestConformal:
    @pytest.mark.parametrize("method", ["lac", "aps"])
    def test_coverage_on_exchangeable_data(self, rng, method):
        logits = rng.normal(0.0, 1.5, (6_000, 5))
        probs = probabilities(logits)
        labels = np.array([rng.choice(5, p=p) for p in probs])
        q = calibrate(probs[:3_000], labels[:3_000], alpha=0.1, method=method)
        sets = prediction_sets(probs[3_000:], q, method)
        assert coverage(sets, labels[3_000:]) == pytest.approx(0.9, abs=0.02)
        assert sets.any(axis=1).all()

    def test_lac_sets_are_smaller_than_aps_for_confident_models(self, rng):
        logits = rng.normal(0.0, 6.0, (6_000, 5))  # confident, often wrong
        probs = probabilities(logits)
        labels = np.where(
            rng.random(6_000) < 0.8, probs.argmax(axis=1), rng.integers(0, 5, 6_000)
        )
        cal, test = slice(0, 3_000), slice(3_000, None)
        sizes = {
            m: prediction_sets(
                probs[test], calibrate(probs[cal], labels[cal], 0.1, m), m
            )
            .sum(axis=1)
            .mean()
            for m in ("lac", "aps")
        }
        assert sizes["lac"] < sizes["aps"]

    def test_scores_and_edge_cases(self):
        probs = np.array([[0.6, 0.3, 0.1]])
        np.testing.assert_allclose(conformal_scores(probs, np.array([1]), "aps"), [0.9])
        np.testing.assert_allclose(conformal_scores(probs, np.array([1]), "lac"), [0.7])
        assert calibrate(probs, np.array([0]), alpha=0.1) == float("inf")
        np.testing.assert_array_equal(
            prediction_sets(probs, 0.0, "aps"), [[True, False, False]]
        )
        np.testing.assert_array_equal(
            prediction_sets(probs, 0.95, "aps"), [[True, True, False]]
        )
        np.testing.assert_array_equal(
            prediction_sets(probs, 0.75, "lac"), [[True, True, False]]
        )
        with pytest.raises(ValueError, match="alpha"):
            calibrate(probs, np.array([0]), alpha=1.0)
        with pytest.raises(ValueError, match="unknown conformal"):
            conformal_scores(probs, np.array([0]), "raps")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="unknown conformal"):
            prediction_sets(probs, 0.5, "raps")  # type: ignore[arg-type]


def test_artifact_round_trip_and_inference(tmp_path, toy_windows):
    signals, labels = toy_windows
    config = TrainConfig(model="tcn", epochs=2, batch_size=64, seed=0)
    result = train_model(
        config, signals, labels, signals, labels, 5, torch.device("cpu")
    )
    artifact = ModelArtifact(
        name="toy",
        model="tcn",
        preprocess=config.preprocess,
        temperature=1.5,
        conformal=ConformalSpec(alpha=0.1, threshold=0.95),
    )
    save_artifact(tmp_path / "a", artifact, result.model)
    loaded, model = load_artifact(tmp_path / "a")
    assert loaded == artifact
    for a, b in zip(
        model.state_dict().values(), result.model.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(a, b.cpu())

    classifier = EventClassifier.load(tmp_path / "a")
    out = classifier.predict(signals[:7])
    assert len(out.labels) == 7
    assert set(out.labels) <= set(EVENT_CLASSES)
    np.testing.assert_allclose(out.probabilities.sum(axis=1), 1.0, atol=1e-9)
    assert out.prediction_sets.any(axis=1).all()
    assert out.energy.shape == (7,)

    unbounded = EventClassifier(artifact.model_copy(update={"conformal": None}), model)
    assert unbounded.predict(signals[:2]).prediction_sets.all()


def test_artifact_rejects_unknown_classes(tmp_path):
    model = build_model("cnn", 2)
    save_artifact(
        tmp_path, ModelArtifact(name="x", model="cnn", classes=("cat", "dog")), model
    )
    with pytest.raises(ValueError, match="taxonomy"):
        load_artifact(tmp_path)
