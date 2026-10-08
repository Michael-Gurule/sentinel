import json

import numpy as np
import pytest
from pydantic import ValidationError

from sentinel.data import (
    DatasetConfig,
    build_dataset,
    content_hash,
    load_dataset_config,
    load_manifest,
    load_split,
)
from sentinel.data.build import compare_manifests, main, sample_rng
from sentinel.data.config import Priors, Range, SplitConfig, deep_merge
from sentinel.data.generate import generate_sample, params_to_json
from sentinel.models.taxonomy import EVENT_CLASSES


def tiny_config(**changes) -> DatasetConfig:
    base = {
        "name": "tiny",
        "version": "0",
        "root_seed": 11,
        "window_s": 48.0,
        "splits": [
            {"name": "train", "samples_per_class": 3},
            {
                "name": "noisy",
                "samples_per_class": 2,
                "overrides": {"sensor": {"nei": {"low": 5.0, "high": 6.0}}},
            },
        ],
    }
    return DatasetConfig.model_validate(base | changes)


class TestConfig:
    def test_range_sampling(self, rng):
        uniform = Range(low=2.0, high=3.0)
        log = Range(low=1e-3, high=1e3, log=True)
        u = [uniform.sample(rng) for _ in range(2_000)]
        g = [log.sample(rng) for _ in range(2_000)]
        assert 2.0 <= min(u) <= max(u) <= 3.0
        assert np.median(np.log10(g)) == pytest.approx(0.0, abs=0.15)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"low": 2, "high": 1}, "high"),
            ({"low": 0, "high": 1, "log": True}, "positive"),
        ],
    )
    def test_range_validation(self, kwargs, message):
        with pytest.raises(ValidationError, match=message):
            Range(**kwargs)

    def test_overrides_merge_and_validate(self):
        config = tiny_config()
        noisy = config.priors_for(config.splits[1])
        assert noisy.sensor.nei == Range(low=5.0, high=6.0, log=True)
        assert noisy.launch == Priors().launch
        with pytest.raises(KeyError, match="unknown override"):
            deep_merge({"a": 1}, {"b": 2})
        with pytest.raises(ValidationError):
            tiny_config(
                splits=[
                    {
                        "name": "x",
                        "samples_per_class": 1,
                        "overrides": {"scene": {"background": 5}},
                    }
                ]
            )

    def test_onsets_must_fall_inside_the_window(self):
        with pytest.raises(ValidationError, match=r"inside the 30\.0 s window"):
            tiny_config(window_s=30.0)

    def test_split_names_unique_and_scaling(self):
        with pytest.raises(ValidationError, match="unique"):
            tiny_config(splits=[{"name": "a", "samples_per_class": 1}] * 2)
        scaled = tiny_config().scaled(0.01)
        assert [s.samples_per_class for s in scaled.splits] == [1, 1]

    def test_versioned_config_loads(self):
        config = load_dataset_config("configs/dataset/opir_v2.yaml")
        assert {s.name for s in config.splits} >= {"train", "validation", "test"}
        assert isinstance(config.splits[0], SplitConfig)


class TestGenerate:
    @pytest.mark.parametrize("label", EVENT_CLASSES)
    def test_every_class(self, label, rng):
        sample = generate_sample(label, Priors(), 48.0, rng)
        assert sample.signal.shape == (480,)
        assert sample.signal.dtype == np.float32
        assert sample.label == EVENT_CLASSES.index(label)
        json.loads(params_to_json(sample.params))
        if label == "background":
            assert sample.metadata["peak_snr"] == 0.0
            assert np.isnan(sample.metadata["onset_s"])
        else:
            assert sample.metadata["peak_snr"] > 0.0

    def test_unknown_label_and_json_specials(self, rng):
        with pytest.raises(ValueError, match="unknown label"):
            generate_sample("volcano", Priors(), 48.0, rng)
        text = params_to_json({"a": float("inf"), "b": [float("nan"), 1.0]})
        assert json.loads(text) == {"a": "inf", "b": ["nan", 1.0]}

    def test_sample_seeds_depend_only_on_coordinates(self):
        a = sample_rng(1, "train", 2, 5).random(3)
        b = sample_rng(1, "train", 2, 5).random(3)
        c = sample_rng(1, "test", 2, 5).random(3)
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)


class TestBuild:
    def test_build_is_deterministic_across_workers_and_split_order(self, tmp_path):
        config = tiny_config()
        one = build_dataset(config, tmp_path / "one", workers=1)
        two = build_dataset(config, tmp_path / "two", workers=2)
        reordered = tiny_config(
            splits=[s.model_dump() for s in reversed(config.splits)]
        )
        three = build_dataset(reordered, tmp_path / "three", workers=1)
        for name in ("train", "noisy"):
            assert (
                one["splits"][name]["content_sha256"]
                == two["splits"][name]["content_sha256"]
            )
            assert (
                one["splits"][name]["content_sha256"]
                == three["splits"][name]["content_sha256"]
            )
        assert compare_manifests(two, one) == []

    def test_load_split_verifies_contents(self, tmp_path):
        build_dataset(tiny_config(), tmp_path)
        split = load_split(tmp_path, "train")
        assert len(split) == 3 * len(EVENT_CLASSES)
        assert split.signals.shape == (15, 480)
        np.testing.assert_array_equal(np.bincount(split.labels), [3] * 5)
        assert "event" in split.sample_params(0)
        assert load_manifest(tmp_path)["splits"]["train"]["num_samples"] == 15

        with np.load(tmp_path / "train.npz") as data:
            arrays = {k: data[k] for k in data.files}
        arrays["signals"][0, 0] += 1.0
        np.savez_compressed(tmp_path / "train.npz", **arrays)
        with pytest.raises(ValueError, match="does not match"):
            load_split(tmp_path, "train")
        load_split(tmp_path, "train", verify=False)

    def test_content_hash_is_sensitive_and_order_free(self):
        a = {"x": np.arange(3), "y": np.ones(2)}
        assert content_hash(a) == content_hash({"y": np.ones(2), "x": np.arange(3)})
        assert content_hash(a) != content_hash({"x": np.arange(3), "y": np.zeros(2)})
        assert content_hash(a) != content_hash(
            {"x": np.arange(3, dtype=float), "y": np.ones(2)}
        )

    def test_splits_are_independent(self, tmp_path):
        build_dataset(tiny_config(), tmp_path)
        train, noisy = load_split(tmp_path, "train"), load_split(tmp_path, "noisy")
        assert not np.array_equal(train.signals[:2], noisy.signals[:2])
        assert noisy.metadata["nei"].min() >= 5.0

    def test_cli_build_and_verify(self, tmp_path, capsys):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(json.dumps(tiny_config().model_dump(mode="json")))
        assert main(["--config", str(config_path), "--out", str(tmp_path / "a")]) == 0
        reference = tmp_path / "a" / "manifest.json"
        args = [
            "--config",
            str(config_path),
            "--out",
            str(tmp_path / "b"),
            "--verify",
            str(reference),
        ]
        assert main(args) == 0
        assert "matches" in capsys.readouterr().out

        tampered = json.loads(reference.read_text())
        tampered["splits"]["train"]["content_sha256"] = "0" * 64
        reference.write_text(json.dumps(tampered))
        assert main(args) == 1
        assert "content hash differs" in capsys.readouterr().err
        assert main([*args[:4], "--scale", "0.5", "--verify", str(reference)]) == 1
