import json

import pytest

from locant.pipeline import PipelineConfig
from locant.runs import RunRegistry, config_hash


def test_completed_run_record(tmp_path):
    registry = RunRegistry(tmp_path)
    config = PipelineConfig()
    with registry.start("demo", config, seed=3, kind="pipeline") as run:
        out = run.directory / "tracks.json"
        out.write_text("[]")
        run.add_artifact(out)
        run.log_metrics({"gospa": 1.5}, nan=float("nan"))
    record = registry.load(run.id)
    assert record.status == "completed"
    assert record.config_sha256 == config_hash(config) == config.sha256()
    assert record.seed == 3
    assert record.metrics == {"gospa": 1.5, "nan": None}
    assert record.artifacts == ["tracks.json"]
    assert record.duration_s is not None
    assert {"python", "numpy", "locant"} <= record.environment.keys()
    assert json.loads((run.directory / "run.json").read_text())["kind"] == "pipeline"


def test_failed_run_is_recorded_and_reraises(tmp_path):
    registry = RunRegistry(tmp_path)
    with pytest.raises(RuntimeError), registry.start("broken", {"a": 1}) as run:
        raise RuntimeError("boom")
    record = registry.load(run.id)
    assert record.status == "failed"
    assert record.error == "RuntimeError: boom"


def test_listing_and_prefix_lookup(tmp_path):
    registry = RunRegistry(tmp_path)
    ids = []
    for name in ("alpha", "beta"):
        with registry.start(name, {"name": name}) as run:
            ids.append(run.id)
    with registry.start("alpha", {"name": "alpha"}) as again:
        pass
    assert again.id != ids[0]  # same second, same config: suffixed, not overwritten
    assert [r.id for r in registry.records()][:2] == ids
    assert registry.load(ids[1][:-2]).name == "beta"
    with pytest.raises(KeyError, match="no run"):
        registry.load("missing")
    with pytest.raises(KeyError, match="ambiguous"):
        registry.load(ids[0][:8])
    assert RunRegistry(tmp_path / "empty").records() == []
