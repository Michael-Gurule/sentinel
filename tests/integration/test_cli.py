"""The command-line interface, end to end."""

import json
from pathlib import Path

from typer.testing import CliRunner

from sentinel import __version__
from sentinel.cli import app

ROOT = Path(__file__).resolve().parents[2]
SCENARIO = str(ROOT / "configs/scenario/multi_int.yaml")
runner = CliRunner()


def test_version_and_help():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output
    assert "simulate" in runner.invoke(app, ["--help"]).output


def test_simulate_writes_a_summary(tmp_path):
    out = tmp_path / "summary.json"
    result = runner.invoke(
        app, ["simulate", SCENARIO, "--seed", "1", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(out.read_text())
    assert summary["opir_sensors"] == 2
    assert set(summary["events"]) == {"launch-1", "aircraft-1", "aircraft-2", "fire-1"}


def test_run_records_a_reproducible_run(tmp_path):
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text("classification: {artifact: null}\n")
    args = [
        "run",
        SCENARIO,
        "--pipeline",
        str(pipeline),
        "--runs-dir",
        str(tmp_path / "runs"),
    ]
    result = runner.invoke(app, ["--log-json", *args])
    assert result.exit_code == 0, result.output
    assert "GOSPA" in result.output
    listing = runner.invoke(app, ["runs", "list", "--runs-dir", str(tmp_path / "runs")])
    assert "pipeline" in listing.output
    assert "completed" in listing.output
    run_id = result.output.strip().split()[-1]
    shown = json.loads(
        runner.invoke(
            app, ["runs", "show", run_id, "--runs-dir", str(tmp_path / "runs")]
        ).output
    )
    assert shown["metrics"]["gospa_mean_m"] > 0
    assert shown["artifacts"] == ["tracks.json"]
    tracks = json.loads((tmp_path / "runs" / run_id / "tracks.json").read_text())
    assert tracks
    unregistered = runner.invoke(app, [*args, "--no-register"])
    assert unregistered.exit_code == 0
    assert "recorded run" not in unregistered.output


def test_runs_show_unknown_and_empty_registry(tmp_path):
    result = runner.invoke(app, ["runs", "show", "nope", "--runs-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert (
        "no runs"
        in runner.invoke(app, ["runs", "list", "--runs-dir", str(tmp_path)]).output
    )


def test_data_build_and_verify(tmp_path):
    config = str(ROOT / "configs/dataset/opir_v2.yaml")
    common = ["--config", config, "--scale", "0.002"]
    out = tmp_path / "data"
    built = runner.invoke(app, ["data", "build", *common, "--out", str(out)])
    assert built.exit_code == 0, built.output
    manifest = out / "manifest.json"

    def verify() -> int:
        args = ["data", "verify", *common, "--out", str(tmp_path / "again")]
        return runner.invoke(app, [*args, "--manifest", str(manifest)]).exit_code

    assert verify() == 0
    tampered = json.loads(manifest.read_text())
    tampered["splits"]["train"]["content_sha256"] = "0" * 64
    manifest.write_text(json.dumps(tampered))
    assert verify() == 1
