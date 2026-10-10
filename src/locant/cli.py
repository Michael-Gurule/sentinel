"""Command-line interface: ``locant --help`` (or ``python -m locant``).

locant simulate configs/scenario/multi_int.yaml --seed 0
locant run configs/scenario/multi_int.yaml --pipeline configs/pipeline/default.yaml
locant data build --config configs/dataset/opir_v2.yaml --out data/opir_v2
locant runs list
"""

import json
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer

from locant import __version__
from locant.core.logs import configure_logging

app = typer.Typer(
    help="Locant: OPIR + RF detection, geolocation, tracking, and fusion.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(help="Build or verify the OPIR dataset.", no_args_is_help=True)
runs_app = typer.Typer(help="Inspect the run registry.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(runs_app, name="runs")

ScenarioPath = Annotated[
    Path, typer.Argument(exists=True, dir_okay=False, help="Scenario YAML.")
]
Seed = Annotated[int, typer.Option(help="Simulation seed (same seed, same scenario).")]
RunsDir = Annotated[Path, typer.Option(help="Run registry directory.")]


def _version(value: bool) -> None:
    if value:
        typer.echo(f"locant {__version__}")
        raise typer.Exit


@app.callback()
def main(
    log_level: Annotated[
        str, typer.Option(help="DEBUG, INFO, WARNING, ...")
    ] = "WARNING",
    log_json: Annotated[
        bool, typer.Option(help="Log one JSON object per line.")
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version, is_eager=True, help="Show version."
        ),
    ] = False,
) -> None:
    configure_logging(log_level, json_lines=log_json)


def _row(*cells: object, widths: tuple[int, ...]) -> str:
    return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths, strict=False))


@app.command()
def simulate(
    scenario: ScenarioPath,
    seed: Seed = 0,
    out: Annotated[Path | None, typer.Option(help="Write a JSON summary here.")] = None,
) -> None:
    """Simulate a scenario and summarize what each sensor observed."""
    from locant.sim import load_scenario, simulate_scenario

    result = simulate_scenario(load_scenario(scenario), seed)
    summary: dict[str, Any] = {
        "scenario": result.config.name,
        "seed": seed,
        "duration_s": result.config.duration_s,
        "opir_sensors": len(result.opir_by_sensor),
        "events": {
            event_id: [
                round(obs[event_id].peak_snr, 1) for obs in result.opir_by_sensor
            ]
            for event_id in result.opir
        },
        "rf_scans": len(result.rf_scans),
    }
    typer.echo(
        f"{summary['scenario']} (seed {seed}): {summary['duration_s']:.0f} s, "
        f"{summary['opir_sensors']} OPIR sensor(s), {summary['rf_scans']} RF scans"
    )
    widths = (14, 30)
    typer.echo(_row("event", "peak SNR per OPIR sensor", widths=widths))
    for event_id, snrs in summary["events"].items():
        typer.echo(_row(event_id, ", ".join(f"{s:.1f}" for s in snrs), widths=widths))
    if out is not None:
        out.write_text(json.dumps(summary, indent=2))
        typer.echo(f"summary written to {out}")


@app.command()
def run(
    scenario: ScenarioPath,
    pipeline: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Pipeline configuration YAML."),
    ] = Path("configs/pipeline/default.yaml"),
    seed: Seed = 0,
    runs_dir: RunsDir = Path("runs"),
    register: Annotated[
        bool, typer.Option(help="Record the run in the registry.")
    ] = True,
) -> None:
    """Run the full pipeline on a simulated scenario and score it against truth."""
    from locant.pipeline import LocantPipeline, load_pipeline_config, run_scenario
    from locant.pipeline.runner import track_to_dict
    from locant.runs import RunRegistry
    from locant.sim import load_scenario, simulate_scenario

    config = load_pipeline_config(pipeline)
    scenario_config = load_scenario(scenario)
    result = simulate_scenario(scenario_config, seed)
    processor = LocantPipeline(config)

    def execute() -> dict[str, Any]:
        outcome = run_scenario(processor, result)
        metrics = outcome.metrics()
        typer.echo(
            f"{scenario_config.name} (seed {seed}): {len(outcome.frames)} frames, "
            f"GOSPA {metrics['gospa_mean_m']:.0f} m, "
            f"false {metrics['false_tracks_per_scan']:.2f}/scan, "
            f"missed {metrics['missed_targets_per_scan']:.2f}/scan, "
            f"NEES {metrics['nees_mean']:.2f}"
        )
        widths = (6, 22, 12, 46)
        typer.echo(
            _row("track", "class", "RMS unc. m", "hits by source", widths=widths)
        )
        for track in processor.confirmed_tracks:
            posterior = track.class_posterior
            label = "unclassified"
            if posterior is not None and processor.classes is not None:
                k = int(np.argmax(posterior))
                label = f"{processor.classes[k]} ({posterior[k]:.2f})"
            typer.echo(
                _row(
                    track.id,
                    label,
                    f"{track.position_rms_uncertainty:.1f}",
                    dict(track.hits_by_source),
                    widths=widths,
                )
            )
        return metrics

    if not register:
        execute()
        return
    registry = RunRegistry(runs_dir)
    run_config = {
        "scenario": scenario_config.model_dump(mode="json"),
        "pipeline": config.model_dump(mode="json"),
    }
    with registry.start(
        scenario_config.name, run_config, seed, kind="pipeline"
    ) as active:
        metrics = execute()
        active.log_metrics(metrics)
        tracks = active.directory / "tracks.json"
        tracks.write_text(
            json.dumps(
                [
                    track_to_dict(t, processor.classes)
                    for t in processor.confirmed_tracks
                ],
                indent=2,
            )
        )
        active.add_artifact(tracks)
    typer.echo(f"recorded run {active.id}")


@app.command("export-onnx")
def export_onnx_command(
    artifact: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="Classifier artifact.")
    ] = Path("models/opir_event_classifier"),
) -> None:
    """Export a classifier artifact to ONNX (``model.onnx`` next to its weights)."""
    from locant.classification.onnx_backend import export_onnx

    typer.echo(f"wrote {export_onnx(artifact)}")


@data_app.command("build")
def data_build(
    config: Annotated[Path, typer.Option(exists=True, help="Dataset YAML config.")],
    out: Annotated[Path, typer.Option(help="Output directory.")],
    workers: Annotated[int, typer.Option(min=1)] = 1,
    scale: Annotated[float, typer.Option(help="Multiply split sizes.")] = 1.0,
) -> None:
    """Build the dataset deterministically and write its manifest."""
    from locant.data.build import main as build_main

    args = ["--config", str(config), "--out", str(out), "--workers", str(workers)]
    raise typer.Exit(build_main([*args, "--scale", str(scale)]))


@data_app.command("verify")
def data_verify(
    config: Annotated[Path, typer.Option(exists=True, help="Dataset YAML config.")],
    out: Annotated[Path, typer.Option(help="Output directory.")],
    manifest: Annotated[Path, typer.Option(exists=True, help="Reference manifest.")],
    workers: Annotated[int, typer.Option(min=1)] = 1,
    scale: Annotated[float, typer.Option(help="Multiply split sizes.")] = 1.0,
) -> None:
    """Rebuild the dataset and fail if it differs from a reference manifest."""
    from locant.data.build import main as build_main

    args = ["--config", str(config), "--out", str(out), "--workers", str(workers)]
    raise typer.Exit(
        build_main([*args, "--scale", str(scale), "--verify", str(manifest)])
    )


@runs_app.command("list")
def runs_list(runs_dir: RunsDir = Path("runs")) -> None:
    """List recorded runs, oldest first."""
    from locant.runs import RunRegistry

    records = RunRegistry(runs_dir).records()
    if not records:
        typer.echo(f"no runs in {runs_dir}")
        return
    widths = (44, 10, 10, 9, 8)
    typer.echo(_row("id", "kind", "status", "seconds", "commit", widths=widths))
    for r in records:
        commit = (r.git.get("commit") or "-")[:7] + ("*" if r.git.get("dirty") else "")
        typer.echo(
            _row(r.id, r.kind, r.status, r.duration_s or "-", commit, widths=widths)
        )


@runs_app.command("show")
def runs_show(
    run_id: Annotated[str, typer.Argument(help="Run id or unique prefix.")],
    runs_dir: RunsDir = Path("runs"),
) -> None:
    """Print a run's record."""
    from dataclasses import asdict

    from locant.runs import RunRegistry

    try:
        record = RunRegistry(runs_dir).load(run_id)
    except KeyError as exc:
        typer.echo(str(exc.args[0]), err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(asdict(record), indent=2, sort_keys=True))
