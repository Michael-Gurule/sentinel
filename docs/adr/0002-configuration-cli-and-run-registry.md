# ADR 0002: Typed configuration, a Typer CLI, and a JSON run registry

- **Status:** accepted (Phase 6)
- **Code:** `sentinel.pipeline` (config, stages, runner), `sentinel.cli`, `sentinel.runs`, `sentinel.core.logs`

## Context

By Phase 5 the system's behavior depended on about 25 numbers, each set by an
experiment: the CFAR threshold (E1), the classifier's onset range (E6),
M-of-N confirmation and stereo ghost handling (E5), the IMM models (E5), the
RF fit test and late-fix policy (E7), and the network's systematic-error
levels (E4). Several problems followed:

- **Hard-coded values.** These numbers were keyword defaults spread across
  modules and an experiment harness.
- **Untraceable demo.** The demo pipeline (`phase3_pipeline.py`, named after
  the build timeline) ran with whatever those defaults happened to be, and
  nothing recorded which values produced a given result.
- **Duplicated fusion rules.** The experiments re-implemented fusion rules
  that the library lacked: the bias floor, fix acceptance, and late-fix
  extrapolation.

## Decision

1. **One typed configuration.** `PipelineConfig` is a frozen pydantic model.
   Extra fields are forbidden, so a typo fails at load time. Each section
   (detection, classification, rf, opir, tracking) documents where its
   defaults come from. A YAML file (`configs/pipeline/default.yaml`) fully
   describes a run, and `PipelineConfig.sha256()` identifies it.
2. **Stages behind protocols.** The runner depends on `Detector`,
   `Classifier`, `RFGeolocator`, and `Tracker` protocols. Defaults are built
   from the configuration, and any stage can be injected. Tests use doubles
   (a fixed-onset detector, a uniform classifier). ONNX Runtime plugs in as
   a second `Classifier` without touching the runner.
3. **Fusion rules live in the library.** Fix acceptance (`accept_fix`),
   late-fix extrapolation (`extrapolate`), and the bias floor
   (`GeolocationResult.systematic_covariance` → `Track.reported`) are library
   functions. The pipeline and the experiments call the same code.
4. **A Typer CLI** (`sentinel simulate | run | export-onnx | data | runs`) is
   installed as a console script and also runs as `python -m sentinel`. It
   is the only place that configures logging.
5. **Structured logging on `logging`.** Library code emits named events with
   fields (`log_event(logger, level, "rf_fix_rejected", chi2=..., dof=...)`).
   The CLI renders them as `key=value` lines or, with `--log-json`, as one
   JSON object per line.
6. **A JSON file run registry.** Every `sentinel run` and every experiment
   run from the command line writes `runs/<id>/run.json`. The record holds:
   - the configuration and its hash, and the seed;
   - the git commit, with a dirty flag;
   - package versions and the command line;
   - the timing, the status (or the error), the metrics, and the artifacts
     the run produced.

## Alternatives considered

- **MLflow (file backend).** It offers a UI, metric history, and artifact
  storage. For a single-author research repository it adds a large
  dependency tree and an opaque store under `mlruns/`, and most of its
  features would go unused: there are no long training curves or model
  stages to manage. A JSON record is diffable, greppable, and readable in
  review. If the project grows into multi-user experimentation, the record
  maps one-to-one onto MLflow params, metrics, and artifacts, so migration
  is mechanical.
- **Hydra for configuration.** Config composition and sweeps are useful,
  but Hydra brings its own CLI and working-directory conventions. The
  experiments already define their sweeps in code, where the questions they
  answer are stated.
- **argparse instead of Typer.** It has no extra dependency, but typed
  commands and generated help take noticeably more code. Typer 0.27 needs
  only `rich` and `shellingham`.
- **structlog.** It is nicer to use, but it is another dependency for what
  `log_event` and two formatters already provide.

## Consequences

- **Reproducibility.** A result is reproducible from its record: re-run the
  stored configuration and seed at the stored commit. A dirty-tree flag
  warns when that is not exact.
- **Visible defaults.** Changing a default is a reviewed change to
  `PipelineConfig`. The default YAML is tested against it.
- **Retired demo.** `phase3_pipeline.py` and its dictionary-based frame API
  are gone. Callers use `SentinelPipeline.process_frame(SensorFrame(...))`,
  which returns a typed `FrameResult`.
- **Untracked runs.** `runs/` is git-ignored. Results meant to be kept
  belong in `reports/`, which the experiments write and the registry
  references.
