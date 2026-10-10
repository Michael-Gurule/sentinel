# Architecture

How Locant is put together: what it talks to, what runs inside it, how one
frame of sensor data flows through it, and how a result is reproduced. The
diagrams are [mermaid](https://mermaid.js.org/) and render on GitHub. The
package layering in §3 is enforced by `tests/unit/test_architecture.py`.

## 1. Context

```mermaid
flowchart LR
    analyst(["Analyst / researcher"])
    subgraph sensors["Sensors (simulated: locant.sim)"]
        opir["OPIR satellites<br/>GEO + Molniya HEO<br/>pixel windows + lines of sight"]
        rf["RF receiver network<br/>TDOA / FDOA per emitter"]
    end
    locant["<b>Locant</b><br/>detection · classification ·<br/>geolocation · tracking · fusion"]
    outputs["Track picture<br/>positions, velocities, calibrated<br/>uncertainty, class posteriors"]
    evidence["Evidence<br/>experiment reports, metrics,<br/>run records"]
    opir --> locant
    rf --> locant
    locant --> outputs
    locant --> evidence
    analyst -- "locant run / make experiments" --> locant
    outputs --> analyst
    evidence --> analyst
```

Locant consumes two very different kinds of sensor product. OPIR gives a
radiometric time series per pixel plus the pixel's line of sight. The RF
network gives time and frequency differences of arrival per emitter. Locant
turns both into one track picture with calibrated uncertainty and calibrated
class labels. All sensor data is simulated by a physics-based simulator
([ADR 0003](adr/0003-synthetic-data-strategy.md)).

## 2. Containers

```mermaid
flowchart TB
    subgraph entry["Entry points"]
        cli["locant CLI<br/>simulate · run · export-onnx · data · runs"]
        exps["experiments/ E1–E7<br/>one script per question"]
        bench["benchmarks/<br/>latency budgets"]
    end
    subgraph lib["locant library (src/locant)"]
        pipeline["pipeline<br/>PipelineConfig · stage protocols ·<br/>LocantPipeline · scenario runner"]
        algos["algorithms<br/>detection · classification ·<br/>geolocation · tracking · fusion · eval"]
        simdata["sim · data<br/>scenarios · sensor models ·<br/>deterministic dataset builder"]
        ops["core.logs · runs<br/>structured logging · run registry"]
    end
    subgraph state["Versioned state"]
        configs[("configs/<br/>scenario · dataset · pipeline YAML")]
        manifest[("data/manifests/<br/>dataset hashes")]
        model[("models/opir_event_classifier/<br/>weights + calibration")]
        reports[("reports/<br/>E1–E7 JSON + figures ·<br/>metrics.json")]
    end
    runs[("runs/<br/>run records (local)")]
    cli --> pipeline
    cli --> simdata
    exps --> pipeline
    exps --> algos
    exps --> simdata
    bench --> algos
    pipeline --> algos
    pipeline --> simdata
    cli --> ops
    exps --> ops
    configs --> cli
    configs --> exps
    simdata --> manifest
    exps --> model
    model --> pipeline
    exps --> reports
    ops --> runs
```

| Container | Responsibility | Key types |
|---|---|---|
| `locant.pipeline` | Orchestrates one frame through every stage; builds stages from configuration | `PipelineConfig`, `LocantPipeline`, `SensorFrame`, `FrameResult`, `run_scenario` |
| `locant.detection` | Window-level event detection at a calibrated false-alarm rate | `Detector` protocol, `CFARDetector`, `CUSUMDetector`, `StepGLRTDetector` |
| `locant.classification` | Calibrated classification, conformal sets, OOD scores; PyTorch and ONNX Runtime backends | `EventClassifier`, `OnnxEventClassifier`, `ModelArtifact` |
| `locant.geolocation` | TDOA/FDOA maximum likelihood, Chan-Ho, CRLB, DOP, systematic-error covariance | `solve_tdoa_fdoa`, `GeolocationResult`, `SystematicErrors` |
| `locant.tracking` | Kalman/EKF/IMM filtering, gating, assignment, track lifecycle, class posteriors, bias floor | `MultiTargetTracker`, `Track`, `Measurement` protocol |
| `locant.fusion` | OPIR line-of-sight geolocation, measurement conversion, fusion engine, T2T fusion | `FusionEngine`, `LineOfSightMeasurement`, `measurements_from_reports` |
| `locant.eval` | Detection, classification, calibration, and tracking metrics (GOSPA/OSPA) | `evaluate_tracking`, `gospa`, `expected_calibration_error` |
| `locant.sim`, `locant.data` | Scenario simulation and the deterministic, hashed dataset | `simulate_scenario`, `build_dataset` |
| `locant.runs`, `locant.core.logs` | Run registry and structured logging | `RunRegistry`, `log_event` |

## 3. Package layering

```mermaid
flowchart BT
    core["core"]
    taxonomy["taxonomy"]
    geolocation["geolocation"] --> core
    detection["detection"] --> core
    tracking["tracking"] --> core
    eval["eval"] --> core
    classification["classification"] --> detection
    classification --> taxonomy
    fusion["fusion"] --> geolocation
    fusion --> tracking
    sim["sim"] --> geolocation
    data["data"] --> sim
    data --> taxonomy
    pipeline["pipeline"] --> fusion
    pipeline --> classification
    pipeline --> eval
    pipeline --> sim
    runs["runs"]
    cli["cli"] --> pipeline
    cli --> data
    cli --> runs
```

Arrows point from a package to what it imports. Every algorithm package also
imports `core` (linear algebra, χ² statistics, errors); those edges are
omitted for legibility. The rules:

- **Algorithms never import orchestration.** `core`, `geolocation`,
  `detection`, `tracking`, `eval`, and `fusion` know nothing about the
  pipeline, the CLI, the simulator, or the experiments. They can be reused
  or tested on their own.
- **The simulator depends only on measurement types.** `sim` imports
  `geolocation` for the receiver and TDOA/FDOA measurement types it
  produces, never the solvers.
- **Experiments live outside the package.** `experiments/` imports
  `locant`; nothing in `locant` imports `experiments`.

A new edge must be added to `ALLOWED` in `tests/unit/test_architecture.py`,
which makes every new dependency a reviewed decision.

## 4. One frame through the pipeline

```mermaid
sequenceDiagram
    autonumber
    participant S as SensorFrame
    participant D as Detector (CFAR)
    participant C as Classifier (TCN)
    participant O as OPIR geolocation
    participant G as RF geolocator
    participant T as Tracker (FusionEngine)
    S->>D: pixel windows (one per satellite and pixel)
    D-->>S: score, onset per window (threshold: E1)
    S->>C: windows that fired (padded to 64 s)
    C-->>S: calibrated probabilities and conformal set, "background" dropped
    Note over C: class evidence kept only if the onset is in the<br/>trained 2–40 s range (E6)
    S->>O: detections with their lines of sight
    O-->>S: stereo positions (unambiguous pairs) + angle-only rays (E5)
    S->>G: TDOA/FDOA observations
    G-->>S: fixes with consider + systematic covariance (E4)
    Note over G: reject unless converged and χ²-consistent,<br/>extrapolate late fixes (E7)
    S->>T: all measurements, one centralized update
    T-->>S: confirmed tracks: state, reported covariance (bias floor),<br/>class posterior
```

- **One tracker update per frame.** All measurements are fused together.
  Updates are grouped by (source, dimension), so each satellite's rays and
  each kind of RF fix get their own χ² gate
  ([ADR 0001](adr/0001-fusion-architecture.md)).
- **Every threshold has a source.** Each number in the diagram is a field of
  `PipelineConfig`, and its documentation names the experiment that set it
  ([ADR 0002](adr/0002-configuration-cli-and-run-registry.md)).
- **Stages are replaceable.** Each stage is a protocol in
  `locant.pipeline.stages`. Tests replace the detector and classifier with
  doubles, and ONNX Runtime replaces PyTorch without touching the runner.

## 5. Reproducibility chain

```mermaid
flowchart LR
    dcfg[("configs/dataset/opir_v2.yaml")] --> build["locant data build<br/>SeedSequence per sample"]
    build --> dataset[("data/opir_v2<br/>+ manifest hashes")]
    dataset --> e1["E1 detection"] --> thr["calibrated threshold"]
    dataset --> e2["E2 classification"] --> e3["E3 calibration +<br/>conformal + export"]
    e3 --> model[("models/opir_event_classifier")]
    scfg[("configs/scenario")] --> e46["E4–E7<br/>(no dataset)"]
    model --> e46
    e1 & e2 & e3 & e46 --> reports[("reports/*.json + figures")]
    reports --> metrics["make metrics"] --> mjson[("reports/metrics.json")]
    mjson --> docs["README + docs tables<br/>(checked by a test)"]
```

- **The dataset is a pure function of its config and seed.** `make
  data-verify` rebuilds it and compares content hashes with the committed
  manifest.
- **Experiments own their numbers.** Each experiment writes a JSON report
  with confidence intervals. `make metrics` gathers the headline numbers
  into `reports/metrics.json` and regenerates every results table in the
  README and docs. `tests/regression/test_metrics_sync.py` fails if a table
  drifts from the reports.
- **Runs are recorded.** Every `locant run` and experiment launched from the
  command line writes `runs/<id>/run.json`: configuration and hash, seed,
  git commit (marked when the tree is dirty), package versions, timing,
  metrics, and artifacts.

## 6. Cross-cutting concerns

| Concern | Approach |
|---|---|
| **Randomness** | No global RNG: every stochastic function takes a `numpy.random.Generator`. Datasets and scenarios derive per-sample streams with `SeedSequence`. |
| **Units and frames** | SI units throughout; positions in a local ENU frame about a WGS-84 origin ([ADR 0006](adr/0006-frames-and-units.md)). |
| **Uncertainty** | Every estimate carries a covariance. Consistency is tested by NEES/NIS against χ² bounds, not assumed. |
| **Errors** | Typed exceptions (`GeometryError`, `InsufficientMeasurementsError`, `NotPositiveDefiniteError`), no silent placeholder values (v1 returned 999 or zeros). The pipeline counts and logs failures instead of raising. |
| **Configuration** | Frozen pydantic models; unknown keys fail at load. |
| **Logging** | `log_event(logger, level, "event_name", **fields)`; the CLI renders key=value lines or JSON. |
| **Performance** | Vectorized gating, stereo association, and geometry. Per-stage latency budgets run in CI (`make bench`). |
| **Quality gates** | ruff, strict mypy, 85% coverage gate on the library, Hypothesis properties, statistical tests (NEES/NIS, CRLB), characterization tests for every v1 defect ([audit](audit_v1.md)), metrics drift test, layering test. |
