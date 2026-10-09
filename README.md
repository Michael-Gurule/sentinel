
<p align="center">
  <img width="500" alt="SENTINEL" src="https://github.com/user-attachments/assets/84043002-a02f-4837-9d49-f3390b61176a" />
<p align="center">
  <strong>Multi-Sensor Fusion for Defense Applications</strong><br>

<p align="center">  
Advanced multi-intelligence fusion system combining Overhead Persistent Infrared (OPIR) thermal detection with Radio Frequency (RF) geolocation for real-time threat detection and tracking
</p>  
<br>

## Project Overview

SENTINEL-XF is a production-grade Machine Learning platform designed for Defense applications, demonstrating expertise in Sensor Fusion, Geolocation Algorithms, and Multi-Sensor Tracking. The system integrates thermal event detection with RF signal processing to provide comprehensive situational awareness.

**Key Capabilities:**

- Real-time OPIR thermal event detection and classification
- RF emitter geolocation using TDOA/FDOA algorithms
- Multi-sensor data fusion with Kalman filtering
- Track quality assessment and uncertainty quantification
- Scalable architecture supporting multiple sensor modalities

## Why Sensor Fusion Matters: The Multiplicative Effect

OPIR satellites see every hot event (launches, fires, aircraft) but measure
only a direction; two satellites triangulate to a few hundred meters. RF
TDOA/FDOA geolocation measures position and velocity to tens of meters, but
only for targets that emit. Neither sensor alone gives the full picture.

SENTINEL fuses them at the measurement level: one tracker takes OPIR stereo
positions, single-satellite lines of sight, and RF fixes, each with its own
covariance. On the multi-target benchmark (E6, 10 seeds), fusion cuts the
GOSPA tracking error from 877 m (OPIR only) and 1961 m (RF only) to 693 m.
It tracks launches and fires that RF cannot see, improves aircraft accuracy
over RF alone (63 m vs 74 m), and puts a calibrated class label on each track.
When the RF network goes down, OPIR lines of sight keep every aircraft track
alive (E7). The [fusion write-up](docs/fusion.md) has the methods and
results, and [ADR 0001](docs/adr/0001-fusion-architecture.md) records why
centralized fusion was chosen over track-to-track fusion.

---

## System Architecture

### Signal Generation & Data Pipeline

- **OPIR Signal Generator**: Physics-based thermal signature modeling
  - 5 event types: missile launches, explosions, wildfires, aircraft, background
  - Realistic temporal dynamics and noise characteristics
- **RF Signal Generator**: Communications and radar signal simulation
- **Training Dataset**: 10,000+ labeled samples organized for PyTorch training

### Detection, Classification & Tracking

- **Detection Algorithms**: 4 complementary methods
  - Temporal Difference Detection
  - Anomaly Detection (MAD & Z-Score)
  - Rise Time Analysis
  - Multi-Method Ensemble
- **CNN Classifier**: 1D Convolutional Neural Network
  - 5-class event classification
  - 256-sample input with batch normalization
  - Dropout regularization for generalization
- **Kalman Filter Tracking**: Multi-target tracking with coasting and pruning

### RF Geolocation & Sensor Fusion

- **TDOA Geolocation**: Time Difference of Arrival positioning
  - Least-squares optimization
  - GDOP computation for quality assessment
- **FDOA Geolocation**: Frequency Difference of Arrival for moving emitters
  - Doppler-based velocity estimation
  - Sensor motion compensation
- **Hybrid TDOA/FDOA**: Combined time and frequency measurements
  - Improved accuracy through complementary data
- **OPIR geolocation**: stereo triangulation (GEO + HEO), line-of-sight
  (angle-only) EKF updates, and altitude intersection, all with covariance
- **Sensor Fusion Engine**: centralized measurement-level fusion
  - IMM tracking (quiet + maneuvering constant-velocity models), χ² gating, GNN assignment
  - M-of-N track confirmation and duplicate-track merging
  - Track class posteriors pooled from calibrated classifier outputs
  - Track-to-track fusion (naive and covariance intersection) for comparison

---

## Project Structure

```
sentinel/
├── src/sentinel/
│   ├── core/            # linear algebra, χ² statistics, errors, structured logging
│   ├── sim/             # WGS-84 geometry, trajectories, OPIR sensor model, RF network, scenarios
│   ├── data/            # deterministic dataset builder + hashed manifests
│   ├── detection/       # CFAR, CUSUM, step GLRT with calibrated thresholds
│   ├── classification/  # features/baselines, CNN/TCN, calibration, conformal, ONNX backend
│   ├── geolocation/     # TDOA/FDOA ML, Chan-Ho, CRLB, DOP, systematic-error covariance
│   ├── tracking/        # EKF, IMM, GNN, M-of-N lifecycle, class posteriors, bias floor
│   ├── fusion/          # OPIR line-of-sight geolocation, fusion engine, T2T fusion
│   ├── eval/            # metrics, GOSPA/OSPA tracking evaluation, reports
│   ├── pipeline/        # PipelineConfig, stage protocols, SentinelPipeline, scenario runner
│   ├── runs.py          # JSON run registry
│   └── cli.py           # `sentinel` command line
├── configs/             # scenario, dataset, and pipeline YAML
├── experiments/         # E1–E7 (one script per question) + shared harness
├── benchmarks/          # per-stage latency budgets (`make bench`)
├── tests/               # unit, property, integration, characterization (v1 audit)
├── reports/             # experiment JSON + figures (versioned)
├── models/              # the shipped, calibrated classifier artifact
└── docs/                # data/model cards, methods write-ups, ADRs
```

## Installation

### Setup

Requires [conda](https://docs.conda.io/) (Miniconda or Anaconda).

```bash
# Clone repository
git clone https://github.com/Michael-Gurule/sentinel.git
cd sentinel

# Create and activate the environment (Python 3.12, pinned dependencies,
# and the `sentinel` package installed in editable mode)
conda env create -f environment.yml
conda activate sentinel

# Verify installation
python -c "import sentinel; print(f'SENTINEL {sentinel.__version__} installed')"
pytest
```

## Usage

### Quick Start: Full System Demo

```bash
sentinel run configs/scenario/multi_int.yaml   # or: make demo
```

Simulates a launch cued by a fire-control radar, two aircraft with datalinks,
and a wildfire, observed in stereo by a GEO and a Molniya OPIR satellite and
by a five-receiver RF network. It then runs every frame through CFAR
detection, calibrated classification, OPIR line-of-sight geolocation, RF
TDOA/FDOA fixes, and centralized fusion. The output is the GOSPA error against
truth and each confirmed track's class and sources. The run is recorded in
`runs/` with its configuration, seed, and git commit.

### Command Line

| Command | Purpose |
|---|---|
| `sentinel simulate SCENARIO [--seed N]` | Simulate a scenario; summarize what each sensor saw |
| `sentinel run SCENARIO [--pipeline YAML] [--seed N]` | Full pipeline, scored against truth, recorded as a run |
| `sentinel export-onnx [ARTIFACT]` | Export the classifier for ONNX Runtime |
| `sentinel data build / verify` | Build the dataset, or check it against its manifest |
| `sentinel runs list / show ID` | Inspect the run registry |

Global options: `--log-level`, and `--log-json` for one JSON event per line.
`python -m sentinel` is equivalent.

### Configuration and Runs

Every tunable number of the processing chain is a field of `PipelineConfig`,
a validated pydantic model. Each section names the experiment that set its
default. `configs/pipeline/default.yaml` is the deployed configuration;
unknown or out-of-range values fail at load time. Each `sentinel run` and each
experiment launched from the command line writes `runs/<id>/run.json`. The
record holds the configuration and its hash, the seed, the git commit (marked
when the tree is dirty), package versions, timing, metrics, and artifacts. See
[ADR 0002](docs/adr/0002-configuration-cli-and-run-registry.md).

### RF Geolocation (TDOA)

```python
import numpy as np

from sentinel.geolocation import Receiver, simulate_tdoa, solve_tdoa, tdoa_dop

receivers = [
    Receiver(0, np.array([0.0, 0.0, 500.0])),
    Receiver(1, np.array([10_000.0, 0.0, 1_500.0])),
    Receiver(2, np.array([10_000.0, 10_000.0, 1_000.0])),
    Receiver(3, np.array([0.0, 10_000.0, 2_000.0])),
    Receiver(4, np.array([5_000.0, -4_000.0, 6_000.0])),
]
emitter = np.array([5_000.0, 5_000.0, 500.0])
rng = np.random.default_rng(0)

# Per-receiver timing error of 10 ns; TDOAs are correlated through the
# reference receiver and carry their full covariance.
measurement = simulate_tdoa(emitter, receivers, toa_std=10e-9, rng=rng)
result = solve_tdoa(receivers, measurement)  # Chan-Ho init + ML refinement

print(f"Position error: {np.linalg.norm(result.position - emitter):.1f} m")
print(f"1-sigma RMS:    {np.sqrt(np.trace(result.position_covariance)):.1f} m")
print(f"GDOP:           {tdoa_dop(emitter, [r.position for r in receivers]).gdop:.2f}")
```

### Joint TDOA/FDOA (Position and Velocity)

```python
import numpy as np

from sentinel.geolocation import Receiver, simulate_fdoa, simulate_tdoa, solve_tdoa_fdoa

positions = [
    [0, 0, 500],
    [10e3, 0, 1500],
    [10e3, 10e3, 1000],
    [0, 10e3, 2000],
    [5e3, -4e3, 6e3],
]
velocities = [[0, 0, 0], [30, 0, 0], [0, -40, 0], [20, 20, 0], [-30, 10, 0]]
receivers = [
    Receiver(i, np.array(p, float), np.array(v, float))
    for i, (p, v) in enumerate(zip(positions, velocities, strict=True))
]
emitter, velocity = np.array([5e3, 5e3, 500.0]), np.array([100.0, 50.0, 0.0])
rng = np.random.default_rng(0)

tdoa = simulate_tdoa(emitter, receivers, toa_std=10e-9, rng=rng)
fdoa = simulate_fdoa(emitter, velocity, receivers, 1e9, frequency_std=1.0, rng=rng)
result = solve_tdoa_fdoa(receivers, tdoa, fdoa)

print(f"Velocity estimate: {result.velocity.round(1)} m/s")
```

Without FDOA the velocity is not observable, and `solve_tdoa_fdoa` returns
`velocity=None` instead of a meaningless estimate.

### Multi-Sensor Tracking

```python
import numpy as np

from sentinel.geolocation import simulate_tdoa
from sentinel.pipeline import RFObservation, SensorFrame, SentinelPipeline

pipeline = SentinelPipeline()  # defaults from PipelineConfig
rng = np.random.default_rng(0)
start, velocity = np.array([5e3, 5e3, 500.0]), np.array([100.0, 50.0, 0.0])

for t in range(10):
    truth = start + velocity * t
    tdoa = simulate_tdoa(truth, pipeline.receivers, 10e-9, rng)
    result = pipeline.process_frame(SensorFrame(float(t), rf=[RFObservation(tdoa)]))

print(pipeline.summary())
```

Tracks are maintained by an IMM filter (quiet and maneuvering
constant-velocity models), with χ² gating on the innovation covariance,
global-nearest-neighbor assignment, and M-of-N confirmation. OPIR windows
passed as `OPIRObservation` with their line of sight are geolocated (stereo
or angle-only) and fused in the same update. Any stage can be replaced by
passing a component that satisfies its protocol in `sentinel.pipeline.stages`,
such as `SentinelPipeline(config, classifier=...)`.

### OPIR Detection & Classification

```python
import numpy as np

from sentinel.classification import EventClassifier
from sentinel.data.config import Priors
from sentinel.data.generate import generate_sample
from sentinel.detection import CFAR_THRESHOLD_PFA_1E2, CFARDetector

# One simulated 64 s pixel window (10 Hz) containing a launch.
sample = generate_sample("launch", Priors(), 64.0, np.random.default_rng(3))

# Stage 1: CFAR detection at a threshold calibrated for 1% false alarms
# per window on glint-free background.
score = CFARDetector().score(sample.signal, 10.0).score[0]
print(f"CFAR score {score:.1f}, alarm: {score > CFAR_THRESHOLD_PFA_1E2}")

# Stage 2: calibrated classification with a 90% conformal prediction set;
# a "background" label rejects the alarm (e.g. a sun glint).
classifier = EventClassifier.load("models/opir_event_classifier")
prediction = classifier.predict(sample.signal)
members = prediction.prediction_sets[0]
print("label:", prediction.labels[0])
print(
    "prediction set:",
    [c for c, m in zip(classifier.classes, members, strict=True) if m],
)
```

### Simulating a Scenario

```python
from sentinel.sim import load_scenario, simulate_scenario

config = load_scenario("configs/scenario/launch_with_radar.yaml")
result = simulate_scenario(config, seed=7)  # same seed, same scenario

for event_id, pixel in result.opir.items():
    print(f"{event_id:12s} peak SNR {pixel.peak_snr:6.1f}")
print(f"{len(result.rf_scans)} RF scans (TDOA, plus FDOA where configured)")
```

A scenario places launches, explosions, fires, and aircraft around a geodetic
origin. A GEO staring sensor observes them (range, atmosphere, clouds, PSF,
clutter, glint, noise), and an RF receiver network with clock biases and survey
errors measures the emitters.

### Building the Dataset

```bash
make data          # builds data/opir_v2 and updates data/manifests/opir_v2.json
make data-verify   # rebuilds and checks every split against the versioned manifest
```

The dataset is a pure function of `configs/dataset/opir_v2.yaml` and its seed.
See the [data card](docs/data_card.md) for the generative model, splits,
domain-shift test sets, and limitations.

### Running the Experiments

```bash
make experiments   # E1–E7, in order
make e5 e6 e7      # tracking and fusion only (~10 min, no dataset needed)
```

Each experiment writes its JSON report and figures under `reports/` and
records the run in `runs/`.

| Experiment | Question | Reports |
|---|---|---|
| E1 | Detection at a calibrated false-alarm rate | `reports/phase3/` |
| E2 | Classification vs baselines, under domain shift (~1 h on Apple MPS) | `reports/phase3/` |
| E3 | Calibration, conformal sets, OOD; exports `models/opir_event_classifier/` | `reports/phase3/` |
| E4 | Geolocation vs the Cramér-Rao bound, geometry, systematic errors | `reports/phase4/` |
| E5 | Tracking under clutter, stereo ghosts, motion models | `reports/phase5/` |
| E6 | Fusion architectures and track classification | `reports/phase5/` |
| E7 | Sensor outages, RF latency, time-correlated RF bias | `reports/phase5/` |

## Testing

```bash
make check   # lint, format check, strict type check, tests, coverage gate, latency budgets
make test    # tests only
make bench   # per-stage latency benchmarks
```

| Suite | Contents |
|---|---|
| `tests/unit/` | Per-module tests, including Monte Carlo consistency checks (NEES/NIS within χ² bounds) and exactness on noiseless data |
| `tests/property/` | Hypothesis property tests: covariance stays PSD, GNN matches brute force, estimates are invariant to measurement order and translation |
| `tests/integration/` | End-to-end pipeline scenarios, the CLI, and experiment smoke runs |
| `tests/characterization/` | One regression test per defect found in the v1 audit (all fixed) |
| `benchmarks/` | Latency budget per stage: a 30-target tracker scan, stereo pairing, RF fix, CFAR, classifier (PyTorch and ONNX Runtime) |

## Results

All numbers are measured on the simulated `opir_v2` dataset
([data card](docs/data_card.md)) by the experiments in `experiments/`
(`make experiments`). Reports with confidence intervals are in
[`reports/phase3/`](reports/phase3/). The shipped classifier is documented in
the [model card](docs/model_card.md). Geolocation methods and results are in
[docs/geolocation.md](docs/geolocation.md), and tracking and fusion in
[docs/fusion.md](docs/fusion.md).

**Fusion (E5–E7).** Centralized fusion of OPIR and RF on a multi-target
scenario (2 launches, 3 aircraft with datalinks, 1 fire; 10 seeds):

| Architecture | GOSPA (m) | Aircraft RMSE | Launches / fires tracked | NEES (3 = consistent) |
|---|---|---|---|---|
| RF only | 1961 | 74 m | no | 3.2 |
| OPIR only | 877 | 286 m | yes (413 m / 276 m RMSE) | 2.7 |
| **Centralized fusion** | **693** | 63 m | yes | 3.1 |
| Track-to-track, covariance intersection | 692 | 57 m | yes | 3.1 |

Track-to-track fusion places aircraft better because the centralized filter
treats biased RF fixes as independent and gives OPIR too little weight.
Centralized fusion is still preferred: GOSPA is equal, identities are 6× more
stable, and only it can use single-satellite rays. Estimating the bias in the
filter is the planned remedy ([ADR 0001](docs/adr/0001-fusion-architecture.md)).

- **Stereo ghosts:** two satellites can pair rays from different targets. Deferring ambiguous pairs to angle-only updates cuts false tracks from 0.31 to 0.09 per scan.
- **Motion model:** an IMM beats a single constant-velocity model (GOSPA 693 m vs 889 m at the best single process noise).
- **Track classification:** using the classifier only on windows inside its training domain is what makes track labels reliable: posterior ECE falls from 0.13–0.20 to 0.05, and at the default pooling weight every target ends correctly labelled.
- **Robustness:** OPIR keeps 100% of aircraft tracked through an RF outage (RF alone: 17%). Extrapolating late RF fixes loses nothing up to 2 s of latency.
- **RF bias:** a fixed 30 ns clock bias makes RF tracks overconfident as they age (NEES 21 → 113), even with consider-covariance fixes. The library carries each fix's systematic covariance to the track as a floor on its reported covariance, which restores NEES 3.2–3.8 at every age.

**Geolocation (E4).** The ML TDOA estimator stays within 3% of the Cramér-Rao
bound from 1 to 100 ns of timing noise (for example 8.0 m RMSE against an
8.3 m bound at 10 ns), and its reported covariance is consistent (NEES ≈ 3).
Joint TDOA/FDOA attains its velocity bound (0.80 m/s against 0.83 m/s at 1 Hz).
Unmodeled clock bias or survey error makes the reported uncertainty too small:
with 50 m survey error the actual RMSE is 131 m against a reported 8.3 m
(NEES 710). Folding the errors into a consider covariance restores consistency
(NEES 2.5–2.9). Geometry dominates accuracy: the median
error bound inside the network is 9 m with mixed-altitude receivers vs 136 m
with ground-only receivers.

**Detection (E1).** Window-level alarms at a calibrated false-alarm rate
(20,000 independent background windows):

| Detector | Pd at Pfa = 1% (glint-free background) | Pfa of the analytic 1% threshold on realistic background |
|---|---|---|
| CFAR | **66%** (explosion 98%, launch 92%) | 33% |
| CUSUM | 55% | 84% |
| Step GLRT | 47% | 65% |
| v1 ensemble | n/a | 100% (alarms on every window) |

Thresholds derived from white-noise theory fail on correlated clutter, so
thresholds are calibrated empirically. Sun glints dominate the remaining false
alarms and are handled by the classifier.

**Classification (E2).** Macro-F1, mean over 5 seeds with 95% CI:

| Model | Test | Low SNR | Out-of-range physics | Heavy clutter |
|---|---|---|---|---|
| Features + logistic regression | 0.703 | 0.550 | 0.556 | 0.455 |
| Features + gradient-boosted trees | 0.720 | 0.584 | 0.560 | 0.463 |
| 1D CNN | 0.790 | 0.618 | 0.750 | 0.575 |
| **TCN (shipped)** | **0.806** [0.788, 0.824] | 0.659 | 0.737 | 0.579 |

**Calibration and two-stage alarms (E3).**

- **Conformal sets:** 90.2% coverage on test with a mean of 1.32 labels, falling to 73–83% under shift.
- **Two-stage false alarms:** CFAR followed by the classifier reduces false alarms on all background from 24% to 5.6% at 63% detection probability.
- **Novel event types:** energy-based OOD scores flag unseen faint events (AUROC 0.71–0.79) but not unseen bright ones (0.38–0.40), a documented limitation.

## Technical Highlights

### Algorithm Implementations

**Detection Algorithms:**

- Temporal differencing with adaptive thresholding
- MAD-based anomaly detection for outlier identification
- Rise-time analysis for signature characterization
- Ensemble voting for robust detection

**Geolocation Methods:**

- Least-squares TDOA positioning with Levenberg-Marquardt optimization
- Doppler-shift FDOA for velocity estimation
- Chan's algorithm for closed-form hyperbolic positioning
- Weighted least squares with covariance estimation

**Sensor Fusion:**

- Centralized measurement-level fusion with χ² gating on the innovation covariance
- Extended Kalman (line-of-sight) updates and IMM filtering
- Covariance intersection for track-to-track fusion under unknown correlation
- GOSPA/OSPA, purity, fragmentation, and NEES for evaluation

## References

**Geolocation Algorithms:**

- Y. T. Chan and K. C. Ho, "A Simple and Efficient Estimator for Hyperbolic Location"
- K. C. Ho and W. Xu, "An Accurate Algebraic Solution for Moving Source Location"

**Sensor Fusion:**

- S. Blackman and R. Popoli, "Design and Analysis of Modern Tracking Systems"
- Y. Bar-Shalom et al., "Estimation with Applications to Tracking and Navigation"

**Signal Processing:**

- S. Kay, "Fundamentals of Statistical Signal Processing: Detection Theory"

<br>

<h1 align="center">LET'S CONNECT!</h1>

<h3 align="center">Michael Gurule</h3>

<p align="center">
  <strong>Data Science | ML Engineering</strong>
</p>
<br>

<div align="center">
  <a href="mailto:michaelgurule1164@gmail.com">
    <img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white"></a>

  <a href="michaelgurule.com">
    <img src="https://custom-icon-badges.demolab.com/badge/MICHAELGURULE.COM-150458?style=for-the-badge&logo=browser&logoColor=white"></a>

  <a href="www.linkedin.com/in/michael-gurule-447aa2134">
    <img src="https://custom-icon-badges.demolab.com/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin-white&logoColor=fff"></a>

  <a href="https://medium.com/@michaelgurule1164">
    <img src="https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white"></a>
</div>
<br>

---

<p align="center">
<img  width="450" alt="Designed By" src="https://github.com/user-attachments/assets/12ddff9c-b9b6-4e69-ace0-5cbc94f1a3ad">
</p>
