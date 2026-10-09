
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

Consider a scenario: An OPIR satellite detects a thermal anomaly consistent with a missile launch. Confidence: 70%. Simultaneously, an RF geolocation system identifies an emitter at coordinates with 50-meter uncertainty. Confidence: 80%.

**Naive approach**: Report both independently.

> **Result:** Analysts must manually correlate information, introducing delays and potential errors.

**Fusion approach**: Combine measurements using covariance weighting.

> **Result**: Single track with 95% confidence, 15-meter position uncertainty, and classified event type.

The mathematics is straightforward but powerful. Using covariance intersection, the fused position uncertainty becomes:
<br>

<p align="center">
<img src="https://latex.codecogs.com/png.latex?%5Chuge%20P_{fused}%20=%20(P_{OPIR}^{-1}%20+%20P_{RF}^{-1})^{-1}">
</p>  

Where `P` represents position covariance matrices. The fused uncertainty is **always lower** than either individual measurement. This isn't just combining data; it's extracting information that neither sensor could provide alone.

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
- **Sensor Fusion Engine**: Multi-sensor track management
  - Data association with Mahalanobis distance gating
  - Covariance-weighted measurement fusion
  - Track quality scoring and confidence estimation
  - Uncertainty quantification (CEP, position/velocity covariance)

---

## Project Structure

```
sentinel/
│
├── src/sentinel/
│   ├── models/
│   │   ├── signal_generator.py       # OPIR thermal signature generation
│   │   ├── rf_generator.py           # RF signal generation
│   │   └── cnn_classifier.py         # Event classification CNN
│   │
│   ├── detection/
│   │   └── opir_detectors.py         # 4 detection algorithms
│   │
│   ├── tracking/
│   │   └── kalman_tracker.py         # Multi-target Kalman tracking
│   │
│   ├── geolocation/
│   │   ├── tdoa_fdoa.py              # TDOA/FDOA geolocation
│   │   └── multilateration.py        # Spherical/hyperbolic positioning
│   │
│   ├── fusion/
│   │   └── sensor_fusion.py          # Multi-sensor fusion engine
│   │
│   ├── training/
│   │   └── train_classifier.py       # CNN training pipeline
│   │
│   └── pipeline/
│       ├── phase2_pipeline.py        # OPIR detection pipeline
│       └── phase3_pipeline.py        # Full multi-sensor pipeline
│
├── scripts/
│   └── generate_opir_dataset.py      # Training data generation
│
├── tests/
│   ├── test_0_generator.py           # Signal generator tests
│   ├── test_1_detection.py           # Detection algorithm tests
│   ├── test_2_cnn.py                 # CNN architecture tests
│   ├── test_3_classifier.py          # Classifier wrapper tests
│   ├── test_4_kalman.py              # Kalman filter tests
│   ├── test_5_tracker.py             # Multi-target tracking tests
│   ├── test_6_tdoa_fdoa.py           # TDOA/FDOA geolocation tests
│   ├── test_7_multilateration.py     # Multilateration tests
│   ├── test_8_sensor_fusion.py       # Sensor fusion tests
│   └── test_9_full_system.py         # Complete system integration test
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│       └── opir/
│           ├── train/                # Training data (5 classes)
│           ├── validation/           # Validation data
│           └── test/                 # Test data
│
└── outputs/
    └── models/                       # Trained model checkpoints
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
python -m sentinel.pipeline.phase3_pipeline
```

Tracks a moving RF emitter for 10 s while running OPIR detection and
classification on each frame.

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
from sentinel.pipeline.phase3_pipeline import SENTINELPhase3Pipeline

pipeline = SENTINELPhase3Pipeline()
rng = np.random.default_rng(0)
start, velocity = np.array([5e3, 5e3, 500.0]), np.array([100.0, 50.0, 0.0])

for t in range(10):
    truth = start + velocity * t
    pipeline.process_multi_sensor_frame(
        opir_signals=[],
        rf_measurements=[simulate_tdoa(truth, pipeline.receivers, 10e-9, rng)],
        sampling_rate=100.0,
        timestamp=float(t),
    )

print(pipeline.get_situation_awareness())
```

Tracks are maintained by a Kalman filter with a constant-velocity
(white-noise acceleration) model, χ² gating on the innovation covariance, and
global-nearest-neighbor assignment. OPIR reports are detected and classified
but not yet fused into tracks: that requires OPIR geolocation, which is on the
roadmap.

### OPIR Detection & Classification

```python
import numpy as np

from sentinel.classification import EventClassifier
from sentinel.data.config import Priors
from sentinel.data.generate import generate_sample
from sentinel.detection import CFARDetector
from sentinel.pipeline.phase3_pipeline import CFAR_THRESHOLD_PFA_1E2

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
make experiments   # E1 detection, E2 classification (~1 h on Apple MPS), E3 calibration + model export
```

Each experiment writes a JSON report and figures to `reports/phase3/`. E3
exports the calibrated classifier to `models/opir_event_classifier/`.

## Testing

```bash
make check   # lint, format check, strict type check, tests, coverage gate
make test    # tests only
```

| Suite | Contents |
|---|---|
| `tests/unit/` | Per-module tests, including Monte Carlo consistency checks (NEES/NIS within χ² bounds) and exactness on noiseless data |
| `tests/property/` | Hypothesis property tests: covariance stays PSD, GNN matches brute force, estimates are invariant to measurement order and translation |
| `tests/integration/` | End-to-end pipeline scenarios |
| `tests/characterization/` | One test per defect found in the v1 audit; open defects are strict expected failures |

## Results

All numbers are measured on the simulated `opir_v2` dataset
([data card](docs/data_card.md)) by the experiments in `experiments/`
(`make experiments`). Reports with confidence intervals are in
[`reports/phase3/`](reports/phase3/). The shipped classifier is documented in
the [model card](docs/model_card.md). Geolocation and fusion benchmarks follow
in later phases.

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

- Mahalanobis distance gating for data association
- Covariance intersection for multi-sensor fusion
- Extended Kalman filtering for track propagation
- Track quality scoring based on confidence, uncertainty, and sensor diversity

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
