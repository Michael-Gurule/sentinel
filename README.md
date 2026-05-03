
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
sentinel-multi-intel-platform/
│
├── src/
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

```
# Clone repository
git clone https://github.com/michael-gurule/sentinel-multi-intel-platform.git
cd sentinel-multi-intel-platform

# Create virtual environment

# Install dependencies
pip install torch torchvision
pip install numpy scipy pandas matplotlib

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
```

## Usage

### Quick Start: Full System Demo

```python
from src.pipeline.phase3_pipeline import demo_phase3_system

# Run complete multi-sensor demonstration
demo_phase3_system()
```

### OPIR Detection & Classification

```python
from src.models.signal_generator import OPIRSignalGenerator
from src.detection.opir_detectors import MultiMethodDetector
from src.models.cnn_classifier import OPIRClassifier

# Generate signal
generator = OPIRSignalGenerator()
signal = generator.generate_launch_signature(start_time=2.0)

# Detect event
detector = MultiMethodDetector()
detection = detector.detect(signal, generator.sampling_rate)

# Classify event
classifier = OPIRClassifier(device='cpu')
classification = classifier.classify(signal)

print(f"Detected: {detection.detected}")
print(f"Event type: {classification.class_name}")
print(f"Confidence: {classification.confidence:.3f}")
```

### RF Geolocation

```python
from src.geolocation.tdoa_fdoa import (
    HybridTDOAFDOA,
    SensorPosition,
    simulate_tdoa_measurements
)
import numpy as np

# Define sensor network (mixed altitude deployment)
sensors = [
    SensorPosition(id=0, position=np.array([0.0, 0.0, 500.0])),
    SensorPosition(id=1, position=np.array([10000.0, 0.0, 1500.0])),
    SensorPosition(id=2, position=np.array([10000.0, 10000.0, 1000.0])),
    SensorPosition(id=3, position=np.array([0.0, 10000.0, 2000.0]))
]

# Simulate measurements
emitter_pos = np.array([5000.0, 5000.0, 500.0])
measurements = simulate_tdoa_measurements(emitter_pos, sensors)

# Geolocate emitter
solver = HybridTDOAFDOA(carrier_freq=1e9)
result = solver.estimate(sensors, measurements)

print(f"Estimated position: {result.position}")
print(f"Position error: {np.linalg.norm(result.position - emitter_pos):.1f} m")
print(f"GDOP: {result.gdop:.3f}")
```

### Multi-Sensor Fusion

```python
from src.pipeline.phase3_pipeline import SENTINELPhase3Pipeline
from src.models.signal_generator import OPIRSignalGenerator
from src.geolocation.tdoa_fdoa import simulate_tdoa_measurements
import numpy as np

# Initialize system
pipeline = SENTINELPhase3Pipeline()

# Generate multi-sensor frame
generator = OPIRSignalGenerator()
opir_signals = [generator.generate_launch_signature(start_time=2.0)]

emitter_pos = np.array([5000.0, 5000.0, 500.0])
rf_measurements = [simulate_tdoa_measurements(emitter_pos, pipeline.sensors)]

# Process frame
result = pipeline.process_multi_sensor_frame(
    opir_signals=opir_signals,
    rf_measurements=rf_measurements,
    sampling_rate=generator.sampling_rate,
    timestamp=0.0
)

print(f"OPIR detections: {result['opir_detections']}")
print(f"RF geolocations: {result['rf_geolocations']}")
print(f"Fused tracks: {result['fused_tracks']}")

# Get situation awareness
sa = pipeline.get_situation_awareness()
print(f"Track quality: {sa['average_track_quality']:.3f}")
```

### Training the CNN Classifier

```python
from src.training.train_classifier import train_model_from_folders

# Train model on generated dataset
history = train_model_from_folders(
    train_dir='data/synthetic/opir/train',
    val_dir='data/synthetic/opir/validation',
    output_dir='outputs/models',
    num_epochs=50,
    batch_size=32,
    device='cpu'  # or 'cuda' for GPU
)

print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
```

## Testing

### Run Individual Component Tests

```bash
# Test signal generation
python tests/test_0_generator.py

# Test detection algorithms
python tests/test_1_detection.py

# Test CNN architecture
python tests/test_2_cnn.py

# Test TDOA/FDOA geolocation
python tests/test_6_tdoa_fdoa.py

# Test multilateration
python tests/test_7_multilateration.py

# Test sensor fusion
python tests/test_8_sensor_fusion.py

# Test complete system
python tests/test_9_full_system.py
```

### Expected Test Results

- Detection algorithms: 75-100% detection rate
- CNN forward pass: Successful with 4-5 classes
- Kalman tracking: <5m mean error over 10 steps
- TDOA geolocation: <50m position error (4 sensors, low noise)
- FDOA velocity estimation: <20 m/s velocity error
- Sensor fusion: Track quality >0.7 for high-confidence tracks
- Full system: Successfully creates and maintains fused tracks

## Performance Metrics

### Geolocation Accuracy

- **Position Error**: 10-50m (depending on sensor geometry and noise)
- **GDOP**: 2-5 (good geometry with mixed-altitude sensors)
- **Convergence Rate**: >95% for 4+ sensors

### Detection Performance

- **Temporal Difference**: 80-90% detection rate
- **Anomaly Detection**: 75-85% detection rate
- **Multi-Method Ensemble**: 90-95% detection rate

### Classification Accuracy (Untrained Model)

- Random baseline: ~20% (5 classes)
- After training: Expected 85-95% validation accuracy

### Sensor Fusion Quality

- **Track Quality**: 0.7-0.9 for multi-sensor tracks
- **Position Uncertainty**: 20-100m CEP (50% confidence)
- **Velocity Uncertainty**: 5-20 m/s standard deviation

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
