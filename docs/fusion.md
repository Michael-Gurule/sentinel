# Tracking and multi-INT fusion: methods and results

How SENTINEL turns OPIR lines of sight and RF fixes into one track picture,
and what each design choice buys. Results come from experiments E5–E7
(`make e5 e6 e7`, about 20 min, no dataset needed). The full numbers, with
95% confidence intervals over 10 seeds, are in
[`reports/phase5/`](../reports/phase5/). The architecture decision is
recorded in [ADR 0001](adr/0001-fusion-architecture.md).

## Benchmark scenario

Each seed draws a 180 s scenario (`experiments/fusion_common.py`):

| Element | Configuration |
|---|---|
| Targets | 2 launches (boost 90–150 s), 3 aircraft at 8–11 km altitude, 200–250 m/s, each with a datalink, 1 ground fire |
| OPIR | GEO + Molniya HEO satellites, 1 Hz reports above SNR 5, 10 µrad line-of-sight noise |
| RF | 5 receivers over a 40 km area, 10 ns timing noise, 5 ns clock bias, 2 m survey error, FDOA at 1 Hz noise |

The truth set at each scan is every target that *any* sensor can see. A
single-sensor architecture is therefore charged for the targets only the
other sensor sees. That coverage question is exactly what fusion should
answer.

**Metrics.** GOSPA (α = 2, cutoff c = 2 km) decomposes into localization
error plus c²/2 per missed target and per false track; an estimate more than
2 km from every target counts as false. The other metrics are per-type
localization RMSE and coverage (the fraction of truth scans with a matched
confirmed track), confirmation latency, fragmentation and identity switches
from a per-scan truth-to-track assignment, and the NEES of matched tracks
(`sentinel.eval.tracking`).

## Methods

### OPIR geolocation (`sentinel.fusion.opir_geoloc`)

An OPIR detection is a unit line of sight **u** from a known satellite
position **s**, with angular noise σ per axis. It is used in three ways:

- **Line-of-sight update.** z = E·u(x) = 0, where E is a 2×3 basis
  perpendicular to the measured ray. The Jacobian is
  H = E (I − uuᵀ) / r, and the extended Kalman update runs in Joseph form.
  One ray cannot start a track because range is unobservable, but it can
  update any track.
- **Stereo triangulation.** Weighted least squares on two or more rays. It
  minimizes Σ ‖Pᵢ(x − sᵢ)‖² / (σᵢrᵢ)², where Pᵢ = I − uᵢuᵢᵀ, so the
  information is Σ Pᵢ / (σᵢrᵢ)². With GEO + HEO the error is about
  300–400 m, mostly along the shallow-angle direction.
- **Altitude intersection.** One ray meets a known altitude plane (for
  ground fires and launch pads). Implemented and tested, but not used in the
  benchmarks.

**Stereo association and ghosts.** A pair of rays from two satellites is
consistent if its normalized miss distance passes a χ²(1) gate. Two
satellites constrain a pair only within their epipolar plane, so rays from
*different* targets near a common epipolar plane also pass and triangulate to
a **ghost**. In the benchmark, global-nearest-neighbor pairing produced
224 ghosts among 8,595 pairs (2.6%). SENTINEL keeps a pair only
if neither ray gates with any other ray. Ambiguous rays become line-of-sight
updates, which the tracker resolves against predicted tracks. This removed
every ghost and kept 89% of the correct pairs.

### Tracker (`sentinel.tracking`)

- **Measurement interface.** Every measurement provides
  `linearize(x) -> (h(x), H)` and, if it determines position,
  `initial_state()`. RF fixes (position, or position and velocity with
  cross-covariance), stereo positions, and lines of sight share one update
  path.
- **Association.** Measurements are grouped by (source, dimension), with
  each OPIR satellite its own source. Each group is gated on the NIS against
  χ²(dim) at 0.99 and assigned by GNN, so a track can take a ray from each
  satellite and an RF fix in the same scan.
- **IMM.** A quiet (q = 25 m²/s³) and a maneuvering (q = 1600 m²/s³)
  constant-velocity model, with mean sojourn times of 60 s and 10 s. Gating
  uses the smallest NIS over the modes, so a maneuver can be gated by the
  model that predicts it.
- **Track management.**
  - Confirmation is M-of-N (default 3 of 5); only confirmed tracks are
    reported.
  - A track is deleted after 10 s without an update.
  - Duplicate tracks on one target, which alternate in claiming its single
    measurement, are merged when their positions agree at χ²(3) = 0.9 under
    the summed covariance.
- **Class posterior.** Each OPIR measurement can carry calibrated classifier
  probabilities (E3). They are pooled log-linearly into the track posterior:
  log π ← log π + w·(log p − log prior), with tempering weight w ≤ 1.

### Track-to-track fusion (`sentinel.fusion.t2t`)

Separate OPIR and RF trackers; confirmed tracks are paired by a χ²(3) gate
on the summed position covariance and GNN. Matched pairs are fused either
**naively**, as if independent, P = (P₁⁻¹ + P₂⁻¹)⁻¹, or by **covariance
intersection**, P⁻¹ = ωP₁⁻¹ + (1−ω)P₂⁻¹ with ω chosen to minimize tr P. CI
is consistent for any unknown cross-correlation. A unit test shows the naive
rule's NEES exceeds its χ² bound under correlated errors, while CI stays
within it.

## Results

### Architectures (E6)

| Architecture | GOSPA (m) | False tracks / scan | Aircraft RMSE | Launch RMSE | Fire RMSE | NEES |
|---|---|---|---|---|---|---|
| RF only | 1961 [1938, 1985] | 0.00 | 74 m | not seen | not seen | 16.3 |
| OPIR only | 877 [822, 931] | 0.08 | 286 m | 413 m | 276 m | **2.7** |
| **Centralized** | **693** [637, 749] | 0.09 | **63 m** | 413 m | 276 m | 10.5 |
| T2T naive | 706 [647, 765] | 0.10 | 67 m | 413 m | 276 m | 10.7 |
| T2T CI | 710 [648, 771] | 0.10 | 73 m | 413 m | 276 m | 10.8 |

![Architectures](../reports/phase5/figures/e6_architectures.png)

- **Fusion is mostly about coverage.** RF never sees the launches or the
  fire. OPIR sees everything, but places aircraft about 4× worse than RF (286 m vs 74 m).
  Centralized fusion gets both.
- **Aircraft accuracy improves modestly** (74 → 63 m), because OPIR stereo
  is about 4× less precise than RF. The gain is largest where the RF
  geometry is weak, with aircraft outside the receiver network.
- **T2T loses information and identity.** Naive T2T is close to
  centralized here because the two trackers' errors are nearly independent.
  CI gives up information that, in this case, was not correlated: its
  aircraft accuracy equals RF alone. Pairing tracks after the fact switches
  identities about 9× as often as centralized fusion (2.7 vs 0.3 per run).
- **OPIR tracks are consistent** (NEES 2.7 against 3). The high NEES of every
  RF-fed architecture comes from the time-correlated RF bias (E7 below), not
  from fusion.

![Scenario](../reports/phase5/figures/e6_scenario.png)

### Track management under clutter (E5)

OPIR false reports are injected uniformly over a 120 km square, at 0–5 per
satellite per scan.

| Clutter rate | 1-of-1: false tracks/scan, GOSPA | 2-of-3 | 3-of-5 |
|---|---|---|---|
| 0 | 0.21, 754 m | 0.11, 685 m | 0.09, 693 m |
| 2 | 1.59, 1691 m | 0.19, 760 m | 0.13, 725 m |
| 5 | 7.58, 3837 m | 0.51, 1035 m | **0.21, 808 m** |

Each extra required hit costs about 1 s of confirmation latency (median for
launches: 2.5 s, 3.5 s, and 4.5 s). Clutter does little to 2-of-3 and 3-of-5
because a random ray from one satellite rarely pairs stereo-consistently
with one from the other, so clutter seldom starts a track. 3-of-5 is the
default.

![Track management](../reports/phase5/figures/e5_track_management.png)

**Stereo ghosts.** Deferring ambiguous pairs cuts false tracks per scan
from 0.31 to 0.09 (centralized) and GOSPA from 832 to 693 m. The cost is
slower launch initiation, with median latency rising from 2.3 to 4.5 s.

![Stereo ghosts](../reports/phase5/figures/e5_stereo_ambiguity.png)

**Motion model.**

| Model | GOSPA | Launch RMSE | Launch coverage | Identity switches |
|---|---|---|---|---|
| Constant velocity, q = 25 | 1404 m | 747 m | 64% | 8.5 |
| Constant velocity, q = 400 | 889 m | 522 m | 92% | 2.6 |
| **IMM (q = 25 / 1600)** | **693 m** | **413 m** | **96%** | **0.3** |

No single process noise suits both cruising aircraft and boosting launches.
The quiet model loses launches, and the noisy one blurs everything else.

### Track classification (E6)

The exported TCN classifies each OPIR report's 64 s window. Two choices
matter: the pooling weight w, and which windows the classifier is applied to.
The classifier was trained on windows with the event onset 2–40 s in, that
is, windows ending 24–62 s after onset. Later windows no longer contain the
onset, and a boosting plume then looks like a steady source, so launches
drift to "aircraft".

| Windows used (detection age) | w | Final label accuracy | Posterior ECE | Median time to confident label |
|---|---|---|---|---|
| all ages | 1.0 | 68% | 0.20 | 18 s |
| all ages | 0.3 | 77% | 0.17 | 24 s |
| 0–62 s | 0.3 | 100% | 0.14 | 24 s |
| **24–62 s (trained range)** | **0.3** | **100%** | **0.06** | 30 s |
| 24–62 s | 1.0 | 100% | 0.05 | 28 s |

"Confident" means a true-class probability of at least 0.9. ECE is computed
over every scan at which a matched track has a posterior.

![Classification](../reports/phase5/figures/e6_classification.png)

- **The domain of validity matters more than the pooling rule.** Restricting
  the classifier to the windows it was trained on fixes the final labels and
  cuts calibration error about 3×, at the cost of about 6 s to the first
  confident label.
- **The tempering weight matters little inside the trained domain.** Its
  role is robustness: with w = 1, out-of-domain evidence (all ages) drives
  wrong labels to certainty. The default is w = 0.3 with the trained range.
- **The pipeline applies the same rule** using the CFAR onset estimate:
  class evidence reaches a track only if the detected onset lies 2–40 s
  into the window (`CLASSIFIER_ONSET_RANGE_S`).

### Robustness (E7)

**Sensor outages**, during 60–120 s:

| Outage | Architecture | GOSPA during | Aircraft coverage | Launch coverage | Fragmentations per run |
|---|---|---|---|---|---|
| none | centralized | 682 m | 100% | 97% | 0 |
| HEO satellite | OPIR only | 2696 m | 100% | 23% | 1.9 |
| HEO satellite | centralized | 2476 m | 100% | 23% | 1.8 |
| RF network | RF only | 3284 m | 17% | – | 3.0 |
| RF network | centralized | 802 m | **100%** | 97% | **0** |
| RF network | T2T CI | 812 m | 100% | 97% | 0 |

![Outages](../reports/phase5/figures/e7_outages.png)

- **An RF outage is absorbed.** OPIR stereo positions and lines of sight
  keep every aircraft track, with RMSE rising from 64 m to 217 m, and RF
  re-acquires the tracks without fragmentation.
- **A satellite outage is not.** With one satellite there is no stereo, so
  OPIR cannot start tracks, and angle-only updates leave range unobserved.
  Existing launch tracks drift in range, which shows up as false and missed
  targets. RF still holds the aircraft (71 m). The remedies are a third
  satellite, altitude-constrained initiation for ground events, or a boost
  dynamics model.

**RF latency.** RF fixes arrive L seconds late, after newer OPIR scans.

| Latency | Drop late fixes | Buffer and predict forward | Extrapolate late fixes |
|---|---|---|---|
| 0 s | 693 m | – | – |
| 1 s | 877 m | 819 m | **692 m** |
| 2 s | 877 m | 968 m | **692 m** |
| 5 s | 877 m | 1542 m | **704 m** |

All values are GOSPA.

- **Dropping** discards all RF, because at 1 Hz OPIR every late fix is out
  of sequence.
- **Buffering** keeps the tracker in time order but delays the whole picture
  by L, which hurts the accelerating launches most.
- **Extrapolating** propagates each fix to the current time with its own
  FDOA velocity (z′ = Fz, R′ = FRFᵀ + Q). It loses nothing up to 2 s.
  Aircraft RMSE grows from 64 m to 94 m at 5 s.

**Time-correlated RF bias.** Clock bias and survey error are fixed for a
scenario. The consider covariance (E4) makes each *fix* consistent, but the
tracker averages many fixes from the same biased network as if their errors
were independent. Its covariance shrinks toward zero while the bias stays.

| 30 ns clock bias + 10 m survey | NEES, track age 0–10 s | 30–60 s | 120–180 s |
|---|---|---|---|
| naive fixes | 256 | 540 | 501 |
| consider fixes | 21 | 77 | 113 |
| **consider fixes + bias floor** | **3.3** | **3.2** | **3.4** |

![Latency and bias](../reports/phase5/figures/e7_latency_bias.png)

Clock bias and survey error have the same (I + 11ᵀ) structure as random
timing noise (docs/geolocation.md). To first order, the systematic part of a
fix's position covariance is therefore k/(1+k) of the consider covariance,
where k = σ_sys²/σ_toa². Adding that part back as a floor on the reported
track covariance restores consistency at every age. At the default 5 ns /
2 m levels, the floor brings NEES from 7.5–19 down to 2.6–3.3. It does not
rescue naive fixes (NEES about 11): gating and IMM mode weights computed with
too-small fix covariances already misweight the measurements. In E5–E7 the
floor lives in the experiment harness; moving it into the library (a
systematic covariance on each RF fix, carried to the track) is Phase 6 work.

## Limits

- **Simulated reports.** OPIR reports are generated per event, at a fixed
  10 µrad noise, from the simulator's pixel signals. Focal-plane detection,
  centroiding, and pixel-level clutter tracking are not modeled; the
  detection age used to gate the classifier stands in for a focal-plane
  tracker's 2-D track age.
- **GNN, not JPDA or MHT.** Hard assignment is adequate at these target
  densities. Crossing targets would need soft association.
- **Launch dynamics.** Launches are tracked with constant-velocity models,
  which explains the 413 m launch RMSE. A boost model, or IMM with a
  constant-acceleration mode, is a stretch item.
- **Bias floor.** The floor uses the nearest RF fix's geometry and assumes
  the bias stays fixed over a track's life. Estimating the bias states
  (Schmidt–Kalman) is the principled extension.
