# ADR 0001: Centralized measurement-level fusion

- **Status:** accepted (Phase 5); amended in Phase 6 with the bias-floor results
- **Evidence:** experiments E5–E7, [`docs/fusion.md`](../fusion.md)

## Context

SENTINEL fuses two very different sensors:

- **OPIR** gives a line of sight (two angles) per satellite at 1 Hz, for every
  hot event: launches, fires, aircraft. Two satellites (GEO + Molniya HEO)
  triangulate to about 300–400 m. One satellite alone cannot measure range.
- **RF** TDOA/FDOA fixes give position and velocity (about 20–80 m) every
  scan, but only for emitting targets (here the aircraft datalinks). The
  fixes share the network's clock bias and survey error, so their errors are
  correlated in time.

v1 fused pre-combined positions and then also ran a Kalman update, which
counted the same information twice. It also used a covariance-intersection
formula that was really the independent-information formula (audit H3).

## Options

1. **Centralized measurement-level fusion.** One tracker; every OPIR stereo
   position, OPIR line of sight, and RF fix updates shared tracks
   sequentially with its own noise model.
2. **Track-to-track (T2T), naive.** Separate OPIR and RF trackers; matched
   tracks are fused as if their errors were independent.
3. **T2T with covariance intersection (CI).** Same, but the fused covariance
   is consistent for *any* cross-correlation between the two tracks.

## Decision

Option 1, centralized fusion. T2T is kept in the library
(`sentinel.fusion.t2t`) for distributed deployments, and E6 benchmarks it.

## Rationale (E6, 10 seeds, mean with 95% CI)

| Architecture | GOSPA (m) | Aircraft RMSE | Identity switches per run |
|---|---|---|---|
| RF only | 1961 [1938, 1985] | 74 m | 0.0 |
| OPIR only | 877 [822, 931] | 286 m | 0.3 |
| **Centralized** | **693** [637, 749] | **63 m** | **0.3** |
| T2T naive | 706 [647, 765] | 67 m | 2.7 |
| T2T CI | 710 [648, 771] | 73 m | 2.7 |

- **Measurement-level is optimal under independent sensor noise.** Each
  update is the exact Bayesian update, so there is no double counting and no
  need to estimate cross-correlations. It gives the best GOSPA and the best
  aircraft accuracy.
- **Line-of-sight measurements can only be used centrally.** A single OPIR
  ray cannot start a track, but it can update a track that RF started. That
  is what keeps aircraft tracked through an RF outage (E7: 100% aircraft
  coverage and no fragmentation, against 17% for RF alone) and what closes
  audit defect C1 in the pipeline. A separate OPIR tracker cannot use these
  rays.
- **T2T loses information and identity.** CI makes no assumption about
  correlation, so it gives up information. Here the two trackers' errors are
  nearly independent, so CI ends up at the RF-only accuracy (73 m vs 74 m).
  Pairing tracks after the fact also churns identities (2.7 switches per run
  vs 0.3).

## Consequences

- **The tracker handles heterogeneous measurements.** Measurements expose
  `linearize(x) -> (h(x), H)` and an optional `initial_state`. Updates are
  grouped by (source, dimension), so each group gets its own χ² gate. Each
  OPIR sensor is its own source, so a track can take a ray from every
  satellite in the same scan.
- **Out-of-sequence data needs a policy.** A central tracker must handle late
  measurements. E7 compares the options: dropping late RF fixes discards RF
  entirely at 1 Hz OPIR rates, and buffering delays the whole picture.
  Extrapolating each late fix to the current time with its FDOA velocity
  loses nothing up to 2 s of latency.
- **Time-correlated RF bias is not white noise.** The consider covariance
  makes each fix consistent, but a filter averaging many fixes from the same
  biased network becomes overconfident: NEES grows with track age to about
  100. Reported covariances therefore carry a bias floor (E7, NEES ≈ 3 at all
  ages; in the library since Phase 6).
- **When to revisit:** if sensors are fused across a network with limited
  bandwidth (track lists, not measurements), switch to T2T with CI, which is
  the safe choice when track correlations are unknown.

## Amendment (Phase 6): the bias floor changes the accuracy comparison

Once tracks report the RF bias floor and T2T fuses those reported
estimates, E6 gives:

| Architecture | GOSPA (m) | Aircraft RMSE | NEES | Identity switches per run |
|---|---|---|---|---|
| **Centralized** | 693 [637, 749] | 63 m | 3.1 | **0.3** |
| T2T naive | 689 [638, 741] | **53 m** | 3.1 | 1.9 |
| T2T CI | 692 [640, 744] | 57 m | 3.1 | 1.9 |

The rationale's claim that centralized fusion gives "the best aircraft
accuracy" no longer holds. Its premise was independent sensor noise. The RF
fixes share a time-correlated bias, so the centralized filter, which treats
them as independent, gives RF too much weight relative to OPIR. T2T fusion
of the reported estimates sees the RF track's bias floor and weights the two
correctly.

**The decision stands**, for the reasons that do not depend on that premise:

- GOSPA is equal within its confidence interval.
- Single-satellite rays can only be used centrally (RF-outage coverage).
- Identities are about 6× more stable.

**The proper remedy** is to model the bias inside the centralized filter
(Schmidt-Kalman consider states, or bias states augmented per RF network),
not to switch architectures. It is recorded as Phase 8 stretch work, and
this comparison is its benchmark.
