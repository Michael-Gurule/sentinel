# ADR 0003: Physics-based synthetic data, with domain-shift test sets

- **Status:** accepted (Phase 2)
- **Evidence:** [data card](../data_card.md), E1–E3, `locant.sim`

## Context

The problem needs labeled OPIR time series (launches, explosions, fires,
aircraft, background) and RF measurements of emitters with known truth.
Real OPIR data is controlled, and no public labeled source exists. The v1
prototype used hand-drawn signal shapes. The classes were trivially
separable, so the classifier scored 100% (audit C8), and the dataset could
not even be regenerated (audit C3).

## Decision

Generate all data with a **physics-based simulator** and treat it as an
experimental instrument, not a stand-in for reality:

- **Model the measurement chain, not the signal shape.** The chain runs from
  event physics (radiant intensity in W/sr, ballistic boost, fire growth)
  through WGS-84 geometry, atmospheric transmittance, cloud occlusion, PSF
  pixel phasing, Earth background, correlated clutter, sun glints, and
  sensor noise. Difficulty then comes from the physics: dim, distant, or
  occluded events, and glints that mimic explosions.
- **Build deterministically and verify.** Each sample's random stream is
  `SeedSequence(root_seed, (split, class, index))`. The manifest stores a
  config hash and per-split content hashes. `make data-verify` rebuilds the
  dataset and compares hashes.
- **Hold out domain shift.** Besides the i.i.d. test split there are three
  shifted test sets: low SNR, event physics outside the training ranges,
  and heavy clutter. Every model is evaluated on all four, so "how does it
  degrade?" is answered rather than assumed.
- **Simulate RF from geometry, with fielded error sources.** Each scenario
  draws per-receiver clock bias, survey error, and LO offsets, plus random
  timing and frequency noise per scan. Solvers are evaluated against the
  Cramér-Rao bound, which needs exact truth.

## Alternatives considered

- **Public proxies** (e.g., FIRMS fire detections, weather-satellite IR).
  They have the wrong sensor, cadence, and classes, and no truth for
  launches or explosions.
- **The v1 hand-drawn shapes.** Rejected: they gave a trivially separable
  task (C8).
- **Generative models** trained on a few examples. There is nothing to
  train them on, and they add unverifiable realism.

## Consequences

- **Results measure methodology, not fielded performance.** Every document
  says so (data card "Intended use"; README). Magnitudes are illustrative
  and unclassified.
- **Measured sim-to-real proxies.** The shift sets show how much the
  learned components degrade: TCN macro-F1 drops from 0.81 to 0.58–0.74,
  and conformal coverage from 90% to 73–83%. This is the kind of
  degradation to expect against real data. The E6 domain-of-validity
  finding comes from the same reasoning.
- **What changes with real data:**
  - recalibrate the detector thresholds on real background (E1's procedure);
  - refit temperature and the conformal threshold on a real calibration
    split (E3's procedure);
  - re-establish the classifier's trained onset range;
  - estimate the network's systematic-error levels from survey and timing
    data.

  The code paths stay the same; the configuration changes
  ([ADR 0002](0002-configuration-cli-and-run-registry.md)).
- **Open gaps (Phase 8):** TDOA/FDOA are simulated from geometry, not
  estimated from waveforms, and the focal plane is not modeled
  (single-pixel events).
