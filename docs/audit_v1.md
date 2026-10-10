# v1 audit: what was wrong and how it was fixed

Locant began as a prototype (tag [`v1.0-prototype`](https://github.com/Michael-Gurule/locant/tree/v1.0-prototype),
then named SENTINEL). Before rebuilding it, I audited the prototype the way
a reviewer would. I read every module, probed each algorithm with inputs
whose answer is known, and checked every claim in the README against the
code.

The audit found 8 critical defects, 11 serious ones, and a list of
engineering problems. This document records each one: what was wrong, the
evidence, how it was fixed, and the test that now guards it. Every
critical and serious defect has a regression test in
[`tests/characterization/test_known_defects.py`](../tests/characterization/test_known_defects.py)
unless noted. Each test asserts the *correct* behavior. Tests of open
defects carried a strict `xfail` marker that was removed with the fix, so
each fix has a recorded before and after.

## Critical: each would undermine the project on its own

| ID | Defect in v1 | Evidence | Fix | Guard |
|---|---|---|---|---|
| C1 | **OPIR never reached the fused tracks.** OPIR measurements were built with `position=None`, and the fusion engine dropped any measurement without a position. The README's "multiplicative effect" of fusion was not implemented. | Probe: after 10 frames the track had 0 OPIR detections and no event type. | OPIR line-of-sight geolocation: stereo triangulation, angle-only EKF updates, stereo ghost handling. Centralized fusion with RF (Phase 5). | `test_c1_opir_contributes_to_fused_tracks` |
| C2 | **The test suite asserted nothing.** All 13 test files only printed results. One test failed with a `TypeError`, and one failed to import. | `grep -c assert tests/*.py` gave 0 in every file. | New suite: unit, property (Hypothesis), statistical (NEES/NIS, CRLB), integration, characterization, regression; 85% coverage gate in CI (Phases 0–1). | The suite itself (about 340 tests) and the CI coverage gate |
| C3 | **The dataset could not be reproduced.** The generation script passed arguments that no committed version of the generator accepted, and the generator referenced an undefined attribute. The 10,000 stored samples and the trained model came from code that was never committed. | `git log -S` on the generator's signature. | Deterministic builder: per-sample `SeedSequence`, content hashes, a versioned manifest, `make data-verify` (Phases 1–2). | `test_c3_dataset_is_reproducible_from_committed_code`, `test_c3_versioned_manifest_matches_committed_config` |
| C4 | **Three conflicting classifier contracts.** Four classes and 256 samples in one module, five classes and 100 samples in another, and the README said a third thing. Both pipelines ran an untrained network. | Class lists and input lengths in the code. | One taxonomy; model artifacts carry their preprocessing, class order, calibration, and provenance, and load with `weights_only=True` (Phases 1, 3). | `test_c4_artifact_carries_the_training_contract` |
| C5 | **The detector fired on pure noise.** The multi-method detector flagged 100 of 100 noise-only signals: one test used an absolute threshold, and another estimated its baseline from 5 samples. The README claimed "90–95% detection" with no false-alarm rate. | Probe: 100/100 alarms on noise. | CFAR, CUSUM, and step GLRT with thresholds calibrated empirically to a stated false-alarm rate (E1, Phase 3). | `test_c5_false_alarm_rate_on_glint_free_background`, `test_c5_detectors_ignore_constant_signal`, `test_c5_detection_does_not_precede_the_event` |
| C6 | **Weighted least squares diverged yet reported success.** The Jacobian sign was wrong, so each Gauss-Newton step moved the wrong way. | Probe: 3.9e13 m error with `success=True`. | Whitened Levenberg-Marquardt with analytic Jacobians and an observability check (Phase 1). In Phase 5 a χ² fit test was added, after a fix far from the truth also reported "converged" (§ Later findings). | `test_c6_wls_recovers_noiseless_position` |
| C7 | **"Chan's algorithm" was not Chan-Ho.** It was an incorrect linearization that dropped the unknown reference range. The README cited Chan & Ho. | Probe: 6 km error on noiseless data. | Two-stage Chan-Ho (Chan & Ho 1994), used to initialize ML (Phases 1, 4). | `test_c7_chan_recovers_noiseless_position` |
| C8 | **The headline result was a red flag.** 100% test accuracy with a perfectly diagonal confusion matrix. The classes were trivially separable, and the model used only one of four input features. | `outputs/evaluation/test_results.json` | A hard task with domain-shift test sets, feature baselines, 5 seeds, bootstrap CIs, and calibration. Shipped TCN: macro-F1 0.806 on test, 0.58–0.74 under shift (E2, E3). | `test_c8_reported_results_are_honest`, `test_pipeline_threshold_matches_e1_report` |

## Serious: algorithmic and modeling flaws

| ID | Defect in v1 | Fix | Guard |
|---|---|---|---|
| H1 | Fusion prediction ignored the elapsed time; the Kalman filter had a fixed `dt = 1`. | Timestamp-driven prediction (Phase 1). | `test_h1_fusion_prediction_uses_elapsed_time` |
| H2 | The association gate used the track covariance instead of the innovation covariance S = HPHᵀ + R, with a gate of "2000 m" in unitless Mahalanobis units, so every measurement passed. | NIS against χ²(dim); GNN by `linear_sum_assignment` (Phase 1). | `test_h2_gate_rejects_distant_measurement` |
| H3 | The README said "covariance intersection". The code fused positions as independent and then ran a Kalman update on top, counting information twice. | Centralized sequential updates; naive vs CI track-to-track fusion compared in E6 ([ADR 0001](adr/0001-fusion-architecture.md)) (Phases 1, 5). | `test_t2t_fusion_consistency_under_correlation` |
| H4 | Process noise used dt⁴/6 instead of dt⁴/4, which made Q indefinite. The covariance update was not in Joseph form. | Continuous white-noise-acceleration Q; Joseph-form update with symmetrization (Phase 1). | `test_h4_process_noise_is_positive_semidefinite` |
| H5 | Four inconsistent GDOP implementations used GPS pseudorange geometry, and returned a magic 999 on failure. | One DOP for differenced TDOA geometry, plus the CRLB (Phases 1, 4). | `test_h5_unobservable_geometry_raises_instead_of_magic_value` |
| H6 | RF fixes had no covariance (fusion used a default 100 m²); velocity was "estimated" from TDOA alone, where it is unobservable; TDOA pairs were treated as independent. | Reference-sensor TDOAs with the full (I + 11ᵀ) covariance; velocity only with FDOA; consider covariance for clock bias and survey error (Phases 1, 4, 6). | `test_velocity_is_none_without_fdoa`, `test_is_reference_invariant_and_scales_with_noise`, `test_rf_measurement_keeps_position_velocity_cross_covariance` |
| H7 | Confidence and "quality" scores came from uncalibrated constants (×0.95 decay, +0.2 per hit). | Removed. Tracks report a calibrated covariance and a calibrated class posterior (Phases 1, 5, 6). | Statistical tests: `test_floor_keeps_long_tracks_consistent`, NEES/NIS tests |
| H8 | The Phase 2 pipeline used random track positions and imported a module that did not exist. | Deleted; replaced by the configurable `LocantPipeline` (Phases 1, 6). | `tests/integration/test_pipeline.py` |
| H9 | The RF generator modulated a 10 GHz "baseband" carrier sampled at 100 MHz, which aliases; geolocation never used it. | True complex baseband waveforms with delay/Doppler operators (Phase 2). **Open:** TDOA/FDOA are still simulated from geometry, not estimated from waveforms by cross-ambiguity (Phase 8 stretch). | `TestWaveforms` in `tests/unit/sim/test_rf.py` |
| H10 | The README's quick-starts did not match the APIs; its performance numbers came from no script; links were broken. | README snippets executed; every results table generated from `reports/metrics.json` (Phases 1, 7). | `tests/regression/test_metrics_sync.py` |
| H11 | *(Found in Phase 1.)* The temporal-difference detector required three consecutive above-threshold differences, which a launch rise at 1 Hz cannot produce. | Replaced by CFAR/CUSUM/GLRT (Phase 3). | `test_h11_detectors_catch_target_events` |

## Engineering

| Problem in v1 | Now |
|---|---|
| No `pyproject.toml`; `sys.path` hacks everywhere; top-level package named `src` | Installable `locant` package (src layout); console script `locant` |
| Unpinned dependencies, heavy unused ones (geopandas, fastapi, streamlit, …), a used one missing | Pinned `requirements*.txt`, conda `environment.yml`, only what is imported |
| Global `np.random`, nothing seeded | Injected `numpy.random.Generator` everywhere; `SeedSequence` streams per sample |
| About 12 bare `except:` blocks; silent "success" values (zeros, 999, 1e6·I) | Typed exceptions; the pipeline counts and logs failures |
| GDOP ×4, assignment ×2, device selection ×3, normalization ×4 | One implementation each; one preprocessing function shared by training and inference |
| Hard-coded constants and paths; `print` for logging | `PipelineConfig` (validated YAML); structured logging ([ADR 0002](adr/0002-configuration-cli-and-run-registry.md)) |
| O(N·W) Python loops | Vectorized hot paths with latency budgets in CI |
| `torch.load` without `weights_only`; batch predictions misaligned on error | `weights_only=True`; batched inference |
| No lint, type checking, CI, or pre-commit | ruff, strict mypy, GitHub Actions, pre-commit |
| `docs/`, `assets/`, notebooks git-ignored | Documentation versioned under `docs/` |
| Modules named after the build timeline (`phase2_pipeline`, `phase3_pipeline`) | Named by domain (`pipeline`, `fusion`, `tracking`, …) |

## Later findings

The audit was not the last word. Building the evidence surfaced more
defects, each fixed and recorded the same way:

- **Chan-Ho could pick the mirrored root (Phase 5).** The solver then ran
  off along a hyperboloid asymptote and reported "converged" at 6×10¹¹ m.
  Fixed with a χ² fit test, a multi-start from the receiver centroid, and a
  unit-invariant conditioning check
  (`test_restarts_when_chan_ho_picks_the_wrong_root`).
- **Stereo ghosts (Phase 5).** Two satellites can pair rays from different
  targets: 224 of 8,595 pairs in the benchmark. Fixed by deferring ambiguous
  pairs (E5).
- **One ray per scan (Phase 5).** All satellites shared one measurement
  source, so a track could take only one satellite's ray per scan. Fixed by
  giving each satellite its own source.
- **Classifier outside its domain (Phase 5).** Once a window had slid past a
  launch's onset, the classifier labeled the launch "aircraft". Fixed by
  using class evidence only inside the trained onset range (E6).
- **Overconfident RF tracks (Phases 5–6).** A time-correlated RF bias makes
  tracks overconfident as they age. Fixed with a bias floor carried from
  the solver to the reported covariance (E7).
- **Padding created false alarms (Phase 6).** Padding short windows before
  CFAR detection created false alarms. Detection now runs on the raw window
  (`test_short_windows_are_padded_without_false_alarms`).
- **A wrong number in the docs (Phase 7).** A hand-copied README table gave
  47% for the step-GLRT detection rate; the report says 49%. Fixed for good
  by generating every results table from the reports.
