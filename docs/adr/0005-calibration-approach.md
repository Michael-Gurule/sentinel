# ADR 0005: Calibrate empirically, report sets, and stay in the trained domain

- **Status:** accepted (Phases 3 and 5)
- **Evidence:** E1, E3, E6; [model card](../model_card.md)

## Context

Every decision Locant reports rests on a probability or a threshold: an
alarm, a class label, a set of plausible classes, a track's class posterior,
a covariance. Each must mean what it says. v1 reported heuristic
"confidence" scores (audit H7) and quoted a detection rate with no
false-alarm rate (C5).

## Decision

1. **Calibrate thresholds empirically, not from white-noise theory.**
   Detector thresholds are set from background windows at a stated
   false-alarm rate (E1). The analytic 1% threshold actually gave CFAR
   about 33% false alarms on realistic background, because clutter is
   correlated and σ is estimated.
2. **Temperature-scale the classifier.** One parameter T is fit on
   validation NLL. It changes no decision and corrects global over- or
   under-confidence.
3. **Report conformal prediction sets (LAC), not just a label.** Split
   conformal at α = 0.1 guarantees 90% coverage on exchangeable data. The
   shipped set uses LAC (least ambiguous classifier) scores.
4. **Pool class evidence on tracks with a tempered log-linear rule,** with
   weight w = 0.3. Consecutive windows overlap, so their outputs are not
   independent evidence.
5. **Use the classifier only inside its trained domain.** Class evidence
   reaches a track only when the detected onset lies 2–40 s into the
   window, as in training (E6).

## Rationale

- **LAC over APS.** APS sets were nearly uninformative: on test they
  covered 99.8% of cases with 3.4 of 5 labels. LAC achieved the 90%
  target (90.2%) with 1.32 labels.
- **Temperature scaling is cheap and safe**, but small here. Test ECE was
  already 0.020, and became 0.019. The fitted T values were 0.93–1.12
  across seeds.
- **Calibration does not survive domain shift.** ECE rises to 0.09–0.12
  under shift, and conformal coverage falls to 73–83%. That is reported,
  not hidden. Conformal guarantees assume exchangeability, which shift
  breaks.
- **The domain of validity mattered more than the pooling rule** (E6). On
  tracks, applying the classifier to every window drove launches to
  "aircraft" once the window slid past the onset: final label accuracy
  68% (w = 1). Restricting it to the trained onset range gave 100% and
  cut the posterior ECE from about 0.17 to 0.06. Within the trained
  range, the pooling weight hardly mattered.
- **Energy OOD scores** flag unseen faint event types (AUROC 0.71–0.79)
  but not unseen bright ones (0.38–0.40). They are reported as a
  limitation, not used as a gate.

## Consequences

- **Recalibrate after a domain change.** On new data, refit the threshold,
  T, and the conformal quantile on a calibration split from the new domain
  before trusting any of them (the E1/E3 procedures are scripted).
- **The domain gate is explicit.** It is a single configuration field
  (`classification.onset_range_s`), tied to the dataset's onset prior.
  Change both together.
- **Track posteriors can wait.** A track can carry no class posterior for
  its first ~24 s. Labels arrive later, but they are reliable.
