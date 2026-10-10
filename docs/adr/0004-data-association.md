# ADR 0004: Global nearest neighbor association, with ambiguity deferred

- **Status:** accepted (Phase 5)
- **Evidence:** E5 ([fusion write-up](../fusion.md)), `locant.tracking`

## Context

Each scan, the tracker must decide which measurement belongs to which
track. Measurements are RF fixes, OPIR stereo positions, and single-satellite
rays. The standard options trade cost for robustness:

- **Global nearest neighbor (GNN):** one hard, globally optimal assignment
  per scan (Hungarian algorithm).
- **JPDA:** a probabilistic, soft assignment averaged over hypotheses.
- **MHT:** keep alternative assignment histories and decide later.

## Decision

GNN, made robust by four measures:

1. **A χ² gate on the innovation.** Gating uses NIS = rᵀS⁻¹r against
   χ²(dim) at 0.99, with S = HPHᵀ + R (v1 gated on P, audit H2). Each
   (source, dimension) group is gated and assigned separately. For IMM
   tracks the NIS is the minimum over modes, so a maneuver can be gated by
   the mode that predicts it.
2. **Ambiguity resolved before it reaches the tracker.** Two satellites can
   pair rays from different targets into stereo "ghosts". A stereo pair is
   kept only if neither ray gates with any other ray. Ambiguous rays become
   angle-only updates, which the tracker resolves against predicted tracks.
3. **M-of-N confirmation (3 of 5).** A false association has to repeat
   before it becomes a reported track.
4. **Duplicate merging.** Two tracks that alternate in claiming one target's
   measurement are merged when their positions agree at χ²(3) = 0.9.

## Rationale (E5, 10 seeds)

- **Low false-track rate.** Confirmed false tracks run at 0.09 per scan
  without clutter, and 0.21 per scan with 5 clutter rays per satellite per
  scan.
- **Stable identities.** 0.3 identity switches per 180 s run.
- **The ghosts were the real association problem.** They were 2.6% of GNN
  pairings. Deferring them removed all of them and cut false tracks from
  0.31 to 0.09 per scan.
- **Low target density.** At 6 targets over about 100 km, gates rarely
  overlap, which is where JPDA and MHT earn their cost.
- **Cost.** A 30-target scan with 90 measurements takes 12 ms (`make bench`).
  JPDA's hypothesis enumeration, or MHT's tree, would cost orders of
  magnitude more for no measured benefit here.

## Consequences

- **Crossing or closely spaced targets will swap identities**, since GNN
  commits to one assignment per scan. Revisit when targets come within
  about 2 gate widths of each other often (dense raids, formation flight).
  JPDA is listed as a Phase 8 stretch item, and E5's metrics (identity
  switches, fragmentation, GOSPA) are its benchmark.
- **Measurements stay unambiguous by construction**, because ambiguity is
  handled upstream: the stereo deferral, and one source per satellite.
