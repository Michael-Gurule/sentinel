# ADR 0006: One local ENU frame and SI units

- **Status:** accepted (Phase 2)
- **Code:** `locant.sim.geometry`, `locant.geolocation`, `locant.tracking`

## Context

v1 mixed latitude/longitude helpers that nothing used with unlabeled
"meters" in an undefined frame. The OPIR line-of-sight, RF geolocation, and
tracking code must agree on frames and units. A silent mismatch, such as
ECEF against ENU or seconds against nanoseconds, produces plausible but
wrong numbers that no test of a single module would catch.

## Decision

**Frames**

| Frame | Use |
|---|---|
| WGS-84 geodetic (lat, lon, height) | Scenario origin and event sites in configuration |
| ECEF (m) | Satellite and target positions in the simulator; ECI only inside Keplerian propagation |
| **Local ENU (m)** about the scenario origin (`LocalFrame`) | Everything after the simulator: receivers, RF fixes, OPIR sensor positions and lines of sight, track states |

- **State vectors** are `[x, y, z, vx, vy, vz]` in ENU (m, m/s).
  Covariances use the same order.
- **Lines of sight** are unit vectors in ENU, from the satellite to the
  target. Angle errors are per axis, in radians, about the measured ray.
- **The flat-plane approximation** (altitude ≈ ENU z) is used only by
  altitude intersection. It is valid within a few hundred kilometres of the
  origin.

**Units**: SI throughout, with these quantities named in the code.

| Quantity | Unit |
|---|---|
| Time | s (scenario time; `t = 0` at scenario start) |
| TDOA / clock bias | s (configuration uses ns, converted at load) |
| FDOA / LO offset | Hz |
| Range differences, range-rate differences (internal) | m, m/s |
| Radiant intensity (events) | W/sr |
| Irradiance at the aperture (OPIR samples) | pW/m² |
| Angles | rad (configuration may use degrees, converted at load) |

Configuration fields that use other units say so in their names
(`clock_bias_ns`, `toa_std_ns`, `raan_deg`, `onset_s`).

## Consequences

- **One frame for every measurement.** OPIR, RF, and track states share one
  frame, so fusion never converts coordinates. Moving the origin (a new
  scenario) moves everything consistently.
- **Some tests check frames directly.** WGS-84 round trips, local-frame
  round trips and the up axis, GEO and Molniya orbits, and line-of-sight
  geometry all have tests (`tests/unit/sim/test_geometry.py`).
- **Wide-area operation would need a different frame.** Scenarios spanning
  thousands of kilometres would need ECEF track states, or a per-region
  frame with handover. That is out of scope here.
