"""E4: geolocation accuracy against the Cramér-Rao bound, geometry, and robustness.

Questions:
1. Does the ML TDOA estimator attain the CRLB, and where does it stop
   (threshold effect at high noise)? How does closed-form Chan-Ho compare?
2. How does accuracy depend on receiver count and network geometry (DOP maps)?
3. What do unmodeled clock bias and receiver survey error do to accuracy and
   to the reported covariance, and does the consider covariance restore
   consistency?
4. Does joint TDOA/FDOA velocity estimation attain its bound?

All Monte Carlo draws use fixed seeds; no dataset is needed.

    python -m experiments.e4_geolocation [--quick]
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from experiments.common import (
    MUTED,
    ROOT,
    SERIES,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    RunOptions,
    run_experiment,
    style,
)
from locant.core import (
    GeometryError,
    InsufficientMeasurementsError,
    mean_nees_bounds,
    nees,
)
from locant.core.constants import SPEED_OF_LIGHT
from locant.eval import write_report
from locant.geolocation import (
    Receiver,
    SystematicErrors,
    chan_ho,
    rms_bound,
    simulate_fdoa,
    simulate_tdoa,
    solve_tdoa,
    solve_tdoa_fdoa,
    tdoa_crlb,
    tdoa_dop,
    tdoa_fdoa_crlb,
)
from locant.sim.rf.network import ReceiverModel, RFNetwork, default_receiver_network
from locant.sim.trajectories import Stationary

REPORT_DIR = ROOT / "reports" / "phase4"
EMITTER = np.array([5_000.0, 5_000.0, 500.0])
VELOCITY = np.array([100.0, 50.0, 0.0])
CARRIER = 1e9
SOLVER_ERRORS = (GeometryError, InsufficientMeasurementsError, ValueError)
# Sequential blue ramp from the reference palette (light → dark = small → large error).
BLUE_RAMP = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]


def networks() -> dict[str, list[Receiver]]:
    base = default_receiver_network()
    xy = np.array([r.position[:2] for r in base])
    return {
        "mixed_altitude": base,
        "ground_only": [
            Receiver(i, np.array([*p, z]))
            for i, (p, z) in enumerate(zip(xy, (0, 50, 100, 30, 80), strict=True))
        ],
        "compact_2km": [
            Receiver(r.id, np.array([*(r.position[:2] * 0.2), r.position[2]]))
            for r in base
        ],
    }


def ring_network(n: int) -> list[Receiver]:
    """n-1 receivers on a 10 km ring at varied altitude, plus a stand-off platform."""
    angles = np.linspace(0.0, 2.0 * np.pi, n - 1, endpoint=False)
    altitudes = np.resize([200.0, 1_500.0, 600.0, 2_500.0], n - 1)
    ring = [
        Receiver(i, np.array([10_000 * np.cos(a), 10_000 * np.sin(a), z]))
        for i, (a, z) in enumerate(zip(angles, altitudes, strict=True))
    ]
    return [*ring, Receiver(n - 1, np.array([0.0, 0.0, 6_000.0]))]


def _stats(
    errors: list[np.ndarray], nees_values: list[float], dim: int
) -> dict[str, Any]:
    e = np.asarray(errors)
    low, high = mean_nees_bounds(dim, max(len(nees_values), 1), confidence=0.95)
    return {
        "runs": len(errors),
        "rmse": float(np.sqrt(np.mean(np.sum(e**2, axis=1))))
        if len(e)
        else float("nan"),
        "bias_norm": float(np.linalg.norm(e.mean(axis=0))) if len(e) else float("nan"),
        "median_error": float(np.median(np.linalg.norm(e, axis=1)))
        if len(e)
        else float("nan"),
        "nees_mean": float(np.mean(nees_values)) if nees_values else float("nan"),
        "nees_median": float(np.median(nees_values)) if nees_values else float("nan"),
        "nees_95_bounds": [low, high],
    }


def noise_sweep(
    runs: int, sigmas_ns: list[float], rng: np.random.Generator
) -> dict[str, Any]:
    receivers = default_receiver_network()
    out: dict[str, Any] = {}
    for sigma_ns in sigmas_ns:
        sigma = sigma_ns * 1e-9
        ml_err, ml_nees, ch_err, ch_nees, failures = [], [], [], [], 0
        for _ in range(runs):
            m = simulate_tdoa(EMITTER, receivers, sigma, rng)
            try:
                ml = solve_tdoa(receivers, m)
                ch = chan_ho(receivers, m)
            except SOLVER_ERRORS:
                failures += 1
                continue
            ml_err.append(ml.position - EMITTER)
            ml_nees.append(nees(ml.position - EMITTER, ml.position_covariance))
            ch_err.append(ch.position - EMITTER)
            ch_nees.append(nees(ch.position - EMITTER, ch.position_covariance))
        bound = rms_bound(tdoa_crlb(EMITTER, receivers, sigma))
        out[f"{sigma_ns:g}ns"] = {
            "toa_std_ns": sigma_ns,
            "crlb_rms": bound,
            "failures": failures,
            "ml": _stats(ml_err, ml_nees, 3),
            "chan_ho": _stats(ch_err, ch_nees, 3),
        }
    return out


def receiver_count(
    runs: int, counts: list[int], rng: np.random.Generator
) -> dict[str, Any]:
    emitter = np.array([3_000.0, 2_000.0, 500.0])
    sigma = 10e-9
    out: dict[str, Any] = {}
    for n in counts:
        receivers = ring_network(n)
        ml_err, ml_nees, ch_err, ch_nees = [], [], [], []
        for _ in range(runs):
            m = simulate_tdoa(emitter, receivers, sigma, rng)
            ml, ch = solve_tdoa(receivers, m), chan_ho(receivers, m)
            ml_err.append(ml.position - emitter)
            ml_nees.append(nees(ml.position - emitter, ml.position_covariance))
            ch_err.append(ch.position - emitter)
            ch_nees.append(nees(ch.position - emitter, ch.position_covariance))
        out[str(n)] = {
            "crlb_rms": rms_bound(tdoa_crlb(emitter, receivers, sigma)),
            "ml": _stats(ml_err, ml_nees, 3),
            "chan_ho": _stats(ch_err, ch_nees, 3),
        }
    return out


def dop_maps(step_m: float) -> dict[str, Any]:
    xs = np.arange(-20_000.0, 30_000.0 + step_m, step_m)
    ys = np.arange(-20_000.0, 30_000.0 + step_m, step_m)
    sigma = 10e-9
    maps: dict[str, Any] = {}
    for name, receivers in networks().items():
        positions = np.array([r.position for r in receivers])
        grid = np.full((ys.size, xs.size), np.nan)
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                try:
                    grid[i, j] = (
                        tdoa_dop(np.array([x, y, 500.0]), positions).gdop
                        * SPEED_OF_LIGHT
                        * sigma
                    )
                except GeometryError:
                    continue
        inside = (np.abs(xs[None, :] - 5_000) <= 5_000) & (
            np.abs(ys[:, None] - 5_000) <= 5_000
        )
        maps[name] = {
            "receivers": positions,
            "rms_bound_m": grid,
            "median_inside_10km_box_m": float(
                np.nanmedian(np.where(inside, grid, np.nan))
            ),
            "median_whole_grid_m": float(np.nanmedian(grid)),
        }
    return {
        "x": xs,
        "y": ys,
        "toa_std_ns": 10.0,
        "emitter_altitude_m": 500.0,
        "networks": maps,
    }


def systematics(runs: int, rng: np.random.Generator, quick: bool) -> dict[str, Any]:
    def sweep(
        network: list[Receiver], clock_ns: list[float], survey_m: list[float]
    ) -> list[dict[str, Any]]:
        rows = []
        for clock, survey in [(c, 0.0) for c in clock_ns] + [
            (0.0, s) for s in survey_m if s > 0
        ]:
            sys = SystematicErrors(
                receiver_position_std=survey, clock_bias_std=clock * 1e-9
            )
            errors, naive, consider, reported = [], [], [], []
            for _ in range(runs):
                net = RFNetwork(
                    [
                        ReceiverModel(
                            r.id,
                            Stationary(r.position),
                            toa_std=10e-9,
                            clock_bias_std=clock * 1e-9,
                            position_error_std=survey,
                        )
                        for r in network
                    ],
                    rng,
                )
                scan = net.scan(0.0, EMITTER, np.zeros(3), rng)
                try:
                    a = solve_tdoa(scan.receivers, scan.tdoa)
                    b = solve_tdoa(scan.receivers, scan.tdoa, systematic=sys)
                except SOLVER_ERRORS:
                    continue
                errors.append(a.position - EMITTER)
                naive.append(nees(a.position - EMITTER, a.position_covariance))
                consider.append(nees(b.position - EMITTER, b.position_covariance))
                reported.append(
                    (rms_bound(a.position_covariance), rms_bound(b.position_covariance))
                )
            stats = _stats(errors, naive, 3)
            rows.append(
                {
                    "clock_bias_ns": clock,
                    "survey_error_m": survey,
                    "rmse": stats["rmse"],
                    "reported_rms_naive": float(np.mean([r[0] for r in reported])),
                    "reported_rms_consider": float(np.mean([r[1] for r in reported])),
                    "nees_naive_mean": stats["nees_mean"],
                    "nees_consider_mean": float(np.mean(consider)),
                    "nees_consider_median": float(np.median(consider)),
                    "nees_95_bounds": stats["nees_95_bounds"],
                    "runs": stats["runs"],
                }
            )
        return rows

    if quick:
        return {"mixed_altitude": sweep(default_receiver_network(), [0, 10], [5])}
    return {
        "mixed_altitude": sweep(
            default_receiver_network(), [0, 3, 10, 30, 100], [2, 5, 10, 20, 50]
        ),
        "ground_only": sweep(networks()["ground_only"], [0, 10], [5, 20]),
    }


def hybrid(
    runs: int, frequency_stds: list[float], rng: np.random.Generator
) -> dict[str, Any]:
    velocities = np.array(
        [[0, 0, 0], [30, 0, 0], [0, -40, 0], [20, 20, 0], [-30, 10, 0]], dtype=float
    )
    receivers = [
        Receiver(r.id, r.position, v)
        for r, v in zip(default_receiver_network(), velocities, strict=True)
    ]
    out: dict[str, Any] = {}
    for f_std in frequency_stds:
        pos_err, vel_err, nees6 = [], [], []
        for _ in range(runs):
            tdoa = simulate_tdoa(EMITTER, receivers, 10e-9, rng)
            fdoa = simulate_fdoa(EMITTER, VELOCITY, receivers, CARRIER, f_std, rng)
            r = solve_tdoa_fdoa(receivers, tdoa, fdoa)
            assert r.velocity is not None
            pos_err.append(r.position - EMITTER)
            vel_err.append(r.velocity - VELOCITY)
            nees6.append(
                nees(np.concatenate([pos_err[-1], vel_err[-1]]), r.state_covariance)
            )
        bound = tdoa_fdoa_crlb(EMITTER, VELOCITY, receivers, 10e-9, f_std, CARRIER)
        low, high = mean_nees_bounds(6, runs, confidence=0.95)
        out[f"{f_std:g}Hz"] = {
            "frequency_std_hz": f_std,
            "position_rmse": float(
                np.sqrt(np.mean(np.sum(np.square(pos_err), axis=1)))
            ),
            "position_crlb_rms": rms_bound(bound[:3, :3]),
            "velocity_rmse": float(
                np.sqrt(np.mean(np.sum(np.square(vel_err), axis=1)))
            ),
            "velocity_crlb_rms": rms_bound(bound[3:, 3:]),
            "nees6_mean": float(np.mean(nees6)),
            "nees6_95_bounds": [low, high],
        }
    return out


def run(options: RunOptions) -> dict[str, Any]:
    quick = options.quick
    rng = np.random.default_rng(4)
    report = {
        "experiment": "E4 geolocation",
        "quick": quick,
        "emitter_position_m": EMITTER,
        "noise_sweep": noise_sweep(
            20 if quick else 500, [10.0] if quick else [1, 3, 10, 30, 100, 300], rng
        ),
        "receiver_count": receiver_count(
            20 if quick else 300, [5] if quick else [5, 6, 8, 12], rng
        ),
        "systematics": systematics(20 if quick else 300, rng, quick),
        "hybrid_tdoa_fdoa": hybrid(
            20 if quick else 300, [1.0] if quick else [0.3, 1, 3, 10], rng
        ),
    }
    maps = dop_maps(10_000.0 if quick else 500.0)
    report["dop_maps"] = {
        name: {k: v for k, v in m.items() if k != "rms_bound_m"}
        for name, m in maps["networks"].items()
    }
    write_report(options.report_dir / "e4_geolocation.json", report)
    if not quick:
        _figures(options.report_dir / "figures", report, maps)
    return report


def _figures(out: Path, report: dict[str, Any], maps: dict[str, Any]) -> None:
    style()
    out.mkdir(parents=True, exist_ok=True)

    sweep = report["noise_sweep"]
    sigma = np.array([v["toa_std_ns"] for v in sweep.values()])
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    ax = axes[0]
    ax.plot(
        sigma,
        [v["crlb_rms"] for v in sweep.values()],
        color=MUTED,
        linestyle="--",
        linewidth=1.5,
        label="CRLB",
    )
    for color, key, label in (
        (SERIES[0], "ml", "ML (Chan-Ho init)"),
        (SERIES[1], "chan_ho", "Chan-Ho closed form"),
    ):
        ax.plot(
            sigma,
            [v[key]["rmse"] for v in sweep.values()],
            color=color,
            marker="o",
            markersize=7,
            linewidth=2.0,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("per-receiver timing noise (ns)")
    ax.set_ylabel("position RMSE (m)")
    ax.legend(loc="upper left")
    ax.set_title("RMSE vs CRLB", loc="left", fontsize=10)
    ax = axes[1]
    low, high = next(iter(sweep.values()))["ml"]["nees_95_bounds"]
    ax.axhspan(low, high, color=MUTED, alpha=0.25, linewidth=0)
    ax.axhline(3.0, color=MUTED, linewidth=1.0)
    for color, key, label in (
        (SERIES[0], "ml", "ML"),
        (SERIES[1], "chan_ho", "Chan-Ho"),
    ):
        ax.plot(
            sigma,
            [v[key]["nees_mean"] for v in sweep.values()],
            color=color,
            marker="o",
            markersize=7,
            linewidth=2.0,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_ylim(2.0, 5.0)
    ax.set_xlabel("per-receiver timing noise (ns)")
    ax.set_ylabel("mean NEES (consistent ≈ 3, band = 95%)")
    ax.legend(loc="upper left")
    ax.set_title("covariance consistency", loc="left", fontsize=10)
    fig.suptitle(
        "E4 TDOA accuracy, 5-receiver mixed-altitude network",
        x=0.01,
        ha="left",
        color=TEXT_PRIMARY,
    )
    fig.tight_layout()
    fig.savefig(out / "e4_rmse_vs_crlb.png", dpi=150)
    plt.close(fig)

    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    names = list(maps["networks"])
    fig, axes = plt.subplots(1, len(names), figsize=(13.0, 4.4), sharey=True)
    norm = LogNorm(vmin=3.0, vmax=3_000.0)
    image = None
    titles = {
        "mixed_altitude": "mixed altitude (default)",
        "ground_only": "ground only (z ≤ 100 m)",
        "compact_2km": "compact (2 km aperture)",
    }
    for ax, name in zip(axes, names, strict=True):
        m = maps["networks"][name]
        extent = (
            maps["x"][0] / 1e3,
            maps["x"][-1] / 1e3,
            maps["y"][0] / 1e3,
            maps["y"][-1] / 1e3,
        )
        image = ax.imshow(
            np.clip(m["rms_bound_m"], 3.0, 3_000.0),
            origin="lower",
            extent=extent,
            cmap=cmap,
            norm=norm,
        )
        ax.grid(False)
        rx = m["receivers"]
        ax.plot(
            rx[:, 0] / 1e3,
            rx[:, 1] / 1e3,
            "^",
            color="#ffffff",
            markeredgecolor=TEXT_PRIMARY,
            markersize=8,
        )
        ax.set_title(
            f"{titles[name]}\nmedian {m['median_inside_10km_box_m']:.0f} m inside the network",
            loc="left",
            fontsize=9,
        )
        ax.set_xlabel("east (km)")
    axes[0].set_ylabel("north (km)")
    assert image is not None
    bar = fig.colorbar(image, ax=axes, shrink=0.85)
    bar.set_label("RMS position error bound at 10 ns (m)", color=TEXT_SECONDARY)
    fig.suptitle(
        "E4 TDOA error bound by emitter location (emitter at 500 m altitude; triangles = receivers)",
        x=0.01,
        ha="left",
        color=TEXT_PRIMARY,
    )
    fig.savefig(out / "e4_dop_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    rows = report["systematics"]["mixed_altitude"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    for ax, key, xlabel in (
        (axes[0], "clock_bias_ns", "clock bias per receiver (ns, 1σ)"),
        (axes[1], "survey_error_m", "survey error per axis (m, 1σ)"),
    ):
        other = "survey_error_m" if key == "clock_bias_ns" else "clock_bias_ns"
        selected = [r for r in rows if r[other] == 0]
        x = np.array([max(r[key], 0.5) for r in selected])
        low, high = selected[0]["nees_95_bounds"]
        ax.axhspan(low, high, color=MUTED, alpha=0.25, linewidth=0)
        ax.plot(
            x,
            [r["nees_naive_mean"] for r in selected],
            color=SERIES[1],
            marker="o",
            markersize=7,
            linewidth=2.0,
            label="ignored (naive covariance)",
        )
        ax.plot(
            x,
            [r["nees_consider_mean"] for r in selected],
            color=SERIES[0],
            marker="o",
            markersize=7,
            linewidth=2.0,
            label="consider covariance",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel + "  (0 plotted at 0.5)")
    axes[0].set_ylabel("mean NEES (consistent ≈ 3)")
    axes[0].legend(loc="upper left")
    fig.suptitle(
        "E4 systematic receiver errors: consistency with and without the consider covariance",
        x=0.01,
        ha="left",
        color=TEXT_PRIMARY,
    )
    fig.tight_layout()
    fig.savefig(out / "e4_systematics.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_experiment(
        "e4_geolocation",
        run,
        (__doc__ or "e4_geolocation").splitlines()[0],
        report_dir=REPORT_DIR,
    )
