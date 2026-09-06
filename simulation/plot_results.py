#!/usr/bin/env python3
"""Publication-quality telemetry plots for SIL benchmark runs.

Figures per input CSV (saved as PNG next to it, or into --out):
  1. beta_r_phase.png  - sideslip/yaw-rate phase portrait with the mu-g
                         friction-ellipse envelope; trajectory colored by time.
  2. barriers.png      - h_obs / h_road / h_drift over time with the zero line.
  3. speed_ttc_mu.png  - speed + TTC (log-ish symlog) + mu_hat vs mu_true.
  4. tracking_cmd.png  - lateral position vs nominal steer/safe steer.

Usage:
  python3 simulation/plot_results.py runs/bench/highway_cut_in_lqr_cbf_mock.csv
  python3 simulation/plot_results.py runs/bench/*.csv --out figs/
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import telemetry as tm  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
import numpy as np  # noqa: E402

G = 9.81


def _finite(xs):
    return np.array([v for v in xs if v == v], dtype=float)


def _colormap_line(ax, x, y, t, label=None):
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="viridis", linewidth=1.6)
    lc.set_array(np.asarray(t[:-1] if len(t) == len(x) else t))
    ax.add_collection(lc)
    ax.autoscale_view()
    if label:
        ax.plot([], [], label=label)
    return lc


def fig_phase(data, mu_ref, path):
    vx = np.array(data["vx"], dtype=float)
    vy = np.array(data["vy"], dtype=float)
    r = np.array(data["r"], dtype=float)
    t = np.array(data["time"], dtype=float)
    beta = np.degrees(np.arctan2(vy, np.maximum(vx, 1.0)))
    ok = np.isfinite(beta) & np.isfinite(r) & np.isfinite(t)
    beta, r, t = beta[ok], r[ok], t[ok]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    # Friction-ellipse envelope: (vx*r)^2 + ay^2 <= (mu g)^2, with
    # ay ~= vx*(r + beta_dot)-ish; draw the quasi-steady box instead:
    # |beta| <= atan(mu*g / vx^2 * L-ish)... use the drift envelope directly:
    # admissible yaw rate |r| <= mu*g / max(vx, eps), and sideslip guide lines.
    v_ref = float(np.median(vx[np.isfinite(vx)])) if np.any(np.isfinite(vx)) else 12.0
    r_lim = mu_ref * G / max(v_ref, 2.0)
    axlim_r = max(float(np.max(np.abs(r))) * 1.15, r_lim * 1.3, 0.1)
    axlim_b = max(float(np.max(np.abs(beta))) * 1.15, 3.0)
    ax.axhspan(-r_lim, r_lim, color="green", alpha=0.08, label=f"drift envelope |r|<={r_lim:.2f}")
    ax.axhline(r_lim, color="green", ls="--", lw=1)
    ax.axhline(-r_lim, color="green", ls="--", lw=1)
    _colormap_line(ax, beta, r, t)
    ax.set_xlabel("sideslip beta [deg]")
    ax.set_ylabel("yaw rate r [rad/s]")
    ax.set_title("beta-r phase portrait (color = time [s])")
    ax.set_xlim(-axlim_b, axlim_b)
    ax.set_ylim(-axlim_r, axlim_r)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.colorbar(ax.collections[0], ax=ax, label="t [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_barriers(data, path):
    t = _finite(data["time"])
    fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    for ax, key, name in zip(
        axes,
        ("h_obs", "h_road", "h_drift", "h_gap"),
        ("obstacle", "road", "drift", "gap (regulating, not gated)"),
    ):
        y = np.array(data[key], dtype=float)
        m = np.isfinite(y)
        ax.plot(t[m], y[m], lw=1.2, label=f"h_{name}")
        ax.axhline(0.0, color="red", ls="--", lw=1, label="boundary")
        ax.set_ylabel(f"h_{name}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("t [s]")
    fig.suptitle("Control barrier values (safe iff >= 0)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_speed_ttc_mu(data, path):
    t = np.array(data["time"], dtype=float)
    vx = np.array(data["vx"], dtype=float)
    ttc = np.array([v if v == v else np.inf for v in data["min_ttc"]], dtype=float)
    mu_hat = np.array(data["mu_hat"], dtype=float)
    mu_true = np.array(data["mu_true"], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True)
    m = np.isfinite(t) & np.isfinite(vx)
    axes[0].plot(t[m], vx[m] * 3.6, lw=1.2)
    axes[0].set_ylabel("speed [km/h]")
    axes[0].grid(True, alpha=0.3)
    fin = np.isfinite(ttc) & (ttc < 1e8)
    axes[1].plot(t[fin], np.minimum(ttc[fin], 30.0), lw=1.2, color="darkorange")
    axes[1].axhline(1.5, color="red", ls="--", lw=1, label="1.5 s warn")
    axes[1].set_ylabel("min TTC [s]")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    mh = np.isfinite(mu_hat)
    mt = np.isfinite(mu_true)
    axes[2].plot(t[mh], mu_hat[mh], lw=1.2, label="mu_hat (UKF)")
    if np.any(mt):
        axes[2].plot(t[mt], mu_true[mt], lw=1.0, ls="--", label="mu_true")
    axes[2].set_ylabel("friction mu")
    axes[2].set_xlabel("t [s]")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    fig.suptitle("Speed, time-to-collision, friction adaptation")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_tracking_cmd(data, path):
    t = np.array(data["time"], dtype=float)
    y = np.array(data["y"], dtype=float)
    dn = np.array(data["delta_nom"], dtype=float)
    ds = np.array(data["delta"], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    m = np.isfinite(t) & np.isfinite(y)
    axes[0].plot(t[m], y[m], lw=1.2, label="y")
    axes[0].set_ylabel("lateral y [m]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    m2 = np.isfinite(t) & np.isfinite(dn) & np.isfinite(ds)
    axes[1].plot(t[m2], np.degrees(dn[m2]), lw=1.0, label="delta_nom")
    axes[1].plot(t[m2], np.degrees(ds[m2]), lw=1.2, label="delta_safe")
    axes[1].set_ylabel("steer [deg]")
    axes[1].set_xlabel("t [s]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle("Lateral position and steering intervention")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Plot benchmark telemetry")
    ap.add_argument("csv", nargs="+", help="run CSV file(s)")
    ap.add_argument("--out", default=None, help="figure directory (default: next to CSV)")
    args = ap.parse_args()
    for csv_path in args.csv:
        data = tm.load_run(csv_path)
        metrics = tm.compute_metrics(data)
        outdir = args.out or os.path.dirname(os.path.abspath(csv_path))
        os.makedirs(outdir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        mu_ref = metrics["mu_true_final"]
        if mu_ref != mu_ref:
            mu_ref = 0.9
        fig_phase(data, mu_ref, os.path.join(outdir, stem + "_phase.png"))
        fig_barriers(data, os.path.join(outdir, stem + "_barriers.png"))
        fig_speed_ttc_mu(data, os.path.join(outdir, stem + "_speed_ttc_mu.png"))
        fig_tracking_cmd(data, os.path.join(outdir, stem + "_tracking_cmd.png"))
        print(f"[{stem}] pass={metrics['pass']} " +
              " ".join(f"{k}={v:.3f}" for k, v in metrics.items()
                        if k in ("min_ttc", "min_clearance", "min_h_obs",
                                 "min_h_road", "min_h_drift", "max_abs_beta_deg",
                                 "solve_us_p99", "solve_us_max")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
