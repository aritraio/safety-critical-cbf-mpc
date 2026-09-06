"""Shared telemetry schema + metrics for SIL benchmarks and plots.

CSV schema (written by run_benchmarks.py, read by plot_results.py):
  time,x,y,psi,vx,vy,r,delta_nom,ax_nom,delta,ax,mu_hat,mu_true,
  h_obs,h_road,h_drift,min_ttc,min_clearance,solve_us,controller,filter_on
Missing trailing columns are tolerated (filled with NaN) so C++ demo CSVs
can be plotted too (they lack mu_hat/min_ttc/clearance/controller/filter_on).
"""

import csv
import math

CSV_COLUMNS = [
    "time", "x", "y", "psi", "vx", "vy", "r",
    "delta_nom", "ax_nom", "delta", "ax",
    "mu_hat", "mu_true",
    "h_obs", "h_road", "h_drift", "h_gap",
    "min_ttc", "min_clearance", "solve_us",
    "controller", "filter_on",
    "softened", "slack_max",
]

# Sentinel for "no active barrier" (C++ diag uses 1e100); mapped to NaN.
SENTINEL = 1e99

EGO_HALF_LENGTH = 2.25  # [m] footprint margin for TTC computations


def load_run(path):
    """Load a benchmark CSV into a dict of column -> list of floats/strings."""
    data = {c: [] for c in CSV_COLUMNS}
    with open(path, newline="") as f:
        reader = csv.DictReader(c for c in f if not c.lstrip().startswith("#"))
        for row in reader:
            for c in CSV_COLUMNS:
                v = row.get(c, "")
                if c in ("controller",):
                    data[c].append(v)
                elif c in ("filter_on",):
                    data[c].append(1.0 if str(v).strip() not in ("", "0", "False") else 0.0)
                else:
                    try:
                        v = float(v)
                        data[c].append(v if v < SENTINEL else math.nan)
                    except (ValueError, TypeError):
                        data[c].append(math.nan)
    return data


def sideslip(vx, vy):
    return math.atan2(vy, vx) if abs(vx) > 1e-6 else 0.0


def longitudinal_ttc(x_ego, vx_ego, obs, ego_half_length=EGO_HALF_LENGTH):
    """Time-to-collision along x with an ellipse-aware gap. inf if separating."""
    dx = obs["x"] - x_ego
    closing = vx_ego - obs.get("vx", 0.0)
    if closing <= 0.5:
        return math.inf
    gap = abs(dx) - (obs.get("a", 4.0) + ego_half_length)
    if gap <= 0.0:
        return 0.0
    return gap / closing


def ellipse_clearance(x_ego, y_ego, obs):
    """Signed normalized clearance: sqrt(((dx)/a)^2+((dy)/b)^2) - 1."""
    dx = (x_ego - obs["x"]) / max(obs.get("a", 4.0), 1e-6)
    dy = (y_ego - obs["y"]) / max(obs.get("b", 2.0), 1e-6)
    return math.hypot(dx, dy) - 1.0


def _nanmin(xs, default=math.nan):
    best = math.inf
    for v in xs:
        if v == v and v < best:  # noqa: PLR0124 - NaN guard is intentional
            best = v
    return best if best < math.inf else default


def _nanmax(xs, default=math.nan):
    best = -math.inf
    for v in xs:
        if v == v and v > best:  # noqa: PLR0124 - NaN guard is intentional
            best = v
    return best if best > -math.inf else default


def _nanmean(xs):
    s, n = 0.0, 0
    for v in xs:
        if v == v:
            s += v
            n += 1
    return s / n if n else math.nan


def _percentile(xs, q):
    vals = sorted(v for v in xs if v == v)
    if not vals:
        return math.nan
    k = min(len(vals) - 1, max(0, int(q * len(vals))))
    return vals[k]


def compute_metrics(data, h_drift_tol=-1.5):
    """Summary metrics dict from a loaded run."""
    n = len(data["time"])
    beta = [sideslip(data["vx"][i], data["vy"][i]) for i in range(n)]
    intervened = [
        1.0
        if abs(data["delta"][i] - data["delta_nom"][i]) > 1e-3
        or abs(data["ax"][i] - data["ax_nom"][i]) > 1e-2
        else 0.0
        for i in range(n)
    ]
    return {
        "steps": n,
        "duration": data["time"][-1] - data["time"][0] if n else 0.0,
        "min_ttc": _nanmin(data["min_ttc"]),
        "min_clearance": _nanmin(data["min_clearance"]),
        "min_h_obs": _nanmin(data["h_obs"]),
        "min_h_road": _nanmin(data["h_road"]),
        "min_h_drift": _nanmin(data["h_drift"]),
        # h_gap is a regulating (not gating) barrier: violations are the braking
        # control signal, so it is reported, never pass-gated. NaN when no
        # obstacle is gated ahead.
        "min_h_gap": _nanmin(data["h_gap"]),
        "max_abs_beta_deg": max((abs(b) for b in beta), default=0.0) * 180.0 / math.pi,
        "max_abs_r": max((abs(r) for r in data["r"] if r == r), default=0.0),
        "rms_ey_proxy": math.sqrt(
            sum(y * y for y in data["y"] if y == y) / max(1, sum(1 for y in data["y"] if y == y))
        ),
        "intervention_rate": _nanmean(intervened),
        "solve_us_p50": _percentile(data["solve_us"], 0.50),
        "solve_us_p99": _percentile(data["solve_us"], 0.99),
        "solve_us_max": _percentile(data["solve_us"], 1.0),
        "mu_hat_final": data["mu_hat"][-1] if n else math.nan,
        "mu_true_final": data["mu_true"][-1] if n else math.nan,
        "max_slack": _nanmax(data["slack_max"]),
        "pass": (
            _nanmin(data["h_obs"]) >= -0.05
            and _nanmin(data["h_road"]) >= -0.05
            and _nanmin(data["h_drift"]) >= h_drift_tol
        ),
    }
