#!/usr/bin/env python3
"""Closed-loop SIL benchmarks: nominal (LQR/MPC) + UKF + CBF-QP at 100 Hz.

One configuration per invocation; run twice (with/without --no-filter) for
baseline contrast. Mock world (default) is deterministic and CI-friendly;
--world carla drives a live Town04 server through the same harness.

Examples:
  python3 simulation/run_benchmarks.py --scenario all --out runs/base
  python3 simulation/run_benchmarks.py --scenario highway_cut_in --controller mpc --ukf --out runs/mpc
  python3 simulation/run_benchmarks.py --scenario all --no-filter --out runs/nofilter
  python3 simulation/run_benchmarks.py --scenario highway_cut_in --world carla --out runs/carla
"""

import argparse
import csv
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build_py"))
sys.path.insert(0, os.path.dirname(__file__))

import telemetry as tm  # noqa: E402
from carla_bridge.carla_runner import make_world  # noqa: E402

try:
    import av_safety_py as av
except ImportError as e:  # pragma: no cover
    sys.exit(
        "av_safety_py not importable. Build with "
        "-DBUILD_PYTHON_BINDINGS=ON and run from the repo root. "
        f"({e})"
    )


def wrap_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def arc_reference(state, R):
    """Project ego onto the left-bend circle about (0, R)."""
    x, y, psi = state["x"], state["y"], state["psi"]
    theta = math.atan2(max(x, 0.0), max(R - y, 1.0))
    rx, ry = R * math.sin(theta), R * (1.0 - math.cos(theta))
    dx, dy = x - rx, y - ry
    # Outward normal (away from center) ~ radial direction.
    nx, ny = math.sin(theta), -math.cos(theta)
    ey = dx * nx + dy * ny
    epsi = wrap_angle(psi - theta)
    return ey, epsi, 1.0 / R


def build_stack(args, vp, use_road=True):
    lc = av.load_lqr_config(args.lqr_yaml)
    mc = av.load_mpc_config(args.mpc_yaml)
    bc = av.load_barrier_config(args.barrier_yaml)
    uc = av.load_ukf_config(args.ukf_yaml)
    fc = av.load_cbf_qp_config(args.barrier_yaml)
    fc.use_road = use_road
    model = av.DynamicBicycleModel(vp)
    lqr = av.LqrTracker(vp, lc)
    mpc = av.MpcTracker(vp, mc)
    ukf = av.FrictionUkf(vp, uc)
    exact = av.EnumerationQpSolver()
    filt = None
    backend_kind = "enumeration"
    try:
        osqp = av.OsqpSolver([fc.H_delta, fc.H_ax])
        if osqp.ready():
            filt_backend = osqp
            backend_kind = "osqp"
        else:
            filt_backend = exact
    except Exception:
        filt_backend = exact
    filt = av.CbfQpFilter(vp, bc, fc, filt_backend)
    if backend_kind == "osqp":
        filt.set_fallback_backend(exact)
    return {
        "model": model, "lqr": lqr, "mpc": mpc, "ukf": ukf,
        "filter": filt, "exact": exact,
        "lc": lc, "mc": mc, "bc": bc, "uc": uc, "fc": fc,
        "backend": backend_kind,
    }


def run_one(args, sc, stack, vp, controller):
    dt = args.dt
    rng = random.Random(1234 + len(sc["name"]))
    world = make_world(args.world, av, vp, host=args.carla_host,
                       port=args.carla_port, dt=dt)
    world.reset(sc)
    st = stack
    vx0 = sc["ego_init"]["vx"]
    ok_lqr = st["lqr"].compute_gains(vx0, sc.get("mu_nominal", 0.9))
    ok_mpc = st["mpc"].compute_gains(vx0, sc.get("mu_nominal", 0.9))
    if controller == "mpc" and not ok_mpc:
        raise RuntimeError("MPC gain computation failed")
    if controller == "lqr" and not ok_lqr:
        raise RuntimeError("LQR gain computation failed")
    st["ukf"].reset()

    tire = av.PacejkaTire(vp.pacejka, vp.mu_nominal)
    steer_max = vp.steer_max
    d_steer = vp.steer_rate_max * dt
    d_ax = vp.accel_rate_max * dt
    u_prev = [0.0, 0.0]  # actuator state (slew model, both branches)

    rows = []
    n = int(sc["duration"] / dt)
    stats = {"interventions": 0, "infeasible": 0, "fallbacks": 0, "softened": 0,
             "worst_us": 0.0}
    R = sc.get("curve_R", None)

    for k in range(n):
        s = world.ego_state
        obs = world.obstacles[:4]
        mu_true = world.true_mu(s["x"], s["y"])

        # Reference tracking error.
        if R is not None:
            ey, epsi, kappa = arc_reference(s, R)
        else:
            ey, epsi, kappa = s["y"] - sc["y_ref"], s["psi"], sc["kappa_ref"]
        err = [ey, epsi, s["vy"], s["r"]]
        if controller == "mpc":
            nom = list(st["mpc"].update(err, s["vx"], kappa, sc["vx_cruise"], dt))
        else:
            nom = list(st["lqr"].update(err, s["vx"], kappa, sc["vx_cruise"], dt))

        # Friction source: UKF estimate or static init.
        if args.ukf:
            a_f = av.PacejkaTire.front_slip_angle(
                u_prev[0], s["vx"], s["vy"], s["r"], vp.lf, vp.vx_epsilon)
            a_r = av.PacejkaTire.rear_slip_angle(
                s["vx"], s["vy"], s["r"], vp.lr, vp.vx_epsilon)
            fyf = tire.lateral_force(a_f, mu_true)
            fyr = tire.lateral_force(a_r, mu_true)
            z = [
                u_prev[1] - s["vy"] * s["r"] + rng.gauss(0, 1) * 0.15,
                (2 * fyf * math.cos(u_prev[0]) + 2 * fyr) / vp.mass
                - s["vx"] * s["r"] + rng.gauss(0, 1) * 0.15,
                s["r"] + rng.gauss(0, 1) * 1e-3,
                s["vx"] + rng.gauss(0, 1) * 0.05,
            ]
            mu_hat = av.ukf_step(st["ukf"], list(u_prev), z, dt)
        else:
            mu_hat = st["uc"].mu_init

        if args.no_filter:
            cmd = nom
            diag = {"feasible": True, "h_obs_min": float("nan"),
                    "h_road": float("nan"), "h_drift": float("nan"),
                    "solve_us": 0.0, "softened": False, "used_fallback": False,
                    "slack_max": 0.0}
        else:
            av_obs = []
            for o in obs:
                oo = av.ObstacleState()
                oo.x, oo.y = o["x"], o["y"]
                oo.vx, oo.vy = o.get("vx", 0.0), o.get("vy", 0.0)
                oo.a, oo.b = o.get("a", 4.0), o.get("b", 2.0)
                av_obs.append(oo)
            st["filter"].set_obstacles(av_obs)
            cmd, diag = av.cbf_filter_step(
                st["filter"], [s["x"], s["y"], s["psi"], s["vx"], s["vy"], s["r"]],
                nom, mu_hat)

        # Actuator slew (identical model both branches).
        cmd = [
            min(max(cmd[0], u_prev[0] - d_steer), u_prev[0] + d_steer),
            min(max(cmd[1], u_prev[1] - d_ax), u_prev[1] + d_ax),
        ]
        cmd = [
            min(max(cmd[0], -steer_max), steer_max),
            min(max(cmd[1], vp.accel_min), vp.accel_max),
        ]
        u_prev = list(cmd)
        world.step(cmd[0], cmd[1], dt)

        h_obs = diag["h_obs_min"]
        if obs and (h_obs != h_obs):  # NaN when unfiltered: compute directly
            h_obs = min(tm.ellipse_clearance(s["x"], s["y"], o) for o in obs)
        h_gap = diag.get("h_gap_min", float("nan"))
        ttc = min([tm.longitudinal_ttc(s["x"], s["vx"], o) for o in obs] or [math.inf])
        clr = min([tm.ellipse_clearance(s["x"], s["y"], o) for o in obs] or [math.inf])
        rows.append([
            world.time, s["x"], s["y"], s["psi"], s["vx"], s["vy"], s["r"],
            nom[0], nom[1], cmd[0], cmd[1], mu_hat, mu_true,
            h_obs, diag["h_road"], diag["h_drift"], h_gap, ttc, clr,
            diag["solve_us"], controller, 0 if args.no_filter else 1,
            1 if diag.get("softened", False) else 0,
            diag.get("slack_max", 0.0),
        ])
        stats["worst_us"] = max(stats["worst_us"], diag["solve_us"])
        stats["interventions"] += 1 if (
            abs(cmd[0] - nom[0]) > 1e-3 or abs(cmd[1] - nom[1]) > 1e-2) else 0
        stats["infeasible"] += 0 if diag["feasible"] else 1
        stats["fallbacks"] += 1 if diag.get("used_fallback", False) else 0
        stats["softened"] += 1 if diag.get("softened", False) else 0
        if world.collision:
            stats["collision_step"] = k
            break

    world.close()
    stats["collision"] = bool(world.collision)
    return rows, stats


def main():
    ap = argparse.ArgumentParser(description="SIL safety benchmarks (mock/CARLA)")
    ap.add_argument("--scenario", default="all",
                    choices=["all", "highway_cut_in", "split_mu_curve", "narrow_corridor"])
    ap.add_argument("--world", default="mock", choices=["mock", "carla"])
    ap.add_argument("--controller", default="auto",
                    choices=["auto", "lqr", "mpc"],
                    help="auto: per-scenario default (hot regimes use MPC)")
    ap.add_argument("--no-filter", action="store_true", help="baseline: nominal only")
    ap.add_argument("--ukf", action="store_true", help="UKF friction in the loop")
    ap.add_argument("--out", default="runs/bench")
    ap.add_argument("--vehicle-yaml", default="config/vehicle_params.yaml")
    ap.add_argument("--lqr-yaml", default="config/lqr_tuning.yaml")
    ap.add_argument("--mpc-yaml", default="config/mpc_tuning.yaml")
    ap.add_argument("--barrier-yaml", default="config/barrier_params.yaml")
    ap.add_argument("--ukf-yaml", default="config/friction_ukf.yaml")
    ap.add_argument("--carla-host", default="localhost")
    ap.add_argument("--carla-port", type=int, default=2000)
    ap.add_argument("--dt", type=float, default=0.01)
    args = ap.parse_args()

    import scenarios as scn
    names = list(scn.SCENARIOS) if args.scenario == "all" else [args.scenario]
    os.makedirs(args.out, exist_ok=True)

    vp = av.load_vehicle_params(args.vehicle_yaml)
    base_tag = f"{args.controller}_{'nofilter' if args.no_filter else 'cbf'}" \
               f"{'_ukf' if args.ukf else ''}_{args.world}"
    summary = {"config": base_tag, "scenarios": {}}
    rc = 0
    for name in names:
        sc = scn.SCENARIOS[name]()
        controller = args.controller
        if controller == "auto":
            controller = sc.get("controller", "lqr")
        cfg_tag = f"{controller}_{'nofilter' if args.no_filter else 'cbf'}" \
                  f"{'_ukf' if args.ukf else ''}_{args.world}"
        # Straight-lane road barrier is invalid on curves (global-Y frame);
        # curve scenarios opt out (documented limitation, Frenet roadmap).
        stack = build_stack(args, vp, use_road=sc.get("use_road", True))
        rows, stats = run_one(args, sc, stack, vp, controller)
        csv_path = os.path.join(args.out, f"{name}_{cfg_tag}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(tm.CSV_COLUMNS)
            w.writerows(rows)
        data = tm.load_run(csv_path)
        metrics = tm.compute_metrics(data)
        metrics.update({k: v for k, v in stats.items()
                        if k in ("interventions", "infeasible", "fallbacks",
                                 "softened", "worst_us", "collision")})
        checks = sc.get("checks", {})
        # Slack verdict on MACROSCOPIC shedding (max slack > 0.05): the binary
        # per-step softened flag also fires on quadratic-penalty micro-shed
        # (~1e-3) at riding constraints, which is expected, not a trade-off.
        macro_shed = metrics["max_slack"] == metrics["max_slack"] and metrics["max_slack"] > 0.05
        if args.no_filter:
            # Baseline has no barrier values (NaN): judge on outcomes only.
            safety = (not stats["collision"]
                      and metrics["min_clearance"] == metrics["min_clearance"]
                      and metrics["min_clearance"] >= -0.02)
            soft_ok = True  # no filter => no slack by construction
        else:
            safety = bool(metrics["pass"]) and not stats["collision"] \
                and stats["infeasible"] == 0
            soft_ok = (macro_shed == checks.get("expect_softened", False))
        verdicts = {
            "safety": safety,
            "softened_as_expected": soft_ok,
        }
        metrics["verdicts"] = verdicts
        metrics["pass"] = verdicts["safety"] and verdicts["softened_as_expected"]
        summary["scenarios"][name] = metrics
        print(f"[{name}] pass={metrics['pass']} " +
              " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in metrics.items()
                        if k in ("min_ttc", "min_clearance", "min_h_obs",
                                 "min_h_road", "min_h_drift", "worst_us") or
                        k in ("interventions", "infeasible", "fallbacks", "collision")))
        rc = rc or (not metrics["pass"])
    with open(os.path.join(args.out, f"summary_{base_tag}.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return rc


if __name__ == "__main__":
    sys.exit(main())
