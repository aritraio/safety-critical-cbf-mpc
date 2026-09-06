"""Benchmark scenario definitions (world-agnostic dicts).

Each builder returns a scenario dict consumed by carla_runner worlds and
run_benchmarks.py:
    {
      "name": str, "town": str, "duration": float,
      "ego_init": {"x","y","psi","vx"},
      "vx_cruise": float, "y_ref": float, "kappa_ref": float,
      "obstacles": [ {"x","y","vx","vy","a","b", ...}, ... ],
      "mu_field": callable(x, y, t) -> mu,   # mock plant friction
      "mu_nominal": float,                    # CARLA-side nominal
      "spawn": optional explicit CARLA spawn,
      "checks": {"min_ttc_warn": .., "expect_softened": bool, ...},
      "description": str,
    }

Reference tracking: straight lanes use y_ref/kappa_ref = const; the split-mu
scenario uses a circular arc (R=150 m) evaluated by run_benchmarks.py, which
projects the ego pose onto the arc for (ey, epsi, kappa).
"""

import math


def highway_cut_in():
    """Scenario 1: 70 km/h highway cruise (Town04); a vehicle cuts in 25 m
    ahead from the adjacent lane and stays slow — ego must brake + offset."""
    v_ego = 70.0 / 3.6  # 19.44 m/s
    return {
        "name": "highway_cut_in",
        "town": "Town04",
        "duration": 9.0,
        "controller": "mpc",  # hot regime: MPC preview + input-rate cost
        "ego_init": {"x": 0.0, "y": 0.0, "psi": 0.0, "vx": v_ego},
        "vx_cruise": v_ego,
        "y_ref": 0.0,
        "kappa_ref": 0.0,
        # Cutter crosses the centerline ~24 m ahead of ego (t ~= 3.5 s) at
        # 15 m/s: genuine 70 km/h emergency (must brake + offset), with enough
        # warning for physical actuators (no teleportation).
        # Cutter crosses the centerline ~24 m ahead (t ~= 3.5 s), then holds
        # the lane at 15 m/s: ego closes from 70 km/h and must brake + follow.
        "obstacles": [
            {"x": 40.0, "y": 3.2, "vx": 15.0, "vy": -0.9, "a": 4.0, "b": 2.0,
             "segments": [[3.5, 15.0, -0.9], [1e9, 15.0, 0.0]]},
        ],
        "mu_field": lambda x, y, t: 0.9,
        "mu_nominal": 0.9,
        "checks": {"min_ttc_warn": 1.5, "expect_softened": False},
        "description": "70 km/h cruise; cutter merges 25 m ahead and holds 12 m/s",
    }


def split_mu_curve():
    """Scenario 2: 150 m-radius left bend; friction collapses 0.9 -> 0.25 in
    the arc sector s in [40, 120] m, with a stalled car mid-curve at s = 85 m.
    Ego enters at 60 km/h and must shed speed for ice + obstacle."""
    R = 150.0
    s_lo, s_hi = 40.0, 120.0
    s_obs = 85.0
    v_entry = 45.0 / 3.6  # survivable ice entry; 60 km/h needs V2X ice warning

    def mu_field(x, y, t):
        # Arc-length approx for the left bend about center (0, R), ego near s.
        # Invert locally: s ~= R * atan2(x, R - y) for the travelled sector.
        s = R * math.atan2(max(x, 0.0), max(R - y, 1.0))
        return 0.25 if (s_lo <= s <= s_hi) else 0.9

    th = s_obs / R
    return {
        "name": "split_mu_curve",
        "town": "Town04",
        "duration": 12.0,
        "curve_R": R,
        # Straight-lane road barrier uses the global-Y frame, invalid on the
        # arc: disabled here (Frenet-frame road barrier is roadmap work).
        "use_road": False,
        "ego_init": {"x": 0.0, "y": 0.0, "psi": 0.0, "vx": v_entry},
        "vx_cruise": v_entry,
        "y_ref": 0.0,
        "kappa_ref": 1.0 / R,
        "obstacles": [
            {
                "x": R * math.sin(th),
                "y": R * (1.0 - math.cos(th)),
                "vx": 0.0, "vy": 0.0,
                "a": 4.0, "b": 2.0,
            },
        ],
        "mu_field": mu_field,
        "mu_nominal": 0.9,
        "checks": {"min_ttc_warn": 1.5, "expect_softened": False},
        "description": "R=150 m bend; ice sector s=[40,120] m; stalled car at s=85 m",
    }


def narrow_corridor():
    """Scenario 3: staggered pair forces a slalom inside the lane margins.
    Full swerve would breach the road boundary, so the soft-CBF slack MUST
    engage (expect_softened=True) while hard input bounds always hold."""
    return {
        "name": "narrow_corridor",
        "town": "Town04",
        "duration": 9.0,
        "controller": "mpc",  # slalom needs preview + input-rate shaping, not bang-bang LQR
        "ego_init": {"x": 0.0, "y": 0.0, "psi": 0.0, "vx": 11.5},
        "vx_cruise": 11.5,
        "y_ref": 0.0,
        "kappa_ref": 0.0,
        "obstacles": [
            {"x": 38.0, "y": 2.25, "vx": 0.0, "vy": 0.0, "a": 4.0, "b": 1.7},
            {"x": 58.0, "y": -2.25, "vx": 0.0, "vy": 0.0, "a": 4.0, "b": 1.7},
        ],
        "mu_field": lambda x, y, t: 0.9,
        "mu_nominal": 0.9,
        "checks": {"min_ttc_warn": 1.0, "expect_softened": True},
        "description": "staggered pair; slalom must trade barrier margin via slack",
    }


SCENARIOS = {
    "highway_cut_in": highway_cut_in,
    "split_mu_curve": split_mu_curve,
    "narrow_corridor": narrow_corridor,
}
