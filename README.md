# Safety-Critical Autonomous Vehicle Control with Control Barrier Functions (CBFs) & LQR/MPC

## Project Overview
An active safety motion control framework for high-speed autonomous ground vehicles that unifies performance-driven trajectory tracking (**LQR / Linear MPC**) with formal safety verification via **Control Barrier Functions (CBFs)**. Implemented as a real-time **CBF-Quadratic Program (CBF-QP)** safety filter at 100 Hz, the system guarantees collision avoidance, roadway boundary containment, and rollover/drift prevention under uncertain road-tire friction conditions.

---

## Target Industry & Value Proposition
* **Target Industries:** Autonomous Vehicles, ADAS Engineering, Mobile Robotics (e.g., Waymo, Zoox, Cruise, Tesla Autopilot, Motional, Aurora).
* **Resume Impact:** Bridges control theory with formal verification (Lyapunov & Nagumo invariance principles), proving mathematically that safety constraints will never be violated during dynamic driving maneuvers.

> [!TIP]
> **Technical Deep Dive & Interview Guide:**
> For an exhaustive, first-principles mathematical derivation, architecture walkthrough, and the **Top 15 Autonomous Driving / Motion Control Technical Interview Questions & Answers**, see [explanation.md](file:///Users/aritra/Code/Languages/C++/Project-3/explanation.md).

---

## Mathematical Foundations & Control Formulation

### 1. Non-linear Dynamic Vehicle Model (Pacejka Tire Formula)
* **State Vector:** $\mathbf{x} = [x, y, \psi, v_x, v_y, r]^\top \in \mathbb{R}^6$ (inertial coordinates, heading angle, body longitudinal/lateral velocity, yaw rate).
* **Control Input:** $\mathbf{u} = [\delta, a_x]^\top \in \mathbb{R}^2$ (front steering angle, longitudinal acceleration).

**Lateral Tire Forces (Pacejka Magic Formula):**

$$
F_{yf} = D \sin\left(C \arctan\left(B\alpha_f - E(B\alpha_f - \arctan(B\alpha_f))\right)\right)
$$

$$
F_{yr} = D \sin\left(C \arctan\left(B\alpha_r - E(B\alpha_r - \arctan(B\alpha_r))\right)\right)
$$

where front and rear slip angles are given by:

$$
\alpha_f = \delta - \arctan\left(\frac{v_y + l_f r}{v_x}\right), \quad \alpha_r = -\arctan\left(\frac{v_y - l_r r}{v_x}\right)
$$

**Equations of Motion:**

$$
m(\dot{v}_x - v_y r) = 2F_{xf} \cos\delta - 2F_{yf} \sin\delta + 2F_{xr}
$$

$$
m(\dot{v}_y + v_x r) = 2F_{xf} \sin\delta + 2F_{yf} \cos\delta + 2F_{yr}
$$

$$
I_z \dot{r} = 2l_f(F_{xf}\sin\delta + F_{yf}\cos\delta) - 2l_r F_{yr}
$$

### 2. Control Barrier Functions (CBFs) & Nagumo Invariance
A safe operating set $\mathcal{C}$ is defined by the 0-superlevel set of a continuously differentiable barrier function $h(\mathbf{x}): \mathbb{R}^n \to \mathbb{R}$:

$$
\mathcal{C} = \{\mathbf{x} \in \mathbb{R}^n \mid h(\mathbf{x}) \ge 0\}
$$

* **Barrier 1: Roadway Boundary Containment:**

$$
h_{\mathrm{road}}(\mathbf{x}) = d_{\mathrm{margin}}^2 - e_y(\mathbf{x})^2 \ge 0
$$

* **Barrier 2: Ellipsoidal Obstacle Collision Avoidance:**

$$
h_{\mathrm{obs}}(\mathbf{x}) = \frac{(x - x_{\mathrm{obs}})^2}{a^2} + \frac{(y - y_{\mathrm{obs}})^2}{b^2} - 1 \ge 0
$$

* **Barrier 3: Dynamic Drift / Stability Envelope:**

$$
h_{\mathrm{drift}}(\mathbf{x}) = (\mu g)^2 - (v_x r)^2 \ge 0
$$

* **Forward Invariance Condition (Nagumo):**
For the set $\mathcal{C}$ to remain forward invariant under closed-loop control, the control input $\mathbf{u}$ must satisfy:

$$
\sup_{\mathbf{u} \in \mathcal{U}} \left[ L_f h(\mathbf{x}) + L_g h(\mathbf{x})\mathbf{u} + \gamma(h(\mathbf{x})) \right] \ge 0
$$

where $L_f h = \frac{\partial h}{\partial \mathbf{x}}\mathbf{f}(\mathbf{x})$, $L_g h = \frac{\partial h}{\partial \mathbf{x}}\mathbf{g}(\mathbf{x})$ are Lie derivatives, and $\gamma(h) = \kappa h$ ($\kappa > 0$) is an extended class $\mathcal{K}$ function.

### 3. Real-Time CBF-QP Safety Filter
The nominal tracking controller (LQR or MPC) proposes an unconstrained or performance-focused control input $\mathbf{u}_{\mathrm{nom}}$. The online safety filter minimally adjusts this command while strictly enforcing safety:

$$
\mathbf{u}^* = \arg\min_{\mathbf{u}} \frac{1}{2}\|\mathbf{u} - \mathbf{u}_{\mathrm{nom}}\|_{\mathbf{H}}^2
$$

**Subject to:**

$$
L_f h_i(\mathbf{x}) + L_g h_i(\mathbf{x})\mathbf{u} \ge -\gamma(h_i(\mathbf{x})), \quad \forall i \in \{1, \dots, M\}
$$

$$
\mathbf{u}_{\min} \le \mathbf{u} \le \mathbf{u}_{\max}
$$

This convex Quadratic Program is solved deterministically in $< 1\,\text{ms}$ using OSQP.

### 4. Adaptive Friction Estimation via UKF
* Online estimation of peak road-tire friction coefficient $\hat{\mu}$ using wheel speed discrepancies and IMU longitudinal/lateral acceleration.
* Adapts the stability barrier envelope $h_{\text{drift}}(\mathbf{x})$ online when transitioning from dry asphalt ($\mu \approx 0.9$) to rain/ice ($\mu \approx 0.3$).

---

## System & Software Architecture

### Directory Structure Blueprint
```text
av-cbf-safety-filter/
├── CMakeLists.txt
├── config/
│   ├── vehicle_params.yaml
│   ├── barrier_params.yaml
│   └── lqr_tuning.yaml
├── include/
│   ├── dynamics/
│   │   ├── dynamic_bicycle_model.hpp
│   │   └── pacejka_tire.hpp
│   ├── barriers/
│   │   ├── control_barrier_function.hpp
│   │   ├── road_boundary_cbf.hpp
│   │   └── obstacle_cbf.hpp
│   ├── controllers/
│   │   ├── lqr_tracker.hpp
│   │   └── cbf_qp_filter.hpp
│   └── estimation/
│       └── friction_ukf.hpp
├── src/
│   ├── dynamics/
│   ├── barriers/
│   ├── controllers/
│   └── cbf_node.cpp
├── tests/
│   ├── test_pacejka.cpp
│   ├── test_lie_derivatives.cpp
│   └── test_qp_solver.cpp
└── simulation/
    └── carla_bridge/
```

### Technical Stack
* **Language:** Modern C++ (C++17, `-Wall -Wextra -Wpedantic -Wconversion`), Eigen3, yaml-cpp, GoogleTest.
* **Optimization Solver:** OSQP (warm-started, preallocated) + exact active-set enumeration fallback; condensed OSQP for MPC.
* **Estimation:** Unscented Kalman Filter (Van der Merwe) for road-tire friction, feeding the drift barrier live.
* **Simulator:** CARLA Simulator (Town04) via Python bridge + deterministic mock world (same interface); standalone C++ demos.
* **Visualization & analysis:** RViz2 marker array (footprint, ellipses, margins, drift status); Python telemetry + publication plots ($\beta$–$r$, $h(t)$, TTC).

---

## Implementation Roadmap

### Phase 1: Dynamics & Baseline LQR Tracker
* Implement dynamic bicycle model with Pacejka non-linear tire curves.
* Linearize vehicle model around operating speed; compute steady-state LQR gain matrix via Riccati equation (`care` solver).
* Verify lateral tracking error along a smooth curve in a standalone C++ simulator.

### Phase 2: CBF Derivation & Verification
* Analytically calculate Lie derivatives $L_f h(\mathbf{x})$ and $L_g h(\mathbf{x})$ for lane margin and obstacle barriers.
* Write automated tests confirming barrier positivity along boundary trajectories.

### Phase 3: Convex QP Safety Filter Integration
* Connect OSQP C++ interface to solve the minimum perturbation problem at 100 Hz.
* Simulate an unsafe trajectory (e.g., nominal controller aims directly at an obstacle); verify that the QP filter overrides the command and safely steers away.

### Phase 4: CARLA Simulation & Adverse Scenarios — DONE (mock-verified SIL)
* `simulation/carla_bridge/` (mock + live CARLA worlds), `simulation/scenarios.py`
  (70 km/h cut-in, split-μ curve, narrow corridor), `simulation/run_benchmarks.py`
  (100 Hz closed loop, CSV + JSON metrics). Live CARLA needs a server; mock runs in CI.

### Module A: Soft-CBF Slack & Slew-Rate QP — DONE
* Decision vector $[\delta, a_x, \xi]$ with exact quadratic slack penalty
  (recovers the hard solution when feasible) + hard slew box around $u_{prev}$.
* Exact enumeration backend (soft-subset + kink-scan) and expanded OSQP backend
  (n = 8, m = 26, row normalization, scaling off, tight-budget + exact-fallback
  latency pattern). Doomed head-ons now shed slack (full braking) instead of
  going infeasible.

### Module B: Adaptive Friction UKF — DONE
* Van der Merwe UKF on $[v_x, v_y, r, \mu]$ with IMU/wheel-speed updates and a
  mean-reverting prior (fixes the straight-cruise observability ridge);
  $\hat{\mu}$ feeds the drift barrier live (verified 0.9 → 0.49 on wet tarmac).

### Module C: Linear MPC Tracker — DONE
* Condensed N = 15 receding-horizon MPC in $[e_y, \dot e_y, e_\psi, \dot e_\psi]$
  coordinates with curvature preview, terminal DARE weight, and LQR fallback;
  `--controller={lqr,mpc}` toggle in the demo.

### Module D/E: Benchmarks, Telemetry & Plots — DONE
* `simulation/telemetry.py` (metrics: TTC, clearance, barrier minima, solve
  stats), `simulation/plot_results.py` ($\beta$–$r$ phase portraits with
  friction envelope, $h(t)$, speed/TTC/$\mu$, tracking/commands).

### Module F: ROS 2 Humble Scaffolding — DONE
* `package.xml`, `launch/cbf_bringup.launch.py` (controller/UKF/RViz args),
  `rviz/cbf_config.rviz`, `/cbf_markers` visualization (footprint, obstacle
  ellipses + contours, road margins, drift/friction status), IMU + UKF wiring
  in the node. Plus: headway (gap) CBF for early highway braking with a
  slew-box robust envelope (scrub-steering perverse incentive removed).

---

## Build, Test & Benchmarks (verified on macOS arm64)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_OSQP=ON   # vendors OSQP v0.6.3 if absent
cmake --build build -j8
ctest --test-dir build            # 63 tests, all passing
./build/cbf_safety_demo config/vehicle_params.yaml config/lqr_tuning.yaml config/barrier_params.yaml config/friction_ukf.yaml config/mpc_tuning.yaml [--controller=mpc] [--ukf]
# Python SIL (needs bindings + numpy/matplotlib venv):
cmake -S . -B build_py -DBUILD_PYTHON_BINDINGS=ON -DBUILD_TESTS=OFF -DBUILD_DEMOS=OFF && cmake --build build_py -j8
python3 simulation/run_benchmarks.py --scenario all --out runs/ --ukf
python3 simulation/plot_results.py runs/*_mock.csv --out figs/
```

* **Correctness:** analytic Pacejka slopes, drift Jacobian, and all Lie / ECBF / gap terms verified against finite differences; enumeration QP cross-checked against brute force and against OSQP; forward invariance holding safe inputs keep $h \ge 0$; UKF converges 0.9 → 0.35 on wet cornering.
* **Safety filter vs. unfiltered baseline:** C++ demo static flank $h_{obs}$ $-0.73 \to +4.26$; Python SIL cut-in (70 km/h) baseline **collides** → filtered clears by 4.5 m; corridor margin doubled; split-μ ice handled with live $\hat{\mu}$.
* **Real-time:** full pipeline (tracker + UKF + barrier linearizations + OSQP solve) mean $\approx 30\,\mu\text{s}$, worst observed $< 500\,\mu\text{s}$ — inside the $1\,\text{ms}$ / $100\,\text{Hz}$ budget with zero heap allocations in the loop (OSQP primary with 800-iter budget + exact enumeration fallback bounds worst-case latency deterministically).
* **Known limits:** quasi-affine linearization assumes tires near the linear regime; road barrier uses the straight-lane frame (curves need a Frenet extension — curve benchmarks opt out); dead-ahead approaches inside the braking distance shed slack under full braking by design.

### ROS 2 Humble deployment
```bash
# Inside a Humble workspace (colcon), this repo builds as package av_cbf_safety_filter:
colcon build --packages-select av_cbf_safety_filter
ros2 launch av_cbf_safety_filter cbf_bringup.launch.py controller:=mpc use_ukf:=true
# Topics: /vehicle/odometry + /vehicle/imu + /perception/obstacles in;
# /safety_cmd [delta, ax], /safety_diag, /cbf_markers (RViz2) out.
```

### Live CARLA (Town04)
```bash
# Terminal 1: CARLA server with Town04 ( Unreal / Docker ).
# Terminal 2: synchronous SIL through the same stack as mock:
python3 simulation/run_benchmarks.py --scenario highway_cut_in --world carla --ukf --out runs/carla
# Mock (default) is bit-deterministic and needs no server; CARLA mode reuses
# identical scenarios, metrics, and plots.
```

---

## Documentation & Code Architecture Tour

* **Technical Interview & Deep Dive:** [explanation.md](file:///Users/aritra/Code/Languages/C++/Project-3/explanation.md) — 10 chapters covering first-principles tire mechanics, Lie algebra, relative degree, ECBF derivations, soft-slack QP proofs, UKF observability, and the top 15 technical interview questions with model answers.
* **Core Headers (`include/av_safety/`):**
  * `dynamics/`: Pacejka non-linear tire curves and 6-DoF planar dynamic bicycle model.
  * `barriers/`: Analytical implementations of $h_{\text{road}}$, $h_{\text{obs}}$, $h_{\text{drift}}$, and $h_{\text{gap}}$.
  * `controllers/`: Nominal LQR tracker (DARE/CARE) and condensed $N=15$ Linear MPC tracker.
  * `qp/`: 2D Active-Set enumeration solver and OSQP sparse backend with soft slacks and slew limits.
  * `estimation/`: 4-DoF Unscented Kalman Filter for real-time road-tire friction ($\mu$) estimation.
* **Simulation & SIL (`simulation/`):**
  * `cbf_safety_demo.cpp` & `lqr_tracking_demo.cpp`: Standalone C++ verification demos.
  * `run_benchmarks.py`: Automated multi-scenario harness testing cut-in, split-$\mu$, and narrow corridors.
  * `carla_bridge/carla_runner.py`: Synchronous CARLA Town04 client with mock fallback.
* **ROS 2 Humble Package (`src/ros/`, `launch/`, `rviz/`):**
  * Standard `package.xml`, launch files, and RViz2 3D marker visualizations.

---

## Resume Talking Points
* *Engineered a 100 Hz active safety motion controller in C++ combining LQR/MPC tracking with Control Barrier Functions (soft CBF-QP with slack + slew limits) in CARLA Town04 and deterministic SIL.*
* *Enforced strict mathematical forward invariance on vehicle states, preventing collisions and roadway departure during aggressive evasive maneuvers; added a headway CBF for early highway braking.*
* *Formulated and solved convex QP safety filters in $<1\,\text{ms}$ using OSQP with an exact enumeration fallback, adapting safety barrier margins dynamically via UKF-estimated road-tire friction.*
* *Built the full verification story: 63 unit tests (finite-difference Lie checks, QP brute-force cross-checks, closed-loop override proofs), Python benchmark harness with TTC/clearance metrics, and publication plots.*
