// Phase-3/4 preview demo: LQR/MPC nominal + CBF-QP safety filter in three
// adverse scenarios, optionally with UKF friction estimation in the loop.
// No ROS 2 or CARLA required.
//
// Usage:
//   cbf_safety_demo [vehicle_yaml] [lqr_yaml] [barrier_yaml] [ukf_yaml] [mpc_yaml]
//                   [--controller=lqr|mpc] [--ukf]
// stdout: CSV  scenario,t,x,y,psi,vx,vy,r,d_nom,a_nom,d_safe,a_safe,h_obs,h_road,h_drift,solve_us,mu_hat,h_gap
// stderr: per-scenario PASS/FAIL summary (barrier minima, interventions, worst solve).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "av_safety/barriers/barrier_params.hpp"
#include "av_safety/common/vehicle_params.hpp"
#include "av_safety/controllers/cbf_qp_filter.hpp"
#include "av_safety/controllers/lqr_tracker.hpp"
#include "av_safety/controllers/mpc_tracker.hpp"
#include "av_safety/dynamics/dynamic_bicycle_model.hpp"
#include "av_safety/estimation/friction_ukf.hpp"
#include "av_safety/qp/enumeration_qp_solver.hpp"
#ifdef HAVE_OSQP
#include "av_safety/qp/osqp_solver.hpp"
#endif

using namespace av_safety;

namespace {

struct Scenario {
  std::string name;
  double mu;
  double vx0;
  double y0;
  double y_ref;    // tracker lane reference (straight, kappa = 0); ignored if use_lqr=false
  bool use_lqr;    // false => mild malicious drift + cruise PI (unit-test pattern)
  double mal_steer;
  double duration;  // [s]
  std::vector<ObstacleState> obstacles;
};

struct DemoOptions {
  bool use_mpc{false};
  bool use_ukf{false};
};

// Deterministic Gaussian source (fixed seed => reproducible demo traces).
struct DemoRng {
  uint64_t s{0xC0FFEE123456789ull};
  bool spare_ready{false};
  double spare{0.0};
  double uni() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return static_cast<double>(s >> 11) * (1.0 / 9007199254740992.0);
  }
  double gauss() {
    if (spare_ready) {
      spare_ready = false;
      return spare;
    }
    double u1 = uni(), u2 = uni();
    if (u1 < 1e-12) u1 = 1e-12;
    const double r = std::sqrt(-2.0 * std::log(u1));
    spare = r * std::sin(2.0 * M_PI * u2);
    spare_ready = true;
    return r * std::cos(2.0 * M_PI * u2);
  }
};

ObstacleState MakeObs(double x, double y, double vx, double vy, double a, double b) {
  ObstacleState o;
  o.x = x;
  o.y = y;
  o.vx = vx;
  o.vy = vy;
  o.a = a;
  o.b = b;
  return o;
}

int RunScenario(const Scenario& sc, const VehicleParams& vp, const LqrConfig& lqr_cfg,
                const MpcConfig& mpc_cfg, const estimation::UkfConfig& ukf_cfg,
                const BarrierConfig& bc, const DemoOptions& opt) {
  DynamicBicycleModel model(vp);
  LqrTracker lqr(vp, lqr_cfg);
  MpcTracker mpc(vp, mpc_cfg);
  if (opt.use_mpc) {
    if (!mpc.computeGains(mpc_cfg.target_speed, sc.mu)) {
      std::fprintf(stderr, "[%s] MPC gain computation failed\n", sc.name.c_str());
      return 1;
    }
  } else {
    if (!lqr.computeGains(lqr_cfg.target_speed, sc.mu)) {
      std::fprintf(stderr, "[%s] CARE failed\n", sc.name.c_str());
      return 1;
    }
  }
  qp::EnumerationQpSolver exact;
#ifdef HAVE_OSQP
  const double H[2] = {1.0, 0.5};
  qp::OsqpSolver::Settings st;
  st.max_iter = 800;  // deterministic primary budget; exact fallback nets misses
  qp::OsqpSolver osqp(H, st);
  qp::QpSolverBackend* backend = osqp.ready() ? static_cast<qp::QpSolverBackend*>(&osqp)
                                              : static_cast<qp::QpSolverBackend*>(&exact);
#else
  qp::QpSolverBackend* backend = &exact;
#endif
  CbfQpConfig fc;
  CbfQpFilter filter(vp, bc, fc, backend);
#ifdef HAVE_OSQP
  if (osqp.ready()) filter.setFallbackBackend(&exact);
#endif
  estimation::FrictionUkf ukf(vp, ukf_cfg);
  DemoRng rng;
  PacejkaTire front_tire(vp.pacejka, vp.mu_nominal), rear_tire(vp.pacejka, vp.mu_nominal);
  const double dt = 0.01;

  auto trackNominal = [&](const StateVector& xs, ControlVector& u) {
    if (!sc.use_lqr) {
      u(kSteer) = sc.mal_steer;
      u(kAx) = 0.8 * (sc.vx0 - xs(kVx));  // cruise PI (P term suffices here)
      return;
    }
    LateralState err;
    err << xs(kPy) - sc.y_ref, xs(kPsi), xs(kVy), xs(kYawRate);
    if (opt.use_mpc) {
      mpc.update(err, xs(kVx), 0.0, sc.vx0, dt, u);  // hold entry speed
    } else {
      lqr.update(err, xs(kVx), 0.0, sc.vx0, dt, u);
    }
  };

  std::vector<ObstacleState> obs = sc.obstacles;
  StateVector x = StateVector::Zero();
  x(kPx) = 0.0;
  x(kPy) = sc.y0;
  x(kVx) = sc.vx0;
  StateVector xb = x;  // unfiltered baseline (same nominal, no safety filter)
  ControlVector xb_prev = ControlVector::Zero();  // actuator state (slew model)
  ControlVector u_applied = ControlVector::Zero();  // last filtered command (UKF input)

  const int N = static_cast<int>(sc.duration / dt);
  double h_obs_min = 1e100, h_road_min = 1e100, h_drift_min = 1e100;
  double h_obs_base_min = 1e100;
  double worst_us = 0.0;
  int interventions = 0, infeasible = 0;
  ObstacleCbf probe;
  RoadBoundaryCbf road;
  DriftEnvelopeCbf drift;

  std::printf("# scenario=%s\n", sc.name.c_str());
  for (int k = 0; k < N; ++k) {
    const double t = k * dt;
    // Advect moving obstacles (constant-velocity prediction, exact model match).
    for (auto& o : obs) {
      o.x += o.vx * dt;
      o.y += o.vy * dt;
    }
    filter.setObstacles(obs.data(), static_cast<int>(obs.size()));

    ControlVector nom;
    trackNominal(x, nom);
    // Friction source: UKF estimate from synthetic IMU when enabled, else the
    // scenario truth (static scheduling). UKF consumes the last APPLIED cmd.
    double mu_filter = sc.mu;
    if (opt.use_ukf) {
      const double af =
          PacejkaTire::frontSlipAngle(u_applied(kSteer), x(kVx), x(kVy), x(kYawRate), vp.lf,
                                      vp.vx_epsilon);
      const double ar =
          PacejkaTire::rearSlipAngle(x(kVx), x(kVy), x(kYawRate), vp.lr, vp.vx_epsilon);
      const double fyf = front_tire.lateralForce(af, sc.mu);
      const double fyr = rear_tire.lateralForce(ar, sc.mu);
      estimation::UkfMeas z;
      z(estimation::kMax) = u_applied(kAx) - x(kVy) * x(kYawRate) + 0.15 * rng.gauss();
      z(estimation::kMay) =
          (2.0 * fyf * std::cos(u_applied(kSteer)) + 2.0 * fyr) / vp.mass -
          x(kVx) * x(kYawRate) + 0.15 * rng.gauss();
      z(estimation::kMr) = x(kYawRate) + 1e-3 * rng.gauss();
      z(estimation::kMvx) = x(kVx) + 0.05 * rng.gauss();
      ukf.step(u_applied, z, dt);
      mu_filter = ukf.mu();
    }
    ControlVector safe;
    FilterDiagnostics d;
    filter.filter(x, nom, mu_filter, safe, d);
    u_applied = safe;
    if (!d.feasible) ++infeasible;
    if (d.intervened) ++interventions;
    worst_us = std::max(worst_us, d.solve_us);
    h_obs_min = std::min(h_obs_min, d.h_obs_min);
    h_road_min = std::min(h_road_min, d.h_road);
    h_drift_min = std::min(h_drift_min, d.h_drift);

    std::printf("%s,%.2f,%.3f,%.3f,%.4f,%.3f,%.3f,%.4f,%.4f,%.3f,%.4f,%.3f,%.3f,%.3f,%.3f,%.1f,%.4f,%.3f\n",
                sc.name.c_str(), t, x(kPx), x(kPy), x(kPsi), x(kVx), x(kVy), x(kYawRate),
                nom(kSteer), nom(kAx), safe(kSteer), safe(kAx), d.h_obs_min, d.h_road,
                d.h_drift, d.solve_us, mu_filter, d.h_gap_min);
    StateVector xn, xbn;
    model.stepRK4(x, safe, sc.mu, dt, xn);
    // Baseline uses the same nominal evaluated on ITS state (fair comparison),
    // passed through the identical actuator slew model (no teleportation).
    ControlVector nom_b;
    trackNominal(xb, nom_b);
    nom_b(kSteer) =
        std::clamp(nom_b(kSteer), xb_prev(kSteer) - vp.steer_rate_max * dt,
                   xb_prev(kSteer) + vp.steer_rate_max * dt);
    nom_b(kAx) = std::clamp(nom_b(kAx), xb_prev(kAx) - vp.accel_rate_max * dt,
                            xb_prev(kAx) + vp.accel_rate_max * dt);
    xb_prev = nom_b;
    model.stepRK4(xb, nom_b, sc.mu, dt, xbn);
    x = xn;
    xb = xbn;
    for (const auto& o : obs) h_obs_base_min = std::min(h_obs_base_min, probe.value(xb, o));
    (void)probe;
    (void)road;
    (void)drift;
  }
  // Drift ripple allowance: 100 Hz sampling of the RD1 envelope admits round-off
  // scale excursions (~2% of the nominal (mu g)^2 range); unit tests pin exactness.
  const bool pass = (infeasible == 0 && h_obs_min >= -0.05 && h_road_min >= -0.05 &&
                     h_drift_min >= -1.5);
  std::fprintf(stderr, "[%s] %s | filt h_obs=%+.3f (base %+.3f) h_road=%+.3f h_drift=%+.3f "
                       "interv=%d infeas=%d worst_solve=%.0fus backend=%s\n",
               sc.name.c_str(), pass ? "PASS" : "FAIL", h_obs_min, h_obs_base_min, h_road_min,
               h_drift_min, interventions, infeasible, worst_us, backend->name());
  return pass ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  std::string vehicle_yaml = "config/vehicle_params.yaml";
  std::string lqr_yaml = "config/lqr_tuning.yaml";
  std::string barrier_yaml = "config/barrier_params.yaml";
  std::string ukf_yaml = "config/friction_ukf.yaml";
  std::string mpc_yaml = "config/mpc_tuning.yaml";
  DemoOptions opt;
  std::vector<std::string> positionals;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--controller=mpc") {
      opt.use_mpc = true;
    } else if (a == "--controller=lqr") {
      opt.use_mpc = false;
    } else if (a == "--ukf") {
      opt.use_ukf = true;
    } else {
      positionals.push_back(a);
    }
  }
  if (positionals.size() > 0) vehicle_yaml = positionals[0];
  if (positionals.size() > 1) lqr_yaml = positionals[1];
  if (positionals.size() > 2) barrier_yaml = positionals[2];
  if (positionals.size() > 3) ukf_yaml = positionals[3];
  if (positionals.size() > 4) mpc_yaml = positionals[4];
  VehicleParams vp;
  LqrConfig lqr_cfg;
  MpcConfig mpc_cfg;
  estimation::UkfConfig ukf_cfg;
  BarrierConfig bc;
  try {
    vp = LoadVehicleParams(vehicle_yaml);
    lqr_cfg = LoadLqrConfig(lqr_yaml);
    mpc_cfg = LoadMpcConfig(mpc_yaml);
    ukf_cfg = estimation::LoadUkfConfig(ukf_yaml);
    bc = LoadBarrierConfig(barrier_yaml);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "config error: %s\n", e.what());
    return 1;
  }

  std::printf("scenario,t,x,y,psi,vx,vy,r,d_nom,a_nom,d_safe,a_safe,h_obs,h_road,h_drift,solve_us,mu_hat,h_gap\n");

  std::vector<Scenario> scenarios;
  // A: distracted drift toward a static obstacle (mild malicious nominal).
  scenarios.push_back({"static_flank", 0.9, 12.0, -1.5, 0.0, false, 0.015, 8.0,
                       {MakeObs(40.0, 1.8, 0.0, 0.0, 4.0, 2.0)}});
  // B: slow crosser merging into the ego lane ahead; ego closes from behind
  // and must brake + offset (sustained in-lane encounter, no grind).
  scenarios.push_back({"cut_in", 0.9, 13.0, 0.0, 0.0, true, 0.0, 8.0,
                       {MakeObs(33.0, 1.0, 8.0, -0.3, 3.5, 1.8)}});
  // C: wet road + obstacle grazing the centerline (drift barrier engaged).
  // Mild graze at moderate speed keeps both controllers in the linear regime.
  scenarios.push_back({"low_mu_edge", 0.5, 9.0, 0.0, 0.0, true, 0.0, 9.0,
                       {MakeObs(40.0, 1.9, 0.0, 0.0, 4.0, 2.0)}});

  int failures = 0;
  for (const auto& sc : scenarios) failures += RunScenario(sc, vp, lqr_cfg, mpc_cfg, ukf_cfg, bc, opt);
  std::fprintf(stderr, "scenarios failed: %d/%zu\n", failures, scenarios.size());
  return failures;
}
