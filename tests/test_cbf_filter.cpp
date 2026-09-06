// Phase-3 verification: the CBF-QP filter overrides unsafe nominal commands
// (obstacle / road / drift scenarios), respects input bounds, degrades
// gracefully when infeasible, and runs inside the 1 ms budget.

#include <chrono>
#include <cmath>
#include <gtest/gtest.h>

#include "av_safety/common/vehicle_params.hpp"
#include "av_safety/controllers/cbf_qp_filter.hpp"
#include "av_safety/controllers/lqr_tracker.hpp"
#include "av_safety/dynamics/dynamic_bicycle_model.hpp"
#include "av_safety/qp/enumeration_qp_solver.hpp"
#ifdef HAVE_OSQP
#include "av_safety/qp/osqp_solver.hpp"
#endif

using namespace av_safety;

namespace {

VehicleParams TestParams() {
  VehicleParams p;
  p.mass = 1500.0;
  p.yaw_inertia = 2500.0;
  p.lf = 1.2;
  p.lr = 1.3;
  p.pacejka = PacejkaParams{};
  p.mu_nominal = 0.9;
  p.drive_bias_front = 0.0;
  p.steer_max = 0.55;
  p.accel_max = 3.0;
  p.accel_min = -6.0;
  p.vx_epsilon = 1.0;
  p.gravity = 9.81;
  return p;
}

BarrierConfig TestBarriers() {
  BarrierConfig b;
  b.road.lane_margin = 3.5;
  b.road.lane_center_y = 0.0;
  b.road.kappa = 2.0;
  b.obstacle.kappa = 2.0;
  b.drift.kappa = 3.0;
  return b;
}

struct Fixture {
  VehicleParams vp = TestParams();
  DynamicBicycleModel model{vp};
  qp::EnumerationQpSolver backend{};
  CbfQpConfig cfg{};
  CbfQpFilter filter{vp, TestBarriers(), cfg, &backend};
};

}  // namespace

// Safe cruise: filter must pass the nominal command through untouched.
TEST(CbfFilter, PassThroughWhenSafe) {
  Fixture fx;
  StateVector x = StateVector::Zero();
  x(kVx) = 15.0;
  ControlVector nom;
  nom << 0.0, 0.0;
  ControlVector safe;
  FilterDiagnostics d;
  fx.filter.filter(x, nom, 0.9, safe, d);
  EXPECT_TRUE(d.feasible);
  EXPECT_TRUE(d.optimal);
  EXPECT_NEAR(safe(kSteer), 0.0, 1e-6);
  EXPECT_NEAR(safe(kAx), 0.0, 1e-6);
  EXPECT_FALSE(d.intervened);
}

// Closed loop: a malicious nominal drifts from the lower lane toward a static
// obstacle whose top extends past the road edge, so passing below is the
// only option. Moderate speed keeps tire response in the linear regime where
// the quasi-affine model is valid. The filter must hold the lower side and
// keep h_obs >= 0 while the unfiltered baseline enters the ellipse.
// (Dead-ahead high-speed approaches are physically doomed once inside the
// braking distance — see InfeasibleDegradesGracefully.)
TEST(CbfFilter, ObstacleOverrideKeepsSafe) {
  Fixture fx;
  ObstacleState obs;
  obs.x = 40.0;
  obs.y = 1.8;
  obs.vx = 0.0;
  obs.vy = 0.0;
  obs.a = 4.0;
  obs.b = 2.0;
  fx.filter.setObstacles(&obs, 1);

  ObstacleCbf probe;
  RoadBoundaryCbf road;
  const ControlVector nom = (ControlVector() << 0.015, 0.0).finished();  // drifts into obstacle
  const double dt = 0.01;
  double h_min = 1e100, h_base_min = 1e100, h_road_min = 1e100;
  bool ever_intervened = false;
  double worst_us = 0.0;

  StateVector x = StateVector::Zero();      // filtered branch
  x << 0.0, -1.5, 0.0, 12.0, 0.0, 0.0;
  StateVector xb = x;                        // unfiltered baseline branch
  for (int k = 0; k < 800; ++k) {            // 8 s
    ControlVector safe;
    FilterDiagnostics d;
    fx.filter.filter(x, nom, 0.9, safe, d);
    EXPECT_TRUE(d.feasible) << "infeasible at k=" << k;
    EXPECT_LE(std::abs(safe(kSteer)), fx.vp.steer_max + 1e-9);
    EXPECT_GE(safe(kAx), fx.vp.accel_min - 1e-9);
    EXPECT_LE(safe(kAx), fx.vp.accel_max + 1e-9);
    ever_intervened = ever_intervened || d.intervened;
    worst_us = std::max(worst_us, d.solve_us);
    h_min = std::min(h_min, probe.value(x, obs));
    h_road_min = std::min(h_road_min, road.value(x));
    StateVector xn, xbn;
    fx.model.stepRK4(x, safe, 0.9, dt, xn);
    fx.model.stepRK4(xb, nom, 0.9, dt, xbn);
    x = xn;
    xb = xbn;
    h_base_min = std::min(h_base_min, probe.value(xb, obs));
    if (x(kPx) > 85.0) break;
  }
  EXPECT_LT(h_base_min, -0.05) << "baseline never entered: scenario proves nothing";
  EXPECT_TRUE(ever_intervened) << "filter never overrode the unsafe nominal";
  EXPECT_GE(h_min, -5e-2) << "ellipse barrier violated";
  EXPECT_GE(h_road_min, -5e-2) << "left the road during avoidance";
  EXPECT_LT(worst_us, 1000.0) << "QP exceeded 1 ms budget";
}

// Nominal steering off the road: closed-loop rollout must stay in the lane,
// and the second consecutive solution must satisfy the formulated road row
// exactly (validates filter row assembly + solver in situ; the row is built
// at the filter's own linearization point, removing relinearization skew).
TEST(CbfFilter, RoadDepartureOverride) {
  Fixture fx;
  StateVector x = StateVector::Zero();
  x << 0.0, 2.6, 0.05, 12.0, 0.2, 0.03;  // near edge, edging outward (moderate)
  const ControlVector nom = (ControlVector() << 0.08, 0.0).finished();
  const double dt = 0.01;
  RoadBoundaryCbf road;
  double h_min = road.value(x);
  bool ever_intervened = false;
  for (int k = 0; k < 200; ++k) {
    ControlVector safe;
    FilterDiagnostics d;
    fx.filter.filter(x, nom, 0.9, safe, d);
    EXPECT_TRUE(d.feasible) << "infeasible at k=" << k;
    ever_intervened = ever_intervened || d.intervened;
    StateVector xn;
    fx.model.stepRK4(x, safe, 0.9, dt, xn);
    x = xn;
    h_min = std::min(h_min, road.value(x));
  }
  EXPECT_TRUE(ever_intervened);
  EXPECT_GE(h_min, -5e-2) << "left the road under outward nominal";

  // In-situ row check in hard-ablation mode: two consecutive calls, verify 2nd
  // solution against a row linearized at the 1st solution (= 2nd call's u_prev).
  CbfQpConfig hard_cfg;
  hard_cfg.use_slack = false;
  CbfQpFilter hard_filter(fx.vp, TestBarriers(), hard_cfg, &fx.backend);
  ControlVector s1, s2;
  FilterDiagnostics d1, d2;
  hard_filter.filter(x, nom, 0.9, s1, d1);
  hard_filter.filter(x, nom, 0.9, s2, d2);
  ASSERT_TRUE(d2.feasible);
  StateDerivative f;
  InputMatrix G;
  StateMatrix J;
  fx.model.affineDecomposition(x, s1, 0.9, f, G);
  fx.model.driftJacobian(x, 0.9, J);
  EcbffConstraint c;
  road.linearize(x, f, J, G, c);
  const double lhs = c.Lf2 + c.LgLf[0] * s2(kSteer) + c.LgLf[1] * s2(kAx) +
                     (c.p1 + c.p2) * c.hdot + c.p1 * c.p2 * c.h;
  EXPECT_GE(lhs, -1e-6);
}

// Doomed head-on (stopping distance exceeded): soft CBF sheds slack instead of
// going infeasible — progressive braking at the slew limit, finite bounded
// output, flagged via slack. (First step can only reach the slew box around
// u_prev = 0, so the test marches the filter to full braking.)
TEST(CbfFilter, SoftSlackShedsGracefully) {
  Fixture fx;
  ObstacleState obs;
  obs.x = 10.0;
  obs.y = 0.0;
  obs.a = 4.0;
  obs.b = 2.0;
  fx.filter.setObstacles(&obs, 1);
  StateVector x = StateVector::Zero();
  x(kVx) = 18.0;  // 8 m from a 4 m ellipse at 18 m/s: cannot stop in time
  ControlVector nom;
  nom << 0.0, 0.0;
  ControlVector safe;
  FilterDiagnostics d;
  for (int k = 0; k < 120; ++k) {
    fx.filter.filter(x, nom, 0.9, safe, d);
    ASSERT_TRUE(d.feasible) << "k=" << k;
    ASSERT_TRUE(std::isfinite(safe(kSteer)) && std::isfinite(safe(kAx)));
    EXPECT_LE(std::abs(safe(kSteer)), fx.vp.steer_max + 1e-9);
    EXPECT_GE(safe(kAx), fx.vp.accel_min - 1e-9);
    EXPECT_LE(safe(kAx), fx.vp.accel_max + 1e-9);
  }
  EXPECT_TRUE(d.softened);
  EXPECT_GT(d.slack_max, 0.5);
  // Best effort ramps to full braking at the slew limit (8 m/s^3 * 0.01 s).
  EXPECT_NEAR(safe(kAx), fx.vp.accel_min, 0.2);
}

// Slew-rate box: per-step command jumps respect rate limits around u_prev.
TEST(CbfFilter, SlewRateRespected) {
  Fixture fx;
  StateVector x = StateVector::Zero();
  x(kVx) = 12.0;
  ControlVector nom, safe, prev;
  FilterDiagnostics d;
  prev << 0.0, 0.0;
  double seed[5][2] = {{0.5, 2.0}, {-0.5, -5.0}, {0.1, 0.0}, {0.0, 2.9}, {-0.2, -1.0}};
  for (const auto& sd : seed) {
    nom << sd[0], sd[1];
    fx.filter.filter(x, nom, 0.9, safe, d);
    ASSERT_TRUE(d.feasible);
    EXPECT_LE(std::abs(safe(kSteer) - prev(kSteer)), fx.vp.steer_rate_max * 0.01 + 1e-9);
    EXPECT_LE(std::abs(safe(kAx) - prev(kAx)), fx.vp.accel_rate_max * 0.01 + 1e-9);
    prev = safe;
  }
}

// Headway barrier: dumb cruise nominal behind a slower lead car must brake
// early and settle into following (no contact, no late panic). The ellipse
// ECBF alone stays quiet until close; the gap row provides the foresight.
TEST(CbfFilter, GapBarrierBrakesEarly) {
  Fixture fx;
  ObstacleState lead;
  lead.x = 45.0;
  lead.y = 0.0;
  lead.vx = 15.0;
  lead.vy = 0.0;
  lead.a = 4.0;
  lead.b = 2.0;
  fx.filter.setObstacles(&lead, 1);

  ObstacleCbf probe;
  StateVector x = StateVector::Zero();
  x(kVx) = 19.4;  // 70 km/h closing on a 15 m/s lead car
  const ControlVector nom = (ControlVector() << 0.0, 0.0).finished();
  const double dt = 0.01;
  double h_min = 1e100, gap_min = 1e100, ax_min = 1e100;
  bool braked_early = false;
  for (int k = 0; k < 900; ++k) {
    lead.x += lead.vx * dt;  // lead holds speed
    fx.filter.setObstacles(&lead, 1);
    ControlVector safe;
    FilterDiagnostics d;
    fx.filter.filter(x, nom, 0.9, safe, d);
    ASSERT_TRUE(d.feasible) << "infeasible at k=" << k;
    h_min = std::min(h_min, probe.value(x, lead));
    gap_min = std::min(gap_min, lead.x - x(kPx));
    ax_min = std::min(ax_min, safe(kAx));
    // Headway need at 19.4 m/s is d_min + t_h*vx = 27.3 m; the row must act
    // as the gap approaches it (~3.5 s), well before the ellipse binds.
    if (k * dt < 5.0 && safe(kAx) < -0.5) braked_early = true;
    StateVector xn;
    fx.model.stepRK4(x, safe, 0.9, dt, xn);
    x = xn;
  }
  EXPECT_TRUE(braked_early) << "gap row failed to brake on approach";
  EXPECT_LT(ax_min, -1.5);
  EXPECT_GE(h_min, -5e-2) << "ellipse violated during follow";
  EXPECT_GE(gap_min, 4.0 - 1.5) << "rear-ended (gap under d_min)";
  EXPECT_NEAR(x(kVx), 15.0, 2.0) << "did not settle to lead speed";
}

// Full pipeline timing: LQR + all barrier linearizations + QP under 1 ms.
TEST(CbfFilter, RealtimeBudget) {
  Fixture fx;
  LqrConfig lc;
  lc.Q.setZero();
  lc.Q(0, 0) = 8.0;
  lc.Q(1, 1) = 6.0;
  lc.Q(2, 2) = 1.0;
  lc.Q(3, 3) = 1.0;
  lc.R = 1.0;
  lc.kp_lon = 0.8;
  lc.ki_lon = 0.15;
  LqrTracker tracker(fx.vp, lc);
  ASSERT_TRUE(tracker.computeGains(20.0, 0.9));
  ObstacleState obs;
  obs.x = 50.0;
  obs.y = 1.0;
  obs.a = 4.0;
  obs.b = 2.0;
  fx.filter.setObstacles(&obs, 1);

  StateVector x = StateVector::Zero();
  x(kVx) = 20.0;
  x(kPy) = 0.5;
  LateralState err;
  err << 0.5, 0.02, 0.1, 0.02;
  ControlVector nom, safe;
  FilterDiagnostics d;
  for (int i = 0; i < 200; ++i) {
    tracker.update(err, 20.0, 0.0, 20.0, 0.01, nom);
    fx.filter.filter(x, nom, 0.9, safe, d);
  }
  const auto t0 = std::chrono::steady_clock::now();
  constexpr int N = 2000;
  for (int i = 0; i < N; ++i) {
    tracker.update(err, 20.0, 0.0, 20.0, 0.01, nom);
    fx.filter.filter(x, nom, 0.9, safe, d);
  }
  const auto t1 = std::chrono::steady_clock::now();
  const double us =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) /
      N / 1000.0;
  EXPECT_LT(us, 1000.0) << "full pipeline exceeds 1 ms";
}

// Deterministic primary failure (stub backend) => exact fallback takes over,
// solution stays correct and the event is diagnosed.
TEST(CbfFilter, FallbackEngages) {
  struct FailBackend : public qp::QpSolverBackend {
    bool solve(const qp::TinyQpProblem&, qp::TinyQpSolution& sol) noexcept override {
      sol.optimal = false;
      return false;
    }
    const char* name() const noexcept override { return "fail"; }
  };
  Fixture fx;
  FailBackend fail;
  fx.filter.setFallbackBackend(&fx.backend);  // exact enumeration net
  // Swap primary to the failing stub via a second filter sharing the fixture.
  CbfQpFilter filter(fx.vp, TestBarriers(), CbfQpConfig{}, &fail);
  filter.setFallbackBackend(&fx.backend);
  StateVector x = StateVector::Zero();
  x(kVx) = 12.0;
  ControlVector nom, safe, ref;
  FilterDiagnostics d, dref;
  nom << 0.05, 0.5;
  filter.filter(x, nom, 0.9, safe, d);
  fx.filter.filter(x, nom, 0.9, ref, dref);  // direct exact reference
  EXPECT_TRUE(d.feasible);
  EXPECT_TRUE(d.used_fallback);
  EXPECT_NEAR(safe(kSteer), ref(kSteer), 1e-9);
  EXPECT_NEAR(safe(kAx), ref(kAx), 1e-9);
}

#ifdef HAVE_OSQP
// OSQP primary + enumeration fallback on the override scenario (production
// latency pattern: tight OSQP budget, exact fallback net).
TEST(CbfFilter, OsqpBackendOverride) {
  VehicleParams vp = TestParams();
  DynamicBicycleModel model(vp);
  const double H[2] = {1.0, 0.5};
  qp::OsqpSolver::Settings st;
  st.max_iter = 800;  // deterministic primary budget (~0.6 ms worst)
  qp::OsqpSolver osqp(H, st);
  ASSERT_TRUE(osqp.ready());
  qp::EnumerationQpSolver exact;
  CbfQpConfig cfg;
  CbfQpFilter filter(vp, TestBarriers(), cfg, &osqp);
  filter.setFallbackBackend(&exact);
  ObstacleState obs;
  obs.x = 40.0;
  obs.y = 1.8;
  obs.a = 4.0;
  obs.b = 2.0;
  filter.setObstacles(&obs, 1);
  ObstacleCbf probe;
  StateVector x = StateVector::Zero();
  x << 0.0, -1.5, 0.0, 12.0, 0.0, 0.0;
  const double dt = 0.01;
  double h_min = 1e100;
  bool ever_intervened = false;
  const ControlVector nom = (ControlVector() << 0.015, 0.0).finished();
  for (int k = 0; k < 800; ++k) {
    ControlVector safe;
    FilterDiagnostics d;
    filter.filter(x, nom, 0.9, safe, d);
    EXPECT_TRUE(d.feasible) << "infeasible at k=" << k;
    ever_intervened = ever_intervened || d.intervened;
    h_min = std::min(h_min, probe.value(x, obs));
    StateVector xn;
    model.stepRK4(x, safe, 0.9, dt, xn);
    x = xn;
    if (x(kPx) > 85.0) break;
  }
  EXPECT_TRUE(ever_intervened);
  EXPECT_GE(h_min, -0.15) << "OSQP-filtered trajectory violated the ellipse";
}
#endif
