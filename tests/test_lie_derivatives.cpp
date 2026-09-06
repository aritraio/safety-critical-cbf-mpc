// Phase-2 verification: analytic Lie derivatives vs. numerics, plus Nagumo /
// forward-invariance spot checks. Uses only the dynamics + barrier layers
// (no QP solver): safe inputs are found by brute-force grid search.

#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>

#include "av_safety/barriers/barrier_params.hpp"
#include "av_safety/barriers/drift_cbf.hpp"
#include "av_safety/barriers/gap_cbf.hpp"
#include "av_safety/barriers/obstacle_cbf.hpp"
#include "av_safety/barriers/road_boundary_cbf.hpp"
#include "av_safety/common/vehicle_params.hpp"
#include "av_safety/dynamics/dynamic_bicycle_model.hpp"

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
  return p;
}

struct LinPoint {
  StateVector x;
  ControlVector u;
  StateDerivative f;
  StateMatrix J;
  InputMatrix G;
};

LinPoint MakeLinPoint(const DynamicBicycleModel& model, const StateVector& x,
                      const ControlVector& u, double mu) {
  LinPoint lp;
  lp.x = x;
  lp.u = u;
  model.affineDecomposition(x, u, mu, lp.f, lp.G);
  model.driftJacobian(x, mu, lp.J);
  return lp;
}

// Representative states: cruise, cornering, low-speed, near-limit.
std::vector<StateVector> ProbeStates() {
  std::vector<StateVector> out;
  StateVector s = StateVector::Zero();
  s << 0.0, 0.5, 0.02, 20.0, 0.1, 0.03;  // cruise, slight offset
  out.push_back(s);
  s << 30.0, -1.5, -0.1, 15.0, -0.8, -0.25;  // cornering
  out.push_back(s);
  s << 5.0, 0.0, 0.5, 3.0, 0.4, 0.3;  // low speed, high yaw
  out.push_back(s);
  s << 100.0, 3.4, 0.0, 25.0, 0.0, 0.0;  // at road edge, fast
  out.push_back(s);
  return out;
}

ControlVector ProbeInput() {
  ControlVector u;
  u << 0.04, 0.5;
  return u;
}

}  // namespace

// Analytic drift Jacobian vs. central finite differences of the drift field.
TEST(LieDerivatives, DriftJacobianVsFiniteDiff) {
  DynamicBicycleModel model(TestParams());
  const ControlVector zero = ControlVector::Zero();
  for (const auto& x : ProbeStates()) {
    StateMatrix Ja;
    model.driftJacobian(x, 0.9, Ja);
    StateMatrix Jn = StateMatrix::Zero();
    const double h = 1e-7;
    for (int j = 0; j < kStateDim; ++j) {
      StateVector xp = x, xm = x;
      xp(j) += h;
      xm(j) -= h;
      StateDerivative fp, fm;
      model.continuousDynamics(xp, zero, 0.9, fp);
      model.continuousDynamics(xm, zero, 0.9, fm);
      Jn.col(j) = (fp - fm) / (2.0 * h);
    }
    for (int i = 0; i < kStateDim; ++i)
      for (int j = 0; j < kStateDim; ++j)
        EXPECT_NEAR(Ja(i, j), Jn(i, j), 1e-5)
            << "J mismatch at (" << i << "," << j << ")";
  }
}

// Road barrier value/rate vs. numerics.
TEST(LieDerivatives, RoadBarrierVsNumeric) {
  RoadBoundaryCbf cbf;
  DynamicBicycleModel model(TestParams());
  for (const auto& x : ProbeStates()) {
    const double h = 3.5 * 3.5 - std::pow(x(kPy), 2);
    EXPECT_NEAR(cbf.value(x), h, 1e-12);
    // hdot vs. directional derivative along full dynamics.
    ControlVector u = ProbeInput();
    StateDerivative xd;
    model.continuousDynamics(x, u, 0.9, xd);
    const double dh = 1e-7;
    StateVector xp = x + dh * xd, xm = x - dh * xd;
    const double fd = (cbf.value(xp) - cbf.value(xm)) / (2.0 * dh);
    StateDerivative f;
    InputMatrix G;
    model.affineDecomposition(x, u, 0.9, f, G);
    EXPECT_NEAR(cbf.hDot(x, f), fd, 1e-5);
  }
}

// Obstacle barrier (static + moving) value/rate vs. numerics.
TEST(LieDerivatives, ObstacleBarrierVsNumeric) {
  ObstacleCbf cbf;
  DynamicBicycleModel model(TestParams());
  ObstacleState obs;
  obs.x = 40.0;
  obs.y = 0.5;
  obs.vx = -2.0;  // oncoming cut-in
  obs.vy = 0.3;
  obs.a = 4.0;
  obs.b = 2.0;
  for (const auto& x : ProbeStates()) {
    ControlVector u = ProbeInput();
    StateDerivative xd;
    model.continuousDynamics(x, u, 0.9, xd);
    // Total derivative dh/dt = grad.f + dh/dt-explicit (obstacle motion).
    const double dt = 1e-7;
    StateVector xp = x + dt * xd, xm = x - dt * xd;
    ObstacleState obsp = obs, obsm = obs;
    obsp.x += dt * obs.vx;
    obsp.y += dt * obs.vy;
    obsm.x -= dt * obs.vx;
    obsm.y -= dt * obs.vy;
    const double fd = (cbf.value(xp, obsp) - cbf.value(xm, obsm)) / (2.0 * dt);
    StateDerivative f;
    InputMatrix G;
    model.affineDecomposition(x, u, 0.9, f, G);
    EXPECT_NEAR(cbf.hDot(x, f, obs), fd, 1e-4);
  }
}

// Drift RD1 Lie derivatives vs. numerics.
TEST(LieDerivatives, DriftLieVsNumeric) {
  DriftEnvelopeCbf cbf;
  DynamicBicycleModel model(TestParams());
  for (const auto& x : ProbeStates()) {
    ControlVector u = ProbeInput();
    const LinPoint lp = MakeLinPoint(model, x, u, 0.9);
    CbfLinearConstraint c;
    cbf.linearize(x, lp.f, lp.G, 0.9, 9.81, c);
    // Lf vs. directional derivative of h along f.
    const double e = 1e-7;
    StateVector xp = x + e * lp.f, xm = x - e * lp.f;
    const double lf_fd = (cbf.value(xp, 0.9, 9.81) - cbf.value(xm, 0.9, 9.81)) / (2.0 * e);
    EXPECT_NEAR(c.Lf, lf_fd, 1e-4);
    // Lg columns vs. directional derivative along G columns.
    for (int j = 0; j < kControlDim; ++j) {
      StateVector gp = x + e * lp.G.col(j), gm = x - e * lp.G.col(j);
      const double lg_fd =
          (cbf.value(gp, 0.9, 9.81) - cbf.value(gm, 0.9, 9.81)) / (2.0 * e);
      EXPECT_NEAR(c.Lg[j], lg_fd, 1e-4);
    }
  }
}

// End-to-end: the barrier's Lie assembly (Lf2, LgLf) matches the exact
// directional derivative of eta(x) = hdot(x) along the affine direction
// w = f + G u. eta is evaluated with fresh drift vectors (no affine
// approximation of the dynamics), so this isolates the barrier math.
// (The residual between the affine and true vector fields is a documented
// modeling approximation handled by relinearizing every cycle in Phase 3.)
TEST(LieDerivatives, EcbfLieAssemblyVsNumeric) {
  DynamicBicycleModel model(TestParams());
  RoadBoundaryCbf road;
  ObstacleCbf obscbf;
  ObstacleState obsStatic;
  obsStatic.x = 60.0;
  obsStatic.y = 0.0;
  obsStatic.a = 4.0;
  obsStatic.b = 2.0;
  ObstacleState obsMoving = obsStatic;
  obsMoving.vx = -3.0;
  obsMoving.vy = 0.4;
  const ControlVector u = ProbeInput();
  const double e = 1e-6;
  for (const auto& x0 : ProbeStates()) {
    const LinPoint lp = MakeLinPoint(model, x0, u, 0.9);
    const StateVector w = lp.f + lp.G * u;  // affine direction
    const StateVector xp = x0 + e * w, xm = x0 - e * w;
    StateDerivative fp, fm;
    InputMatrix dummy;
    const ControlVector zero = ControlVector::Zero();
    model.continuousDynamics(xp, zero, 0.9, fp);
    model.continuousDynamics(xm, zero, 0.9, fm);

    // Road (static eta field).
    EcbffConstraint cr;
    road.linearize(x0, lp.f, lp.J, lp.G, cr);
    const double r_num = (road.hDot(xp, fp) - road.hDot(xm, fm)) / (2.0 * e);
    const double r_pred = cr.Lf2 + cr.LgLf[0] * u(0) + cr.LgLf[1] * u(1);
    EXPECT_NEAR(r_num, r_pred, 1e-3 + 1e-6 * std::abs(r_pred)) << "road ECBF assembly";

    // Obstacle, static and moving (eta field frozen/advected in time).
    for (const auto& obs : {obsStatic, obsMoving}) {
      EcbffConstraint co;
      obscbf.linearize(x0, obs, lp.f, lp.J, lp.G, co);
      ObstacleState obsp = obs, obsm = obs;
      obsp.x += e * obs.vx;
      obsp.y += e * obs.vy;
      obsm.x -= e * obs.vx;
      obsm.y -= e * obs.vy;
      const double o_num =
          (obscbf.hDot(xp, fp, obsp) - obscbf.hDot(xm, fm, obsm)) / (2.0 * e);
      const double o_pred = co.Lf2 + co.LgLf[0] * u(0) + co.LgLf[1] * u(1);
      EXPECT_NEAR(o_num, o_pred, 1e-3 + 1e-6 * std::abs(o_pred)) << "obstacle ECBF assembly";
    }
  }
}

// Nagumo prerequisite: at near-boundary states the CBF condition must be
// feasible (sup over box inputs of LHS >= 0), checked by dense grid search.
TEST(LieDerivatives, NagumoFeasibilityOnBoundary) {
  DynamicBicycleModel model(TestParams());
  RoadBoundaryCbf road;
  ObstacleCbf obscbf;
  DriftEnvelopeCbf drift;
  ObstacleState obs;
  obs.x = 10.0;
  obs.y = 0.0;
  obs.a = 4.0;
  obs.b = 2.0;

  StateVector xr = StateVector::Zero();
  xr << 0.0, 3.3, 0.0, 20.0, 0.6, 0.05;  // near road edge, drifting outward
  StateVector xo = StateVector::Zero();
  xo << 2.0, 1.8, 0.0, 15.0, 0.0, 0.0;  // passing the ellipse flank (avoidable)
  StateVector xd = StateVector::Zero();
  xd << 0.0, 0.0, 0.0, 25.0, 0.0, 0.3;  // near drift limit (vx*r = 7.5)

  const ControlVector u0 = ControlVector::Zero();
  const std::vector<std::pair<StateVector, int>> cases = {{xr, 0}, {xo, 1}, {xd, 2}};
  for (const auto& [x, kind] : cases) {
    const LinPoint lp = MakeLinPoint(model, x, u0, 0.9);
    EcbffConstraint cr, co;
    CbfLinearConstraint cd;
    road.linearize(x, lp.f, lp.J, lp.G, cr);
    obscbf.linearize(x, obs, lp.f, lp.J, lp.G, co);
    drift.linearize(x, lp.f, lp.G, 0.9, 9.81, cd);
    double best = -1e100;
    for (int i = 0; i <= 20; ++i) {
      for (int j = 0; j <= 20; ++j) {
        ControlVector u;
        u << -0.55 + 1.1 * i / 20.0, -6.0 + 9.0 * j / 20.0;
        double lhs = 0.0;
        if (kind == 0)
          lhs = cr.Lf2 + cr.LgLf[0] * u(0) + cr.LgLf[1] * u(1) +
                (cr.p1 + cr.p2) * cr.hdot + cr.p1 * cr.p2 * cr.h;
        else if (kind == 1)
          lhs = co.Lf2 + co.LgLf[0] * u(0) + co.LgLf[1] * u(1) +
                (co.p1 + co.p2) * co.hdot + co.p1 * co.p2 * co.h;
        else
          lhs = cd.Lf + cd.Lg[0] * u(0) + cd.Lg[1] * u(1) + cd.kappa * cd.h;
        best = std::max(best, lhs);
      }
    }
    EXPECT_GE(best, -1e-6) << "CBF condition infeasible for case " << kind;
  }
}

// Forward invariance: a constraint-satisfying constant input keeps h >= 0.
TEST(LieDerivatives, InvarianceUnderSafeControl) {
  DynamicBicycleModel model(TestParams());
  RoadBoundaryCbf road;
  StateVector x = StateVector::Zero();
  x << 0.0, 2.8, 0.0, 18.0, 0.8, 0.1;  // close to edge, moving outward
  // Grid-search a u with good ECBF margin, hold 0.6 s, require h >= 0.
  ControlVector best_u = ControlVector::Zero();
  double best_lhs = -1e100;
  {
    StateDerivative f;
    InputMatrix G;
    StateMatrix J;
    model.affineDecomposition(x, best_u, 0.9, f, G);
    model.driftJacobian(x, 0.9, J);
    EcbffConstraint c;
    road.linearize(x, f, J, G, c);
    for (int i = 0; i <= 40; ++i)
      for (int j = 0; j <= 40; ++j) {
        ControlVector u;
        u << -0.55 + 1.1 * i / 40.0, -6.0 + 9.0 * j / 40.0;
        const double lhs = c.Lf2 + c.LgLf[0] * u(0) + c.LgLf[1] * u(1) +
                           (c.p1 + c.p2) * c.hdot + c.p1 * c.p2 * c.h;
        if (lhs > best_lhs) {
          best_lhs = lhs;
          best_u = u;
        }
      }
    ASSERT_GT(best_lhs, 0.0);
  }
  const double dt = 0.005;
  double h_min = road.value(x);
  for (int k = 0; k < 120; ++k) {
    StateVector xn;
    model.stepRK4(x, best_u, 0.9, dt, xn);
    x = xn;
    h_min = std::min(h_min, road.value(x));
  }
  EXPECT_GE(h_min, -1e-3) << "safe input failed to keep h >= 0";
}

// Barrier config loader reads the shipped YAML (works from repo root or build/).
TEST(LieDerivatives, BarrierConfigLoads) {
  const char* candidates[] = {"config/barrier_params.yaml", "../config/barrier_params.yaml"};
  const char* found = nullptr;
  for (const char* p : candidates) {
    if (FILE* f = std::fopen(p, "r")) {
      std::fclose(f);
      found = p;
      break;
    }
  }
  if (!found) GTEST_SKIP() << "barrier_params.yaml not found; skipping";
  const BarrierConfig c = LoadBarrierConfig(found);
  EXPECT_NEAR(c.road.lane_margin, 3.5, 1e-12);
  EXPECT_GT(c.road.kappa, 0.0);
  EXPECT_GT(c.obstacle.kappa, 0.0);
  EXPECT_GT(c.drift.kappa, 0.0);
  EXPECT_GT(c.gap.t_headway, 0.0);
  EXPECT_GE(c.gap.d_min, 0.0);
  EXPECT_GT(c.gap.kappa, 0.0);
  EXPECT_GT(c.gap.lateral_gate, 0.0);
}

// Gap-barrier value/gating/Lie derivatives vs. numerics.
TEST(LieDerivatives, GapBarrierVsNumeric) {
  GapBarrierParams gp;
  gp.t_headway = 1.2;
  gp.d_min = 4.0;
  gp.kappa = 2.0;
  gp.lateral_gate = 2.0;
  GapBarrierCbf cbf(gp);
  DynamicBicycleModel model(TestParams());

  ObstacleState lead;
  lead.x = 40.0;
  lead.y = 0.0;
  lead.vx = 15.0;
  lead.vy = 0.0;
  lead.a = 4.0;
  lead.b = 2.0;

  // Gating: ahead + near-lane applies; behind or far-lane does not.
  StateVector x = StateVector::Zero();
  x(kVx) = 19.0;
  EXPECT_TRUE(cbf.applies(x, lead));
  ObstacleState behind = lead;
  behind.x = -10.0;
  EXPECT_FALSE(cbf.applies(x, behind));
  ObstacleState far = lead;
  far.y = 5.0;
  EXPECT_FALSE(cbf.applies(x, far));

  // Value: gap - d_min - t_h * vx.
  EXPECT_NEAR(cbf.value(x, lead), 40.0 - 4.0 - 1.2 * 19.0, 1e-12);

  // Lie derivatives vs. central differences along f and G columns.
  ControlVector u;
  u << 0.02, -1.0;
  StateDerivative f;
  InputMatrix G;
  StateMatrix J;
  model.affineDecomposition(x, u, 0.9, f, G);
  model.driftJacobian(x, 0.9, J);
  CbfLinearConstraint c;
  cbf.linearize(x, lead, f, G, c);
  // NOTE: Lf includes the obstacle-motion rate (vx_obs, cf. moving-obstacle
  // hdot): the FD below must advect the obstacle, like ObstacleBarrierVsNumeric.
  const double e = 1e-7;
  ObstacleState lead_p = lead, lead_m = lead;
  lead_p.x += e * lead.vx;
  lead_m.x -= e * lead.vx;
  StateVector xp = x + e * f, xm = x - e * f;
  const double lf_fd =
      (cbf.value(xp, lead_p) - cbf.value(xm, lead_m)) / (2.0 * e);
  EXPECT_NEAR(c.Lf, lf_fd, 1e-4);
  for (int j = 0; j < kControlDim; ++j) {
    StateVector gp2 = x + e * G.col(j), gm = x - e * G.col(j);
    const double lg_fd = (cbf.value(gp2, lead) - cbf.value(gm, lead)) / (2.0 * e);
    EXPECT_NEAR(c.Lg[j], lg_fd, 1e-4);
  }
  // Ax channel dominates (RWD): braking authority = -t_h exactly. Steering
  // couples weakly POSITIVE (tire scrub: more steer drags speed, opening the
  // headway) — correct physics, counterintuitive sign, pinned here.
  EXPECT_NEAR(c.Lg[1], -1.2, 1e-6);
  EXPECT_GT(c.Lg[0], 0.0);
  EXPECT_LT(c.Lg[0], 5.0);
}
