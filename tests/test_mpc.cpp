// Module-C verification: condensed MPC tracker — solver sanity, closed-loop
// tracking (straight + curve), input-box respect, preview consistency,
// post-solve safety without gains, and the 1 ms budget.

#include <chrono>
#include <cmath>
#include <gtest/gtest.h>

#include "av_safety/common/vehicle_params.hpp"
#include "av_safety/controllers/lqr_tracker.hpp"
#include "av_safety/controllers/mpc_tracker.hpp"
#include "av_safety/dynamics/dynamic_bicycle_model.hpp"
#ifdef HAVE_OSQP
#include "av_safety/qp/mpc_osqp_solver.hpp"
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

MpcConfig TestMpcConfig() {
  MpcConfig c;
  c.Q.setZero();
  c.Q(0, 0) = 8.0;
  c.Q(1, 1) = 1.0;
  c.Q(2, 2) = 6.0;
  c.Q(3, 3) = 1.0;
  c.R = 1.0;
  c.R_delta = 0.5;
  c.dt = 0.05;
  c.kp_lon = 0.8;
  c.ki_lon = 0.15;
  c.integrator_max = 2.0;
  return c;
}

}  // namespace

#ifdef HAVE_OSQP
// Condensed solver sanity: unconstrained-inside optimum recovered exactly.
TEST(MpcSolver, RecoversUnconstrainedOptimum) {
  qp::MpcOsqpSolver solver(-0.55, 0.55);
  ASSERT_TRUE(solver.ready());
  Eigen::Matrix<double, qp::kMpcHorizon, qp::kMpcHorizon> H =
      Eigen::Matrix<double, qp::kMpcHorizon, qp::kMpcHorizon>::Identity();
  H(0, 0) = 2.0;
  Eigen::Matrix<double, qp::kMpcHorizon, 1> g, Ustar, U;
  for (int i = 0; i < qp::kMpcHorizon; ++i) g(i, 0) = 0.01 * (i + 1);
  Ustar = -H.ldlt().solve(g);  // strictly inside the box
  for (int i = 0; i < qp::kMpcHorizon; ++i) ASSERT_LT(std::abs(Ustar(i, 0)), 0.5);
  int iters = 0;
  ASSERT_TRUE(solver.solve(H, g, U, iters));
  for (int i = 0; i < qp::kMpcHorizon; ++i) EXPECT_NEAR(U(i, 0), Ustar(i, 0), 1e-4);
}

// Box respected: huge gradient pushes every input to the bound.
TEST(MpcSolver, RespectsBox) {
  qp::MpcOsqpSolver solver(-0.55, 0.55);
  ASSERT_TRUE(solver.ready());
  Eigen::Matrix<double, qp::kMpcHorizon, qp::kMpcHorizon> H =
      Eigen::Matrix<double, qp::kMpcHorizon, qp::kMpcHorizon>::Identity();
  Eigen::Matrix<double, qp::kMpcHorizon, 1> g, U;
  for (int i = 0; i < qp::kMpcHorizon; ++i) g(i, 0) = 10.0;  // push negative
  int iters = 0;
  ASSERT_TRUE(solver.solve(H, g, U, iters));
  for (int i = 0; i < qp::kMpcHorizon; ++i) EXPECT_NEAR(U(i, 0), -0.55, 1e-4);
}
#endif

TEST(MpcTracker, GainsComputeAndStabilize) {
  MpcTracker mpc(TestParams(), TestMpcConfig());
  ASSERT_TRUE(mpc.computeGains(15.0, 0.9));
  EXPECT_TRUE(mpc.gainsComputed());
#ifdef HAVE_OSQP
  EXPECT_TRUE(mpc.mpcReady());
#endif
  // Discrete model sanity: nontrivial dynamics, bounded entries.
  const MpcStateMatrix& Ad = mpc.lastAd();
  const double ad_move = (Ad - MpcStateMatrix::Identity()).norm();
  EXPECT_GT(ad_move, 0.1);
  EXPECT_LT(ad_move, 50.0);
  EXPECT_LT(Ad.norm(), 50.0);
  EXPECT_GT(mpc.lastBd().norm(), 1e-6);  // steering authority present
  // Terminal weight PSD.
  Eigen::LLT<MpcStateMatrix> llt(mpc.terminalP());
  EXPECT_EQ(llt.info(), Eigen::Success);
}

TEST(MpcTracker, StraightLineErrorConverges) {
  const VehicleParams vp = TestParams();
  DynamicBicycleModel model(vp);
  MpcTracker mpc(vp, TestMpcConfig());
  ASSERT_TRUE(mpc.computeGains(15.0, 0.9));

  StateVector x = StateVector::Zero();
  x(kPx) = 0.0;
  x(kPy) = 1.0;
  x(kVx) = 15.0;
  const double dt = 0.01;
  double ey = 1.0;
  for (int k = 0; k < 1000; ++k) {
    LateralState err;
    err << ey, x(kPsi), x(kVy), x(kYawRate);
    ControlVector u;
    mpc.update(err, x(kVx), 0.0, 15.0, dt, u);
    EXPECT_LE(std::abs(u(kSteer)), vp.steer_max + 1e-9);
    StateVector xn;
    model.stepRK4(x, u, 0.9, dt, xn);
    x = xn;
    ey = x(kPy);
  }
  EXPECT_LT(std::abs(ey), 0.15) << "MPC failed to pull 1 m offset back to lane";
  EXPECT_LT(std::abs(x(kPsi)), 0.08);
}

TEST(MpcTracker, CurvedRoadTracking) {
  const VehicleParams vp = TestParams();
  DynamicBicycleModel model(vp);
  MpcTracker mpc(vp, TestMpcConfig());
  ASSERT_TRUE(mpc.computeGains(15.0, 0.9));

  const double kappa = 1.0 / 200.0;
  StateVector x = StateVector::Zero();
  x(kVx) = 15.0;
  const double dt = 0.01;
  double ref_psi = 0.0, ref_x = 0.0, ref_y = 0.0;
  double max_ey = 0.0;
  for (int k = 0; k < 2000; ++k) {
    const double dx = x(kPx) - ref_x, dy = x(kPy) - ref_y;
    const double ex = std::cos(ref_psi), eyd = std::sin(ref_psi);
    const double ey = -dx * eyd + dy * ex;
    double epsi = x(kPsi) - ref_psi;
    while (epsi > M_PI) epsi -= 2 * M_PI;
    while (epsi < -M_PI) epsi += 2 * M_PI;
    max_ey = std::max(max_ey, std::abs(ey));
    LateralState err;
    err << ey, epsi, x(kVy), x(kYawRate);
    ControlVector u;
    mpc.update(err, x(kVx), kappa, 15.0, dt, u);
    StateVector xn;
    model.stepRK4(x, u, 0.9, dt, xn);
    x = xn;
    ref_psi += 15.0 * kappa * dt;
    ref_x += 15.0 * std::cos(ref_psi) * dt;
    ref_y += 15.0 * std::sin(ref_psi) * dt;
  }
  EXPECT_LT(max_ey, 1.0) << "MPC curve error too large";
}

TEST(MpcTracker, PreviewConsistency) {
  MpcTracker mpc(TestParams(), TestMpcConfig());
  ASSERT_TRUE(mpc.computeGains(15.0, 0.9));
  LateralState err;
  err << 0.3, 0.02, 0.1, 0.01;
  ControlVector u_scalar, u_preview;
  mpc.clearCurvaturePreview();
  mpc.update(err, 15.0, 0.005, 15.0, 0.01, u_scalar);
  double kv[qp::kMpcHorizon];
  for (int i = 0; i < qp::kMpcHorizon; ++i) kv[i] = 0.005;
  mpc.setCurvaturePreview(kv, qp::kMpcHorizon);
  mpc.resetIntegrator();  // identical longitudinal state for a fair comparison
  // NOTE: delta_prev differs between the calls (stateful MPC); compare only
  // that both are finite, bounded, and agree in sign/magnitude coarsely.
  mpc.update(err, 15.0, 0.005, 15.0, 0.01, u_preview);
  for (int j = 0; j < 2; ++j) {
    EXPECT_TRUE(std::isfinite(u_preview(j)));
    EXPECT_LE(std::abs(u_preview(j)), TestParams().steer_max + 1e-9);
  }
  EXPECT_NEAR(u_preview(kSteer), u_scalar(kSteer), 0.05);
  mpc.clearCurvaturePreview();
}

TEST(MpcTracker, SafeWithoutGains) {
  MpcTracker mpc(TestParams(), TestMpcConfig());
  LateralState err;
  err << 5.0, 0.5, 2.0, 0.5;  // absurd errors, no gains computed
  ControlVector u;
  mpc.update(err, 15.0, 0.0, 15.0, 0.01, u);
  EXPECT_TRUE(std::isfinite(u(kSteer)) && std::isfinite(u(kAx)));
  EXPECT_LE(std::abs(u(kSteer)), TestParams().steer_max + 1e-9);
}

TEST(MpcTracker, RealtimeBudget) {
  const VehicleParams vp = TestParams();
  MpcTracker mpc(vp, TestMpcConfig());
  ASSERT_TRUE(mpc.computeGains(15.0, 0.9));
  LateralState err;
  err << 0.3, 0.02, 0.1, 0.02;
  ControlVector u;
  for (int i = 0; i < 50; ++i) mpc.update(err, 15.0, 0.0, 15.0, 0.01, u);
  const auto t0 = std::chrono::steady_clock::now();
  constexpr int N = 500;
  for (int i = 0; i < N; ++i) mpc.update(err, 15.0, 0.0, 15.0, 0.01, u);
  const auto t1 = std::chrono::steady_clock::now();
  const double us =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) /
      N / 1000.0;
  EXPECT_LT(us, 1000.0) << "MPC update exceeds 1 ms";
}

TEST(MpcTracker, ConfigLoads) {
  const char* candidates[] = {"config/mpc_tuning.yaml", "../config/mpc_tuning.yaml"};
  const char* found = nullptr;
  for (const char* p : candidates) {
    if (FILE* f = std::fopen(p, "r")) {
      std::fclose(f);
      found = p;
      break;
    }
  }
  if (!found) GTEST_SKIP() << "mpc_tuning.yaml not found; skipping";
  const MpcConfig c = LoadMpcConfig(found);
  EXPECT_GT(c.R, 0.0);
  EXPECT_NEAR(c.dt, 0.05, 1e-12);
}
