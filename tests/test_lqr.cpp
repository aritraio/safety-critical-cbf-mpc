#include <chrono>
#include <cmath>
#include <gtest/gtest.h>

#include "av_safety/common/vehicle_params.hpp"
#include "av_safety/controllers/lqr_tracker.hpp"
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

LqrConfig TestConfig() {
  LqrConfig c;
  c.Q.setZero();
  c.Q(0, 0) = 8.0;
  c.Q(1, 1) = 6.0;
  c.Q(2, 2) = 1.0;
  c.Q(3, 3) = 1.0;
  c.R = 1.0;
  c.kp_lon = 0.8;
  c.ki_lon = 0.15;
  c.integrator_max = 2.0;
  c.use_feedforward = true;
  return c;
}
}  // namespace

// CARE solver: residual small, K stabilizing on the textbook linearization.
TEST(LQR, CareSolutionStabilizes) {
  const VehicleParams vp = TestParams();
  DynamicBicycleModel model(vp);
  LateralMatrix A;
  LateralInputMatrix B;
  model.linearizedLateralMatrices(20.0, 0.9, A, B);
  const LqrConfig cfg = TestConfig();
  LateralMatrix P;
  LateralGain K;
  ASSERT_TRUE(LqrTracker::solveContinuousCARE(A, B, cfg.Q, cfg.R, P, K));
  // P symmetric positive definite.
  EXPECT_NEAR((P - P.transpose()).norm(), 0.0, 1e-9);
  EXPECT_GT(P.determinant(), 0.0);
  // Closed loop Hurwitz.
  EXPECT_LT(LqrTracker::closedLoopMaxRealPart(A, B, K), -0.05);
  // Sign convention: delta = -K*e, so K > 0 on (e_y, e_psi) steers back
  // toward the path (positive lateral/heading error -> negative steer).
  EXPECT_GT(K(0, 0), 0.0);
  EXPECT_GT(K(0, 1), 0.0);
}

// Invalid R must be rejected, not crash.
TEST(LQR, CareRejectsBadWeights) {
  LateralMatrix A = LateralMatrix::Zero(), P;
  LateralInputMatrix B = LateralInputMatrix::Zero();
  LateralGain K;
  EXPECT_FALSE(LqrTracker::solveContinuousCARE(A, B, LateralMatrix::Identity(), 0.0, P, K));
  EXPECT_FALSE(LqrTracker::solveContinuousCARE(A, B, LateralMatrix::Identity(), -1.0, P, K));
}

// Corrective action: left-of-path / pointing-left errors -> steer right.
TEST(LQR, CorrectiveSteeringDirection) {
  LqrTracker tracker(TestParams(), TestConfig());
  ASSERT_TRUE(tracker.computeGains(20.0, 0.9));
  LateralState err;
  err << 1.0, 0.1, 0.0, 0.0;
  ControlVector u;
  tracker.update(err, 20.0, 0.0, 20.0, 0.01, u);
  EXPECT_LT(u(kSteer), 0.0);
  err << -1.0, -0.1, 0.0, 0.0;
  tracker.update(err, 20.0, 0.0, 20.0, 0.01, u);
  EXPECT_GT(u(kSteer), 0.0);
}
TEST(LQR, RespectsLimits) {
  LqrTracker tracker(TestParams(), TestConfig());
  ASSERT_TRUE(tracker.computeGains(20.0, 0.9));
  LateralState big;
  big << 50.0, 3.0, 5.0, 1.0;  // absurd errors
  ControlVector u;
  tracker.update(big, 20.0, 0.0, 20.0, 0.01, u);
  EXPECT_LE(std::abs(u(kSteer)), TestParams().steer_max + 1e-12);
  EXPECT_GE(u(kAx), TestParams().accel_min - 1e-12);
  EXPECT_LE(u(kAx), TestParams().accel_max + 1e-12);
}

// Closed-loop lateral tracking: 1 m initial offset must decay on straight road.
TEST(LQR, StraightLineErrorConverges) {
  const VehicleParams vp = TestParams();
  DynamicBicycleModel model(vp);
  LqrTracker tracker(vp, TestConfig());
  ASSERT_TRUE(tracker.computeGains(15.0, 0.9));

  StateVector x = StateVector::Zero();
  x(kPx) = 0.0;
  x(kPy) = 1.0;  // 1 m left of centerline
  x(kVx) = 15.0;
  const double dt = 0.01;  // 100 Hz
  double ey = 1.0;
  for (int k = 0; k < 1000; ++k) {  // 10 s
    LateralState err;
    err << ey, x(kPsi), x(kVy), x(kYawRate);
    ControlVector u;
    tracker.update(err, x(kVx), 0.0, 15.0, dt, u);
    StateVector xn;
    model.stepRK4(x, u, 0.9, dt, xn);
    x = xn;
    ey = x(kPy);  // straight centerline y = 0
  }
  EXPECT_LT(std::abs(ey), 0.15) << "LQR failed to pull 1 m offset back to lane";
  EXPECT_LT(std::abs(x(kPsi)), 0.05);
}

// Constant-curvature road: feedforward should hold steady-state error small.
TEST(LQR, CurvedRoadTracking) {
  const VehicleParams vp = TestParams();
  DynamicBicycleModel model(vp);
  LqrTracker tracker(vp, TestConfig());
  ASSERT_TRUE(tracker.computeGains(15.0, 0.9));

  const double kappa = 1.0 / 200.0;  // R = 200 m gentle highway curve
  const double vx_ref = 15.0;
  StateVector x = StateVector::Zero();
  x(kVx) = vx_ref;
  const double dt = 0.01;
  // Reference: circle of radius R; track lateral error vs arc-length approx.
  double ref_psi = 0.0, ref_x = 0.0, ref_y = 0.0;
  double max_ey = 0.0;
  for (int k = 0; k < 2000; ++k) {  // 20 s
    // Closest-point approx: project onto moving circular reference.
    const double dx = x(kPx) - ref_x, dy = x(kPy) - ref_y;
    const double ex = std::cos(ref_psi), ey_dir = std::sin(ref_psi);
    const double ey = -dx * ey_dir + dy * ex;
    double epsi = x(kPsi) - ref_psi;
    while (epsi > M_PI) epsi -= 2 * M_PI;
    while (epsi < -M_PI) epsi += 2 * M_PI;
    max_ey = std::max(max_ey, std::abs(ey));
    LateralState err;
    err << ey, epsi, x(kVy), x(kYawRate);
    ControlVector u;
    tracker.update(err, x(kVx), kappa, vx_ref, dt, u);
    StateVector xn;
    model.stepRK4(x, u, 0.9, dt, xn);
    x = xn;
    // Advance circular reference at reference speed.
    ref_psi += vx_ref * kappa * dt;
    ref_x += vx_ref * std::cos(ref_psi) * dt;
    ref_y += vx_ref * std::sin(ref_psi) * dt;
  }
  EXPECT_LT(max_ey, 1.0) << "steady-state curve error too large (check feedforward)";
}

// Real-time budget: dynamics RK4 + LQR update must complete well under 1 ms.
TEST(LQR, RealtimeBudget) {
  const VehicleParams vp = TestParams();
  DynamicBicycleModel model(vp);
  LqrTracker tracker(vp, TestConfig());
  ASSERT_TRUE(tracker.computeGains(20.0, 0.9));
  StateVector x = StateVector::Zero();
  x(kVx) = 20.0;
  LateralState err;
  err << 0.3, 0.02, 0.1, 0.01;
  ControlVector u;
  u << 0.01, 0.0;
  StateVector xn;
  ControlVector un;

  // Warm up, then time 1000 iterations.
  for (int i = 0; i < 100; ++i) {
    tracker.update(err, 20.0, 0.0, 20.0, 0.01, un);
    model.stepRK4(x, un, 0.9, 0.01, xn);
  }
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < 1000; ++i) {
    tracker.update(err, 20.0, 0.0, 20.0, 0.01, un);
    model.stepRK4(x, un, 0.9, 0.01, xn);
  }
  const auto t1 = std::chrono::steady_clock::now();
  const double us_per_iter =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) /
      1000.0 / 1000.0;
  EXPECT_LT(us_per_iter, 1000.0) << "control step exceeds 1 ms budget";
}
