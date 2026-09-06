#include <cmath>
#include <gtest/gtest.h>

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
  p.drive_bias_front = 0.0;  // RWD
  p.vx_epsilon = 1.0;
  return p;
}
}  // namespace

// Straight-line equilibrium: zero steer/accel, no lateral velocity or yaw rate
// -> lateral + yaw accelerations vanish, longitudinal holds speed.
TEST(Dynamics, StraightLineEquilibrium) {
  DynamicBicycleModel model(TestParams());
  StateVector x = StateVector::Zero();
  x(kVx) = 20.0;
  ControlVector u = ControlVector::Zero();
  StateDerivative xd;
  model.continuousDynamics(x, u, 0.9, xd);
  EXPECT_NEAR(xd(kVy), 0.0, 1e-9);
  EXPECT_NEAR(xd(kYawRate), 0.0, 1e-9);
  EXPECT_NEAR(xd(kVx), 0.0, 1e-9);
  EXPECT_NEAR(xd(kPx), 20.0, 1e-9);  // heading 0 -> moving +x
  EXPECT_NEAR(xd(kPy), 0.0, 1e-9);
  EXPECT_NEAR(xd(kPsi), 0.0, 1e-9);
}

// Heading kinematics: psi = 90 deg -> forward motion maps to +y.
TEST(Dynamics, HeadingKinematics) {
  DynamicBicycleModel model(TestParams());
  StateVector x = StateVector::Zero();
  x(kPsi) = M_PI / 2.0;
  x(kVx) = 10.0;
  ControlVector u = ControlVector::Zero();
  StateDerivative xd;
  model.continuousDynamics(x, u, 0.9, xd);
  EXPECT_NEAR(xd(kPx), 0.0, 1e-9);
  EXPECT_NEAR(xd(kPy), 10.0, 1e-9);
}

// Longitudinal: RWD ax maps (nearly) 1:1 into vx_dot at zero steer.
TEST(Dynamics, LongitudinalGainRWD) {
  DynamicBicycleModel model(TestParams());
  StateVector x = StateVector::Zero();
  x(kVx) = 15.0;
  ControlVector u;
  u << 0.0, 2.0;
  StateDerivative xd;
  model.continuousDynamics(x, u, 0.9, xd);
  EXPECT_NEAR(xd(kVx), 2.0, 1e-9);
}

// Steering left (delta > 0) at speed must produce positive lateral accel + yaw.
TEST(Dynamics, SteeringSignConvention) {
  DynamicBicycleModel model(TestParams());
  StateVector x = StateVector::Zero();
  x(kVx) = 15.0;
  ControlVector u;
  u << 0.05, 0.0;
  StateDerivative xd;
  model.continuousDynamics(x, u, 0.9, xd);
  EXPECT_GT(xd(kVy), 0.0);
  EXPECT_GT(xd(kYawRate), 0.0);
}

// RK4: no input -> constant-speed straight motion; yaw integrates.
TEST(Dynamics, RK4StraightMotion) {
  DynamicBicycleModel model(TestParams());
  StateVector x = StateVector::Zero();
  x(kVx) = 10.0;
  ControlVector u = ControlVector::Zero();
  StateVector xn;
  model.stepRK4(x, u, 0.9, 0.01, xn);  // 100 Hz step
  EXPECT_NEAR(xn(kPx), 0.1, 1e-9);
  EXPECT_NEAR(xn(kVx), 10.0, 1e-9);
  EXPECT_NEAR(xn(kVy), 0.0, 1e-9);
  EXPECT_TRUE(xn.allFinite());
}

// Low-speed / standstill safety: no NaN anywhere.
TEST(Dynamics, StandstillFinite) {
  DynamicBicycleModel model(TestParams());
  StateVector x = StateVector::Zero();
  ControlVector u;
  u << 0.1, 1.0;
  StateDerivative xd;
  StateVector xn;
  model.continuousDynamics(x, u, 0.9, xd);
  model.stepRK4(x, u, 0.9, 0.01, xn);
  EXPECT_TRUE(xd.allFinite());
  EXPECT_TRUE(xn.allFinite());
}

// Linearization sanity: A/B match the textbook bicycle structure and the
// steering column of the affine G matrix matches finite differences.
TEST(Dynamics, LinearizationStructure) {
  DynamicBicycleModel model(TestParams());
  LateralMatrix A;
  LateralInputMatrix B;
  model.linearizedLateralMatrices(20.0, 0.9, A, B);
  EXPECT_GT(A(0, 1), 0.0);   // ey_dot += vx * epsi
  EXPECT_EQ(A(0, 2), 1.0);   // ey_dot += vy
  EXPECT_EQ(A(1, 3), 1.0);   // epsi_dot += r
  EXPECT_LT(A(2, 2), 0.0);   // stable tire damping
  EXPECT_LT(A(3, 3), 0.0);
  EXPECT_GT(B(2, 0), 0.0);   // left steer -> +vy_dot
  EXPECT_GT(B(3, 0), 0.0);   // left steer -> +yaw

  // Affine G steering column vs finite difference of full dynamics.
  StateVector x = StateVector::Zero();
  x(kVx) = 15.0;
  x(kVy) = 0.5;
  x(kYawRate) = 0.05;
  ControlVector ue;
  ue << 0.03, 1.0;
  StateDerivative f;
  InputMatrix G;
  model.affineDecomposition(x, ue, 0.9, f, G);
  const double h = 1e-7;
  ControlVector up = ue, um = ue;
  up(kSteer) += h;
  um(kSteer) -= h;
  StateDerivative xp, xm;
  model.continuousDynamics(x, up, 0.9, xp);
  model.continuousDynamics(x, um, 0.9, xm);
  const StateDerivative fd = (xp - xm) / (2.0 * h);
  for (int i = 0; i < kStateDim; ++i) EXPECT_NEAR(G(i, kSteer), fd(i), 1e-5);
  // Longitudinal column is exact: d(vx_dot)/d(ax) = 1 for RWD at delta=0.
  ControlVector straight;
  straight << 0.0, 0.0;
  model.affineDecomposition(x, straight, 0.9, f, G);
  EXPECT_NEAR(G(kVx, kAx), 1.0, 1e-9);
}
