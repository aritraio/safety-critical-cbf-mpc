// Module-B verification: UKF friction estimation under a matched model.
// A ground-truth Pacejka bicycle (true mu) is driven with sustained lateral
// excitation; synthetic IMU/wheel measurements (spec measurement model +
// Gaussian noise) feed the UKF, which must recover mu. Straight cruising
// (no excitation) must keep mu_hat bounded near init.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>

#include "av_safety/common/vehicle_params.hpp"
#include "av_safety/dynamics/dynamic_bicycle_model.hpp"
#include "av_safety/estimation/friction_ukf.hpp"

using namespace av_safety;
using namespace av_safety::estimation;

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
  p.operating_speed = 12.0;
  return p;
}

UkfConfig TestUkfConfig() {
  UkfConfig c;
  c.Q.setZero();
  c.Q(0, 0) = 1e-4;
  c.Q(1, 1) = 1e-4;
  c.Q(2, 2) = 1e-6;
  c.Q(3, 3) = 1e-6;  // mu random walk: lets updates move it within seconds
  c.R.setZero();
  c.R(0, 0) = 0.09;
  c.R(1, 1) = 0.09;
  c.R(2, 2) = 1e-6;
  c.R(3, 3) = 0.0025;
  c.P0.setZero();
  c.P0(0, 0) = 1.0;
  c.P0(1, 1) = 1.0;
  c.P0(2, 2) = 0.1;
  c.P0(3, 3) = 0.25;
  c.alpha = 1e-3;
  c.beta = 2.0;
  c.kappa = 0.0;
  c.mu_init = 0.9;
  c.mu_min = 0.1;
  c.mu_max = 1.3;
  return c;
}

// Deterministic Gaussian source (xorshift + Box-Muller), fixed seed.
struct GaussRng {
  uint64_t s{0x243F6A8885A308D3ull};
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

}  // namespace

TEST(FrictionUkf, ConvergesUnderLateralExcitation) {
  const VehicleParams vp = TestParams();
  DynamicBicycleModel truth(vp);
  FrictionUkf ukf(vp, TestUkfConfig());
  GaussRng rng;

  // Truth: steady cornering on mu = 0.35 (wet) at ~12 m/s.
  constexpr double kMuTrue = 0.35;
  StateVector x = StateVector::Zero();
  x(kVx) = 12.0;
  ControlVector u;
  u << 0.07, 0.0;
  PacejkaTire front(vp.pacejka, vp.mu_nominal), rear(vp.pacejka, vp.mu_nominal);

  const double dt = 0.01;
  double mu_sum = 0.0;
  int mu_cnt = 0;
  bool moved_down = false;
  for (int k = 0; k < 800; ++k) {  // 8 s
    StateVector xn;
    truth.stepRK4(x, u, kMuTrue, dt, xn);
    x = xn;
    // Synthetic IMU/wheel measurement from the truth trajectory.
    const double af = PacejkaTire::frontSlipAngle(u(kSteer), x(kVx), x(kVy), x(kYawRate),
                                                 vp.lf, vp.vx_epsilon);
    const double ar = PacejkaTire::rearSlipAngle(x(kVx), x(kVy), x(kYawRate), vp.lr,
                                                vp.vx_epsilon);
    const double fyf = front.lateralForce(af, kMuTrue);
    const double fyr = rear.lateralForce(ar, kMuTrue);
    const double cd = std::cos(u(kSteer));
    UkfMeas z;
    z(kMax) = u(kAx) - x(kVy) * x(kYawRate) + 0.3 * rng.gauss();
    z(kMay) = (2.0 * fyf * cd + 2.0 * fyr) / vp.mass - x(kVx) * x(kYawRate) + 0.3 * rng.gauss();
    z(kMr) = x(kYawRate) + 1e-3 * rng.gauss();
    z(kMvx) = x(kVx) + 0.05 * rng.gauss();
    ukf.step(u, z, dt);

    EXPECT_GE(ukf.mu(), 0.1 - 1e-12);
    EXPECT_LE(ukf.mu(), 1.3 + 1e-12);
    EXPECT_TRUE(ukf.state().allFinite());
    EXPECT_TRUE(ukf.covariance().allFinite());
    if (k == 400) moved_down = ukf.mu() < 0.7;
    if (k >= 600) {
      mu_sum += ukf.mu();
      ++mu_cnt;
    }
  }
  EXPECT_TRUE(moved_down) << "mu_hat never left the dry init, mu=" << ukf.mu();
  const double mu_mean = mu_sum / mu_cnt;
  EXPECT_NEAR(mu_mean, kMuTrue, 0.06) << "UKF failed to converge to wet mu";
}

TEST(FrictionUkf, StaysBoundedWithoutExcitation) {
  const VehicleParams vp = TestParams();
  DynamicBicycleModel truth(vp);
  FrictionUkf ukf(vp, TestUkfConfig());
  GaussRng rng;
  PacejkaTire front(vp.pacejka, vp.mu_nominal), rear(vp.pacejka, vp.mu_nominal);

  // Straight cruise on mu = 0.9: no lateral excitation, mu unobservable.
  StateVector x = StateVector::Zero();
  x(kVx) = 12.0;
  ControlVector u;
  u << 0.0, 0.0;
  const double dt = 0.01;
  for (int k = 0; k < 800; ++k) {
    StateVector xn;
    truth.stepRK4(x, u, 0.9, dt, xn);
    x = xn;
    const double af = PacejkaTire::frontSlipAngle(u(kSteer), x(kVx), x(kVy), x(kYawRate),
                                                 vp.lf, vp.vx_epsilon);
    const double ar = PacejkaTire::rearSlipAngle(x(kVx), x(kVy), x(kYawRate), vp.lr,
                                                vp.vx_epsilon);
    const double fyf = front.lateralForce(af, 0.9);
    const double fyr = rear.lateralForce(ar, 0.9);
    UkfMeas z;
    z(kMax) = u(kAx) - x(kVy) * x(kYawRate) + 0.3 * rng.gauss();
    z(kMay) = (2.0 * fyf + 2.0 * fyr) / vp.mass - x(kVx) * x(kYawRate) + 0.3 * rng.gauss();
    z(kMr) = x(kYawRate) + 1e-3 * rng.gauss();
    z(kMvx) = x(kVx) + 0.05 * rng.gauss();
    ukf.step(u, z, dt);
    EXPECT_GE(ukf.mu(), 0.1 - 1e-12);
    EXPECT_LE(ukf.mu(), 1.3 + 1e-12);
  }
  // Weak excitation => estimate must not wander far from the dry init.
  EXPECT_NEAR(ukf.mu(), 0.9, 0.25);
}

TEST(FrictionUkf, PredictUpdateSmoke) {
  const VehicleParams vp = TestParams();
  FrictionUkf ukf(vp, TestUkfConfig());
  ControlVector u;
  u << 0.02, 0.1;
  UkfMeas z;
  z << 0.1, 0.5, 0.02, 12.0;
  ukf.predict(u, 0.01);
  EXPECT_TRUE(ukf.state().allFinite());
  ukf.update(z, u);
  EXPECT_TRUE(ukf.state().allFinite());
  EXPECT_GE(ukf.mu(), 0.1 - 1e-12);
  EXPECT_LE(ukf.mu(), 1.3 + 1e-12);
  ukf.reset();
  EXPECT_NEAR(ukf.mu(), 0.9, 1e-12);
}

TEST(FrictionUkf, ConfigLoads) {
  const char* candidates[] = {"config/friction_ukf.yaml", "../config/friction_ukf.yaml"};
  const char* found = nullptr;
  for (const char* p : candidates) {
    if (FILE* f = std::fopen(p, "r")) {
      std::fclose(f);
      found = p;
      break;
    }
  }
  if (!found) GTEST_SKIP() << "friction_ukf.yaml not found; skipping";
  const UkfConfig c = LoadUkfConfig(found);
  EXPECT_NEAR(c.alpha, 1e-3, 1e-12);
  EXPECT_NEAR(c.beta, 2.0, 1e-12);
  EXPECT_NEAR(c.mu_init, 0.9, 1e-12);
  EXPECT_GT(c.R(0, 0), 0.0);
  EXPECT_GT(c.R_prior, 0.0);
}

TEST(FrictionUkf, RealtimeBudget) {
  const VehicleParams vp = TestParams();
  FrictionUkf ukf(vp, TestUkfConfig());
  ControlVector u;
  u << 0.03, 0.2;
  UkfMeas z;
  z << 0.2, 1.5, 0.05, 12.0;
  for (int i = 0; i < 100; ++i) ukf.step(u, z, 0.01);  // warm up
  const auto t0 = std::chrono::steady_clock::now();
  constexpr int N = 2000;
  for (int i = 0; i < N; ++i) ukf.step(u, z, 0.01);
  const auto t1 = std::chrono::steady_clock::now();
  const double us =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) /
      N / 1000.0;
  EXPECT_LT(us, 1000.0) << "UKF step exceeds 1 ms budget";
}
