#include <cmath>
#include <gtest/gtest.h>

#include "av_safety/dynamics/pacejka_tire.hpp"

using av_safety::PacejkaParams;
using av_safety::PacejkaTire;

namespace {
PacejkaTire DefaultTire() { return PacejkaTire(PacejkaParams{}, 0.9); }
constexpr double kTol = 1e-9;
}  // namespace

// Zero slip -> zero force (tire at rest laterally).
TEST(Pacejka, ZeroSlipZeroForce) {
  const auto tire = DefaultTire();
  EXPECT_NEAR(tire.lateralForce(0.0, 0.9), 0.0, kTol);
  EXPECT_NEAR(tire.lateralForce(0.0, 0.3), 0.0, kTol);
}

// Odd symmetry: Fy(-a) = -Fy(a).
TEST(Pacejka, OddSymmetry) {
  const auto tire = DefaultTire();
  for (double a : {0.01, 0.05, 0.1, 0.2}) {
    EXPECT_NEAR(tire.lateralForce(-a, 0.9), -tire.lateralForce(a, 0.9), 1e-9);
  }
}

// Peak force approx D at moderate slip, and force saturates (does not blow up).
TEST(Pacejka, PeakAndSaturation) {
  const auto tire = DefaultTire();
  double peak = 0.0;
  for (double a = 0.0; a <= 0.5; a += 0.005) peak = std::max(peak, tire.lateralForce(a, 0.9));
  // Peak should be within ~15% of D (shape factors shift the exact max).
  EXPECT_GT(peak, 0.85 * 3500.0);
  EXPECT_LT(peak, 1.15 * 3500.0);
  // Far beyond the peak the curve decreases/plateaus but stays bounded by D.
  EXPECT_LE(std::abs(tire.lateralForce(0.5, 0.9)), 3500.0);
}

// Friction scaling: lower mu -> proportionally lower forces and stiffness.
TEST(Pacejka, FrictionScalingMonotonic) {
  const auto tire = DefaultTire();
  const double f_dry = tire.lateralForce(0.05, 0.9);
  const double f_wet = tire.lateralForce(0.05, 0.45);
  const double f_ice = tire.lateralForce(0.05, 0.2);
  EXPECT_GT(f_dry, f_wet);
  EXPECT_GT(f_wet, f_ice);
  EXPECT_NEAR(f_wet / f_dry, 0.5, 1e-9);  // linear D scaling by construction
  EXPECT_NEAR(tire.corneringStiffness(0.45) / tire.corneringStiffness(0.9), 0.5, 1e-9);
}

// Analytic stiffness matches central finite differences (validates Lie inputs).
TEST(Pacejka, AnalyticStiffnessVsFiniteDiff) {
  const auto tire = DefaultTire();
  for (double a : {-0.15, -0.03, 0.0, 0.04, 0.12}) {
    const double h = 1e-6;
    const double fd =
        (tire.lateralForce(a + h, 0.9) - tire.lateralForce(a - h, 0.9)) / (2.0 * h);
    EXPECT_NEAR(tire.lateralStiffness(a, 0.9), fd, 1e-4 * std::max(1.0, std::abs(fd)));
  }
  // Slope at origin equals B*C*D.
  EXPECT_NEAR(tire.lateralStiffness(0.0, 0.9), 10.0 * 1.3 * 3500.0, 1e-6);
}

// Slip-angle helpers: straight running gives ~0; clamped at low speed (no NaN).
TEST(Pacejka, SlipAngleHelpers) {
  EXPECT_NEAR(PacejkaTire::frontSlipAngle(0.0, 20.0, 0.0, 0.0, 1.2, 1.0), 0.0, kTol);
  EXPECT_NEAR(PacejkaTire::rearSlipAngle(20.0, 0.0, 0.0, 1.3, 1.0), 0.0, kTol);
  // Steering directly adds to front slip.
  EXPECT_NEAR(PacejkaTire::frontSlipAngle(0.1, 20.0, 0.0, 0.0, 1.2, 1.0), 0.1, kTol);
  // Standstill must not produce NaN/Inf (denominator clamped to eps).
  const double a = PacejkaTire::frontSlipAngle(0.0, 0.0, 0.0, 0.0, 1.2, 1.0);
  EXPECT_TRUE(std::isfinite(a));
}
