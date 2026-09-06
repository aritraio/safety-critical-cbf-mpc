#pragma once
/// @file pacejka_tire.hpp
/// @brief Pacejka Magic Formula lateral tire (per-tire, bicycle model).
///
///   Fy(a) = D_eff * sin(C * atan(B*a - E*(B*a - atan(B*a))))
///   D_eff = D * (mu / mu_nominal)
///
/// All methods are inline / noexcept / allocation-free and safe to call in the
/// 100 Hz loop. Slip-angle helpers clamp the longitudinal velocity denominator
/// so the model is well-defined down to standstill.

#include "av_safety/common/vehicle_params.hpp"

namespace av_safety {

class PacejkaTire {
 public:
  explicit PacejkaTire(const PacejkaParams& p = PacejkaParams{}, double mu_nominal = 0.9) noexcept;

  /// Lateral force [N] for slip angle alpha [rad] and friction mu [-].
  double lateralForce(double alpha, double mu) const noexcept;

  /// Analytic slope dFy/dAlpha [N/rad] (exact derivative, used for A/B/G matrices).
  double lateralStiffness(double alpha, double mu) const noexcept;

  /// Cornering stiffness at alpha = 0: C0 = B*C*D_eff [N/rad].
  double corneringStiffness(double mu) const noexcept;

  // -- Slip angles (bicycle kinematics) -------------------------------------
  /// Front slip angle: a_f = delta - atan((vy + lf*r) / max(vx, eps)).
  static double frontSlipAngle(double steer, double vx, double vy, double yaw_rate, double lf,
                               double vx_eps) noexcept;
  /// Rear slip angle: a_r = -atan((vy - lr*r) / max(vx, eps)).
  static double rearSlipAngle(double vx, double vy, double yaw_rate, double lr,
                              double vx_eps) noexcept;

  const PacejkaParams& params() const noexcept { return params_; }
  double muNominal() const noexcept { return mu_nominal_; }

 private:
  PacejkaParams params_;
  double mu_nominal_{0.9};

  double effectiveD(double mu) const noexcept;
};

}  // namespace av_safety
