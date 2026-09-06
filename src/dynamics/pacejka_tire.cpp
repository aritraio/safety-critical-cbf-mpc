#include "av_safety/dynamics/pacejka_tire.hpp"

#include <cmath>

namespace av_safety {

PacejkaTire::PacejkaTire(const PacejkaParams& p, double mu_nominal) noexcept
    : params_(p), mu_nominal_(mu_nominal > 0.0 ? mu_nominal : 0.9) {}

double PacejkaTire::effectiveD(double mu) const noexcept {
  if (mu < 0.0) mu = 0.0;
  if (mu > 2.0) mu = 2.0;
  return params_.D * (mu / mu_nominal_);
}

double PacejkaTire::lateralForce(double alpha, double mu) const noexcept {
  const double d = effectiveD(mu);
  const double x = params_.B * alpha;
  const double atan_x = std::atan(x);
  const double inner = x - params_.E * (x - atan_x);
  return d * std::sin(params_.C * std::atan(inner));
}

double PacejkaTire::lateralStiffness(double alpha, double mu) const noexcept {
  const double d = effectiveD(mu);
  const double b = params_.B;
  const double x = b * alpha;
  const double atan_x = std::atan(x);
  const double inner = x - params_.E * (x - atan_x);
  // d(inner)/d(alpha) = B*(1-E) + E*B/(1+x^2)
  const double dinner = b * (1.0 - params_.E) + params_.E * b / (1.0 + x * x);
  const double cos_term = std::cos(params_.C * std::atan(inner));
  return d * cos_term * params_.C * dinner / (1.0 + inner * inner);
}

double PacejkaTire::corneringStiffness(double mu) const noexcept {
  // alpha = 0: inner = 0, cos = 1, dinner = B.
  return effectiveD(mu) * params_.C * params_.B;
}

double PacejkaTire::frontSlipAngle(double steer, double vx, double vy, double yaw_rate, double lf,
                                   double vx_eps) noexcept {
  const double denom = vx >= vx_eps ? vx : vx_eps;
  return steer - std::atan((vy + lf * yaw_rate) / denom);
}

double PacejkaTire::rearSlipAngle(double vx, double vy, double yaw_rate, double lr,
                                  double vx_eps) noexcept {
  const double denom = vx >= vx_eps ? vx : vx_eps;
  return -std::atan((vy - lr * yaw_rate) / denom);
}

}  // namespace av_safety
