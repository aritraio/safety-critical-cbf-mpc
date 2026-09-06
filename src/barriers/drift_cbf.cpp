#include "av_safety/barriers/drift_cbf.hpp"

namespace av_safety {

DriftEnvelopeCbf::DriftEnvelopeCbf(const DriftBarrierParams& p) noexcept : params_(p) {}

double DriftEnvelopeCbf::value(const StateVector& x, double mu, double gravity) const noexcept {
  const double w = x(kVx) * x(kYawRate);
  const double lim = mu * gravity;
  return lim * lim - w * w;
}

void DriftEnvelopeCbf::linearize(const StateVector& x, const StateDerivative& f,
                                 const InputMatrix& G, double mu, double gravity,
                                 CbfLinearConstraint& out) const noexcept {
  const double vx = x(kVx);
  const double r = x(kYawRate);
  const double w = vx * r;
  const double lim = mu * gravity;
  // dh = [0,0,0, -2 w r, 0, -2 w vx].
  const double dh_dvx = -2.0 * w * r;
  const double dh_dr = -2.0 * w * vx;

  out.h = lim * lim - w * w;
  out.Lf = dh_dvx * f(kVx) + dh_dr * f(kYawRate);
  for (int j = 0; j < kControlDim; ++j) {
    out.Lg[j] = dh_dvx * G(kVx, j) + dh_dr * G(kYawRate, j);
  }
  out.kappa = params_.kappa;
}

}  // namespace av_safety
