#include "av_safety/barriers/gap_cbf.hpp"

namespace av_safety {

GapBarrierCbf::GapBarrierCbf(const GapBarrierParams& p) noexcept : params_(p) {}

double GapBarrierCbf::value(const StateVector& x, const ObstacleState& obs) const noexcept {
  return (obs.x - x(kPx)) - params_.d_min - params_.t_headway * x(kVx);
}

bool GapBarrierCbf::applies(const StateVector& x, const ObstacleState& obs) const noexcept {
  const double dx = obs.x - x(kPx);
  double dy = obs.y - x(kPy);
  if (dy < 0.0) dy = -dy;
  return dx > 0.0 && dy < params_.lateral_gate;
}

void GapBarrierCbf::linearize(const StateVector& x, const ObstacleState& obs,
                              const StateDerivative& f, const InputMatrix& G,
                              CbfLinearConstraint& out) const noexcept {
  const double th = params_.t_headway;
  // Lf = (vx_obs - f_x) - t_h f_vx ;  Lg = -t_h G_vx (G pose rows vanish).
  const double lf = (obs.vx - f(kPx)) - th * f(kVx);
  const double lg0 = -th * G(kVx, 0);
  const double lg1 = -th * G(kVx, 1);

  out.h = (obs.x - x(kPx)) - params_.d_min - th * x(kVx);
  out.Lf = lf;
  out.Lg[0] = lg0;
  out.Lg[1] = lg1;
  out.kappa = params_.kappa;
}

}  // namespace av_safety
