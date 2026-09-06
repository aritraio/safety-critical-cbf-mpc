#pragma once
/// @file drift_cbf.hpp
/// @brief Lateral-stability envelope: h = (mu g)^2 - (vx r)^2 (relative degree 1).
///
///   dh = [0,0,0, -2 w r, 0, -2 w vx],  w = vx r
///   Lf = dh.f,  Lg = dh.G  =>  Lf + Lg.u + kappa h >= 0.
///
/// The gradient vanishes at w = 0, but there h = (mu g)^2 > 0 strictly, so the
/// constraint reduces to kappa h > 0 (satisfied for every u): the apparent
/// singularity lies strictly inside the safe set and needs no regularization.

#include "av_safety/barriers/barrier_params.hpp"
#include "av_safety/barriers/control_barrier_function.hpp"

namespace av_safety {

class DriftEnvelopeCbf {
 public:
  explicit DriftEnvelopeCbf(const DriftBarrierParams& p = DriftBarrierParams{}) noexcept;

  double value(const StateVector& x, double mu, double gravity) const noexcept;

  /// First-order (RD1) linearization. Hot-loop safe.
  void linearize(const StateVector& x, const StateDerivative& f, const InputMatrix& G,
                 double mu, double gravity, CbfLinearConstraint& out) const noexcept;

  const DriftBarrierParams& params() const noexcept { return params_; }

 private:
  DriftBarrierParams params_;
};

}  // namespace av_safety
