#pragma once
/// @file gap_cbf.hpp
/// @brief Longitudinal headway barrier for car-following safety (rel. degree 1).
///
///   h = (x_obs - x) - d_min - t_h * vx,   applied only when the obstacle is
///   ahead (dx > 0) and near-lane (|dy| < lateral_gate; both gated by caller).
///
/// Straight-road (x-axis) formulation, documented approximation on curves:
///   hdot = (vx_obs - xdot) - t_h * vx_dot
///        = [vx_obs - f_x] - t_h * f_vx  +  Lg . u
///   Lf = (vx_obs - f_x) - t_h f_vx,   Lg = -t_h G_vx  (pose rows of G vanish)
///   =>  Lf + Lg.u + kappa h >= 0.
///
/// With RWD at small steer, Lg ~= [~0, -t_h]: the row is a direct BRAKING
/// bound ax <= (Lf + kappa h)/t_h. Unlike the ellipse ECBF (proximity in
/// position space, myopic at highway closing speeds), the headway row reacts
/// as soon as the gap/rate balance demands it, giving early progressive
/// braking instead of last-instant swerves. Steady following (h = hdot = 0)
/// holds ax = 0 exactly.

#include "av_safety/barriers/barrier_params.hpp"
#include "av_safety/barriers/control_barrier_function.hpp"

namespace av_safety {

class GapBarrierCbf {
 public:
  explicit GapBarrierCbf(const GapBarrierParams& p = GapBarrierParams{}) noexcept;

  /// Barrier value (caller gates dx > 0, |dy| < gate before using).
  double value(const StateVector& x, const ObstacleState& obs) const noexcept;

  /// Gating predicate: obstacle ahead and near-lane.
  bool applies(const StateVector& x, const ObstacleState& obs) const noexcept;

  /// First-order (RD1) linearization. Hot-loop safe.
  void linearize(const StateVector& x, const ObstacleState& obs, const StateDerivative& f,
                 const InputMatrix& G, CbfLinearConstraint& out) const noexcept;

  const GapBarrierParams& params() const noexcept { return params_; }

 private:
  GapBarrierParams params_;
};

}  // namespace av_safety
