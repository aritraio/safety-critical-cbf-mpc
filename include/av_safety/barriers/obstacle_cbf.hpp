#pragma once
/// @file obstacle_cbf.hpp
/// @brief Ellipsoidal collision-avoidance barrier (relative degree 2).
///
///   h = ((X-xo)/a)^2 + ((Y-yo)/b)^2 - 1, with the obstacle center
///   (xo(t), yo(t)) moving at constant velocity. With gx = 2(X-xo)/a^2,
///   gy = 2(Y-yo)/b^2:
///   hdot = gx (f0-vxo) + gy (f1-vyo)
///   dEta = spatial grad of (gx f0 + gy f1 + ht), ht = -gx vxo - gy vyo
///        = [(2/a^2)(f0-vxo), (2/b^2)(f1-vyo),
///           gx J02 + gy J12, gx cosP + gy sinP, -gx sinP + gy cosP, 0]
///   eta_t = -(2/a^2) vxo (f0-vxo) - (2/b^2) vyo (f1-vyo)  (explicit time part)
///   Lf2  = dEta.f + eta_t,   LgLf = dEta.G.
/// Pose rows of G vanish, so no input coupling is dropped: the decomposition
/// is exact. Static obstacles are the v = 0 special case.

#include "av_safety/barriers/barrier_params.hpp"
#include "av_safety/barriers/control_barrier_function.hpp"

namespace av_safety {

class ObstacleCbf {
 public:
  explicit ObstacleCbf(const ObstacleBarrierParams& p = ObstacleBarrierParams{}) noexcept;

  double value(const StateVector& x, const ObstacleState& obs) const noexcept;
  double hDot(const StateVector& x, const StateDerivative& f,
              const ObstacleState& obs) const noexcept;

  /// Full second-order (ECBF) linearization incl. obstacle motion. Hot-loop safe.
  void linearize(const StateVector& x, const ObstacleState& obs, const StateDerivative& f,
                 const StateMatrix& J, const InputMatrix& G, EcbffConstraint& out) const noexcept;

  const ObstacleBarrierParams& params() const noexcept { return params_; }

 private:
  ObstacleBarrierParams params_;
};

}  // namespace av_safety
