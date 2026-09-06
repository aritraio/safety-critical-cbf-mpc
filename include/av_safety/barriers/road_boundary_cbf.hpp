#pragma once
/// @file road_boundary_cbf.hpp
/// @brief Lane-containment barrier: h = M^2 - (Y - yc)^2 (relative degree 2).
///
/// Lie structure (f = drift, J = df/dx, G = dxdot/du shared from the model):
///   hdot = gy f1,                    gy = -2(Y - yc)
///   dEta = f'H + dh'J  (row vector, H = diag(0,-2,0,...))
///        = [0, -2 f1, gy J12, gy sinP, gy cosP, 0]
///   Lf2  = dEta . f,   LgLf = dEta . G  (pose/yaw rows of G vanish).

#include "av_safety/barriers/barrier_params.hpp"
#include "av_safety/barriers/control_barrier_function.hpp"

namespace av_safety {

class RoadBoundaryCbf {
 public:
  explicit RoadBoundaryCbf(const RoadBarrierParams& p = RoadBarrierParams{}) noexcept;

  double value(const StateVector& x) const noexcept;
  double hDot(const StateVector& x, const StateDerivative& f) const noexcept;

  /// Full second-order (ECBF) linearization. Hot-loop safe.
  void linearize(const StateVector& x, const StateDerivative& f, const StateMatrix& J,
                 const InputMatrix& G, EcbffConstraint& out) const noexcept;

  const RoadBarrierParams& params() const noexcept { return params_; }

 private:
  RoadBarrierParams params_;
};

}  // namespace av_safety
