#pragma once
/// @file control_barrier_function.hpp
/// @brief Shared types for the CBF layer (Phase 2).
///
/// Relative-degree taxonomy used here:
/// - RD1 (drift envelope): constraint Lf h + Lg h . u + k h >= 0.
/// - RD2 (road boundary, obstacles): exponential-CBF constraint
///     Lf2 h + LgLf h . u + (p1+p2) hdot + p1 p2 h >= 0
///   with poles p1, p2 > 0. This renders the set
///     { h >= 0, hdot + p1 h >= 0 }
///   forward invariant (Nguyen & Sreenath, 2016). With p1 = p2 = kappa the
///   response is critically damped and `kappa` plays the same role as the
///   class-K slope in the RD1 case.
///
/// All barrier methods are noexcept / allocation-free. The filter evaluates
/// f, J = df/dx and G once per cycle and shares them across barriers so the
/// Lie algebra is computed on a single consistent linearization point.

#include "av_safety/common/types.hpp"

namespace av_safety {

/// Smooth (linear) extended class-K function.
inline double ClassK(double h, double kappa) noexcept { return kappa * h; }

/// RD1 constraint data: Lf + Lg*u + kappa*h >= 0  <=>  Lg*u >= -Lf - kappa*h.
struct CbfLinearConstraint {
  double h{0.0};
  double Lf{0.0};
  double Lg[kControlDim]{0.0, 0.0};
  double kappa{1.0};
};

/// RD2 (exponential CBF) constraint data:
///   Lf2 + LgLf*u + (p1+p2)*hdot + p1*p2*h >= 0.
struct EcbffConstraint {
  double h{0.0};
  double hdot{0.0};
  double Lf2{0.0};
  double LgLf[kControlDim]{0.0, 0.0};
  double p1{1.0};
  double p2{1.0};
};

/// Moving-obstacle kinematic state (constant-velocity prediction model).
struct ObstacleState {
  double x{0.0};
  double y{0.0};
  double vx{0.0};
  double vy{0.0};
  double a{4.0};  // [m] longitudinal semi-axis (incl. ego footprint margin)
  double b{2.0};  // [m] lateral semi-axis
};

}  // namespace av_safety
