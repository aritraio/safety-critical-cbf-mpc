#pragma once
/// @file mpc_osqp_solver.hpp
/// @brief Fixed-size OSQP backend for the condensed linear MPC tracker.
///
/// Problem: min  1/2 U' H U + g'U   s.t.  u_min <= U <= u_max,
/// with U in R^N (N = kMpcHorizon steering sequence), H dense SPD (condensed
/// Hessian, upper triangle stored), box bounds fixed at construction.
/// Warm-started, preallocated, noexcept/allocation-free per solve.
/// Requires HAVE_OSQP (vendored osqp v0.6.3 when not installed system-wide).

#include "av_safety/common/types.hpp"

namespace av_safety {
namespace qp {

inline constexpr int kMpcHorizon = 15;

#ifdef HAVE_OSQP

class MpcOsqpSolver {
 public:
  struct Settings {
    double eps_abs{1e-4};
    double eps_rel{1e-4};
    int max_iter{4000};
    bool polish{false};
    double rho{1.0};
    bool adaptive_rho{true};
    int scaling{0};  // off: condensed MPC matrices arrive pre-scaled
  };

  /// u_min/u_max: fixed input box (vehicle steer limits at construction).
  MpcOsqpSolver(double u_min, double u_max);
  MpcOsqpSolver(double u_min, double u_max, const Settings& s);
  ~MpcOsqpSolver();

  MpcOsqpSolver(const MpcOsqpSolver&) = delete;
  MpcOsqpSolver& operator=(const MpcOsqpSolver&) = delete;

  /// Solve with fresh condensed Hessian (upper triangle used) and gradient.
  /// U is warm-started from the previous solution (shifted by the caller).
  bool solve(const Eigen::Matrix<double, kMpcHorizon, kMpcHorizon>& H,
             const Eigen::Matrix<double, kMpcHorizon, 1>& g,
             Eigen::Matrix<double, kMpcHorizon, 1>& U_out, int& iters_out) noexcept;
  bool ready() const noexcept;

 private:
  struct Impl;
  Impl* impl_{nullptr};
};

#else

// Without OSQP there is no MPC solver; MpcTracker falls back to its LQR law
// (see mpc_tracker.hpp). No alias: call sites must check ready().

#endif

}  // namespace qp
}  // namespace av_safety
