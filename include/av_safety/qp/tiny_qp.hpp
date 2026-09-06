#pragma once
/// @file tiny_qp.hpp
/// @brief Fixed-size soft-CBF quadratic program interface for the safety filter.
///
/// Problem (soft CBF-QP with slew limits):
///   min  1/2 (u - u_nom)' H (u - u_nom) + 1/2 sum_i w_i xi_i^2
///   s.t. A_soft[i] u + xi_i >= b_soft[i],   xi_i >= 0     (soft barrier rows)
///        C u >= d                                            (hard rows: box)
///        slew_lo <= u <= slew_hi                             (when use_slew)
///
/// z = [delta, ax] with slacks eliminated analytically by exact backends
/// (xi = max(0, b_i - A_i u)) or carried explicitly by OSQP
/// (z = [u, xi], fixed max structure n = 2 + kMaxCbfRows).
/// Box limits are passed as ordinary hard rows ([1,0] >= lo, ...), so every
/// backend sees a uniform inequality problem. The quadratic slack penalty is
/// an EXACT penalty: whenever the hard (zero-slack) problem is feasible, the
/// soft solution coincides with it (xi = 0); otherwise violations are
/// distributed by weight instead of flagging infeasible.
///
/// Backends: exact enumeration (always available, used for verification) and
/// OSQP with warm starting (HAVE_OSQP). Both are noexcept / allocation-free
/// in solve().
///
/// Scaling contract: soft rows SHOULD be normalized (||A_i||_inf on the order
/// of 1, as CbfQpFilter emits). The enumeration backend is exact for arbitrary
/// row scales; first-order ADMM (OSQP) is not — raw Lie rows spanning 1e3 with
/// 1e3+ weights stall it. Keep slack weights <= ~1e4 on normalized rows.

#include "av_safety/common/types.hpp"

namespace av_safety {
namespace qp {

inline constexpr int kMaxCbfRows = 10;  // road + drift + 4x(ellipse + gap)
inline constexpr int kMaxObsGapPairs = 4;  // one ellipse + one gap row per obstacle max
inline constexpr int kMaxBoxRows = 4;      // delta/ax lower+upper
inline constexpr int kMaxQpRows = kMaxCbfRows + kMaxBoxRows;  // hard rows
inline constexpr int kMaxSlewRows = 4;                        // slew box as rows
inline constexpr int kMaxHardRows = kMaxQpRows + kMaxSlewRows;

struct TinyQpProblem {
  double H_diag[kControlDim]{1.0, 0.5};
  double u_nom[kControlDim]{0.0, 0.0};
  // Hard rows: C z >= d (box limits; filter appends at most kMaxQpRows).
  double C[kMaxQpRows][kControlDim]{{0.0, 0.0}};
  double d[kMaxQpRows]{0.0};
  int num_rows{0};
  // Soft barrier rows: A_soft[i] u + xi >= b_soft[i], xi >= 0,
  // cost += 0.5 * slack_w[i] * xi^2. Requires 0 <= num_soft <= kMaxCbfRows
  // and slack_w[i] > 0 for i < num_soft.
  double A_soft[kMaxCbfRows][kControlDim]{{0.0, 0.0}};
  double b_soft[kMaxCbfRows]{0.0};
  double slack_w[kMaxCbfRows]{0.0};
  int num_soft{0};
  // Slew-rate box (absolute bounds, typically u_prev +- rate*dt).
  double slew_lo[kControlDim]{0.0, 0.0};
  double slew_hi[kControlDim]{0.0, 0.0};
  bool use_slew{false};
};

struct TinyQpSolution {
  double u[kControlDim]{0.0, 0.0};
  double slack[kMaxCbfRows]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  bool optimal{false};
  int iters{0};
  double objective{0.0};
};

class QpSolverBackend {
 public:
  virtual ~QpSolverBackend() = default;
  virtual bool solve(const TinyQpProblem& problem, TinyQpSolution& sol) noexcept = 0;
  virtual const char* name() const noexcept = 0;
};

}  // namespace qp
}  // namespace av_safety
