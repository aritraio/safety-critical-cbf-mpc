#pragma once
/// @file enumeration_qp_solver.hpp
/// @brief Exact inequality-QP solver for n = 2 via active-set enumeration.
///
/// Hard-only problems (num_soft == 0): the optimum is the least-cost feasible
/// point among the unconstrained minimizer, single-constraint projections
/// (closed form for diagonal H), and pairwise intersections.
///
/// Soft problems (num_soft > 0): slacks are eliminated analytically,
///   xi(u) = max(0, b_i - A_i u),
/// reducing to min over u of the convex piecewise-quadratic
///   F(u) = 1/2 (u-un)'H(u-un) + 1/2 sum w_i [b_i - A_i u]_+^2
/// over the hard polygon. The global minimizer is found exactly by:
///  (a) soft-subset enumeration (2^M <= 64 smooth 2x2 KKT solves, interior),
///  (b) exact 1-D kink-scan along every active hard row (edges),
///  (c) hard-row pairwise intersections (vertices).
/// O(m^2 + 2^M) candidates, fixed loop bounds, no allocations. Serves as the
/// always-available backend and as the ground truth for the OSQP cross-check.

#include "av_safety/qp/tiny_qp.hpp"

namespace av_safety {
namespace qp {

class EnumerationQpSolver : public QpSolverBackend {
 public:
  bool solve(const TinyQpProblem& problem, TinyQpSolution& sol) noexcept override;
  const char* name() const noexcept override { return "enumeration"; }
};

}  // namespace qp
}  // namespace av_safety
