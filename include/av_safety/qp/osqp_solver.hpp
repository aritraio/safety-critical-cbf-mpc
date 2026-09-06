#pragma once
/// @file osqp_solver.hpp
/// @brief OSQP backend for the soft-CBF QP (warm-started, preallocated).
///
/// Fixed problem structure (n = 2 + kMaxCbfRows slacks, m = 2*kMaxCbfRows +
/// kMaxQpRows + kMaxSlewRows, dense A pattern) so the workspace is set up once
/// at init; every cycle only pushes new values through osqp_update_* (no
/// allocation, warm start retained). Requires HAVE_OSQP (vendored osqp v0.6.3
/// when not installed system-wide).

#include "av_safety/qp/enumeration_qp_solver.hpp"
#include "av_safety/qp/tiny_qp.hpp"

namespace av_safety {
namespace qp {

#ifdef HAVE_OSQP

class OsqpSolver : public QpSolverBackend {
 public:
  struct Settings {
    double eps_abs{1e-4};
    double eps_rel{1e-4};
    int max_iter{4000};
    bool polish{false};
    double rho{1.0};  // initial ADMM step; 1.0 suits our normalized scale
    bool adaptive_rho{true};
    // Ruiz equilibration iterations at setup (0 = off). Default OFF: the
    // filter normalizes soft rows upstream, and equilibration computed from
    // setup-time seed data was measured to inflate iterations ~30x on shed
    // corners (3225 vs 100) by freezing stale scale factors into updates.
    int scaling{0};
  };

  /// H_diag: fixed Hessian diagonal (weights change => reconstruct solver).
  explicit OsqpSolver(const double H_diag[kControlDim]);
  explicit OsqpSolver(const double H_diag[kControlDim], const Settings& s);
  ~OsqpSolver() override;

  OsqpSolver(const OsqpSolver&) = delete;
  OsqpSolver& operator=(const OsqpSolver&) = delete;

  bool solve(const TinyQpProblem& problem, TinyQpSolution& sol) noexcept override;
  const char* name() const noexcept override { return "osqp"; }
  bool ready() const noexcept;

 private:
  struct Impl;
  Impl* impl_{nullptr};
};

#else

// Without OSQP the filter/tests use EnumerationQpSolver directly (see
// tiny_qp.hpp); no alias here to avoid constructor-signature confusion.

#endif

}  // namespace qp
}  // namespace av_safety
