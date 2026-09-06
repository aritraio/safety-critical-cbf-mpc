#include "av_safety/qp/osqp_solver.hpp"

#ifdef HAVE_OSQP

#include <osqp.h>

#include <cmath>
#include <new>
#include <utility>

static_assert(sizeof(c_float) == sizeof(double),
              "OSQP must be built with double precision (DFLOAT off)");

namespace av_safety {
namespace qp {

// Expanded soft-CBF structure (fixed at setup; values updated per solve):
//   vars: [delta, ax, xi_0..xi_{M-1}],  M = kMaxCbfRows (n = 8)
//   rows: 0..M-1          soft rows   [a_i, e_i]      >= b_i
//         M..2M-1         xi >= 0     [0, e_i]        >= 0
//         2M..2M+9        hard rows   [C, 0]          >= d   (kMaxQpRows slots)
//         2M+10..2M+13    slew rows   (delta/ax box)  (4 slots)
inline constexpr int kSoftN = kControlDim + kMaxCbfRows;              // 8
inline constexpr int kSoftM = 2 * kMaxCbfRows + kMaxQpRows + kMaxSlewRows;  // 26

struct OsqpSolver::Impl {
  OSQPWorkspace* work{nullptr};
  OSQPData data{};
  OSQPSettings settings{};
  // Fixed-structure CSC storage (members => no per-solve allocation).
  c_int Pp[kSoftN + 1]{0};
  c_int Pi[kSoftN]{0};
  c_float Px[kSoftN]{1.0};
  csc Pmat{0, 0, 0, nullptr, nullptr, nullptr, -1};
  c_int Ap[kSoftN + 1]{0};
  c_int Ai[kSoftM * kSoftN]{0};
  c_float Ax[kSoftM * kSoftN]{0.0};
  csc Amat{0, 0, 0, nullptr, nullptr, nullptr, -1};
  c_float q[kSoftN]{0.0};
  c_float l[kSoftM]{0.0};
  c_float u[kSoftM]{0.0};
};

OsqpSolver::OsqpSolver(const double H_diag[kControlDim]) : OsqpSolver(H_diag, Settings{}) {}

OsqpSolver::OsqpSolver(const double H_diag[kControlDim], const Settings& s) {
  Impl* impl = new (std::nothrow) Impl();
  if (!impl) return;
  impl_ = impl;

  // P: diagonal (upper-triangular CSC); values refreshed per solve.
  for (int c = 0; c < kSoftN; ++c) {
    impl->Pp[c] = c;
    impl->Pi[c] = c;
    impl->Px[c] = (c < kControlDim) ? static_cast<c_float>(H_diag[c]) : 1.0;
  }
  impl->Pp[kSoftN] = kSoftN;
  impl->Pmat.m = kSoftN;
  impl->Pmat.n = kSoftN;
  impl->Pmat.nzmax = kSoftN;
  impl->Pmat.x = impl->Px;
  impl->Pmat.i = impl->Pi;
  impl->Pmat.p = impl->Pp;
  impl->Pmat.nz = -1;

  // A: m x n dense pattern (explicit zeros allowed), column-major.
  for (int c = 0; c < kSoftN; ++c) {
    impl->Ap[c] = c * kSoftM;
    for (int r = 0; r < kSoftM; ++r) impl->Ai[c * kSoftM + r] = r;
  }
  impl->Ap[kSoftN] = kSoftN * kSoftM;
  impl->Amat.m = kSoftM;
  impl->Amat.n = kSoftN;
  impl->Amat.nzmax = kSoftN * kSoftM;
  impl->Amat.x = impl->Ax;
  impl->Amat.i = impl->Ai;
  impl->Amat.p = impl->Ap;
  impl->Amat.nz = -1;

  // Seed with REPRESENTATIVE magnitudes (not zeros): OSQP computes its Ruiz
  // equilibration once at setup and reuses those factors on every update, so
  // scaling must see the real orders of magnitude (normalized rows O(1),
  // slack weights O(1e3), bounds O(1..10)). Placeholder zeros would leave the
  // solver effectively unscaled on live data.
  for (int c = 0; c < kSoftN; ++c) {
    impl->Px[c] = (c < kControlDim) ? static_cast<c_float>(H_diag[c]) : 1000.0;
  }
  for (int i = 0; i < kMaxCbfRows; ++i) {
    impl->Ax[i] = 1.0;                                                              // soft row col 0
    impl->Ax[(kControlDim + i) * kSoftM + i] = 1.0;                                  // soft slack col
    impl->Ax[(kControlDim + i) * kSoftM + kMaxCbfRows + i] = 1.0;                    // xi >= 0
  }
  for (int r = 0; r < kMaxQpRows; ++r) impl->Ax[2 * kMaxCbfRows + r] = 1.0;          // hard col 0
  for (int k = 0; k < kMaxSlewRows; ++k) {
    impl->Ax[2 * kMaxCbfRows + kMaxQpRows + k] = 1.0;                                // slew col 0
  }
  for (int r = 0; r < kSoftM; ++r) {
    impl->l[r] = -1.0;
    impl->u[r] = OSQP_INFTY;
  }
  for (int i = 0; i < kMaxCbfRows; ++i) impl->l[kMaxCbfRows + i] = 0.0;  // xi >= 0

  impl->data.n = kSoftN;
  impl->data.m = kSoftM;
  impl->data.P = &impl->Pmat;
  impl->data.A = &impl->Amat;
  impl->data.q = impl->q;
  impl->data.l = impl->l;
  impl->data.u = impl->u;

  osqp_set_default_settings(&impl->settings);
  impl->settings.verbose = 0;
  impl->settings.warm_start = 1;
  impl->settings.polish = s.polish ? 1 : 0;
  impl->settings.eps_abs = s.eps_abs;
  impl->settings.eps_rel = s.eps_rel;
  impl->settings.max_iter = s.max_iter;
  impl->settings.rho = s.rho > 0.0 ? s.rho : 1.0;
  impl->settings.adaptive_rho = s.adaptive_rho ? 1 : 0;
  impl->settings.scaling = s.scaling >= 0 ? s.scaling : 10;

  c_int flag = osqp_setup(&impl->work, &impl->data, &impl->settings);
  if (flag != 0 || !impl->work) {
    if (impl->work) {
      osqp_cleanup(impl->work);
      impl->work = nullptr;
    }
  }
}

OsqpSolver::~OsqpSolver() {
  if (impl_) {
    if (impl_->work) osqp_cleanup(impl_->work);
    delete impl_;
  }
}

bool OsqpSolver::ready() const noexcept { return impl_ && impl_->work; }

bool OsqpSolver::solve(const TinyQpProblem& problem, TinyQpSolution& sol) noexcept {
  sol.optimal = false;
  sol.iters = 0;
  for (int i = 0; i < kMaxCbfRows; ++i) sol.slack[i] = 0.0;
  if (!ready()) return false;
  if (!(problem.H_diag[0] > 0.0 && problem.H_diag[1] > 0.0)) return false;
  if (problem.num_soft < 0 || problem.num_soft > kMaxCbfRows) return false;
  int mh = problem.num_rows < 0 ? 0 : problem.num_rows;
  if (mh > kMaxQpRows) return false;
  for (int i = 0; i < problem.num_soft; ++i) {
    if (!(problem.slack_w[i] > 0.0)) return false;
  }
  Impl* impl = impl_;
  const int M = problem.num_soft;

  auto at = [&](int col, int row) noexcept -> c_float& { return impl->Ax[col * kSoftM + row]; };
  // Clear pattern values (structure is fixed dense; values rewritten below).
  for (int k = 0; k < kSoftN * kSoftM; ++k) impl->Ax[k] = 0.0;

  // Soft rows 0..M-1: [a_i0, a_i1, 0..1..0] >= b_i.
  for (int i = 0; i < kMaxCbfRows; ++i) {
    if (i < M) {
      at(0, i) = static_cast<c_float>(problem.A_soft[i][0]);
      at(1, i) = static_cast<c_float>(problem.A_soft[i][1]);
      at(kControlDim + i, i) = 1.0;
      impl->l[i] = static_cast<c_float>(problem.b_soft[i]);
    } else {
      impl->l[i] = -OSQP_INFTY;  // no-op row
    }
    impl->u[i] = OSQP_INFTY;
  }
  // Slack non-negativity rows M..2M-1.
  for (int i = 0; i < kMaxCbfRows; ++i) {
    const int r = kMaxCbfRows + i;
    at(kControlDim + i, r) = 1.0;
    impl->l[r] = 0.0;
    impl->u[r] = OSQP_INFTY;
  }
  // Hard rows.
  for (int r = 0; r < kMaxQpRows; ++r) {
    const int rr = 2 * kMaxCbfRows + r;
    if (r < mh) {
      at(0, rr) = static_cast<c_float>(problem.C[r][0]);
      at(1, rr) = static_cast<c_float>(problem.C[r][1]);
      impl->l[rr] = static_cast<c_float>(problem.d[r]);
    } else {
      impl->l[rr] = -OSQP_INFTY;
    }
    impl->u[rr] = OSQP_INFTY;
  }
  // Slew rows (absolute box).
  {
    const int base = 2 * kMaxCbfRows + kMaxQpRows;
    if (problem.use_slew) {
      at(0, base) = 1.0;
      impl->l[base] = static_cast<c_float>(problem.slew_lo[0]);
      at(0, base + 1) = -1.0;
      impl->l[base + 1] = static_cast<c_float>(-problem.slew_hi[0]);
      at(1, base + 2) = 1.0;
      impl->l[base + 2] = static_cast<c_float>(problem.slew_lo[1]);
      at(1, base + 3) = -1.0;
      impl->l[base + 3] = static_cast<c_float>(-problem.slew_hi[1]);
      for (int k = 0; k < 4; ++k) impl->u[base + k] = OSQP_INFTY;
    } else {
      for (int k = 0; k < 4; ++k) {
        impl->l[base + k] = -OSQP_INFTY;
        impl->u[base + k] = OSQP_INFTY;
      }
    }
  }

  // Hessian diagonal + linear cost.
  impl->Px[0] = static_cast<c_float>(problem.H_diag[0]);
  impl->Px[1] = static_cast<c_float>(problem.H_diag[1]);
  for (int i = 0; i < kMaxCbfRows; ++i) {
    impl->Px[kControlDim + i] =
        (i < M) ? static_cast<c_float>(problem.slack_w[i]) : 1.0;
  }
  impl->q[0] = static_cast<c_float>(-problem.H_diag[0] * problem.u_nom[0]);
  impl->q[1] = static_cast<c_float>(-problem.H_diag[1] * problem.u_nom[1]);
  for (int c = kControlDim; c < kSoftN; ++c) impl->q[c] = 0.0;

  // Whole-matrix / whole-vector updates (NULL index arrays). NOTE: NULL idx
  // still dereferences the value buffers, so P values are passed explicitly
  // alongside the new A values.
  if (osqp_update_P_A(impl->work, impl->Px, nullptr, 0, impl->Ax, nullptr, 0) != 0)
    return false;
  if (osqp_update_lin_cost(impl->work, impl->q) != 0) return false;
  if (osqp_update_bounds(impl->work, impl->l, impl->u) != 0) return false;

  // Natural OSQP warm start: x, y, z persist in the workspace across solves.
  // NOTE: do NOT reset duals here — zeroing y every cycle destroys the active
  // set history and stalls ADMM for thousands of iterations on marching
  // (slew-rate) constraints.
  osqp_solve(impl->work);
  sol.iters = static_cast<int>(impl->work->info->iter);
  const c_int status = impl->work->info->status_val;
  if (status != OSQP_SOLVED && status != OSQP_SOLVED_INACCURATE) return false;
  if (!impl->work->solution || !impl->work->solution->x) return false;

  const double z0 = static_cast<double>(impl->work->solution->x[0]);
  const double z1 = static_cast<double>(impl->work->solution->x[1]);
  if (!std::isfinite(z0) || !std::isfinite(z1)) return false;
  sol.u[0] = z0;
  sol.u[1] = z1;
  double obj = 0.5 * (problem.H_diag[0] * (z0 - problem.u_nom[0]) * (z0 - problem.u_nom[0]) +
                      problem.H_diag[1] * (z1 - problem.u_nom[1]) * (z1 - problem.u_nom[1]));
  for (int i = 0; i < kMaxCbfRows; ++i) {
    double xi = 0.0;
    if (i < M) {
      xi = static_cast<double>(impl->work->solution->x[kControlDim + i]);
      if (!std::isfinite(xi)) return false;
      if (xi < 0.0 && xi > -1e-7) xi = 0.0;  // snap solver dust
      obj += 0.5 * problem.slack_w[i] * xi * xi;
    }
    sol.slack[i] = xi;
  }
  sol.objective = obj;
  sol.optimal = (status == OSQP_SOLVED);
  return true;
}

}  // namespace qp
}  // namespace av_safety

#endif  // HAVE_OSQP
