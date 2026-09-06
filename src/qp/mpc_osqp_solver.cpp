#include "av_safety/qp/mpc_osqp_solver.hpp"

#ifdef HAVE_OSQP

#include <osqp.h>

#include <cmath>
#include <new>

static_assert(sizeof(c_float) == sizeof(double),
              "OSQP must be built with double precision (DFLOAT off)");

namespace av_safety {
namespace qp {

// Fixed structure: n = N vars, m = 2N box rows [I; -I], dense patterns.
// H stored upper-triangular (N(N+1)/2 entries, column-major CSC).
inline constexpr int kMpcN = kMpcHorizon;
inline constexpr int kMpcM = 2 * kMpcHorizon;
inline constexpr int kMpcHnnz = kMpcHorizon * (kMpcHorizon + 1) / 2;

struct MpcOsqpSolver::Impl {
  OSQPWorkspace* work{nullptr};
  OSQPData data{};
  OSQPSettings settings{};
  c_int Pp[kMpcN + 1]{0};
  c_int Pi[kMpcHnnz]{0};
  c_float Px[kMpcHnnz]{1.0};
  csc Pmat{0, 0, 0, nullptr, nullptr, nullptr, -1};
  c_int Ap[kMpcN + 1]{0};
  c_int Ai[kMpcM]{0};  // col c holds rows {c, N+c}
  c_float Ax[kMpcM]{0.0};
  csc Amat{0, 0, 0, nullptr, nullptr, nullptr, -1};
  c_float q[kMpcN]{0.0};
  c_float l[kMpcM]{0.0};
  c_float u[kMpcM]{0.0};
  double u_min{-0.55}, u_max{0.55};
};

MpcOsqpSolver::MpcOsqpSolver(double u_min, double u_max) : MpcOsqpSolver(u_min, u_max, Settings{}) {
}

MpcOsqpSolver::MpcOsqpSolver(double u_min, double u_max, const Settings& s) {
  Impl* impl = new (std::nothrow) Impl();
  if (!impl) return;
  impl_ = impl;
  impl->u_min = u_min;
  impl->u_max = u_max;

  // P pattern: upper triangle, column-major.
  int k = 0;
  for (int c = 0; c < kMpcN; ++c) {
    impl->Pp[c] = k;
    for (int r = 0; r <= c; ++r) {
      impl->Pi[k] = r;
      impl->Px[k] = (r == c) ? 1.0 : 0.0;  // representative seed (see below)
      ++k;
    }
  }
  impl->Pp[kMpcN] = kMpcHnnz;
  impl->Pmat.m = kMpcN;
  impl->Pmat.n = kMpcN;
  impl->Pmat.nzmax = kMpcHnnz;
  impl->Pmat.x = impl->Px;
  impl->Pmat.i = impl->Pi;
  impl->Pmat.p = impl->Pp;
  impl->Pmat.nz = -1;

  // A pattern: col c holds rows c (+1) and N+c (-1).
  for (int c = 0; c < kMpcN; ++c) {
    impl->Ap[c] = 2 * c;
    impl->Ai[2 * c] = c;
    impl->Ai[2 * c + 1] = kMpcN + c;
    impl->Ax[2 * c] = 1.0;
    impl->Ax[2 * c + 1] = -1.0;
  }
  impl->Ap[kMpcN] = kMpcM;
  impl->Amat.m = kMpcM;
  impl->Amat.n = kMpcN;
  impl->Amat.nzmax = kMpcM;
  impl->Amat.x = impl->Ax;
  impl->Amat.i = impl->Ai;
  impl->Amat.p = impl->Ap;
  impl->Amat.nz = -1;

  for (int c = 0; c < kMpcN; ++c) {
    impl->l[c] = static_cast<c_float>(u_min);
    impl->u[c] = static_cast<c_float>(u_max);
    impl->l[kMpcN + c] = static_cast<c_float>(-u_max);
    impl->u[kMpcN + c] = static_cast<c_float>(-u_min);
  }

  impl->data.n = kMpcN;
  impl->data.m = kMpcM;
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
  impl->settings.scaling = s.scaling >= 0 ? s.scaling : 0;

  c_int flag = osqp_setup(&impl->work, &impl->data, &impl->settings);
  if (flag != 0 || !impl->work) {
    if (impl->work) {
      osqp_cleanup(impl->work);
      impl->work = nullptr;
    }
  }
}

MpcOsqpSolver::~MpcOsqpSolver() {
  if (impl_) {
    if (impl_->work) osqp_cleanup(impl_->work);
    delete impl_;
  }
}

bool MpcOsqpSolver::ready() const noexcept { return impl_ && impl_->work; }

bool MpcOsqpSolver::solve(const Eigen::Matrix<double, kMpcHorizon, kMpcHorizon>& H,
                          const Eigen::Matrix<double, kMpcHorizon, 1>& g,
                          Eigen::Matrix<double, kMpcHorizon, 1>& U_out,
                          int& iters_out) noexcept {
  iters_out = 0;
  if (!ready()) return false;
  Impl* impl = impl_;
  // Pack upper triangle (H must be symmetric; symmetrize defensively).
  int k = 0;
  for (int c = 0; c < kMpcN; ++c) {
    for (int r = 0; r <= c; ++r) {
      const double v = 0.5 * (H(r, c) + H(c, r));
      if (!std::isfinite(v)) return false;
      impl->Px[k++] = static_cast<c_float>(v);
    }
  }
  for (int c = 0; c < kMpcN; ++c) {
    if (!std::isfinite(g(c, 0))) return false;
    impl->q[c] = static_cast<c_float>(g(c, 0));
  }
  if (osqp_update_P_A(impl->work, impl->Px, nullptr, 0, impl->Ax, nullptr, 0) != 0)
    return false;
  if (osqp_update_lin_cost(impl->work, impl->q) != 0) return false;

  osqp_solve(impl->work);
  iters_out = static_cast<int>(impl->work->info->iter);
  const c_int status = impl->work->info->status_val;
  if (status != OSQP_SOLVED && status != OSQP_SOLVED_INACCURATE) return false;
  if (!impl->work->solution || !impl->work->solution->x) return false;
  for (int c = 0; c < kMpcN; ++c) {
    const double v = static_cast<double>(impl->work->solution->x[c]);
    if (!std::isfinite(v)) return false;
    U_out(c, 0) = v;
  }
  return status == OSQP_SOLVED;
}

}  // namespace qp
}  // namespace av_safety

#endif  // HAVE_OSQP
