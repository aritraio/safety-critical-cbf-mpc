#include "av_safety/qp/enumeration_qp_solver.hpp"

#include <cmath>

namespace av_safety {
namespace qp {
namespace {

// Hard-row workspace: problem rows + optional slew box as general rows.
constexpr int kHardMax = kMaxQpRows + kMaxSlewRows;  // 14
constexpr double kFeasTol = 1e-9;
constexpr double kConsTol = 1e-7;  // subset-consistency tolerance

struct HardSet {
  double C[kHardMax][kControlDim]{{0.0, 0.0}};
  double d[kHardMax]{0.0};
  int m{0};
};

struct SoftSet {
  double A[kMaxCbfRows][kControlDim]{{0.0, 0.0}};
  double b[kMaxCbfRows]{0.0};
  double w[kMaxCbfRows]{0.0};
  int M{0};
  double H0{1.0}, H1{0.5};
  double un0{0.0}, un1{0.0};
};

bool HardFeasible(const HardSet& h, double z0, double z1) noexcept {
  for (int i = 0; i < h.m; ++i) {
    if (h.C[i][0] * z0 + h.C[i][1] * z1 < h.d[i] - 1e-7) return false;
  }
  return true;
}

// Full soft-QP objective at u (slacks eliminated analytically).
double SoftCost(const SoftSet& s, double z0, double z1) noexcept {
  const double e0 = z0 - s.un0, e1 = z1 - s.un1;
  double c = 0.5 * (s.H0 * e0 * e0 + s.H1 * e1 * e1);
  for (int i = 0; i < s.M; ++i) {
    const double viol = s.b[i] - (s.A[i][0] * z0 + s.A[i][1] * z1);
    if (viol > 0.0) c += 0.5 * s.w[i] * viol * viol;
  }
  return c;
}

void SoftSlacks(const SoftSet& s, double z0, double z1, double xi[kMaxCbfRows]) noexcept {
  for (int i = 0; i < s.M; ++i) {
    const double viol = s.b[i] - (s.A[i][0] * z0 + s.A[i][1] * z1);
    xi[i] = viol > 0.0 ? viol : 0.0;
  }
}

}  // namespace

bool EnumerationQpSolver::solve(const TinyQpProblem& problem, TinyQpSolution& sol) noexcept {
  sol.optimal = false;
  sol.iters = 0;
  for (int i = 0; i < kMaxCbfRows; ++i) sol.slack[i] = 0.0;
  if (!(problem.H_diag[0] > 0.0 && problem.H_diag[1] > 0.0)) return false;
  if (problem.num_soft < 0 || problem.num_soft > kMaxCbfRows) return false;

  SoftSet soft;
  soft.H0 = problem.H_diag[0];
  soft.H1 = problem.H_diag[1];
  soft.un0 = problem.u_nom[0];
  soft.un1 = problem.u_nom[1];
  soft.M = problem.num_soft;
  for (int i = 0; i < soft.M; ++i) {
    if (!(problem.slack_w[i] > 0.0)) return false;  // slack must be penalized
    soft.A[i][0] = problem.A_soft[i][0];
    soft.A[i][1] = problem.A_soft[i][1];
    soft.b[i] = problem.b_soft[i];
    soft.w[i] = problem.slack_w[i];
  }

  // Fold slew box into the hard-row list (single source of truth downstream).
  HardSet hard;
  {
    int m = problem.num_rows < 0 ? 0 : problem.num_rows;
    if (m > kMaxQpRows) m = kMaxQpRows;
    for (int i = 0; i < m; ++i) {
      hard.C[i][0] = problem.C[i][0];
      hard.C[i][1] = problem.C[i][1];
      hard.d[i] = problem.d[i];
    }
    hard.m = m;
    if (problem.use_slew && hard.m + 4 <= kHardMax) {
      const int k = hard.m;
      hard.C[k][0] = 1.0;
      hard.C[k][1] = 0.0;
      hard.d[k] = problem.slew_lo[0];
      hard.C[k + 1][0] = -1.0;
      hard.C[k + 1][1] = 0.0;
      hard.d[k + 1] = -problem.slew_hi[0];
      hard.C[k + 2][0] = 0.0;
      hard.C[k + 2][1] = 1.0;
      hard.d[k + 2] = problem.slew_lo[1];
      hard.C[k + 3][0] = 0.0;
      hard.C[k + 3][1] = -1.0;
      hard.d[k + 3] = -problem.slew_hi[1];
      hard.m = k + 4;
    }
  }

  double best = 1e100;
  double bz0 = soft.un0, bz1 = soft.un1;

  auto consider = [&](double z0, double z1) noexcept {
    ++sol.iters;
    if (!std::isfinite(z0) || !std::isfinite(z1)) return;
    if (!HardFeasible(hard, z0, z1)) return;
    const double c = SoftCost(soft, z0, z1);
    if (c < best) {
      best = c;
      bz0 = z0;
      bz1 = z1;
    }
  };

  if (soft.M == 0) {
    // ---- Hard-only fast path (bit-identical semantics to the original) ----
    consider(soft.un0, soft.un1);
    // Single-constraint projections (diagonal-H closed form).
    for (int i = 0; i < hard.m; ++i) {
      const double a0 = hard.C[i][0], a1 = hard.C[i][1];
      const double denom = a0 * a0 / soft.H0 + a1 * a1 / soft.H1;
      if (denom < 1e-12) continue;
      const double viol = hard.d[i] - (a0 * soft.un0 + a1 * soft.un1);
      const double t = viol / denom;
      consider(soft.un0 + t * a0 / soft.H0, soft.un1 + t * a1 / soft.H1);
    }
  } else {
    // ---- (a) Soft-subset enumeration: smooth KKT per violated set --------
    const int nsub = 1 << soft.M;  // <= 1024 (soft.M <= kMaxCbfRows)
    for (int mask = 0; mask < nsub; ++mask) {
      ++sol.iters;
      // (H + S w a a') u = H un + S w b a
      double m00 = soft.H0, m01 = 0.0, m11 = soft.H1;
      double r0 = soft.H0 * soft.un0, r1 = soft.H1 * soft.un1;
      for (int i = 0; i < soft.M; ++i) {
        if (mask & (1 << i)) {
          const double a0 = soft.A[i][0], a1 = soft.A[i][1], w = soft.w[i];
          m00 += w * a0 * a0;
          m01 += w * a0 * a1;
          m11 += w * a1 * a1;
          r0 += w * soft.b[i] * a0;
          r1 += w * soft.b[i] * a1;
        }
      }
      const double det = m00 * m11 - m01 * m01;
      if (!(det > 0.0) || !std::isfinite(det)) continue;  // H>0 => det>0 normally
      const double z0 = (r0 * m11 - m01 * r1) / det;
      const double z1 = (m00 * r1 - m01 * r0) / det;
      if (!std::isfinite(z0) || !std::isfinite(z1)) continue;
      // Consistency: assumed violated set must match actual (within tol).
      bool ok = HardFeasible(hard, z0, z1);
      for (int i = 0; ok && i < soft.M; ++i) {
        const double viol = soft.b[i] - (soft.A[i][0] * z0 + soft.A[i][1] * z1);
        if (mask & (1 << i)) {
          if (viol < -kConsTol) ok = false;  // assumed violated but satisfied
        } else {
          if (viol > kConsTol) ok = false;  // assumed satisfied but violated
        }
      }
      if (!ok) continue;
      const double c = SoftCost(soft, z0, z1);
      if (c < best) {
        best = c;
        bz0 = z0;
        bz1 = z1;
      }
    }

    // ---- (b) Exact 1-D minimization along each active hard row ------------
    for (int j = 0; j < hard.m; ++j) {
      const double a0 = hard.C[j][0], a1 = hard.C[j][1];
      const double n2 = a0 * a0 + a1 * a1;
      if (n2 < 1e-14) continue;
      const double nl = std::sqrt(n2);
      // Line point + unit direction: u(t) = u0 + t*v.
      const double u0 = a0 * hard.d[j] / n2, u1 = a1 * hard.d[j] / n2;
      const double v0 = -a1 / nl, v1 = a0 / nl;
      // Feasible t-interval from all hard rows.
      double tlo = -1e9, thi = 1e9;
      for (int k = 0; k < hard.m; ++k) {
        const double s0 = hard.C[k][0] * u0 + hard.C[k][1] * u1 - hard.d[k];
        const double s1 = hard.C[k][0] * v0 + hard.C[k][1] * v1;
        if (std::abs(s1) < 1e-14) {
          if (s0 < -1e-9) {
            tlo = 1.0;
            thi = 0.0;
            break;
          }
          continue;
        }
        const double tb = -s0 / s1;
        if (s1 > 0.0) {
          if (tb > tlo) tlo = tb;
        } else {
          if (tb < thi) thi = tb;
        }
      }
      if (tlo > thi) continue;
      // Kinks where soft rows switch activity.
      double kink[kMaxCbfRows];
      int nk = 0;
      for (int i = 0; i < soft.M; ++i) {
        const double g0 = soft.A[i][0] * u0 + soft.A[i][1] * u1 - soft.b[i];
        const double g1 = soft.A[i][0] * v0 + soft.A[i][1] * v1;
        if (std::abs(g1) < 1e-14) continue;
        const double tk = -g0 / g1;
        if (tk >= tlo && tk <= thi && nk < kMaxCbfRows) kink[nk++] = tk;
      }
      // Insertion sort (tiny).
      for (int p = 1; p < nk; ++p) {
        const double key = kink[p];
        int q = p - 1;
        while (q >= 0 && kink[q] > key) {
          kink[q + 1] = kink[q];
          --q;
        }
        kink[q + 1] = key;
      }
      // Segment boundaries: tlo, kinks, thi.
      double bounds[kMaxCbfRows + 2];
      bounds[0] = tlo;
      for (int p = 0; p < nk; ++p) bounds[p + 1] = kink[p];
      bounds[nk + 1] = thi;
      consider(u0 + tlo * v0, u1 + tlo * v1);
      consider(u0 + thi * v0, u1 + thi * v1);
      for (int p = 0; p < nk; ++p) consider(u0 + kink[p] * v0, u1 + kink[p] * v1);
      for (int sgm = 0; sgm <= nk; ++sgm) {
        const double ta = bounds[sgm], tb = bounds[sgm + 1];
        if (tb <= ta) continue;
        // Active set at segment midpoint (constant inside open segment).
        const double tm = 0.5 * (ta + tb);
        const double mu0 = u0 + tm * v0, mu1 = u1 + tm * v1;
        // 1-D quadratic coefficients of F along the line for this set.
        double qa = v0 * v0 * soft.H0 + v1 * v1 * soft.H1;
        double qb = 2.0 * (soft.H0 * (u0 - soft.un0) * v0 + soft.H1 * (u1 - soft.un1) * v1);
        for (int i = 0; i < soft.M; ++i) {
          const double viol = soft.b[i] - (soft.A[i][0] * mu0 + soft.A[i][1] * mu1);
          if (viol > 0.0) {
            const double sv = soft.A[i][0] * v0 + soft.A[i][1] * v1;
            const double rs = soft.b[i] - (soft.A[i][0] * u0 + soft.A[i][1] * u1);
            qa += soft.w[i] * sv * sv;
            qb += -2.0 * soft.w[i] * rs * sv;
          }
        }
        if (qa > 1e-14) {
          double ts = -qb / (2.0 * qa);
          if (ts < ta) ts = ta;
          if (ts > tb) ts = tb;
          consider(u0 + ts * v0, u1 + ts * v1);
        }
      }
    }
  }

  // ---- (c) Hard-row pairwise intersections (vertices) ----------------------
  for (int i = 0; i < hard.m; ++i) {
    for (int j = i + 1; j < hard.m; ++j) {
      const double a0 = hard.C[i][0], a1 = hard.C[i][1];
      const double b0 = hard.C[j][0], b1 = hard.C[j][1];
      const double det = a0 * b1 - a1 * b0;
      if (std::abs(det) < 1e-12) continue;  // parallel
      consider((hard.d[i] * b1 - a1 * hard.d[j]) / det,
               (a0 * hard.d[j] - hard.d[i] * b0) / det);
    }
  }

  if (best > 1e99) return false;  // hard-infeasible
  sol.u[0] = bz0;
  sol.u[1] = bz1;
  SoftSlacks(soft, bz0, bz1, sol.slack);
  sol.objective = best;
  sol.optimal = true;
  (void)kFeasTol;
  return true;
}

}  // namespace qp
}  // namespace av_safety
