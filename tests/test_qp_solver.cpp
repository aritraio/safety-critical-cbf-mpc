// Phase-3 verification: exact enumeration solver (optimality vs. brute force,
// infeasibility detection) and OSQP cross-check (when HAVE_OSQP).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>

#include "av_safety/qp/enumeration_qp_solver.hpp"
#ifdef HAVE_OSQP
#include "av_safety/qp/osqp_solver.hpp"
#endif

using namespace av_safety::qp;

namespace {

// Deterministic PRNG (xorshift64) — reproducible random QPs, no <random> bloat.
struct Rng {
  uint64_t s{0x9E3779B97F4A7C15ull};
  double uni() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return static_cast<double>(s >> 11) * (1.0 / 9007199254740992.0);
  }
  double range(double lo, double hi) { return lo + (hi - lo) * uni(); }
};

TinyQpProblem RandomFeasibleProblem(Rng& rng) {
  TinyQpProblem p;
  p.H_diag[0] = rng.range(0.2, 3.0);
  p.H_diag[1] = rng.range(0.2, 3.0);
  p.u_nom[0] = rng.range(-0.5, 0.5);
  p.u_nom[1] = rng.range(-4.0, 2.0);
  // Random feasible point, then rows that all contain it (=> feasible by design).
  const double zf0 = rng.range(-0.4, 0.4), zf1 = rng.range(-4.0, 2.0);
  int m = 0;
  const int nrows = 3 + static_cast<int>(rng.uni() * 5);  // 3..7
  for (int i = 0; i < nrows; ++i) {
    const double a0 = rng.range(-1.0, 1.0), a1 = rng.range(-1.0, 1.0);
    p.C[m][0] = a0;
    p.C[m][1] = a1;
    p.d[m] = a0 * zf0 + a1 * zf1 - rng.range(0.0, 1.0);  // zf strictly feasible
    ++m;
  }
  p.num_rows = m;
  return p;
}

double BruteForceCost(const TinyQpProblem& p) {
  // Dense grid + local refinement: tight upper bound on the true optimum.
  double best = 1e100;
  double bz0 = 0.0, bz1 = 0.0;
  for (int pass = 0; pass < 3; ++pass) {
    const double ext = pass == 0 ? 1.0 : 0.05;
    const double c0 = pass == 0 ? 0.0 : bz0, c1 = pass == 0 ? 0.0 : bz1;
    const int N = pass == 0 ? 200 : 100;
    for (int i = 0; i <= N; ++i) {
      for (int j = 0; j <= N; ++j) {
        const double z0 = c0 - ext + 2 * ext * i / N;
        const double z1 = c1 - 6 * ext + 12 * ext * j / N;
        bool ok = true;
        for (int r = 0; r < p.num_rows; ++r)
          if (p.C[r][0] * z0 + p.C[r][1] * z1 < p.d[r] - 1e-9) {
            ok = false;
            break;
          }
        if (!ok) continue;
        const double e0 = z0 - p.u_nom[0], e1 = z1 - p.u_nom[1];
        const double c = 0.5 * (p.H_diag[0] * e0 * e0 + p.H_diag[1] * e1 * e1);
        if (c < best) {
          best = c;
          bz0 = z0;
          bz1 = z1;
        }
      }
    }
  }
  return best;
}

// Failure diagnostic: dump a TinyQpProblem so ADMM-hostile instances can be
// reproduced offline. Only called on failure paths.
void DumpProblem(const TinyQpProblem& p) {
  std::fprintf(stderr, "  H=(%.4f,%.4f) un=(%.4f,%.4f) rows=%d soft=%d slew=%d\n", p.H_diag[0],
               p.H_diag[1], p.u_nom[0], p.u_nom[1], p.num_rows, p.num_soft, (int)p.use_slew);
  for (int r = 0; r < p.num_rows; ++r)
    std::fprintf(stderr, "  hard%d: (%.4f,%.4f) >= %.4f\n", r, p.C[r][0], p.C[r][1], p.d[r]);
  for (int i = 0; i < p.num_soft; ++i)
    std::fprintf(stderr, "  soft%d: (%.4f,%.4f) >= %.4f w=%.1f\n", i, p.A_soft[i][0],
                 p.A_soft[i][1], p.b_soft[i], p.slack_w[i]);
  if (p.use_slew)
    std::fprintf(stderr, "  slew: [%.4f,%.4f] x [%.4f,%.4f]\n", p.slew_lo[0], p.slew_hi[0],
                 p.slew_lo[1], p.slew_hi[1]);
}

}  // namespace

TEST(QpSolver, BoxClampBehaviour) {
  EnumerationQpSolver solver;
  TinyQpProblem p;
  p.H_diag[0] = 1.0;
  p.H_diag[1] = 1.0;
  p.u_nom[0] = 2.0;  // outside box
  p.u_nom[1] = -10.0;
  p.C[0][0] = 1.0;
  p.C[0][1] = 0.0;
  p.d[0] = -0.55;
  p.C[1][0] = -1.0;
  p.C[1][1] = 0.0;
  p.d[1] = -0.55;
  p.C[2][0] = 0.0;
  p.C[2][1] = 1.0;
  p.d[2] = -6.0;
  p.C[3][0] = 0.0;
  p.C[3][1] = -1.0;
  p.d[3] = -3.0;
  p.num_rows = 4;
  TinyQpSolution s;
  ASSERT_TRUE(solver.solve(p, s));
  EXPECT_TRUE(s.optimal);
  EXPECT_NEAR(s.u[0], 0.55, 1e-9);
  EXPECT_NEAR(s.u[1], -6.0, 1e-9);
}

TEST(QpSolver, PassThroughWhenFeasible) {
  EnumerationQpSolver solver;
  TinyQpProblem p;
  p.u_nom[0] = 0.1;
  p.u_nom[1] = 0.5;
  p.num_rows = 0;  // unconstrained
  TinyQpSolution s;
  ASSERT_TRUE(solver.solve(p, s));
  EXPECT_NEAR(s.u[0], 0.1, 1e-12);
  EXPECT_NEAR(s.u[1], 0.5, 1e-12);
}

TEST(QpSolver, DetectsInfeasible) {
  EnumerationQpSolver solver;
  TinyQpProblem p;
  p.C[0][0] = 1.0;
  p.C[0][1] = 0.0;
  p.d[0] = 1.0;
  p.C[1][0] = -1.0;
  p.C[1][1] = 0.0;
  p.d[1] = 0.0;  // z0 >= 1 and z0 <= 0
  p.num_rows = 2;
  TinyQpSolution s;
  EXPECT_FALSE(solver.solve(p, s));
}

TEST(QpSolver, MatchesBruteForce) {
  EnumerationQpSolver solver;
  Rng rng;
  for (int trial = 0; trial < 50; ++trial) {
    const TinyQpProblem p = RandomFeasibleProblem(rng);
    TinyQpSolution s;
    ASSERT_TRUE(solver.solve(p, s)) << "trial " << trial;
    const double ref = BruteForceCost(p);
    EXPECT_LE(s.objective, ref + 1e-6) << "trial " << trial << " (solver worse than grid)";
  }
}

namespace {

TinyQpProblem RandomSoftProblem(Rng& rng) {
  TinyQpProblem p;
  p.H_diag[0] = rng.range(0.2, 3.0);
  p.H_diag[1] = rng.range(0.2, 3.0);
  p.u_nom[0] = rng.range(-0.5, 0.5);
  p.u_nom[1] = rng.range(-4.0, 2.0);
  // Hard box rows around a feasible point (box always consistent here).
  const double zf0 = rng.range(-0.4, 0.4), zf1 = rng.range(-4.0, 2.0);
  int m = 0;
  const double lo0 = zf0 - rng.range(0.1, 0.5), hi0 = zf0 + rng.range(0.1, 0.5);
  const double lo1 = zf1 - rng.range(0.5, 2.0), hi1 = zf1 + rng.range(0.5, 2.0);
  p.C[m][0] = 1.0; p.C[m][1] = 0.0; p.d[m] = lo0; ++m;
  p.C[m][0] = -1.0; p.C[m][1] = 0.0; p.d[m] = -hi0; ++m;
  p.C[m][0] = 0.0; p.C[m][1] = 1.0; p.d[m] = lo1; ++m;
  p.C[m][0] = 0.0; p.C[m][1] = -1.0; p.d[m] = -hi1; ++m;
  p.num_rows = m;
  // Soft rows: NORMALIZED (||a||_inf <= 1), matching the filter contract
  // (CbfQpFilter normalizes Lie rows before packing; see tiny_qp.hpp).
  // Raw huge-gradient rows are out of contract for first-order ADMM.
  const int ns = 1 + static_cast<int>(rng.uni() * 3);  // 1..3
  for (int i = 0; i < ns; ++i) {
    p.A_soft[i][0] = rng.range(-1.0, 1.0);
    p.A_soft[i][1] = rng.range(-1.0, 1.0);
    p.b_soft[i] = rng.range(-2.0, 2.0);
    const double ws[3] = {10.0, 1000.0, 1000.0};
    p.slack_w[i] = ws[static_cast<int>(rng.uni() * 3)];
  }
  p.num_soft = ns;
  if (rng.uni() < 0.5) {
    p.use_slew = true;
    p.slew_lo[0] = zf0 - rng.range(0.02, 0.2);
    p.slew_hi[0] = zf0 + rng.range(0.02, 0.2);
    p.slew_lo[1] = zf1 - rng.range(0.2, 1.0);
    p.slew_hi[1] = zf1 + rng.range(0.2, 1.0);
  }
  return p;
}

// Brute force for soft problems: dense grid over u, analytic slacks, hard
// rows (incl. slew) as feasibility checks. Tight upper bound on the optimum.
double SoftBruteForceCost(const TinyQpProblem& p) {
  double best = 1e100;
  double bz0 = 0.0, bz1 = 0.0;
  for (int pass = 0; pass < 3; ++pass) {
    const double ext = pass == 0 ? 1.0 : 0.05;
    const double c0 = pass == 0 ? 0.0 : bz0, c1 = pass == 0 ? 0.0 : bz1;
    const int N = pass == 0 ? 200 : 100;
    for (int i = 0; i <= N; ++i) {
      for (int j = 0; j <= N; ++j) {
        const double z0 = c0 - ext + 2 * ext * i / N;
        const double z1 = c1 - 6 * ext + 12 * ext * j / N;
        bool ok = true;
        for (int r = 0; r < p.num_rows; ++r)
          if (p.C[r][0] * z0 + p.C[r][1] * z1 < p.d[r] - 1e-9) {
            ok = false;
            break;
          }
        if (ok && p.use_slew) {
          if (z0 < p.slew_lo[0] - 1e-9 || z0 > p.slew_hi[0] + 1e-9 ||
              z1 < p.slew_lo[1] - 1e-9 || z1 > p.slew_hi[1] + 1e-9)
            ok = false;
        }
        if (!ok) continue;
        double c = 0.5 * (p.H_diag[0] * (z0 - p.u_nom[0]) * (z0 - p.u_nom[0]) +
                          p.H_diag[1] * (z1 - p.u_nom[1]) * (z1 - p.u_nom[1]));
        for (int k = 0; k < p.num_soft; ++k) {
          const double viol = p.b_soft[k] - (p.A_soft[k][0] * z0 + p.A_soft[k][1] * z1);
          if (viol > 0.0) c += 0.5 * p.slack_w[k] * viol * viol;
        }
        if (c < best) {
          best = c;
          bz0 = z0;
          bz1 = z1;
        }
      }
    }
  }
  return best;
}

}  // namespace

TEST(QpSolver, SoftSlackMatchesBruteForce) {
  EnumerationQpSolver solver;
  Rng rng;
  for (int trial = 0; trial < 40; ++trial) {
    const TinyQpProblem p = RandomSoftProblem(rng);
    TinyQpSolution s;
    ASSERT_TRUE(solver.solve(p, s)) << "trial " << trial;
    const double ref = SoftBruteForceCost(p);
    EXPECT_LE(s.objective, ref + 1e-4) << "trial " << trial << " (solver worse than grid)";
    // Slack consistency: reported slacks match analytic elimination.
    for (int i = 0; i < p.num_soft; ++i) {
      const double viol = p.b_soft[i] - (p.A_soft[i][0] * s.u[0] + p.A_soft[i][1] * s.u[1]);
      EXPECT_NEAR(s.slack[i], viol > 0.0 ? viol : 0.0, 1e-6) << "trial " << trial;
    }
    // Slew respected whenever enabled.
    if (p.use_slew) {
      EXPECT_GE(s.u[0], p.slew_lo[0] - 1e-9);
      EXPECT_LE(s.u[0], p.slew_hi[0] + 1e-9);
      EXPECT_GE(s.u[1], p.slew_lo[1] - 1e-9);
      EXPECT_LE(s.u[1], p.slew_hi[1] + 1e-9);
    }
  }
}

// Exact-penalty property: hard-feasible problems recover xi = 0 and the hard
// optimum when solved through the soft path.
TEST(QpSolver, SoftRecoversHardSolution) {
  EnumerationQpSolver solver;
  Rng rng;
  for (int trial = 0; trial < 20; ++trial) {
    TinyQpProblem p = RandomFeasibleProblem(rng);
    TinyQpSolution hard;
    ASSERT_TRUE(solver.solve(p, hard));
    // Re-solve with every hard row ALSO softened (huge weights).
    for (int i = 0; i < p.num_rows && i < kMaxCbfRows; ++i) {
      p.A_soft[i][0] = p.C[i][0];
      p.A_soft[i][1] = p.C[i][1];
      p.b_soft[i] = p.d[i];
      p.slack_w[i] = 1e6;
    }
    p.num_soft = p.num_rows < kMaxCbfRows ? p.num_rows : kMaxCbfRows;
    TinyQpSolution soft;
    ASSERT_TRUE(solver.solve(p, soft)) << "trial " << trial;
    EXPECT_NEAR(soft.u[0], hard.u[0], 1e-6) << "trial " << trial;
    EXPECT_NEAR(soft.u[1], hard.u[1], 1e-6) << "trial " << trial;
    for (int i = 0; i < p.num_soft; ++i) EXPECT_NEAR(soft.slack[i], 0.0, 1e-6);
  }
}

// A hard-infeasible problem becomes feasible once its rows are softened.
TEST(QpSolver, SoftenedInfeasibleBecomesFeasible) {
  EnumerationQpSolver solver;
  TinyQpProblem p;
  p.C[0][0] = 1.0;
  p.C[0][1] = 0.0;
  p.d[0] = 1.0;
  p.C[1][0] = -1.0;
  p.C[1][1] = 0.0;
  p.d[1] = 0.0;  // z0 >= 1 and z0 <= 0: hard-infeasible
  p.num_rows = 2;
  TinyQpSolution s;
  EXPECT_FALSE(solver.solve(p, s));
  // Soften row 0 (keep row 1 hard): feasible, slack absorbs the conflict.
  p.A_soft[0][0] = 1.0;
  p.A_soft[0][1] = 0.0;
  p.b_soft[0] = 1.0;
  p.slack_w[0] = 100.0;
  p.num_soft = 1;
  p.num_rows = 1;  // only row 1 stays hard (z0 <= 0)
  // NB: rows beyond num_rows are ignored; move row 1 to slot 0.
  p.C[0][0] = -1.0;
  p.C[0][1] = 0.0;
  p.d[0] = 0.0;
  ASSERT_TRUE(solver.solve(p, s));
  EXPECT_TRUE(s.optimal);
  EXPECT_LE(s.u[0], 0.0);
  EXPECT_GT(s.slack[0], 0.9);  // ~1.0: full violation shed into slack
}

#ifdef HAVE_OSQP
TEST(QpSolver, OsqpAgreesWithEnumeration) {
  EnumerationQpSolver exact;
  const double H[2] = {1.0, 0.5};
  OsqpSolver osqp(H, OsqpSolver::Settings{});
  ASSERT_TRUE(osqp.ready());
  Rng rng;
  for (int trial = 0; trial < 30; ++trial) {
    TinyQpProblem p = RandomFeasibleProblem(rng);
    p.H_diag[0] = H[0];
    p.H_diag[1] = H[1];
    TinyQpSolution se, so;
    ASSERT_TRUE(exact.solve(p, se)) << "trial " << trial;
    ASSERT_TRUE(osqp.solve(p, so)) << "trial " << trial;
    EXPECT_TRUE(so.optimal) << "trial " << trial;
    EXPECT_NEAR(so.u[0], se.u[0], 2e-3) << "trial " << trial;
    EXPECT_NEAR(so.u[1], se.u[1], 2e-3) << "trial " << trial;
  }
}

TEST(QpSolver, OsqpDetectsInfeasible) {
  const double H[2] = {1.0, 0.5};
  OsqpSolver osqp(H, OsqpSolver::Settings{});
  ASSERT_TRUE(osqp.ready());
  TinyQpProblem p;
  p.H_diag[0] = H[0];
  p.H_diag[1] = H[1];
  p.C[0][0] = 1.0;
  p.C[0][1] = 0.0;
  p.d[0] = 1.0;
  p.C[1][0] = -1.0;
  p.C[1][1] = 0.0;
  p.d[1] = 0.0;
  p.num_rows = 2;
  TinyQpSolution s;
  EXPECT_FALSE(osqp.solve(p, s));
  EXPECT_FALSE(s.optimal);
}

TEST(QpSolver, OsqpSoftAgreesWithEnumeration) {
  EnumerationQpSolver exact;
  const double H[2] = {1.0, 0.5};
  Rng rng;
  for (int trial = 0; trial < 20; ++trial) {
    TinyQpProblem p = RandomSoftProblem(rng);
    p.H_diag[0] = H[0];
    p.H_diag[1] = H[1];
    TinyQpSolution se, so;
    ASSERT_TRUE(exact.solve(p, se)) << "trial " << trial;
    // Fresh instance per trial: cross-problem warm-start poisoning is a
    // shared-workspace effect covered at filter level (800-step rollouts),
    // not a per-problem agreement question.
    OsqpSolver::Settings st;
    st.max_iter = 2000;
    OsqpSolver osqp(H, st);
    ASSERT_TRUE(osqp.ready());
    const bool ok = osqp.solve(p, so);
    if (!ok) DumpProblem(p);
    ASSERT_TRUE(ok) << "trial " << trial;
    EXPECT_TRUE(so.optimal) << "trial " << trial;
    EXPECT_NEAR(so.u[0], se.u[0], 2e-3) << "trial " << trial;
    EXPECT_NEAR(so.u[1], se.u[1], 2e-3) << "trial " << trial;
    // Objective comparison is tolerance-limited: penalty weights amplify
    // 1e-4-scale ADMM residuals, so use a relative bound.
    const double obj_tol = 1e-3 * (1.0 + std::abs(se.objective));
    EXPECT_NEAR(so.objective, se.objective, obj_tol) << "trial " << trial;
  }
}

// The exact backend is immune to raw row scaling (out-of-contract rows that
// first-order ADMM cannot be asked to swallow); the filter normalizes.
TEST(QpSolver, EnumerationHandlesHugeGradientRows) {
  EnumerationQpSolver solver;
  TinyQpProblem p;
  p.H_diag[0] = 1.0;
  p.H_diag[1] = 0.5;
  p.u_nom[0] = 0.0;
  p.u_nom[1] = 0.0;
  p.C[0][0] = 1.0; p.C[0][1] = 0.0; p.d[0] = -0.55;
  p.C[1][0] = -1.0; p.C[1][1] = 0.0; p.d[1] = -0.55;
  p.C[2][0] = 0.0; p.C[2][1] = 1.0; p.d[2] = -6.0;
  p.C[3][0] = 0.0; p.C[3][1] = -1.0; p.d[3] = -3.0;
  p.num_rows = 4;
  p.A_soft[0][0] = -1063.0;  // drift-like yaw gradient, raw scale
  p.A_soft[0][1] = 0.2;
  p.b_soft[0] = 250.0;  // violated at origin => optimum rides the row
  p.slack_w[0] = 1000.0;
  p.num_soft = 1;
  TinyQpSolution s;
  ASSERT_TRUE(solver.solve(p, s));
  EXPECT_TRUE(s.optimal);
  // Exact boundary optimum: -1063*z0 + 0.2*z1 = 250 minimized over the box
  // gives z1 = 8.85e-5, z0 = -0.23518 (the 0.2 coupling genuinely tilts it).
  // Tolerances honor double-precision cancellation in the 1e9-scale subset KKT.
  EXPECT_NEAR(s.u[0], -0.23518, 1e-5);
  EXPECT_NEAR(s.u[1], 8.85e-5, 1e-5);
  EXPECT_NEAR(s.slack[0], 0.0, 1e-6);
}
#endif
