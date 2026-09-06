#include "av_safety/controllers/cbf_qp_filter.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace av_safety {

// Slot budgets (provably exact; pushBarrier/box guards backstop anyway).
static_assert(qp::kMaxCbfRows >= 2 + 2 * kMaxFilterObstacles,
              "soft slots must fit road + drift + N ellipse + N gap rows");
static_assert(qp::kMaxQpRows >= 2 + 2 * kMaxFilterObstacles + 4,
              "hard slots must fit barrier rows + box");

CbfQpConfig LoadCbfQpConfig(const std::string& yaml_path) {
  YAML::Node n;
  try {
    n = YAML::LoadFile(yaml_path);
  } catch (const std::exception& e) {
    throw std::runtime_error("cbf_qp: cannot load '" + yaml_path + "': " + e.what());
  }
  CbfQpConfig c;
  const auto q = n["cbf_qp"];
  if (!q) return c;  // all defaults
  if (q["H_diag"]) {
    const auto hd = q["H_diag"].as<std::vector<double>>();
    if (hd.size() != 2 || !(hd[0] > 0.0) || !(hd[1] > 0.0))
      throw std::runtime_error("cbf_qp: H_diag must be 2 positive entries");
    c.H_delta = hd[0];
    c.H_ax = hd[1];
  }
  c.slack_weight = q["slack_weight"].as<double>(c.slack_weight);
  if (!(c.slack_weight > 0.0)) throw std::runtime_error("cbf_qp: slack_weight must be > 0");
  c.use_slack = q["use_slack"].as<bool>(c.use_slack);
  c.use_slew = q["use_slew"].as<bool>(c.use_slew);
  c.dt = q["dt"].as<double>(c.dt);
  if (!(c.dt > 0.0)) throw std::runtime_error("cbf_qp: dt must be > 0");
  return c;
}

CbfQpFilter::CbfQpFilter(const VehicleParams& vehicle, const BarrierConfig& barriers,
                         const CbfQpConfig& cfg, qp::QpSolverBackend* backend) noexcept
    : vehicle_(vehicle),
      barriers_(barriers),
      cfg_(cfg),
      backend_(backend),
      model_(vehicle),
      road_(barriers.road),
      obstacle_(barriers.obstacle),
      drift_(barriers.drift),
      gap_(barriers.gap) {}

void CbfQpFilter::setObstacles(const ObstacleState* obs, int n) noexcept {
  if (n < 0) n = 0;
  if (n > kMaxFilterObstacles) n = kMaxFilterObstacles;
  num_obstacles_ = n;
  for (int i = 0; i < n; ++i) obstacles_[i] = obs[i];
}

void CbfQpFilter::filter(const StateVector& x, const ControlVector& u_nom, double mu,
                         ControlVector& u_safe_out, FilterDiagnostics& diag_out) noexcept {
  diag_out = FilterDiagnostics{};
  if (!backend_) {
    u_safe_out = u_nom;
    return;
  }

  // Single consistent linearization point for all barriers.
  StateDerivative f;
  InputMatrix G;
  StateMatrix J;
  model_.affineDecomposition(x, u_prev_, mu, f, G);
  model_.driftJacobian(x, mu, J);

  qp::TinyQpProblem prob;
  prob.H_diag[0] = cfg_.H_delta;
  prob.H_diag[1] = cfg_.H_ax;
  prob.u_nom[0] = u_nom(kSteer);
  prob.u_nom[1] = u_nom(kAx);
  int m = 0;  // hard rows
  int s = 0;  // soft rows
  double row_scale[qp::kMaxCbfRows];
  for (int i = 0; i < qp::kMaxCbfRows; ++i) row_scale[i] = 1.0;

  // Slew-rate box around the previously applied command (hard). Computed
  // once here: the QP slew rows AND the gap-row robust envelope below share
  // these identical bounds. Defaults to the absolute box when slew is off.
  double slew_lo[2] = {-vehicle_.steer_max, vehicle_.accel_min};
  double slew_hi[2] = {vehicle_.steer_max, vehicle_.accel_max};
  if (cfg_.use_slew && cfg_.dt > 0.0) {
    slew_lo[0] = u_prev_(kSteer) - vehicle_.steer_rate_max * cfg_.dt;
    slew_hi[0] = u_prev_(kSteer) + vehicle_.steer_rate_max * cfg_.dt;
    slew_lo[1] = u_prev_(kAx) - vehicle_.accel_rate_max * cfg_.dt;
    slew_hi[1] = u_prev_(kAx) + vehicle_.accel_rate_max * cfg_.dt;
    prob.slew_lo[0] = slew_lo[0];
    prob.slew_hi[0] = slew_hi[0];
    prob.slew_lo[1] = slew_lo[1];
    prob.slew_hi[1] = slew_hi[1];
    prob.use_slew = true;
  }

  // Barrier rows: soft (slacked) by default, hard in ablation mode.
  // Soft rows are normalized by s_i = max(||a_i||_inf, 1) so raw Lie rows
  // spanning ~0.1 (quiet cruise) to ~1e3 (drift-barrier yaw gradient) cannot
  // stall first-order ADMM. The slack weight applies on this normalized scale
  // (equal priority per barrier, scale-invariant trade-off); small rows are
  // never inflated (one-sided) to avoid amplifying solver dust. The quadratic
  // penalty stays exact: hard-feasible problems still recover xi = 0.
  // Slot budget (road + drift + N ellipse + N' gap, N <= 4) fits kMaxCbfRows
  // exactly; the guards below make overflow impossible by construction.
  auto pushBarrier = [&](const double* LgLf_or_Lg, double rhs) noexcept {
    if (cfg_.use_slack) {
      if (s >= qp::kMaxCbfRows) return;
      double sc = std::abs(LgLf_or_Lg[0]);
      const double a1 = std::abs(LgLf_or_Lg[1]);
      if (a1 > sc) sc = a1;
      if (!(sc >= 1.0)) sc = 1.0;
      if (!std::isfinite(sc)) sc = 1.0;
      prob.A_soft[s][0] = LgLf_or_Lg[0] / sc;
      prob.A_soft[s][1] = LgLf_or_Lg[1] / sc;
      prob.b_soft[s] = rhs / sc;
      prob.slack_w[s] = cfg_.slack_weight;
      row_scale[s] = sc;
      ++s;
    } else {
      if (m >= qp::kMaxQpRows) return;
      prob.C[m][0] = LgLf_or_Lg[0];
      prob.C[m][1] = LgLf_or_Lg[1];
      prob.d[m] = rhs;
      ++m;
    }
  };

  if (cfg_.use_road) {
    EcbffConstraint c;
    road_.linearize(x, f, J, G, c);
    diag_out.h_road = c.h;
    const double Lg[2] = {c.LgLf[0], c.LgLf[1]};
    pushBarrier(Lg, -c.Lf2 - (c.p1 + c.p2) * c.hdot - c.p1 * c.p2 * c.h);
  }
  if (cfg_.use_drift) {
    CbfLinearConstraint c;
    drift_.linearize(x, f, G, mu, vehicle_.gravity, c);
    diag_out.h_drift = c.h;
    const double Lg[2] = {c.Lg[0], c.Lg[1]};
    pushBarrier(Lg, -c.Lf - c.kappa * c.h);
  }
  diag_out.h_obs_min = 1e100;
  diag_out.h_gap_min = 1e100;
  int n_gap = 0;
  for (int i = 0; i < num_obstacles_; ++i) {
    EcbffConstraint c;
    obstacle_.linearize(x, obstacles_[i], f, J, G, c);
    if (c.h < diag_out.h_obs_min) diag_out.h_obs_min = c.h;
    const double Lg[2] = {c.LgLf[0], c.LgLf[1]};
    pushBarrier(Lg, -c.Lf2 - (c.p1 + c.p2) * c.hdot - c.p1 * c.p2 * c.h);
  }
  if (num_obstacles_ == 0) diag_out.h_obs_min = 1e100;
  // Headway (gap) rows: at most one per obstacle, gated ahead + near-lane.
  // Robust envelope over the slew box (see header): the raw row's steering
  // column (tire scrub coupling) would otherwise let the QP "brake" by
  // steering -- cheaper in control effort, catastrophic in yaw. Assume the
  // worst steer in this step's slew interval and make braking cover it.
  if (cfg_.use_gap) {
    for (int i = 0; i < num_obstacles_; ++i) {
      if (!gap_.applies(x, obstacles_[i])) continue;
      CbfLinearConstraint c;
      gap_.linearize(x, obstacles_[i], f, G, c);
      if (c.h < diag_out.h_gap_min) diag_out.h_gap_min = c.h;
      double Lg[2] = {c.Lg[0], c.Lg[1]};
      double rhs = -c.Lf - c.kappa * c.h;
      if (cfg_.use_slew && cfg_.dt > 0.0) {
        const double worst = std::min(Lg[0] * slew_lo[0], Lg[0] * slew_hi[0]);
        rhs -= worst;
        Lg[0] = 0.0;
      }
      pushBarrier(Lg, rhs);
      ++n_gap;
    }
  }
  if (n_gap == 0) diag_out.h_gap_min = 1e100;
  prob.num_soft = s;

  // Box rows: C z >= d form.
  prob.C[m][0] = 1.0;
  prob.C[m][1] = 0.0;
  prob.d[m] = -vehicle_.steer_max;
  ++m;
  prob.C[m][0] = -1.0;
  prob.C[m][1] = 0.0;
  prob.d[m] = -vehicle_.steer_max;
  ++m;
  prob.C[m][0] = 0.0;
  prob.C[m][1] = 1.0;
  prob.d[m] = vehicle_.accel_min;
  ++m;
  prob.C[m][0] = 0.0;
  prob.C[m][1] = -1.0;
  prob.d[m] = -vehicle_.accel_max;
  ++m;
  prob.num_rows = m;

  diag_out.num_rows = m;
  diag_out.num_soft = s;

  qp::TinyQpSolution sol;
  const auto t0 = std::chrono::steady_clock::now();
  bool ok = backend_->solve(prob, sol);
  bool primary_ok = ok && sol.optimal;
  if (!primary_ok && fallback_ != nullptr) {
    // Deadline net: exact fallback (enumeration) bounds worst-case latency
    // and guarantees a solution whenever the hard box is consistent.
    qp::TinyQpSolution fb;
    if (fallback_->solve(prob, fb) && fb.optimal) {
      sol = fb;
      ok = true;
      primary_ok = true;
      diag_out.used_fallback = true;
    }
  }
  const auto t1 = std::chrono::steady_clock::now();
  diag_out.solve_us =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) /
      1000.0;
  diag_out.iters = sol.iters;
  diag_out.optimal = sol.optimal;

  if (ok && sol.optimal && std::isfinite(sol.u[0]) && std::isfinite(sol.u[1])) {
    u_safe_out(kSteer) = sol.u[0];
    u_safe_out(kAx) = sol.u[1];
    diag_out.feasible = true;
    double smax = 0.0, snorm = 0.0;
    for (int i = 0; i < s; ++i) {
      // Backend slacks live in normalized-row units; report physical units.
      const double xi = (sol.slack[i] > 0.0 ? sol.slack[i] : 0.0) * row_scale[i];
      if (xi > smax) smax = xi;
      snorm += xi * xi;
    }
    diag_out.slack_max = smax;
    diag_out.slack_norm = std::sqrt(snorm);
    diag_out.softened = smax > 1e-6;
  } else {
    // Solver failure (not constraint conflict — slacks keep the problem
    // feasible): hold the nominal clamped into box AND slew box, flag
    // infeasible. Slew-clamping keeps the u_prev chain (and the gap-row
    // robust envelope, which assumes it) consistent. Never NaN.
    const double lo0 = std::max(-vehicle_.steer_max, slew_lo[0]);
    const double hi0 = std::min(vehicle_.steer_max, slew_hi[0]);
    const double lo1 = std::max(vehicle_.accel_min, slew_lo[1]);
    const double hi1 = std::min(vehicle_.accel_max, slew_hi[1]);
    u_safe_out(kSteer) = std::clamp(u_nom(kSteer), lo0, hi0);
    u_safe_out(kAx) = std::clamp(u_nom(kAx), lo1, hi1);
    diag_out.feasible = false;
  }
  const double e0 = u_safe_out(kSteer) - u_nom(kSteer);
  const double e1 = u_safe_out(kAx) - u_nom(kAx);
  diag_out.intervened = (cfg_.H_delta * e0 * e0 + cfg_.H_ax * e1 * e1) > 1e-6;
  u_prev_ = u_safe_out;
}

}  // namespace av_safety
