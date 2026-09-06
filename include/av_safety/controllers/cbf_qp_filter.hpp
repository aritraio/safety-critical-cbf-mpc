#pragma once
/// @file cbf_qp_filter.hpp
/// @brief Real-time soft-CBF-QP safety filter.
///
///   u*,xi* = argmin  1/2 ||u - u_nom||_H^2 + 1/2 sum_i w_i xi_i^2
///        s.t.    LgLf_i u + xi_i >= -Lf2_i - (p1+p2) hdot_i - p1 p2 h_i (soft ECBF)
///                Lg_j u + xi_j   >= -Lf_j - kappa h_j                  (soft RD1)
///                xi >= 0
///                u_min <= u <= u_max                                    (hard box)
///                u_prev - du_max <= u <= u_prev + du_max                (hard slew)
///
/// RD1 rows cover the drift envelope and the per-obstacle headway (gap)
/// barrier; the gap row gives early progressive braking at highway closing
/// speeds where the ellipse ECBF is myopic. The gap row's steering column
/// (tire scrub coupling) is replaced by a robust envelope over the slew box:
/// otherwise the QP discovers scrub-steering as a "cheap brake" and injects
/// yaw to open headway. Fallback holds the slew-clamped nominal so the
/// u_prev chain (and the envelope soundness that rests on it) never breaks.
///
/// The quadratic slack penalty is exact: when the hard problem is feasible the
/// solution coincides with it (xi = 0); otherwise barrier violations are shed
/// gracefully by weight instead of flagging infeasible, so the filter never
/// hits an infeasibility cliff when an obstacle appears inside the braking
/// distance. Slew limits bound per-step actuator jumps, which additionally
/// keeps the sequential quasi-affine linearization (f, J, G at u_prev) valid.
/// Soft rows are normalized to unit-inf-norm (weight applied on the normalized
/// scale) so raw Lie coefficients spanning ~0.1 to ~1e3 cannot stall ADMM.
///
/// The filter relinearizes (f, J, G) every cycle at the previously applied
/// command. filter() is noexcept and performs zero heap allocations: all rows
/// live in a fixed-size QP problem and the backend workspace is preallocated.
/// Deterministic latency pattern: OSQP primary with a tight iteration budget
/// plus an exact enumeration fallback (setFallbackBackend). If the primary
/// misses its budget, the fallback bounds worst-case latency while preserving
/// exactness. Total solver-failure (both backends) falls back to the
/// box-clamped nominal and flags feasible = false (no silent NaN).
///
/// Validity notes: the affine model is first-order exact in steer angle, so
/// large single-step steering jumps (deep tire saturation, Pacejka slope
/// collapse/sign-flip) degrade the guarantee until the next relinearization;
/// the slew bound is the primary guard here. Likewise, CBF feasibility under
/// input bounds is not recursive: approaches that are physically doomed (e.g.
/// head-on inside the braking distance) now shed slack (diagnosed via
/// slack_max) instead of going infeasible.

#include "av_safety/barriers/barrier_params.hpp"
#include "av_safety/barriers/drift_cbf.hpp"
#include "av_safety/barriers/gap_cbf.hpp"
#include "av_safety/barriers/obstacle_cbf.hpp"
#include "av_safety/barriers/road_boundary_cbf.hpp"
#include "av_safety/common/types.hpp"
#include "av_safety/common/vehicle_params.hpp"
#include "av_safety/dynamics/dynamic_bicycle_model.hpp"
#include "av_safety/qp/tiny_qp.hpp"

#include <string>

namespace av_safety {

inline constexpr int kMaxFilterObstacles = 4;

struct CbfQpConfig {
  double H_delta{1.0};
  double H_ax{0.5};
  bool use_road{true};
  bool use_drift{true};
  bool use_gap{true};  // headway rows (one per gated obstacle)
  double slack_weight{1000.0};  // exact-penalty weight on every barrier slack
  bool use_slack{true};         // false => legacy hard-barrier rows (ablation)
  bool use_slew{true};          // hard slew-rate box around u_prev
  double dt{0.01};              // controller step [s] for slew bounds
};

/// Load cbf_qp section from the barrier YAML (throws on invalid). Missing
/// keys fall back to the struct defaults above.
CbfQpConfig LoadCbfQpConfig(const std::string& yaml_path);

struct FilterDiagnostics {
  double solve_us{0.0};  // backend solve() wall time [us] (primary + fallback)
  int iters{0};
  bool optimal{false};
  bool feasible{false};
  bool intervened{false};  // ||u_safe - u_nom||_H > threshold
  bool softened{false};    // any barrier slack > slack_tol
  bool used_fallback{false};
  double slack_max{0.0};   // max barrier slack (0 => hard constraints met)
  double slack_norm{0.0};  // sqrt(sum xi^2)
  double h_road{0.0};
  double h_drift{0.0};
  double h_obs_min{0.0};
  double h_gap_min{0.0};
  int num_rows{0};   // hard rows sent to the solver
  int num_soft{0};   // soft (slacked) barrier rows sent to the solver
};

class CbfQpFilter {
 public:
  /// backend: non-owning pointer; must outlive the filter.
  CbfQpFilter(const VehicleParams& vehicle, const BarrierConfig& barriers,
              const CbfQpConfig& cfg, qp::QpSolverBackend* backend) noexcept;

  /// Deadline safety net: exact fallback backend (typically enumeration) used
  /// when the primary fails or misses its iteration budget. Optional.
  void setFallbackBackend(qp::QpSolverBackend* fallback) noexcept { fallback_ = fallback; }

  void setObstacles(const ObstacleState* obs, int n) noexcept;
  void reset() noexcept { u_prev_ = ControlVector::Zero(); }

  /// Hot loop: minimally invasive safe command. Zero allocations, noexcept.
  void filter(const StateVector& x, const ControlVector& u_nom, double mu,
              ControlVector& u_safe_out, FilterDiagnostics& diag_out) noexcept;

 private:
  VehicleParams vehicle_;
  BarrierConfig barriers_;
  CbfQpConfig cfg_;
  qp::QpSolverBackend* backend_{nullptr};
  qp::QpSolverBackend* fallback_{nullptr};
  DynamicBicycleModel model_;
  RoadBoundaryCbf road_;
  ObstacleCbf obstacle_;
  DriftEnvelopeCbf drift_;
  GapBarrierCbf gap_;
  ObstacleState obstacles_[kMaxFilterObstacles]{};
  int num_obstacles_{0};
  ControlVector u_prev_{ControlVector::Zero()};
};

}  // namespace av_safety
