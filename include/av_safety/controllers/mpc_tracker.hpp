#pragma once
/// @file mpc_tracker.hpp
/// @brief Receding-horizon linear MPC lateral tracker + longitudinal PI.
///
/// Drop-in complement to LqrTracker (same update() signature): toggle nominal
/// controllers via --controller {lqr, mpc} in the demo.
///
/// Model (spec coordinates e = [e_y, edot_y, e_psi, edot_psi]), obtained from
/// the LQR lateral linearization (A_l, B_l) by the change of coordinates
///   T = [[1,0,0,0],[0,vx,1,0],[0,1,0,0],[0,0,0,1]]  (edot_y = vx e_psi + vy)
/// as A_m = T A_l T^-1, B_m = T B_l, with curvature disturbance
///   E_c = [0, vx A_l[2,3], 0, vx A_l[3,3]]'  (exact: r = edot_psi + vx kappa).
/// Euler-discretized at dt (default 0.05 s), condensed over N = 15 steps:
///   min  sum_{k<N} (e_k'Q e_k + R d_k^2 + Rd (d_k - d_{k-1})^2) + e_N'P e_N
///   s.t. e_{k+1} = Ad e_k + Bd d_k + Ed kappa_k,  |d_k| <= steer_max.
/// e_N weight P solves the discrete ARE at init. Curvature preview defaults
/// to the scalar kappa (constant over the horizon); setCurvaturePreview()
/// latches a varying preview. Solved by MpcOsqpSolver (warm-started,
/// preallocated). Without HAVE_OSQP the tracker falls back to an LQR law from
/// the same linearization (same PI longitudinally), so call sites are safe.

#include "av_safety/common/types.hpp"
#include "av_safety/common/vehicle_params.hpp"
#include "av_safety/qp/mpc_osqp_solver.hpp"

#include <string>

namespace av_safety {

using MpcState = Eigen::Matrix<double, kLateralDim, 1>;
using MpcStateMatrix = Eigen::Matrix<double, kLateralDim, kLateralDim>;
using MpcInputMatrix = Eigen::Matrix<double, kLateralDim, 1>;
using MpcHorizonVec = Eigen::Matrix<double, qp::kMpcHorizon, 1>;
using MpcHorizonMat = Eigen::Matrix<double, qp::kMpcHorizon, qp::kMpcHorizon>;
using MpcPredMat = Eigen::Matrix<double, kLateralDim * (qp::kMpcHorizon + 1), qp::kMpcHorizon>;
using MpcFreeMat = Eigen::Matrix<double, kLateralDim * (qp::kMpcHorizon + 1), kLateralDim>;

struct MpcConfig {
  MpcStateMatrix Q{MpcStateMatrix::Identity()};
  double R{1.0};
  double R_delta{0.5};
  double dt{0.05};  // model step [s] (control may run faster)
  double target_speed{20.0};
  double kp_lon{0.8};
  double ki_lon{0.15};
  double integrator_max{2.0};
  double max_ey_clamp{5.0};
  double max_epsi_clamp{0.6};
  double rebuild_dvx{0.5};  // re-condense when |vx - vx_build| exceeds this
};

/// Load tuning from YAML (throws on invalid). Q built from Q_diag vector.
MpcConfig LoadMpcConfig(const std::string& yaml_path);

class MpcTracker {
 public:
  MpcTracker(const VehicleParams& vehicle, const MpcConfig& cfg) noexcept;
  ~MpcTracker();

  /// Linearize at (vx0, mu), condense the QP, solve the terminal DARE and
  /// cache an LQR fallback gain. Init-time only (may allocate internally).
  bool computeGains(double vx0, double mu);

  /// Hot loop: LQR-compatible signature (err = [e_y, e_psi, v_y, r]).
  /// No allocations, noexcept. Solves the condensed QP when the OSQP backend
  /// is ready, else the cached LQR fallback law.
  void update(const LateralState& err, double vx, double kappa, double vx_ref, double dt,
              ControlVector& u_out) noexcept;

  /// Latch a curvature preview (copied, up to N entries; tail held constant).
  void setCurvaturePreview(const double* kappa, int n) noexcept;
  void clearCurvaturePreview() noexcept { use_preview_ = false; }

  void resetIntegrator() noexcept { speed_int_ = 0.0; }
  /// Full reset: integrator, previous-steer memory, preview latch.
  void reset() noexcept;

  // -- Accessors -------------------------------------------------------------
  bool gainsComputed() const noexcept { return gains_computed_; }
  bool mpcReady() const noexcept;
  double lastVx0() const noexcept { return vx0_; }
  double lastMu() const noexcept { return mu_; }
  const MpcStateMatrix& lastAd() const noexcept { return Ad_; }
  const MpcInputMatrix& lastBd() const noexcept { return Bd_; }
  const MpcInputMatrix& lastEd() const noexcept { return Ed_; }
  const MpcStateMatrix& terminalP() const noexcept { return P_; }
  int lastIters() const noexcept { return last_iters_; }

 private:
  void rebuildCondensed() noexcept;  // Su/H/P from Ad/Bd/Ed (no allocs)
  void longitudinalPI(double vx, double vx_ref, double dt, double& ax_out) noexcept;

  VehicleParams vehicle_;
  MpcConfig cfg_;
  // Continuous/discrete spec-coordinate model + terminal weight.
  MpcStateMatrix Ac_{MpcStateMatrix::Zero()};
  MpcInputMatrix Bc_{MpcInputMatrix::Zero()};
  MpcInputMatrix Ec_{MpcInputMatrix::Zero()};
  MpcStateMatrix Ad_{MpcStateMatrix::Identity()};
  MpcInputMatrix Bd_{MpcInputMatrix::Zero()};
  MpcInputMatrix Ed_{MpcInputMatrix::Zero()};
  MpcStateMatrix P_{MpcStateMatrix::Identity()};
  // Condensed QP data (rebuilt when the linearization point moves).
  MpcPredMat Su_{MpcPredMat::Zero()};
  MpcFreeMat Sx_{MpcFreeMat::Zero()};
  MpcPredMat Sk_{MpcPredMat::Zero()};
  MpcHorizonMat H_{MpcHorizonMat::Identity()};
  // LQR fallback from the same (A_l, B_l).
  LateralGain K_lqr_{LateralGain::Zero()};
  bool has_lqr_fallback_{false};
  double vx0_{0.0}, mu_{0.9}, vx_build_{0.0};
  bool gains_computed_{false};
  bool use_preview_{false};
  double kappa_preview_[qp::kMpcHorizon]{0.0};
  double delta_prev_{0.0};
  double speed_int_{0.0};
  int last_iters_{0};
#ifdef HAVE_OSQP
  qp::MpcOsqpSolver* solver_{nullptr};  // owned, init-time allocation only
#endif
};

}  // namespace av_safety
