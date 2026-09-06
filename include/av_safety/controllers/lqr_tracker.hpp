#pragma once
/// @file lqr_tracker.hpp
/// @brief Lateral LQR + longitudinal PI path tracker (nominal controller).
///
/// The CBF-QP safety filter (Phase 3) minimally perturbs THIS controller's
/// output `u_nom`, so its interface is frozen now: `update()` is noexcept,
/// allocation-free, and safe at 100 Hz.
///
/// Lateral law (error state e = [e_y, e_psi, v_y, r]):
///   delta = clamp(-K e + delta_ff(kappa), +-steer_max)
///   delta_ff = (L + Kus vx^2) kappa,  Kus = m(lr/Caf - lf/Car)/L  (understeer)
/// Longitudinal law (PI on speed):
///   ax = clamp(kp(v_ref - vx) + ki*I, [ax_min, ax_max]) with anti-windup.
///
/// Gain K solves the continuous ARE A'P + PA - PBR^-1B'P + Q = 0 via the
/// Hamiltonian stable-subspace method at init time (allocations OK there).

#include "av_safety/common/types.hpp"
#include "av_safety/common/vehicle_params.hpp"

namespace av_safety {

struct LqrConfig {
  LateralMatrix Q{LateralMatrix::Identity()};
  double R{1.0};
  double target_speed{20.0};  // [m/s]
  double kp_lon{0.8};
  double ki_lon{0.15};
  double integrator_max{2.0};
  bool use_feedforward{true};
  double max_ey_clamp{5.0};
  double max_epsi_clamp{0.6};
};

/// Load tuning from YAML (throws on invalid). Q built from Q_diag vector.
LqrConfig LoadLqrConfig(const std::string& yaml_path);

class LqrTracker {
 public:
  LqrTracker(const VehicleParams& vehicle, const LqrConfig& cfg) noexcept;

  /// Solve CARE at (vx0, mu) and cache A, B, K. Returns false if the
  /// Hamiltonian decomposition fails or the loop is not stabilizable.
  /// Init-time only (may allocate inside Eigen's eigensolver).
  bool computeGains(double vx0, double mu);

  /// Hot loop: map tracking error -> nominal control. No allocations, noexcept.
  /// @param err      [e_y, e_psi, v_y, r] (r absolute; reference folded into ff)
  /// @param vx       current longitudinal speed [m/s]
  /// @param kappa    reference path curvature [1/m] at closest point
  /// @param vx_ref   reference speed [m/s]
  /// @param dt       controller dt [s] (integrator + anti-windup)
  /// @param u_out    [delta, ax] nominal command
  void update(const LateralState& err, double vx, double kappa, double vx_ref, double dt,
              ControlVector& u_out) noexcept;

  void resetIntegrator() noexcept { speed_int_ = 0.0; }

  // -- Accessors -------------------------------------------------------------
  const LateralGain& gain() const noexcept { return K_; }
  bool gainsComputed() const noexcept { return gains_computed_; }
  const LateralMatrix& lastA() const noexcept { return A_; }
  const LateralInputMatrix& lastB() const noexcept { return B_; }
  double lastVx0() const noexcept { return vx0_; }
  double lastMu() const noexcept { return mu_; }

  /// Solve continuous ARE. Static so tests can call it without a tracker.
  /// Returns false on numerical failure.
  static bool solveContinuousCARE(const LateralMatrix& A, const LateralInputMatrix& B,
                                  const LateralMatrix& Q, double R, LateralMatrix& P_out,
                                  LateralGain& K_out);

  /// Max real part of eig(A - B*K). Negative <=> asymptotically stable.
  static double closedLoopMaxRealPart(const LateralMatrix& A, const LateralInputMatrix& B,
                                      const LateralGain& K);

 private:
  VehicleParams vehicle_;
  LqrConfig cfg_;
  LateralGain K_{LateralGain::Zero()};
  LateralMatrix A_{LateralMatrix::Zero()};
  LateralInputMatrix B_{LateralInputMatrix::Zero()};
  double vx0_{0.0};
  double mu_{0.9};
  bool gains_computed_{false};
  double speed_int_{0.0};
};

}  // namespace av_safety
