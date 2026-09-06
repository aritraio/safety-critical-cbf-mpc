#pragma once
/// @file dynamic_bicycle_model.hpp
/// @brief 6-state dynamic bicycle model with Pacejka lateral tires.
///
/// State x = [x, y, psi, vx, vy, r], input u = [delta, ax].
///
/// Equations of motion (two-track lumped into bicycle, factor 2 = left+right):
///   x_dot  = vx cos psi - vy sin psi
///   y_dot  = vx sin psi + vy cos psi
///   psi_dot = r
///   vx_dot = vy r + (2 Fxf cos d - 2 Fyf sin d + 2 Fxr)/m
///   vy_dot = -vx r + (2 Fxf sin d + 2 Fyf cos d + 2 Fyr)/m
///   r_dot  = (2 lf (Fxf sin d + Fyf cos d) - 2 lr Fyr)/Iz
///
/// Longitudinal allocation (RWD default): total Fx = m*ax split by
/// drive_bias_front bf: Fxf = bf*m*ax/2 (per front wheel), Fxr = (1-bf)*m*ax/2.
/// Lateral forces Fyf/Fyr come from the Pacejka model evaluated at the
/// current slip angles; mu scales the peak D.
///
/// Control-affine view for the Phase-3 CBF-QP:
///   x_dot = f(x; mu) + G(x, u_eval; mu) * u + O(delta^2 residual)
/// where G's steering column is the EXACT analytic partial d(xdot)/d(delta)
/// at the evaluation point and the ax column is exact (Fx is linear in ax).
/// f(x) is the drift at u = 0. Documented residual: steering enters Fyf
/// nonlinearly, so f + G*u reproduces xdot exactly at u_eval only to first
/// order in delta; this is the standard CBF-QP input-affine approximation and
/// its Jacobian is exact (no finite differences anywhere in the loop).

#include "av_safety/common/types.hpp"
#include "av_safety/common/vehicle_params.hpp"
#include "av_safety/dynamics/pacejka_tire.hpp"

namespace av_safety {

class DynamicBicycleModel {
 public:
  explicit DynamicBicycleModel(const VehicleParams& params) noexcept;

  // -- Nonlinear dynamics (hot loop: noexcept, no allocations) ---------------

  /// xdot = f_full(x, u; mu). Output via reference; never allocates.
  void continuousDynamics(const StateVector& x, const ControlVector& u, double mu,
                          StateDerivative& xdot_out) const noexcept;

  /// Fixed-step RK4 integration: x_next = RK4(x, u, mu, dt).
  void stepRK4(const StateVector& x, const ControlVector& u, double mu, double dt,
               StateVector& x_next_out) const noexcept;

  /// Per-tire lateral forces at the current state (for logging/tests).
  void tireForces(const StateVector& x, double steer, double mu, double& fyf_out,
                  double& fyr_out) const noexcept;

  // -- Linearizations (init / slow-rate; may be called at operating point) ---

  /// Per-axle cornering stiffnesses [N/rad]: Caf = 2*C0_front, Car = 2*C0_rear.
  void axleCorneringStiffness(double mu, double& caf_out, double& car_out) const noexcept;

  /// 4-state lateral error model e = [e_y, e_psi, v_y, r] about straight
  /// running at vx0: e_dot = A* e + B*delta (+ E*kappa disturbance, excluded).
  /// A33 = -(Caf+Car)/(m vx), A34 = -(Caf lf-Car lr)/(m vx) - vx, ...
  void linearizedLateralMatrices(double vx0, double mu, LateralMatrix& A_out,
                                 LateralInputMatrix& B_out) const noexcept;

  /// Control-affine decomposition: f = dynamics(x, u=0), G = d(xdot)/du.
  /// G evaluated at (x, u_eval). Exact analytic partials (see header math).
  void affineDecomposition(const StateVector& x, const ControlVector& u_eval, double mu,
                           StateDerivative& f_out, InputMatrix& G_out) const noexcept;

  /// Analytic Jacobian J = df/dx of the drift vector f(x; mu) (6x6).
  /// Exact closed-form partials through the Pacejka slopes and slip-angle
  /// kinematics (see .cpp for the derivation). In the clamped low-speed
  /// region (vx < vx_epsilon) the subgradient d(tilde_vx)/d(vx) = 0 is used.
  /// Hot-loop safe: noexcept, no allocations.
  void driftJacobian(const StateVector& x, double mu, StateMatrix& J_out) const noexcept;

  const VehicleParams& params() const noexcept { return params_; }

 private:
  VehicleParams params_;
  PacejkaTire front_tire_;
  PacejkaTire rear_tire_;

  void longitudinalForces(double ax, double& fxf_out, double& fxr_out) const noexcept;
};

}  // namespace av_safety
