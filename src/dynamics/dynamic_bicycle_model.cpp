#include "av_safety/dynamics/dynamic_bicycle_model.hpp"

#include <cmath>

namespace av_safety {

DynamicBicycleModel::DynamicBicycleModel(const VehicleParams& params) noexcept
    : params_(params),
      front_tire_(params.pacejka, params.mu_nominal),
      rear_tire_(params.pacejka, params.mu_nominal) {}

void DynamicBicycleModel::longitudinalForces(double ax, double& fxf_out,
                                             double& fxr_out) const noexcept {
  const double bf = params_.drive_bias_front;
  // Total longitudinal force m*ax split across 2 wheels per axle in the
  // bicycle equations (hence the /2: 2*Fxf + 2*Fxr = m*ax at delta = 0).
  fxf_out = bf * params_.mass * ax * 0.5;
  fxr_out = (1.0 - bf) * params_.mass * ax * 0.5;
}

void DynamicBicycleModel::tireForces(const StateVector& x, double steer, double mu,
                                     double& fyf_out, double& fyr_out) const noexcept {
  const double vx = x(kVx), vy = x(kVy), r = x(kYawRate);
  const double af =
      PacejkaTire::frontSlipAngle(steer, vx, vy, r, params_.lf, params_.vx_epsilon);
  const double ar = PacejkaTire::rearSlipAngle(vx, vy, r, params_.lr, params_.vx_epsilon);
  fyf_out = front_tire_.lateralForce(af, mu);
  fyr_out = rear_tire_.lateralForce(ar, mu);
}

void DynamicBicycleModel::continuousDynamics(const StateVector& x, const ControlVector& u,
                                             double mu, StateDerivative& xdot_out) const noexcept {
  const double px_vel_x = x(kVx), vy = x(kVy), psi = x(kPsi), r = x(kYawRate);
  const double delta = u(kSteer), ax = u(kAx);

  double fxf = 0.0, fxr = 0.0;
  longitudinalForces(ax, fxf, fxr);

  double fyf = 0.0, fyr = 0.0;
  tireForces(x, delta, mu, fyf, fyr);

  const double cos_d = std::cos(delta), sin_d = std::sin(delta);
  const double m = params_.mass, iz = params_.yaw_inertia;
  const double lf = params_.lf, lr = params_.lr;

  const double fx_body = 2.0 * fxf * cos_d - 2.0 * fyf * sin_d + 2.0 * fxr;
  const double fy_body = 2.0 * fxf * sin_d + 2.0 * fyf * cos_d + 2.0 * fyr;
  const double mz = 2.0 * lf * (fxf * sin_d + fyf * cos_d) - 2.0 * lr * fyr;

  const double cos_psi = std::cos(psi), sin_psi = std::sin(psi);
  xdot_out(kPx) = px_vel_x * cos_psi - vy * sin_psi;
  xdot_out(kPy) = px_vel_x * sin_psi + vy * cos_psi;
  xdot_out(kPsi) = r;
  xdot_out(kVx) = vy * r + fx_body / m;
  xdot_out(kVy) = -px_vel_x * r + fy_body / m;
  xdot_out(kYawRate) = mz / iz;
}

void DynamicBicycleModel::stepRK4(const StateVector& x, const ControlVector& u, double mu,
                                  double dt, StateVector& x_next_out) const noexcept {
  // Stack-only temporaries: fixed-size Eigen objects, no heap.
  StateDerivative k1, k2, k3, k4;
  StateVector tmp;
  continuousDynamics(x, u, mu, k1);
  tmp.noalias() = x + (0.5 * dt) * k1;
  continuousDynamics(tmp, u, mu, k2);
  tmp.noalias() = x + (0.5 * dt) * k2;
  continuousDynamics(tmp, u, mu, k3);
  tmp.noalias() = x + dt * k3;
  continuousDynamics(tmp, u, mu, k4);
  x_next_out.noalias() = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

void DynamicBicycleModel::axleCorneringStiffness(double mu, double& caf_out,
                                                 double& car_out) const noexcept {
  caf_out = 2.0 * front_tire_.corneringStiffness(mu);
  car_out = 2.0 * rear_tire_.corneringStiffness(mu);
}

void DynamicBicycleModel::linearizedLateralMatrices(double vx0, double mu, LateralMatrix& A_out,
                                                    LateralInputMatrix& B_out) const noexcept {
  if (vx0 < params_.vx_epsilon) vx0 = params_.vx_epsilon;
  double caf = 0.0, car = 0.0;
  axleCorneringStiffness(mu, caf, car);
  const double m = params_.mass, iz = params_.yaw_inertia;
  const double lf = params_.lf, lr = params_.lr;

  const double a33 = -(caf + car) / (m * vx0);
  const double a34 = -(caf * lf - car * lr) / (m * vx0) - vx0;
  const double a43 = -(caf * lf - car * lr) / (iz * vx0);
  const double a44 = -(caf * lf * lf + car * lr * lr) / (iz * vx0);

  A_out.setZero();
  A_out(0, 1) = vx0;  // e_y_dot = v_y + vx * e_psi
  A_out(0, 2) = 1.0;
  A_out(1, 3) = 1.0;  // e_psi_dot = r (- vx*kappa treated as known disturbance)
  A_out(2, 2) = a33;
  A_out(2, 3) = a34;
  A_out(3, 2) = a43;
  A_out(3, 3) = a44;

  B_out.setZero();
  B_out(2, 0) = caf / m;
  B_out(3, 0) = caf * lf / iz;
}

void DynamicBicycleModel::driftJacobian(const StateVector& x, double mu,
                                          StateMatrix& J_out) const noexcept {
  // Drift f(x): u = 0 => Fxf = Fxr = 0, sin(delta) = 0, so
  //   f0 = vx cosP - vy sinP, f1 = vx sinP + vy cosP, f2 = r,
  //   f3 = vy r, f4 = -vx r + (2 Fyf0 + 2 Fyr0)/m, f5 = (2 lf Fyf0 - 2 lr Fyr0)/Iz
  // with Fyf0 = Pacejka(af0), af0 = -atan(xif), xif = (vy + lf r)/tvx (and rear).
  // Chain rule: dF/d(vx,vy,r) = C0 * d(alpha)/d(...), where C0 is the analytic
  // Pacejka slope at the evaluation slip angle and, for vx > eps,
  //   d(af0)/dvy = -wf/tvx, d(af0)/dr = -lf wf/tvx, d(af0)/dvx = +xif wf/tvx,
  //   d(ar)/dvy  = -wr/tvx, d(ar)/dr  = +lr wr/tvx, d(ar)/dvx  = +xir wr/tvx,
  // with wf = 1/(1+xif^2), tvx = max(vx, eps).
  const double vx = x(kVx), vy = x(kVy), psi = x(kPsi), r = x(kYawRate);
  const double eps = params_.vx_epsilon;
  const double tvx = vx >= eps ? vx : eps;
  const double low_speed = (vx >= eps) ? 1.0 : 0.0;

  const double xif = (vy + params_.lf * r) / tvx;
  const double xir = (vy - params_.lr * r) / tvx;
  const double wf = 1.0 / (1.0 + xif * xif);
  const double wr = 1.0 / (1.0 + xir * xir);

  const double af0 = -std::atan(xif);
  const double ar0 = -std::atan(xir);
  const double cf0 = front_tire_.lateralStiffness(af0, mu);
  const double cr0 = rear_tire_.lateralStiffness(ar0, mu);

  const double m = params_.mass, iz = params_.yaw_inertia;
  const double lf = params_.lf, lr = params_.lr;
  const double sin_psi = std::sin(psi), cos_psi = std::cos(psi);

  // Slip-angle partials.
  const double daf_dvy = -wf / tvx;
  const double daf_dr = -lf * wf / tvx;
  const double daf_dvx = low_speed * xif * wf / tvx;
  const double dar_dvy = -wr / tvx;
  const double dar_dr = lr * wr / tvx;
  const double dar_dvx = low_speed * xir * wr / tvx;

  J_out.setZero();
  // Pose kinematics rows.
  J_out(kPx, kPsi) = -vx * sin_psi - vy * cos_psi;
  J_out(kPx, kVx) = cos_psi;
  J_out(kPx, kVy) = -sin_psi;
  J_out(kPy, kPsi) = vx * cos_psi - vy * sin_psi;
  J_out(kPy, kVx) = sin_psi;
  J_out(kPy, kVy) = cos_psi;
  J_out(kPsi, kYawRate) = 1.0;
  // Longitudinal drift f3 = vy * r.
  J_out(kVx, kVy) = r;
  J_out(kVx, kYawRate) = vy;
  // Lateral drift row.
  J_out(kVy, kVx) = -r + (2.0 / m) * (cf0 * daf_dvx + cr0 * dar_dvx);
  J_out(kVy, kVy) = (2.0 / m) * (cf0 * daf_dvy + cr0 * dar_dvy);
  J_out(kVy, kYawRate) = -vx + (2.0 / m) * (cf0 * daf_dr + cr0 * dar_dr);
  // Yaw drift row.
  const double c1 = 2.0 * lf / iz, c2 = 2.0 * lr / iz;
  J_out(kYawRate, kVx) = c1 * cf0 * daf_dvx - c2 * cr0 * dar_dvx;
  J_out(kYawRate, kVy) = c1 * cf0 * daf_dvy - c2 * cr0 * dar_dvy;
  J_out(kYawRate, kYawRate) = c1 * cf0 * daf_dr - c2 * cr0 * dar_dr;
}

void DynamicBicycleModel::affineDecomposition(const StateVector& x, const ControlVector& u_eval,
                                              double mu, StateDerivative& f_out,
                                              InputMatrix& G_out) const noexcept {
  const ControlVector zero = ControlVector::Zero();
  continuousDynamics(x, zero, mu, f_out);

  const double delta = u_eval(kSteer);
  const double vx = x(kVx), vy = x(kVy), r = x(kYawRate);
  const double m = params_.mass, iz = params_.yaw_inertia;
  const double lf = params_.lf, bf = params_.drive_bias_front;

  double fxf = 0.0, fxr = 0.0;
  longitudinalForces(u_eval(kAx), fxf, fxr);

  const double af =
      PacejkaTire::frontSlipAngle(delta, vx, vy, r, params_.lf, params_.vx_epsilon);
  double fyf = front_tire_.lateralForce(af, mu);
  const double dfyf = front_tire_.lateralStiffness(af, mu);  // dFyf/d(delta)

  const double cos_d = std::cos(delta), sin_d = std::sin(delta);

  // Steering column: exact analytic partials d(xdot)/d(delta).
  const double dvx_dd = (-2.0 * dfyf * sin_d - 2.0 * fyf * cos_d - 2.0 * fxf * sin_d) / m;
  const double dvy_dd = (2.0 * fxf * cos_d + 2.0 * dfyf * cos_d - 2.0 * fyf * sin_d) / m;
  const double dr_dd =
      (2.0 * lf * (fxf * cos_d + dfyf * cos_d - fyf * sin_d)) / iz;

  // Longitudinal column: exact (Fx linear in ax).
  const double dvx_da = bf * cos_d + (1.0 - bf);
  const double dvy_da = bf * sin_d;
  const double dr_da = lf * bf * m * sin_d / iz;

  G_out.setZero();
  G_out(kVx, kSteer) = dvx_dd;
  G_out(kVy, kSteer) = dvy_dd;
  G_out(kYawRate, kSteer) = dr_dd;
  G_out(kVx, kAx) = dvx_da;
  G_out(kVy, kAx) = dvy_da;
  G_out(kYawRate, kAx) = dr_da;
  // Pose rows (x, y, psi) do not depend directly on u.
}

}  // namespace av_safety
