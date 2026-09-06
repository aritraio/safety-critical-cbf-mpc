#include "av_safety/barriers/obstacle_cbf.hpp"

#include <cmath>

namespace av_safety {

ObstacleCbf::ObstacleCbf(const ObstacleBarrierParams& p) noexcept : params_(p) {}

double ObstacleCbf::value(const StateVector& x, const ObstacleState& obs) const noexcept {
  const double dx = (x(kPx) - obs.x) / obs.a;
  const double dy = (x(kPy) - obs.y) / obs.b;
  return dx * dx + dy * dy - 1.0;
}

double ObstacleCbf::hDot(const StateVector& x, const StateDerivative& f,
                         const ObstacleState& obs) const noexcept {
  const double gx = 2.0 * (x(kPx) - obs.x) / (obs.a * obs.a);
  const double gy = 2.0 * (x(kPy) - obs.y) / (obs.b * obs.b);
  return gx * (f(kPx) - obs.vx) + gy * (f(kPy) - obs.vy);
}

void ObstacleCbf::linearize(const StateVector& x, const ObstacleState& obs,
                            const StateDerivative& f, const StateMatrix& J, const InputMatrix& G,
                            EcbffConstraint& out) const noexcept {
  const double a2 = obs.a * obs.a;
  const double b2 = obs.b * obs.b;
  const double gx = 2.0 * (x(kPx) - obs.x) / a2;
  const double gy = 2.0 * (x(kPy) - obs.y) / b2;
  const double sin_psi = std::sin(x(kPsi));
  const double cos_psi = std::cos(x(kPsi));

  // dEta = spatial gradient of eta(x,t) = gx f0 + gy f1 + ht.
  // Note ht = -gx vxo - gy vyo also depends on x, so
  //   dEta_X = (2/a^2)(f0 - vxo) + gx J(0,0),  (J pose cols vanish)
  // and the explicit time part eta_t = d(eta)/dt|_x
  //          = -(2/a^2) vxo (f0 - vxo) - (2/b^2) vyo (f1 - vyo).
  // J pose rows: J(0,:) = [0,0,J02,cosP,-sinP,0], J(1,:) = [0,0,J12,sinP,cosP,0].
  const double eta0 = (2.0 / a2) * (f(kPx) - obs.vx);
  const double eta1 = (2.0 / b2) * (f(kPy) - obs.vy);
  const double eta2 = gx * J(kPx, kPsi) + gy * J(kPy, kPsi);
  const double eta3 = gx * cos_psi + gy * sin_psi;
  const double eta4 = -gx * sin_psi + gy * cos_psi;
  const double eta_t = -(2.0 / a2) * obs.vx * (f(kPx) - obs.vx) -
                       (2.0 / b2) * obs.vy * (f(kPy) - obs.vy);

  const double lf2 = eta0 * f(kPx) + eta1 * f(kPy) + eta2 * f(kPsi) + eta3 * f(kVx) +
                     eta4 * f(kVy) + eta_t;

  const double dx = (x(kPx) - obs.x) / obs.a;
  const double dy = (x(kPy) - obs.y) / obs.b;
  out.h = dx * dx + dy * dy - 1.0;
  out.hdot = gx * (f(kPx) - obs.vx) + gy * (f(kPy) - obs.vy);
  out.Lf2 = lf2;
  for (int j = 0; j < kControlDim; ++j) {
    out.LgLf[j] = eta3 * G(kVx, j) + eta4 * G(kVy, j);
  }
  out.p1 = params_.kappa;
  out.p2 = params_.kappa;
}

}  // namespace av_safety
