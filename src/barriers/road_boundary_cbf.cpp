#include "av_safety/barriers/road_boundary_cbf.hpp"

#include <cmath>

namespace av_safety {

RoadBoundaryCbf::RoadBoundaryCbf(const RoadBarrierParams& p) noexcept : params_(p) {}

double RoadBoundaryCbf::value(const StateVector& x) const noexcept {
  const double e = x(kPy) - params_.lane_center_y;
  return params_.lane_margin * params_.lane_margin - e * e;
}

double RoadBoundaryCbf::hDot(const StateVector& x, const StateDerivative& f) const noexcept {
  const double gy = -2.0 * (x(kPy) - params_.lane_center_y);
  return gy * f(kPy);
}

void RoadBoundaryCbf::linearize(const StateVector& x, const StateDerivative& f,
                                const StateMatrix& J, const InputMatrix& G,
                                EcbffConstraint& out) const noexcept {
  const double e = x(kPy) - params_.lane_center_y;
  const double gy = -2.0 * e;
  const double sin_psi = std::sin(x(kPsi));
  const double cos_psi = std::cos(x(kPsi));

  // dEta = f'H + dh'J = [0, -2 f1, gy*J(1,2), gy*sinP, gy*cosP, 0].
  const double eta0 = 0.0;
  const double eta1 = -2.0 * f(kPy);
  const double eta2 = gy * J(kPy, kPsi);
  const double eta3 = gy * sin_psi;
  const double eta4 = gy * cos_psi;
  constexpr double eta5 = 0.0;

  const double lf2 = eta1 * f(kPy) + eta2 * f(kPsi) + eta3 * f(kVx) + eta4 * f(kVy);
  // (eta0/eta5 rows of G vanish; psi row of G vanishes.)

  out.h = params_.lane_margin * params_.lane_margin - e * e;
  out.hdot = gy * f(kPy);
  out.Lf2 = lf2;
  for (int j = 0; j < kControlDim; ++j) {
    out.LgLf[j] = eta3 * G(kVx, j) + eta4 * G(kVy, j);
  }
  out.p1 = params_.kappa;
  out.p2 = params_.kappa;
  (void)eta0;
  (void)eta5;
}

}  // namespace av_safety
