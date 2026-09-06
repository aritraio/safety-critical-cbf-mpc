#include "av_safety/estimation/friction_ukf.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace av_safety {
namespace estimation {

UkfConfig LoadUkfConfig(const std::string& yaml_path) {
  YAML::Node n;
  try {
    n = YAML::LoadFile(yaml_path);
  } catch (const std::exception& e) {
    throw std::runtime_error("friction_ukf: cannot load '" + yaml_path + "': " + e.what());
  }
  UkfConfig c;
  const auto vdm = n["van_der_merwe"];
  if (vdm) {
    c.alpha = vdm["alpha"].as<double>(c.alpha);
    c.beta = vdm["beta"].as<double>(c.beta);
    c.kappa = vdm["kappa"].as<double>(c.kappa);
  }
  if (!(c.alpha > 0.0 && c.alpha <= 1.0))
    throw std::runtime_error("friction_ukf: alpha must be in (0, 1]");
  c.mu_init = n["mu_init"].as<double>(c.mu_init);
  c.mu_min = n["mu_min"].as<double>(c.mu_min);
  c.mu_max = n["mu_max"].as<double>(c.mu_max);
  c.R_prior = n["R_prior"].as<double>(c.R_prior);
  if (!(c.mu_min > 0.0 && c.mu_max > c.mu_min && c.mu_init >= c.mu_min && c.mu_init <= c.mu_max))
    throw std::runtime_error("friction_ukf: inconsistent mu bounds/init");
  auto readDiag = [&](const char* key, UkfCov& M, bool positive) {
    const auto node = n[key];
    if (!node) return;
    const auto v = node.as<std::vector<double>>();
    if (v.size() != 4) throw std::runtime_error(std::string("friction_ukf: ") + key + " needs 4 entries");
    M.setZero();
    for (int i = 0; i < 4; ++i) {
      if (positive && !(v[static_cast<size_t>(i)] > 0.0))
        throw std::runtime_error(std::string("friction_ukf: ") + key + " must be > 0");
      if (!std::isfinite(v[static_cast<size_t>(i)]))
        throw std::runtime_error(std::string("friction_ukf: ") + key + " non-finite");
      M(i, i) = v[static_cast<size_t>(i)];
    }
  };
  readDiag("Q_diag", c.Q, false);
  c.Q = 0.5 * (c.Q + c.Q.transpose());
  readDiag("R_diag", c.R, true);
  readDiag("P0_diag", c.P0, true);
  return c;
}

FrictionUkf::FrictionUkf(const VehicleParams& vehicle, const UkfConfig& cfg) noexcept
    : vehicle_(vehicle),
      cfg_(cfg),
      front_tire_(vehicle.pacejka, vehicle.mu_nominal),
      rear_tire_(vehicle.pacejka, vehicle.mu_nominal) {
  const double n = static_cast<double>(kUkfStateDim);
  const double lambda = cfg_.alpha * cfg_.alpha * (n + cfg_.kappa) - n;
  const double nlambda = n + lambda;
  wm0_ = lambda / nlambda;
  wc0_ = wm0_ + (1.0 - cfg_.alpha * cfg_.alpha + cfg_.beta);
  wi_ = 1.0 / (2.0 * nlambda);
  gamma_ = std::sqrt(nlambda > 1e-12 ? nlambda : 1e-12);
  reset();
}

void FrictionUkf::reset() noexcept {
  x_.setZero();
  x_(kUvx) = vehicle_.operating_speed;
  x_(kUmu) = cfg_.mu_init;
  P_ = cfg_.P0;
}

void FrictionUkf::processDerivative(const UkfState& xs, const ControlVector& u,
                                    UkfState& xd_out) const noexcept {
  const double vx = xs(kUvx), vy = xs(kUvy), r = xs(kUr);
  double mu = xs(kUmu);
  if (mu < 0.05) mu = 0.05;  // sigma guard: negative mu flips force signs
  if (mu > 1.5) mu = 1.5;
  const double delta = u(kSteer), ax = u(kAx);
  const double af = PacejkaTire::frontSlipAngle(delta, vx, vy, r, vehicle_.lf, vehicle_.vx_epsilon);
  const double ar = PacejkaTire::rearSlipAngle(vx, vy, r, vehicle_.lr, vehicle_.vx_epsilon);
  const double fyf = front_tire_.lateralForce(af, mu);
  const double fyr = rear_tire_.lateralForce(ar, mu);
  const double m = vehicle_.mass, iz = vehicle_.yaw_inertia;
  const double cos_d = std::cos(delta), sin_d = std::sin(delta);
  xd_out(kUvx) = ax + vy * r - 2.0 * fyf * sin_d / m;
  xd_out(kUvy) = -vx * r + (2.0 * fyf * cos_d + 2.0 * fyr) / m;
  xd_out(kUr) = (2.0 * vehicle_.lf * fyf * cos_d - 2.0 * vehicle_.lr * fyr) / iz;
  xd_out(kUmu) = 0.0;
}

UkfMeas FrictionUkf::measurementModel(const UkfState& xs, const ControlVector& u) const
    noexcept {
  const double vx = xs(kUvx), vy = xs(kUvy), r = xs(kUr);
  double mu = xs(kUmu);
  if (mu < 0.05) mu = 0.05;
  if (mu > 1.5) mu = 1.5;
  const double delta = u(kSteer), ax = u(kAx);
  const double af = PacejkaTire::frontSlipAngle(delta, vx, vy, r, vehicle_.lf, vehicle_.vx_epsilon);
  const double ar = PacejkaTire::rearSlipAngle(vx, vy, r, vehicle_.lr, vehicle_.vx_epsilon);
  const double fyf = front_tire_.lateralForce(af, mu);
  const double fyr = rear_tire_.lateralForce(ar, mu);
  const double m = vehicle_.mass;
  const double cos_d = std::cos(delta);
  UkfMeas z_out;
  z_out(kMax) = ax - vy * r;
  z_out(kMay) = (2.0 * fyf * cos_d + 2.0 * fyr) / m - vx * r;
  z_out(kMr) = r;
  z_out(kMvx) = vx;
  return z_out;
}

void FrictionUkf::computeSigmaPoints(UkfSigmaMat& S) const noexcept {
  S.col(0) = x_;
  // Cholesky with progressive jitter fallback (keeps PSD under roundoff).
  Eigen::LLT<UkfCov> llt(P_);
  if (llt.info() != Eigen::Success) {
    UkfCov Pj = P_;
    for (int j = 0; j < 4; ++j) Pj(j, j) += 1e-9;
    llt.compute(Pj);
    if (llt.info() != Eigen::Success) {
      for (int j = 0; j < 4; ++j) Pj(j, j) += 1e-6;
      llt.compute(Pj);
    }
  }
  const UkfCov L =
      (llt.info() == Eigen::Success) ? UkfCov(llt.matrixL()) : UkfCov::Identity() * 1e-3;
  for (int i = 0; i < kUkfStateDim; ++i) {
    S.col(1 + i) = x_ + gamma_ * L.col(i);
    S.col(1 + kUkfStateDim + i) = x_ - gamma_ * L.col(i);
  }
}

void FrictionUkf::predict(const ControlVector& u, double dt) noexcept {
  if (!(dt > 0.0)) return;
  UkfSigmaMat S;
  computeSigmaPoints(S);
  UkfSigmaMat Sp;
  for (int i = 0; i < kUkfSigmaCount; ++i) {
    UkfState xd;
    processDerivative(S.col(i), u, xd);
    Sp.col(i) = S.col(i) + dt * xd;
    if (Sp(kUmu, i) < 0.05) Sp(kUmu, i) = 0.05;
    if (Sp(kUmu, i) > 1.5) Sp(kUmu, i) = 1.5;
  }
  x_.noalias() = wm0_ * Sp.col(0);
  for (int i = 1; i < kUkfSigmaCount; ++i) x_.noalias() += wi_ * Sp.col(i);
  P_.setZero();
  UkfState d0 = Sp.col(0) - x_;
  P_.noalias() += wc0_ * d0 * d0.transpose();
  for (int i = 1; i < kUkfSigmaCount; ++i) {
    UkfState di = Sp.col(i) - x_;
    P_.noalias() += wi_ * di * di.transpose();
  }
  P_.noalias() += cfg_.Q;
  P_ = 0.5 * (P_ + P_.transpose());  // symmetrize
}

void FrictionUkf::update(const UkfMeas& z, const ControlVector& u) noexcept {
  UkfSigmaMat S;
  computeSigmaPoints(S);
  UkfMeasSigmaMat Z;
  for (int i = 0; i < kUkfSigmaCount; ++i) Z.col(i) = measurementModel(S.col(i), u);
  UkfMeas zhat = wm0_ * Z.col(0);
  for (int i = 1; i < kUkfSigmaCount; ++i) zhat.noalias() += wi_ * Z.col(i);

  Eigen::Matrix<double, kUkfMeasDim, kUkfMeasDim> Pzz = cfg_.R;
  Eigen::Matrix<double, kUkfStateDim, kUkfMeasDim> Pxz;
  Pxz.setZero();
  {
    UkfMeas e0 = Z.col(0) - zhat;
    Pzz.noalias() += wc0_ * e0 * e0.transpose();
    UkfState d0 = S.col(0) - x_;
    Pxz.noalias() += wc0_ * d0 * e0.transpose();
  }
  for (int i = 1; i < kUkfSigmaCount; ++i) {
    UkfMeas ei = Z.col(i) - zhat;
    Pzz.noalias() += wi_ * ei * ei.transpose();
    UkfState di = S.col(i) - x_;
    Pxz.noalias() += wi_ * di * ei.transpose();
  }
  Eigen::LLT<Eigen::Matrix<double, kUkfMeasDim, kUkfMeasDim>> llt(Pzz);
  if (llt.info() != Eigen::Success) return;  // keep prediction on numerical failure
  const Eigen::Matrix<double, kUkfStateDim, kUkfMeasDim> K =
      Pxz * llt.solve(Eigen::Matrix<double, kUkfMeasDim, kUkfMeasDim>::Identity());
  x_.noalias() += K * (z - zhat);
  P_.noalias() -= K * Pzz * K.transpose();  // Joseph-lite (Pzz includes R)
  P_ = 0.5 * (P_ + P_.transpose());
  // Mean-reverting prior (pseudo-measurement mu ~= mu_nominal, H = e_mu^T).
  // P-consistent Kalman sub-step; see header for why straight cruise needs it.
  {
    const double Rpr = cfg_.R_prior;
    if (Rpr > 0.0 && std::isfinite(Rpr)) {
      const double S = P_(kUmu, kUmu) + Rpr;
      if (S > 1e-12) {
        const Eigen::Matrix<double, kUkfStateDim, 1> Kc = P_.col(kUmu) / S;
        x_.noalias() += Kc * (vehicle_.mu_nominal - x_(kUmu));
        P_.noalias() -= Kc * P_.row(kUmu);
        P_ = 0.5 * (P_ + P_.transpose());
      }
    }
  }
  if (x_(kUmu) < cfg_.mu_min) x_(kUmu) = cfg_.mu_min;
  if (x_(kUmu) > cfg_.mu_max) x_(kUmu) = cfg_.mu_max;
}

void FrictionUkf::step(const ControlVector& u, const UkfMeas& z, double dt) noexcept {
  predict(u, dt);
  update(z, u);
}

}  // namespace estimation
}  // namespace av_safety
