#include "av_safety/controllers/lqr_tracker.hpp"
#include "av_safety/dynamics/dynamic_bicycle_model.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

namespace av_safety {

LqrConfig LoadLqrConfig(const std::string& yaml_path) {
  YAML::Node n;
  try {
    n = YAML::LoadFile(yaml_path);
  } catch (const std::exception& e) {
    throw std::runtime_error("lqr_tuning: cannot load '" + yaml_path + "': " + e.what());
  }
  LqrConfig c;
  if (!n["Q_diag"]) throw std::runtime_error("lqr_tuning: missing 'Q_diag'");
  const auto qd = n["Q_diag"].as<std::vector<double>>();
  if (qd.size() != 4) throw std::runtime_error("lqr_tuning: Q_diag must have 4 entries");
  c.Q.setZero();
  for (int i = 0; i < 4; ++i) {
    if (!(qd[static_cast<size_t>(i)] > 0.0))
      throw std::runtime_error("lqr_tuning: Q_diag entries must be > 0");
    c.Q(i, i) = qd[static_cast<size_t>(i)];
  }
  if (!n["R"]) throw std::runtime_error("lqr_tuning: missing 'R'");
  c.R = n["R"].as<double>();
  if (!(c.R > 0.0)) throw std::runtime_error("lqr_tuning: R must be > 0");

  const auto lon = n["longitudinal"];
  if (!lon) throw std::runtime_error("lqr_tuning: missing 'longitudinal'");
  c.kp_lon = lon["kp"].as<double>(0.8);
  c.ki_lon = lon["ki"].as<double>(0.15);
  c.integrator_max = lon["integrator_max"].as<double>(2.0);
  if (c.integrator_max <= 0.0) throw std::runtime_error("lqr_tuning: integrator_max > 0");

  c.target_speed = n["target_speed"].as<double>(20.0);
  c.use_feedforward = n["use_feedforward"].as<bool>(true);
  c.max_ey_clamp = n["max_lateral_error_clamp"].as<double>(5.0);
  c.max_epsi_clamp = n["max_heading_error_clamp"].as<double>(0.6);
  return c;
}

LqrTracker::LqrTracker(const VehicleParams& vehicle, const LqrConfig& cfg) noexcept
    : vehicle_(vehicle), cfg_(cfg) {}

bool LqrTracker::solveContinuousCARE(const LateralMatrix& A, const LateralInputMatrix& B,
                                     const LateralMatrix& Q, double R, LateralMatrix& P_out,
                                     LateralGain& K_out) {
  if (!(R > 0.0)) return false;
  constexpr int N = kLateralDim;
  constexpr int H2 = 2 * kLateralDim;
  using HamMatrix = Eigen::Matrix<double, H2, H2>;

  const double r_inv = 1.0 / R;
  HamMatrix H = HamMatrix::Zero();
  H.topLeftCorner(N, N) = A;
  H.topRightCorner(N, N).noalias() = -r_inv * (B * B.transpose());
  H.bottomLeftCorner(N, N) = -Q;
  H.bottomRightCorner(N, N) = -A.transpose();

  Eigen::ComplexEigenSolver<HamMatrix> es(H);
  if (es.info() != Eigen::Success) return false;

  // Collect stable eigenvectors (Re(lambda) < 0). Need exactly N of them.
  Eigen::Matrix<std::complex<double>, H2, kLateralDim> U;
  int col = 0;
  for (int i = 0; i < H2 && col < N; ++i) {
    if (es.eigenvalues()(i).real() < -1e-9) {
      U.col(col++) = es.eigenvectors().col(i);
    }
  }
  if (col != N) return false;  // not stabilizable / Hamiltonian defect

  const auto U1 = U.topRows(N);
  const auto U2 = U.bottomRows(N);
  // U1 must be invertible.
  Eigen::FullPivLU<Eigen::Matrix<std::complex<double>, N, N>> lu(U1);
  if (!lu.isInvertible()) return false;
  const Eigen::Matrix<std::complex<double>, N, N> P_complex =
      U2 * U1.inverse();
  LateralMatrix P = P_complex.real();
  // Symmetrize (kills ~1e-15 skew from complex arithmetic).
  P = 0.5 * (P + P.transpose());

  // PSD check.
  Eigen::LLT<LateralMatrix> llt(P);
  if (llt.info() != Eigen::Success) return false;

  // CARE residual check: ||A'P + PA - PBR^-1B'P + Q||_F small relative to ||Q||.
  LateralMatrix residual = A.transpose() * P + P * A;
  residual.noalias() -= r_inv * (P * B * (B.transpose() * P));
  residual += Q;
  const double denom = Q.norm() > 0.0 ? Q.norm() : 1.0;
  if (!(residual.norm() / denom < 1e-6)) return false;

  P_out = P;
  K_out.noalias() = r_inv * (B.transpose() * P);
  return true;
}

double LqrTracker::closedLoopMaxRealPart(const LateralMatrix& A, const LateralInputMatrix& B,
                                         const LateralGain& K) {
  LateralMatrix acl = A;
  acl.noalias() -= B * K;
  Eigen::EigenSolver<LateralMatrix> es(acl);
  double worst = -1e100;
  for (int i = 0; i < kLateralDim; ++i) worst = std::max(worst, es.eigenvalues()(i).real());
  return worst;
}

bool LqrTracker::computeGains(double vx0, double mu) {
  if (vx0 < 1.0) vx0 = 1.0;  // keep the linearization well-conditioned
  if (mu <= 0.0) return false;
  DynamicBicycleModel model(vehicle_);
  model.linearizedLateralMatrices(vx0, mu, A_, B_);
  LateralMatrix P;
  if (!solveContinuousCARE(A_, B_, cfg_.Q, cfg_.R, P, K_)) {
    gains_computed_ = false;
    return false;
  }
  vx0_ = vx0;
  mu_ = mu;
  gains_computed_ = true;
  speed_int_ = 0.0;
  return true;
}

void LqrTracker::update(const LateralState& err, double vx, double kappa, double vx_ref,
                        double dt, ControlVector& u_out) noexcept {
  // Saturate incoming errors (robustness against estimator jumps).
  double ey = err(kEy);
  double epsi = err(kEpsi);
  ey = std::clamp(ey, -cfg_.max_ey_clamp, cfg_.max_ey_clamp);
  epsi = std::clamp(epsi, -cfg_.max_epsi_clamp, cfg_.max_epsi_clamp);
  LateralState e;
  e << ey, epsi, err(kLatVy), err(kLatR);

  // Lateral: state feedback + steady-state cornering feedforward.
  double delta = 0.0;
  if (gains_computed_) {
    delta = -(K_ * e)(0, 0);
  }
  if (cfg_.use_feedforward) {
    DynamicBicycleModel model(vehicle_);  // cheap: two Pacejka param copies
    double caf = 0.0, car = 0.0;
    model.axleCorneringStiffness(mu_, caf, car);
    const double L = vehicle_.lf + vehicle_.lr;
    double kus = 0.0;
    if (caf > 1.0 && car > 1.0 && L > 0.0) {
      kus = vehicle_.mass * (vehicle_.lr / caf - vehicle_.lf / car) / L;
    }
    const double v = vx >= 0.0 ? vx : 0.0;
    const double delta_ff = (L + kus * v * v) * kappa;
    delta += delta_ff;
  }
  delta = std::clamp(delta, -vehicle_.steer_max, vehicle_.steer_max);

  // Longitudinal: PI on speed with anti-windup (integrator frozen on saturation).
  double ax_unsat = 0.0;
  {
    const double e_v = vx_ref - vx;
    const double p_term = cfg_.kp_lon * e_v;
    double int_new = speed_int_ + e_v * dt;
    int_new = std::clamp(int_new, -cfg_.integrator_max / (std::abs(cfg_.ki_lon) + 1e-9),
                         cfg_.integrator_max / (std::abs(cfg_.ki_lon) + 1e-9));
    const double i_term = cfg_.ki_lon * int_new;
    ax_unsat = p_term + i_term;
    const double ax_sat = std::clamp(ax_unsat, vehicle_.accel_min, vehicle_.accel_max);
    // Conditional integration: accept integrator update only if not worsening saturation.
    if (ax_sat == ax_unsat || (ax_sat > ax_unsat && e_v < 0.0) || (ax_sat < ax_unsat && e_v > 0.0)) {
      speed_int_ = int_new;
    }
    ax_unsat = ax_sat;
  }

  u_out(kSteer) = delta;
  u_out(kAx) = ax_unsat;
}

}  // namespace av_safety
