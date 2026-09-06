#include "av_safety/controllers/mpc_tracker.hpp"
#include "av_safety/controllers/lqr_tracker.hpp"
#include "av_safety/dynamics/dynamic_bicycle_model.hpp"

#include <algorithm>
#include <cmath>
#include <new>
#include <stdexcept>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace av_safety {

MpcConfig LoadMpcConfig(const std::string& yaml_path) {
  YAML::Node n;
  try {
    n = YAML::LoadFile(yaml_path);
  } catch (const std::exception& e) {
    throw std::runtime_error("mpc_tuning: cannot load '" + yaml_path + "': " + e.what());
  }
  MpcConfig c;
  if (!n["Q_diag"]) throw std::runtime_error("mpc_tuning: missing 'Q_diag'");
  const auto qd = n["Q_diag"].as<std::vector<double>>();
  if (qd.size() != 4) throw std::runtime_error("mpc_tuning: Q_diag must have 4 entries");
  c.Q.setZero();
  for (int i = 0; i < 4; ++i) {
    if (!(qd[static_cast<size_t>(i)] > 0.0))
      throw std::runtime_error("mpc_tuning: Q_diag entries must be > 0");
    c.Q(i, i) = qd[static_cast<size_t>(i)];
  }
  if (!n["R"]) throw std::runtime_error("mpc_tuning: missing 'R'");
  c.R = n["R"].as<double>();
  if (!(c.R > 0.0)) throw std::runtime_error("mpc_tuning: R must be > 0");
  c.R_delta = n["R_delta"].as<double>(c.R_delta);
  if (!(c.R_delta >= 0.0)) throw std::runtime_error("mpc_tuning: R_delta must be >= 0");
  c.dt = n["dt"].as<double>(c.dt);
  if (!(c.dt > 0.0 && c.dt <= 0.25)) throw std::runtime_error("mpc_tuning: dt must be in (0, 0.25]");
  const auto lon = n["longitudinal"];
  if (!lon) throw std::runtime_error("mpc_tuning: missing 'longitudinal'");
  c.kp_lon = lon["kp"].as<double>(0.8);
  c.ki_lon = lon["ki"].as<double>(0.15);
  c.integrator_max = lon["integrator_max"].as<double>(2.0);
  if (c.integrator_max <= 0.0) throw std::runtime_error("mpc_tuning: integrator_max > 0");
  c.target_speed = n["target_speed"].as<double>(20.0);
  c.max_ey_clamp = n["max_lateral_error_clamp"].as<double>(5.0);
  c.max_epsi_clamp = n["max_heading_error_clamp"].as<double>(0.6);
  c.rebuild_dvx = n["rebuild_dvx"].as<double>(0.5);
  if (!(c.rebuild_dvx > 0.0)) throw std::runtime_error("mpc_tuning: rebuild_dvx > 0");
  return c;
}

MpcTracker::MpcTracker(const VehicleParams& vehicle, const MpcConfig& cfg) noexcept
    : vehicle_(vehicle), cfg_(cfg) {
#ifdef HAVE_OSQP
  solver_ = new (std::nothrow)
      qp::MpcOsqpSolver(-vehicle_.steer_max, vehicle_.steer_max);
#endif
}

MpcTracker::~MpcTracker() {
#ifdef HAVE_OSQP
  delete solver_;
  solver_ = nullptr;
#endif
}

namespace {
// Discrete ARE by value iteration (init-time only). B is a column vector, so
// the inverse is scalar; no factorizations needed.
bool SolveDiscreteARE(const MpcStateMatrix& Ad, const MpcInputMatrix& Bd,
                      const MpcStateMatrix& Q, double R, MpcStateMatrix& P_out) {
  MpcStateMatrix P = Q;
  for (int it = 0; it < 500; ++it) {
    const MpcStateMatrix AtP = Ad.transpose() * P;
    const double denom = R + (Bd.transpose() * P * Bd)(0, 0);
    if (!(denom > 0.0)) return false;
    const MpcStateMatrix Pn = Q + AtP * Ad - (AtP * Bd) * (Bd.transpose() * P * Ad) / denom;
    double diff = 0.0;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) diff = std::max(diff, std::abs(Pn(i, j) - P(i, j)));
    P = 0.5 * (Pn + Pn.transpose());
    if (diff < 1e-12) break;
    if (!P.allFinite()) return false;
  }
  // PSD + residual sanity.
  Eigen::LLT<MpcStateMatrix> llt(P);
  if (llt.info() != Eigen::Success) return false;
  const MpcStateMatrix AtP = Ad.transpose() * P;
  const double denom = R + (Bd.transpose() * P * Bd)(0, 0);
  const MpcStateMatrix res =
      Q + AtP * Ad - (AtP * Bd) * (Bd.transpose() * P * Ad) / denom - P;
  if (res.norm() / (Q.norm() > 0.0 ? Q.norm() : 1.0) > 1e-6) return false;
  P_out = P;
  return true;
}
}  // namespace

bool MpcTracker::computeGains(double vx0, double mu) {
  if (vx0 < 1.0) vx0 = 1.0;
  if (mu <= 0.0) return false;
  DynamicBicycleModel model(vehicle_);
  LateralMatrix Al;
  LateralInputMatrix Bl;
  model.linearizedLateralMatrices(vx0, mu, Al, Bl);

  // Spec coordinates via T (edot_y = vx e_psi + vy, edot_psi state).
  MpcStateMatrix T = MpcStateMatrix::Zero();
  T(0, 0) = 1.0;
  T(1, 1) = vx0;
  T(1, 2) = 1.0;
  T(2, 1) = 1.0;
  T(3, 3) = 1.0;
  MpcStateMatrix Tinv = MpcStateMatrix::Zero();
  Tinv(0, 0) = 1.0;
  Tinv(1, 2) = 1.0;
  Tinv(2, 1) = 1.0;
  Tinv(2, 2) = -vx0;
  Tinv(3, 3) = 1.0;
  Ac_.noalias() = T * Al * Tinv;
  Bc_.noalias() = T * Bl;
  // E_c = [0, vx A_l[2,3], 0, vx A_l[3,3]]' (exact, see header derivation).
  Ec_.setZero();
  Ec_(1, 0) = vx0 * Al(2, 3);
  Ec_(3, 0) = vx0 * Al(3, 3);

  Ad_.noalias() = MpcStateMatrix::Identity() + Ac_ * cfg_.dt;
  Bd_.noalias() = Bc_ * cfg_.dt;
  Ed_.noalias() = Ec_ * cfg_.dt;

  if (!SolveDiscreteARE(Ad_, Bd_, cfg_.Q, cfg_.R, P_)) {
    gains_computed_ = false;
    return false;
  }
  // LQR fallback from the same linearization (cost mapped across coordinates:
  // e_m' Q e_m with e_m = T e_l  =>  Q_lqr = T' Q T).
  {
    const LateralMatrix Ql = T.transpose() * cfg_.Q * T;
    LateralMatrix P_lqr;
    LateralGain K;
    if (!LqrTracker::solveContinuousCARE(Al, Bl, Ql, cfg_.R, P_lqr, K)) {
      gains_computed_ = false;
      return false;
    }
    K_lqr_ = K;
    has_lqr_fallback_ = true;
  }
  vx0_ = vx0;
  mu_ = mu;
  vx_build_ = vx0;
  gains_computed_ = true;
  speed_int_ = 0.0;
  delta_prev_ = 0.0;
  rebuildCondensed();
  return true;
}

void MpcTracker::rebuildCondensed() noexcept {
  constexpr int N = qp::kMpcHorizon;
  constexpr int NX = kLateralDim * (N + 1);  // 64
  // Powers of Ad.
  MpcStateMatrix AdP[N + 1];
  AdP[0] = MpcStateMatrix::Identity();
  for (int k = 1; k <= N; ++k) AdP[k].noalias() = AdP[k - 1] * Ad_;
  Sx_.setZero();
  Su_.setZero();
  Sk_.setZero();
  for (int k = 0; k <= N; ++k) {
    Sx_.template block<kLateralDim, kLateralDim>(k * kLateralDim, 0) = AdP[k];
    for (int j = 0; j < k && j < N; ++j) {
      Su_.template block<kLateralDim, 1>(k * kLateralDim, j) = AdP[k - 1 - j] * Bd_;
      Sk_.template block<kLateralDim, 1>(k * kLateralDim, j) = AdP[k - 1 - j] * Ed_;
    }
  }
  // H = Su' Qbar Su + Rbar. Stage weights Q, terminal P.
  H_.setZero();
  // Accumulate Su' Qbar Su column-pair wise via weighted Su.
  for (int k = 0; k <= N; ++k) {
    const MpcStateMatrix& W = (k < N) ? cfg_.Q : P_;
    for (int i = 0; i < N; ++i) {
      const Eigen::Matrix<double, kLateralDim, 1> wi =
          W * Su_.template block<kLateralDim, 1>(k * kLateralDim, i);
      for (int j = 0; j < N; ++j) {
        H_(i, j) += Su_.template block<kLateralDim, 1>(k * kLateralDim, j).dot(wi);
      }
    }
  }
  // R on the diagonal + R_delta band (d_k - d_{k-1}), d_{-1} = delta_prev.
  for (int k = 0; k < N; ++k) H_(k, k) += 2.0 * cfg_.R;
  if (cfg_.R_delta > 0.0) {
    H_(0, 0) += 2.0 * cfg_.R_delta;
    for (int k = 1; k < N; ++k) {
      H_(k, k) += 4.0 * cfg_.R_delta;
      H_(k, k - 1) += -2.0 * cfg_.R_delta;
      H_(k - 1, k) += -2.0 * cfg_.R_delta;
    }
  }
  H_ = 0.5 * (H_ + H_.transpose());
  (void)NX;
}

void MpcTracker::longitudinalPI(double vx, double vx_ref, double dt, double& ax_out) noexcept {
  const double e_v = vx_ref - vx;
  const double p_term = cfg_.kp_lon * e_v;
  double int_new = speed_int_ + e_v * dt;
  int_new = std::clamp(int_new, -cfg_.integrator_max / (std::abs(cfg_.ki_lon) + 1e-9),
                       cfg_.integrator_max / (std::abs(cfg_.ki_lon) + 1e-9));
  const double i_term = cfg_.ki_lon * int_new;
  double ax = p_term + i_term;
  const double ax_sat = std::clamp(ax, vehicle_.accel_min, vehicle_.accel_max);
  if (ax_sat == ax || (ax_sat > ax && e_v < 0.0) || (ax_sat < ax && e_v > 0.0)) {
    speed_int_ = int_new;
  }
  ax_out = ax_sat;
}

void MpcTracker::setCurvaturePreview(const double* kappa, int n) noexcept {
  if (kappa == nullptr || n <= 0) {
    use_preview_ = false;
    return;
  }
  const int m = n < qp::kMpcHorizon ? n : qp::kMpcHorizon;
  for (int i = 0; i < m; ++i) kappa_preview_[i] = kappa[i];
  for (int i = m; i < qp::kMpcHorizon; ++i) kappa_preview_[i] = kappa[m - 1];
  use_preview_ = true;
}

void MpcTracker::reset() noexcept {
  speed_int_ = 0.0;
  delta_prev_ = 0.0;
  use_preview_ = false;
}

bool MpcTracker::mpcReady() const noexcept {
#ifdef HAVE_OSQP
  return gains_computed_ && solver_ != nullptr && solver_->ready();
#else
  return false;
#endif
}

void MpcTracker::update(const LateralState& err, double vx, double kappa, double vx_ref,
                        double dt, ControlVector& u_out) noexcept {
  double ey = std::clamp(err(kEy), -cfg_.max_ey_clamp, cfg_.max_ey_clamp);
  double epsi = std::clamp(err(kEpsi), -cfg_.max_epsi_clamp, cfg_.max_epsi_clamp);
  const double vx_pos = vx >= 0.0 ? vx : 0.0;

  double delta = 0.0;
  if (gains_computed_) {
#ifdef HAVE_OSQP
    const bool use_mpc = (solver_ != nullptr) && solver_->ready();
#else
    const bool use_mpc = false;
#endif
    if (use_mpc) {
      // Re-condense if the speed scheduling point drifted.
      if (std::abs(vx_pos - vx_build_) > cfg_.rebuild_dvx && vx_pos >= 1.0) {
        // NOTE: rebuild at the measured speed with cached mu (documented:
        // re-call computeGains to move both scheduling variables).
        DynamicBicycleModel model(vehicle_);
        LateralMatrix Al;
        LateralInputMatrix Bl;
        model.linearizedLateralMatrices(vx_pos, mu_, Al, Bl);
        MpcStateMatrix T = MpcStateMatrix::Zero();
        T(0, 0) = 1.0;
        T(1, 1) = vx_pos;
        T(1, 2) = 1.0;
        T(2, 1) = 1.0;
        T(3, 3) = 1.0;
        MpcStateMatrix Tinv = MpcStateMatrix::Zero();
        Tinv(0, 0) = 1.0;
        Tinv(1, 2) = 1.0;
        Tinv(2, 1) = 1.0;
        Tinv(2, 2) = -vx_pos;
        Tinv(3, 3) = 1.0;
        Ac_.noalias() = T * Al * Tinv;
        Bc_.noalias() = T * Bl;
        Ec_.setZero();
        Ec_(1, 0) = vx_pos * Al(2, 3);
        Ec_(3, 0) = vx_pos * Al(3, 3);
        Ad_.noalias() = MpcStateMatrix::Identity() + Ac_ * cfg_.dt;
        Bd_.noalias() = Bc_ * cfg_.dt;
        Ed_.noalias() = Ec_ * cfg_.dt;
        vx_build_ = vx_pos;
        rebuildCondensed();
      }
      // Map error to spec coordinates: [ey, vx*epsi + vy, epsi, r].
      MpcState em;
      em << ey, vx_pos * epsi + err(kLatVy), epsi, err(kLatR);
      // Free response F = Sx x0 + Sk K.
      constexpr int N = qp::kMpcHorizon;
      Eigen::Matrix<double, kLateralDim * (N + 1), 1> F;
      F.noalias() = Sx_ * em;
      if (use_preview_) {
        for (int j = 0; j < N; ++j) F.noalias() += Sk_.col(j) * kappa_preview_[j];
      } else if (kappa != 0.0) {
        for (int j = 0; j < N; ++j) F.noalias() += Sk_.col(j) * kappa;
      }
      // g = Su' Qbar F (+ R_delta linear term from delta_prev).
      MpcHorizonVec g = MpcHorizonVec::Zero();
      for (int k = 0; k <= N; ++k) {
        const MpcStateMatrix& W = (k < N) ? cfg_.Q : P_;
        const Eigen::Matrix<double, kLateralDim, 1> wgt =
            W * F.template segment<kLateralDim>(k * kLateralDim);
        for (int j = 0; j < N; ++j) {
          g(j, 0) += Su_.template block<kLateralDim, 1>(k * kLateralDim, j).dot(wgt);
        }
      }
      if (cfg_.R_delta > 0.0) g(0, 0) += -2.0 * cfg_.R_delta * delta_prev_;
#ifdef HAVE_OSQP
      MpcHorizonVec U;
      int iters = 0;
      if (solver_->solve(H_, g, U, iters)) {
        last_iters_ = iters;
        delta = std::clamp(U(0, 0), -vehicle_.steer_max, vehicle_.steer_max);
      } else if (has_lqr_fallback_) {
        LateralState el;
        el << ey, epsi, err(kLatVy), err(kLatR);
        delta = -(K_lqr_ * el)(0, 0);
        delta = std::clamp(delta, -vehicle_.steer_max, vehicle_.steer_max);
      }
#else
      (void)g;
#endif
    } else if (has_lqr_fallback_) {
      LateralState el;
      el << ey, epsi, err(kLatVy), err(kLatR);
      delta = -(K_lqr_ * el)(0, 0);
      delta = std::clamp(delta, -vehicle_.steer_max, vehicle_.steer_max);
    }
  }
  delta_prev_ = delta;
  double ax = 0.0;
  longitudinalPI(vx, vx_ref, dt, ax);
  u_out(kSteer) = delta;
  u_out(kAx) = ax;
}

}  // namespace av_safety
