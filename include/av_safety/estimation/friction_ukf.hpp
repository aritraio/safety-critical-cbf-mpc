#pragma once
/// @file friction_ukf.hpp
/// @brief Online road-tire friction estimation via Unscented Kalman Filter.
///
/// Augmented state x_ukf = [vx, vy, r, mu] in R^4, input u = [delta, ax],
/// measurement z = [ax_imu, ay_imu, r_imu, vx_wheels] in R^4.
///
/// Process model (exact for RWD drive bias; longitudinal force lumped in ax):
///   vx_dot = ax + vy r - 2 Fyf(mu) sin(delta)/m
///   vy_dot = -vx r + (2 Fyf(mu) cos(delta) + 2 Fyr(mu))/m
///   r_dot  = (2 lf Fyf(mu) cos(delta) - 2 lr Fyr(mu))/Iz
///   mu_dot = 0 + w_mu
/// Measurement model:
///   h = [ax - vy r, (2 Fyf(mu) cos(delta) + 2 Fyr(mu))/m - vx r, r, vx]
/// (Conventions: flat road, IMU co-located at CG, wheel speed ~= vx. The
/// filter is verified under this matched model; plant mismatch appears as
/// process noise in deployment.)
///
/// Standard Van der Merwe unscented transform (alpha/beta/kappa), Joseph-form
/// covariance update with symmetrization, a mean-reverting prior on mu
/// (pseudo-measurement mu ~= mu_nominal, see R_prior), and a projection guard
/// enforcing mu_hat in [mu_min, mu_max] after every update. The prior anchors
/// mu when lateral excitation is weak: on straight cruise ay ~= 0 cannot
/// distinguish "no slip" (vy = 0) from "no grip" (mu -> 0), so the joint
/// (vy, mu) posterior has a ridge down which plain UKFs slide to the mu
/// clamp. Cornering data overwhelms the weak prior when excited. All hot-loop
/// storage is fixed-size Eigen (4x4 covariances, 9 sigma points);
/// predict/update/step are noexcept and allocation-free. mu_hat feeds
/// CbfQpFilter::filter(), adapting h_drift(x, mu_hat) online (split-mu
/// transitions included).

#include "av_safety/common/types.hpp"
#include "av_safety/common/vehicle_params.hpp"
#include "av_safety/dynamics/pacejka_tire.hpp"

#include <string>

namespace av_safety {
namespace estimation {

inline constexpr int kUkfStateDim = 4;  // [vx, vy, r, mu]
inline constexpr int kUkfMeasDim = 4;   // [ax_imu, ay_imu, r_imu, vx_wheels]
inline constexpr int kUkfSigmaCount = 2 * kUkfStateDim + 1;

enum UkfStateIdx : int { kUvx = 0, kUvy = 1, kUr = 2, kUmu = 3 };
enum UkfMeasIdx : int { kMax = 0, kMay = 1, kMr = 2, kMvx = 3 };

using UkfState = Eigen::Matrix<double, kUkfStateDim, 1>;
using UkfMeas = Eigen::Matrix<double, kUkfMeasDim, 1>;
using UkfCov = Eigen::Matrix<double, kUkfStateDim, kUkfStateDim>;
using UkfSigmaMat = Eigen::Matrix<double, kUkfStateDim, kUkfSigmaCount>;
using UkfMeasSigmaMat = Eigen::Matrix<double, kUkfMeasDim, kUkfSigmaCount>;

struct UkfConfig {
  UkfCov Q{UkfCov::Identity()};   // process noise (discrete, per step)
  UkfCov R{UkfCov::Identity()};   // measurement noise
  UkfCov P0{UkfCov::Identity()};  // initial covariance
  double alpha{1e-3};             // Van der Merwe spread
  double beta{2.0};               // prior kurtosis match (Gaussian)
  double kappa{0.0};              // secondary scaling
  double mu_init{0.9};
  double mu_min{0.1};
  double mu_max{1.3};
  double R_prior{1.0};  // mean-reversion variance (<= 0 disables the prior)
};

/// Load tuning from YAML (throws on invalid). Missing keys keep defaults.
UkfConfig LoadUkfConfig(const std::string& yaml_path);

class FrictionUkf {
 public:
  FrictionUkf(const VehicleParams& vehicle, const UkfConfig& cfg) noexcept;

  void reset() noexcept;

  /// Time update: propagate sigma points through Euler-discretized process.
  void predict(const ControlVector& u, double dt) noexcept;
  /// Measurement update with z = [ax_imu, ay_imu, r_imu, vx_wheels].
  void update(const UkfMeas& z, const ControlVector& u) noexcept;
  /// predict + update in one call.
  void step(const ControlVector& u, const UkfMeas& z, double dt) noexcept;

  double mu() const noexcept { return x_(kUmu); }
  const UkfState& state() const noexcept { return x_; }
  const UkfCov& covariance() const noexcept { return P_; }

 private:
  // Continuous process derivative for one (sigma-point) state.
  void processDerivative(const UkfState& xs, const ControlVector& u,
                         UkfState& xd_out) const noexcept;
  // Expected measurement for one (propagated sigma-point) state.
  // Return-by-value (fixed size: stack only, no heap) so Eigen column blocks
  // can be passed directly.
  UkfMeas measurementModel(const UkfState& xs, const ControlVector& u) const noexcept;
  void computeSigmaPoints(UkfSigmaMat& S) const noexcept;

  VehicleParams vehicle_;
  UkfConfig cfg_;
  UkfState x_{UkfState::Zero()};
  UkfCov P_{UkfCov::Identity()};
  PacejkaTire front_tire_;
  PacejkaTire rear_tire_;
  double wm0_{0.0}, wc0_{0.0}, wi_{0.0}, gamma_{1.0};
};

}  // namespace estimation
}  // namespace av_safety
