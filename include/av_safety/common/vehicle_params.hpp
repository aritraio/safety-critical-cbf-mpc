#pragma once
/// @file vehicle_params.hpp
/// @brief Vehicle + tire parameter struct with validated YAML loading.
///
/// Exceptions are allowed HERE (init time only). The hot control loop never
/// touches the filesystem or throws: it holds a const copy of VehicleParams.

#include <string>

namespace av_safety {

struct PacejkaParams {
  double B{10.0};
  double C{1.3};
  double D{3500.0};  // [N] per-tire peak at mu_nominal
  double E{-1.0};
};

struct VehicleParams {
  double mass{1500.0};          // [kg]
  double yaw_inertia{2500.0};   // [kg m^2]
  double lf{1.2};               // [m] CG -> front axle
  double lr{1.3};               // [m] CG -> rear axle
  double track_width{1.6};      // [m]
  double wheel_radius{0.33};    // [m]
  double gravity{9.81};         // [m/s^2]

  PacejkaParams pacejka{};
  double mu_nominal{0.9};
  double drive_bias_front{0.0};  // 0=RWD .. 0.5=AWD .. 1=FWD

  double steer_max{0.55};      // [rad]
  double steer_rate_max{3.0};  // [rad/s]
  double accel_max{3.0};       // [m/s^2]
  double accel_min{-6.0};      // [m/s^2]
  double accel_rate_max{8.0};  // [m/s^3] longitudinal slew limit
  double vx_epsilon{1.0};      // [m/s] slip-angle denominator clamp
  double operating_speed{20.0};  // [m/s]

  double wheelbase() const noexcept { return lf + lr; }
};

/// Load + validate. Throws std::runtime_error on missing/invalid entries.
VehicleParams LoadVehicleParams(const std::string& yaml_path);

}  // namespace av_safety
