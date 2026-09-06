#include "av_safety/common/vehicle_params.hpp"

#include <cmath>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

namespace av_safety {
namespace {

double RequirePositive(const YAML::Node& n, const char* key) {
  if (!n[key]) throw std::runtime_error(std::string("vehicle_params: missing key '") + key + "'");
  const double v = n[key].as<double>();
  if (!(v > 0.0)) throw std::runtime_error(std::string("vehicle_params: '") + key + "' must be > 0");
  return v;
}

double RequireFinite(const YAML::Node& n, const char* key) {
  if (!n[key]) throw std::runtime_error(std::string("vehicle_params: missing key '") + key + "'");
  const double v = n[key].as<double>();
  if (!std::isfinite(v)) throw std::runtime_error(std::string("vehicle_params: '") + key + "' non-finite");
  return v;
}

}  // namespace

VehicleParams LoadVehicleParams(const std::string& yaml_path) {
  YAML::Node n;
  try {
    n = YAML::LoadFile(yaml_path);
  } catch (const std::exception& e) {
    throw std::runtime_error("vehicle_params: cannot load '" + yaml_path + "': " + e.what());
  }

  VehicleParams p;
  p.mass = RequirePositive(n, "mass");
  p.yaw_inertia = RequirePositive(n, "yaw_inertia");
  p.lf = RequirePositive(n, "wheelbase_front");
  p.lr = RequirePositive(n, "wheelbase_rear");
  p.track_width = RequirePositive(n, "track_width");
  p.wheel_radius = RequirePositive(n, "wheel_radius");
  p.gravity = RequirePositive(n, "gravity");

  const auto pj = n["pacejka"];
  if (!pj) throw std::runtime_error("vehicle_params: missing key 'pacejka'");
  p.pacejka.B = RequireFinite(pj, "B");
  p.pacejka.C = RequireFinite(pj, "C");
  p.pacejka.D = RequirePositive(pj, "D");
  p.pacejka.E = RequireFinite(pj, "E");
  if (p.pacejka.B <= 0.0 || p.pacejka.C <= 0.0)
    throw std::runtime_error("vehicle_params: pacejka B and C must be > 0");

  p.mu_nominal = RequirePositive(n, "mu_nominal");
  p.drive_bias_front = n["drive_bias_front"] ? n["drive_bias_front"].as<double>() : 0.0;
  if (p.drive_bias_front < 0.0 || p.drive_bias_front > 1.0)
    throw std::runtime_error("vehicle_params: drive_bias_front must be in [0,1]");

  p.steer_max = RequirePositive(n, "steer_max");
  p.steer_rate_max = RequirePositive(n, "steer_rate_max");
  p.accel_max = RequirePositive(n, "accel_max");
  if (!n["accel_min"]) throw std::runtime_error("vehicle_params: missing key 'accel_min'");
  p.accel_min = n["accel_min"].as<double>();
  if (!(p.accel_min < 0.0 && p.accel_max > 0.0))
    throw std::runtime_error("vehicle_params: need accel_min < 0 < accel_max");
  p.accel_rate_max = n["accel_rate_max"] ? n["accel_rate_max"].as<double>() : 8.0;
  if (!(p.accel_rate_max > 0.0))
    throw std::runtime_error("vehicle_params: accel_rate_max must be > 0");
  p.vx_epsilon = RequirePositive(n, "vx_epsilon");
  p.operating_speed = RequirePositive(n, "operating_speed");
  return p;
}

}  // namespace av_safety
