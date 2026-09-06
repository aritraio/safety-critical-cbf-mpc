#include "av_safety/barriers/barrier_params.hpp"

#include <cmath>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

namespace av_safety {

BarrierConfig LoadBarrierConfig(const std::string& yaml_path) {
  YAML::Node n;
  try {
    n = YAML::LoadFile(yaml_path);
  } catch (const std::exception& e) {
    throw std::runtime_error("barrier_params: cannot load '" + yaml_path + "': " + e.what());
  }
  BarrierConfig c;
  const auto road = n["road_boundary"];
  if (!road) throw std::runtime_error("barrier_params: missing 'road_boundary'");
  c.road.lane_margin = road["lane_margin"].as<double>(3.5);
  c.road.lane_center_y = road["lane_center_y"].as<double>(0.0);
  c.road.kappa = road["kappa_road"].as<double>(2.0);
  if (!(c.road.lane_margin > 0.0)) throw std::runtime_error("barrier_params: lane_margin > 0");
  if (!(c.road.kappa > 0.0)) throw std::runtime_error("barrier_params: kappa_road > 0");

  const auto obs = n["obstacle"];
  if (obs) {
    c.obstacle.kappa = obs["kappa_obs"].as<double>(2.0);
  }
  if (!(c.obstacle.kappa > 0.0)) throw std::runtime_error("barrier_params: kappa_obs > 0");

  const auto drift = n["drift_envelope"];
  if (drift) {
    c.drift.kappa = drift["kappa_drift"].as<double>(3.0);
  }
  if (!(c.drift.kappa > 0.0)) throw std::runtime_error("barrier_params: kappa_drift > 0");

  const auto gap = n["gap_barrier"];
  if (gap) {
    c.gap.t_headway = gap["t_headway"].as<double>(1.2);
    c.gap.d_min = gap["d_min"].as<double>(4.0);
    c.gap.kappa = gap["kappa_gap"].as<double>(2.0);
    c.gap.lateral_gate = gap["lateral_gate"].as<double>(2.0);
  }
  if (!(c.gap.t_headway > 0.0)) throw std::runtime_error("barrier_params: t_headway > 0");
  if (!(c.gap.d_min >= 0.0)) throw std::runtime_error("barrier_params: d_min >= 0");
  if (!(c.gap.kappa > 0.0)) throw std::runtime_error("barrier_params: kappa_gap > 0");
  if (!(c.gap.lateral_gate > 0.0)) throw std::runtime_error("barrier_params: lateral_gate > 0");
  return c;
}

}  // namespace av_safety
