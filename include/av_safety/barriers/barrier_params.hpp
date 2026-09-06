#pragma once
/// @file barrier_params.hpp
/// @brief Validated YAML loading for CBF tuning (init time; may throw).

#include <string>

namespace av_safety {

struct RoadBarrierParams {
  double lane_margin{3.5};  // [m] |Y - center| <= margin
  double lane_center_y{0.0};
  double kappa{2.0};  // ECBF pole (p1 = p2 = kappa)
};

struct ObstacleBarrierParams {
  double kappa{2.0};  // ECBF pole shared by all obstacle instances
};

struct DriftBarrierParams {
  double kappa{3.0};  // class-K slope for the RD1 drift constraint
};

struct GapBarrierParams {
  double t_headway{1.2};    // [s] desired time headway
  double d_min{4.0};        // [m] standstill gap
  double kappa{2.0};        // class-K slope for the RD1 gap constraint
  double lateral_gate{2.0};  // [m] row applies only if |dy| < gate (near-lane)
};

struct BarrierConfig {
  RoadBarrierParams road{};
  ObstacleBarrierParams obstacle{};
  DriftBarrierParams drift{};
  GapBarrierParams gap{};
};

/// Load + validate. Throws std::runtime_error on missing/invalid entries.
BarrierConfig LoadBarrierConfig(const std::string& yaml_path);

}  // namespace av_safety
