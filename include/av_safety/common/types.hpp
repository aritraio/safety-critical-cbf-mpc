#pragma once
/// @file types.hpp
/// @brief Fixed-size Eigen aliases + state/control indexing for the safety stack.
///
/// Real-time rule: every hot-loop type is a compile-time-sized Eigen object.
/// No `Eigen::VectorXd`, no `std::vector` in `*_update / *_step` paths.

#include <Eigen/Dense>

namespace av_safety {

// Full nonlinear state: x = [x, y, psi, vx, vy, r]  (6x1)
inline constexpr int kStateDim = 6;
inline constexpr int kControlDim = 2;
inline constexpr int kLateralDim = 4;  // error state e = [e_y, e_psi, v_y, r]

enum StateIdx : int { kPx = 0, kPy = 1, kPsi = 2, kVx = 3, kVy = 4, kYawRate = 5 };
enum ControlIdx : int { kSteer = 0, kAx = 1 };
enum LateralErrIdx : int { kEy = 0, kEpsi = 1, kLatVy = 2, kLatR = 3 };

using StateVector = Eigen::Matrix<double, kStateDim, 1>;
using StateDerivative = Eigen::Matrix<double, kStateDim, 1>;
using ControlVector = Eigen::Matrix<double, kControlDim, 1>;
using StateMatrix = Eigen::Matrix<double, kStateDim, kStateDim>;
using InputMatrix = Eigen::Matrix<double, kStateDim, kControlDim>;

using LateralState = Eigen::Matrix<double, kLateralDim, 1>;
using LateralMatrix = Eigen::Matrix<double, kLateralDim, kLateralDim>;
using LateralInputMatrix = Eigen::Matrix<double, kLateralDim, 1>;
using LateralGain = Eigen::Matrix<double, 1, kLateralDim>;

}  // namespace av_safety
