cmake_minimum_required(VERSION 3.16)
project(av_cbf_safety_filter VERSION 0.1.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(BUILD_TESTS "Build GoogleTest unit tests" ON)
option(BUILD_DEMOS "Build standalone simulation demos" ON)
option(USE_ROS2 "Build ROS 2 nodes (requires rclcpp)" OFF)
option(USE_OSQP "Enable OSQP QP solver if available (required for Phase 3)" ON)
option(BUILD_PYTHON_BINDINGS "Build pybind11 SIL bindings (needs network once)" OFF)

# ---------------------------------------------------------------------------
# Language / toolchain
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
# Real-time hot loop: optimize + keep frame pointers for profiling, no RTTI/exceptions stripped
# (we keep exceptions at init/config time; hot-loop methods are noexcept).

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(yaml-cpp REQUIRED)

if(USE_OSQP)
  find_package(osqp QUIET)
  if(osqp_FOUND)
    message(STATUS "OSQP found: enabling HAVE_OSQP")
    add_compile_definitions(HAVE_OSQP=1)
    set(OSQP_TARGET osqp)
  else()
    message(STATUS "OSQP not installed: fetching v0.6.3 via FetchContent")
    include(FetchContent)
    # osqp-0.6.3 predates CMake 4.x minimum-version rules; keep our floor.
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    set(UNITTESTS OFF CACHE BOOL "" FORCE)
    set(PRINTING ON CACHE BOOL "" FORCE)
    FetchContent_Declare(osqp
      GIT_REPOSITORY https://github.com/osqp/osqp.git
      GIT_TAG v0.6.3
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(osqp)
    set(OSQP_VENDORED TRUE)
    add_compile_definitions(HAVE_OSQP=1)
    if(TARGET osqpstatic)
      set(OSQP_TARGET osqpstatic)
    else()
      set(OSQP_TARGET osqp)
    endif()
    message(STATUS "OSQP vendored: using target ${OSQP_TARGET}")
  endif()
endif()

# ---------------------------------------------------------------------------
# Core library (Phase 1: dynamics + LQR; extension points for barriers/QP/UKF)
# ---------------------------------------------------------------------------
add_library(av_safety_core STATIC
  src/common/vehicle_params.cpp
  src/dynamics/pacejka_tire.cpp
  src/dynamics/dynamic_bicycle_model.cpp
  src/controllers/lqr_tracker.cpp
  src/controllers/cbf_qp_filter.cpp
  src/controllers/mpc_tracker.cpp
  src/estimation/friction_ukf.cpp
  src/barriers/barrier_params.cpp
  src/barriers/road_boundary_cbf.cpp
  src/barriers/obstacle_cbf.cpp
  src/barriers/drift_cbf.cpp
  src/barriers/gap_cbf.cpp
  src/qp/enumeration_qp_solver.cpp
  src/qp/osqp_solver.cpp
  src/qp/mpc_osqp_solver.cpp
)

target_include_directories(av_safety_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(av_safety_core PUBLIC
  Eigen3::Eigen
  yaml-cpp
)

if(USE_OSQP AND (osqp_FOUND OR TARGET ${OSQP_TARGET}))
  target_link_libraries(av_safety_core PUBLIC ${OSQP_TARGET})
  # Vendored headers live in _deps; system installs provide their own path.
  if(OSQP_VENDORED)
    target_include_directories(av_safety_core PUBLIC
      $<BUILD_INTERFACE:${osqp_SOURCE_DIR}/include>)
  endif()
endif()

target_compile_options(av_safety_core PRIVATE -O3)

# ---------------------------------------------------------------------------
# Standalone demos (no ROS 2 required)
# ---------------------------------------------------------------------------
if(BUILD_DEMOS)
  add_executable(lqr_tracking_demo simulation/lqr_tracking_demo.cpp)
  target_link_libraries(lqr_tracking_demo PRIVATE av_safety_core)
  target_compile_options(lqr_tracking_demo PRIVATE -O3)

  add_executable(cbf_safety_demo simulation/cbf_safety_demo.cpp)
  target_link_libraries(cbf_safety_demo PRIVATE av_safety_core)
  target_compile_options(cbf_safety_demo PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)
  include(GoogleTest)

  function(av_add_test name src)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE av_safety_core GTest::gtest GTest::gtest_main)
    target_compile_options(${name} PRIVATE -O2)
    gtest_discover_tests(${name})
  endfunction()

  av_add_test(test_pacejka tests/test_pacejka.cpp)
  av_add_test(test_dynamics tests/test_dynamics.cpp)
  av_add_test(test_lqr tests/test_lqr.cpp)
  av_add_test(test_lie_derivatives tests/test_lie_derivatives.cpp)
  av_add_test(test_qp_solver tests/test_qp_solver.cpp)
  av_add_test(test_cbf_filter tests/test_cbf_filter.cpp)
  av_add_test(test_friction_ukf tests/test_friction_ukf.cpp)
  av_add_test(test_mpc tests/test_mpc.cpp)
endif()

# ---------------------------------------------------------------------------
# ROS 2 (optional; Phase 4). Off by default — macOS CI has no ROS 2.
# Enable inside a Humble+ workspace: cmake -DUSE_ROS2=ON (requires rclcpp,
# nav_msgs, visualization_msgs, std_msgs, geometry_msgs).
# ---------------------------------------------------------------------------
if(USE_ROS2)
  add_compile_definitions(USE_ROS2=1)
  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(nav_msgs REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(visualization_msgs REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  add_executable(cbf_safety_node src/ros/cbf_safety_node.cpp)
  target_link_libraries(cbf_safety_node PRIVATE av_safety_core)
  ament_target_dependencies(cbf_safety_node
    rclcpp nav_msgs sensor_msgs visualization_msgs std_msgs geometry_msgs)
  install(TARGETS cbf_safety_node
    DESTINATION lib/${PROJECT_NAME})
  install(DIRECTORY config launch rviz
    DESTINATION share/${PROJECT_NAME})
  ament_package()
endif()

# ---------------------------------------------------------------------------
# Python SIL bindings (Module D; optional, default OFF to keep default builds
# network-free). Built with: cmake -DBUILD_PYTHON_BINDINGS=ON ...
# ---------------------------------------------------------------------------
if(BUILD_PYTHON_BINDINGS)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  include(FetchContent)
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
  FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.12.0
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(pybind11)
  pybind11_add_module(av_safety_py python/bindings.cpp)
  target_link_libraries(av_safety_py PRIVATE av_safety_core)
  target_compile_options(av_safety_py PRIVATE -O2)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Eigen: ${Eigen3_VERSION} | BUILD_TESTS=${BUILD_TESTS} BUILD_DEMOS=${BUILD_DEMOS} USE_ROS2=${USE_ROS2}")
