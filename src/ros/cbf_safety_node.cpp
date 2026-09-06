// ROS 2 safety-filter node: nominal tracker (LQR/MPC) + UKF friction
// adaptation + CBF-QP safety filter @100 Hz, with RViz diagnostics.
//
// NOTE: compiled only with -DUSE_ROS2=ON inside a ROS 2 workspace (Humble+).
// Wiring:
//   sub: /odom (nav_msgs/Odometry), /imu (sensor_msgs/Imu),
//        /obstacles (visualization_msgs/MarkerArray)
//   pub: /safety_cmd (std_msgs/Float64MultiArray [delta, ax]),
//        /safety_diag (std_msgs/Float64MultiArray [h_obs, h_road, h_drift, ...]),
//        /cbf_markers (visualization_msgs/MarkerArray for RViz2)
// The 100 Hz timer path is allocation-light: fixed-size Eigen, no heap ops
// (marker message building allocates once per cycle outside the control math).

#ifdef USE_ROS2

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "geometry_msgs/msg/pose.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include "av_safety/barriers/barrier_params.hpp"
#include "av_safety/barriers/obstacle_cbf.hpp"
#include "av_safety/common/vehicle_params.hpp"
#include "av_safety/controllers/cbf_qp_filter.hpp"
#include "av_safety/controllers/lqr_tracker.hpp"
#include "av_safety/controllers/mpc_tracker.hpp"
#include "av_safety/dynamics/dynamic_bicycle_model.hpp"
#include "av_safety/estimation/friction_ukf.hpp"
#include "av_safety/qp/enumeration_qp_solver.hpp"

namespace av_safety {
namespace ros {

class CbfSafetyNode : public rclcpp::Node {
 public:
  CbfSafetyNode() : rclcpp::Node("cbf_safety_filter") {
    declare_parameter("vehicle_yaml", "config/vehicle_params.yaml");
    declare_parameter("lqr_yaml", "config/lqr_tuning.yaml");
    declare_parameter("mpc_yaml", "config/mpc_tuning.yaml");
    declare_parameter("barrier_yaml", "config/barrier_params.yaml");
    declare_parameter("ukf_yaml", "config/friction_ukf.yaml");
    declare_parameter("controller", "lqr");  // lqr | mpc
    declare_parameter("use_ukf", true);
    declare_parameter("mu", 0.9);
    declare_parameter("vx_ref", 15.0);
    declare_parameter("y_ref", 0.0);
    declare_parameter("rate_hz", 100.0);

    const auto vehicle_yaml = get_parameter("vehicle_yaml").as_string();
    const auto lqr_yaml = get_parameter("lqr_yaml").as_string();
    const auto mpc_yaml = get_parameter("mpc_yaml").as_string();
    const auto barrier_yaml = get_parameter("barrier_yaml").as_string();
    const auto ukf_yaml = get_parameter("ukf_yaml").as_string();
    controller_ = get_parameter("controller").as_string();
    use_ukf_ = get_parameter("use_ukf").as_bool();
    mu_ = get_parameter("mu").as_double();
    vx_ref_ = get_parameter("vx_ref").as_double();
    y_ref_ = get_parameter("y_ref").as_double();
    const double rate = get_parameter("rate_hz").as_double();
    dt_ = 1.0 / rate;

    vp_ = LoadVehicleParams(vehicle_yaml);
    lqr_cfg_ = LoadLqrConfig(lqr_yaml);
    mpc_cfg_ = LoadMpcConfig(mpc_yaml);
    bc_ = LoadBarrierConfig(barrier_yaml);
    ukf_cfg_ = estimation::LoadUkfConfig(ukf_yaml);
    model_ = std::make_unique<DynamicBicycleModel>(vp_);
    tracker_ = std::make_unique<LqrTracker>(vp_, lqr_cfg_);
    if (!tracker_->computeGains(vx_ref_, mu_)) {
      RCLCPP_FATAL(get_logger(), "CARE solve failed; shutting down");
      throw std::runtime_error("cbf_safety_node: CARE failed");
    }
    mpc_ = std::make_unique<MpcTracker>(vp_, mpc_cfg_);
    if (controller_ == "mpc" && !mpc_->computeGains(vx_ref_, mu_)) {
      RCLCPP_FATAL(get_logger(), "MPC setup failed; shutting down");
      throw std::runtime_error("cbf_safety_node: MPC failed");
    }
    ukf_ = std::make_unique<estimation::FrictionUkf>(vp_, ukf_cfg_);
    filter_ = std::make_unique<CbfQpFilter>(vp_, bc_, CbfQpConfig{}, &backend_);
    x_.setZero();
    x_(kVx) = vx_ref_;
    have_odom_ = false;
    have_imu_ = false;
    u_applied_.setZero();

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "/odom", 10, std::bind(&CbfSafetyNode::onOdom, this, std::placeholders::_1));
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        "/imu", 10, std::bind(&CbfSafetyNode::onImu, this, std::placeholders::_1));
    obs_sub_ = create_subscription<visualization_msgs::msg::MarkerArray>(
        "/obstacles", 10, std::bind(&CbfSafetyNode::onObstacles, this, std::placeholders::_1));
    cmd_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/safety_cmd", 10);
    diag_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/safety_diag", 10);
    markers_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/cbf_markers", 10);

    const auto period = std::chrono::duration<double>(1.0 / rate);
    timer_ = create_wall_timer(std::chrono::duration_cast<std::chrono::nanoseconds>(period),
                               std::bind(&CbfSafetyNode::onTimer, this));
    RCLCPP_INFO(get_logger(), "cbf_safety_filter up at %.0f Hz (%s%s, mu=%.2f)", rate,
                controller_.c_str(), use_ukf_ ? "+ukf" : "", mu_);
  }

 private:
  void onImu(const sensor_msgs::msg::Imu::SharedPtr msg) {
    imu_ax_ = msg->linear_acceleration.x;
    imu_ay_ = msg->linear_acceleration.y;
    imu_r_ = msg->angular_velocity.z;
    have_imu_ = true;
  }
  void onOdom(const nav_msgs::msg::Odometry::SharedPtr msg) {
    // Planar odometry -> [x, y, psi, vx, vy, r]. (Production: replace with a
    // proper localization + sideslip estimator; see friction_ukf roadmap.)
    const auto& p = msg->pose.pose.position;
    const auto& q = msg->pose.pose.orientation;
    const double psi = 2.0 * std::atan2(q.z, q.w);  // yaw-only convention
    const double vx = msg->twist.twist.linear.x;
    const double vy = msg->twist.twist.linear.y;
    x_(kPx) = p.x;
    x_(kPy) = p.y;
    x_(kPsi) = psi;
    x_(kVx) = vx;
    x_(kVy) = vy;
    x_(kYawRate) = msg->twist.twist.angular.z;
    have_odom_ = true;
  }

  void onObstacles(const visualization_msgs::msg::MarkerArray::SharedPtr msg) {
    // Marker i: position = center, scale.x/y = full extents (a = sx/2 + margin).
    num_obs_ = 0;
    for (const auto& m : msg->markers) {
      if (num_obs_ >= kMaxFilterObstacles) break;
      if (m.action == visualization_msgs::msg::Marker::DELETE) continue;
      ObstacleState o;
      o.x = m.pose.position.x;
      o.y = m.pose.position.y;
      o.vx = 0.0;
      o.vy = 0.0;  // roadmap: obstacle velocity tracker (constant-velocity now)
      o.a = m.scale.x * 0.5 + 1.0;  // + ego half-length margin
      o.b = m.scale.y * 0.5 + 0.6;  // + ego half-width margin
      obstacles_[num_obs_++] = o;
    }
    filter_->setObstacles(obstacles_.data(), num_obs_);
  }

  void onTimer() {
    if (!have_odom_) return;
    // Friction source: UKF on IMU+odometry when enabled and fed, else static.
    double mu_filter = mu_;
    if (use_ukf_ && have_imu_) {
      estimation::UkfMeas z;
      z(estimation::kMax) = imu_ax_;
      z(estimation::kMay) = imu_ay_;
      z(estimation::kMr) = imu_r_;
      z(estimation::kMvx) = x_(kVx);
      ukf_->step(u_applied_, z, dt_);
      mu_filter = ukf_->mu();
    }
    LateralState err;
    err << x_(kPy) - y_ref_, x_(kPsi), x_(kVy), x_(kYawRate);
    ControlVector nom, safe;
    FilterDiagnostics diag;
    if (controller_ == "mpc") {
      mpc_->update(err, x_(kVx), 0.0, vx_ref_, dt_, nom);
    } else {
      tracker_->update(err, x_(kVx), 0.0, vx_ref_, dt_, nom);
    }
    filter_->filter(x_, nom, mu_filter, safe, diag);
    u_applied_ = safe;

    std_msgs::msg::Float64MultiArray cmd;
    cmd.data = {safe(kSteer), safe(kAx)};
    cmd_pub_->publish(cmd);
    std_msgs::msg::Float64MultiArray diag_msg;
    diag_msg.data = {diag.h_obs_min,   diag.h_road,  diag.h_drift, diag.h_gap_min,
                     diag.slack_max,   diag.solve_us, mu_filter,  diag.feasible ? 1.0 : 0.0};
    diag_pub_->publish(diag_msg);
    publishMarkers(diag, mu_filter);
    if (!diag.feasible) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "CBF-QP infeasible; holding clamp");
    }
    if (diag.softened) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "CBF slack active (max %.3f); nominal infeasible", diag.slack_max);
    }
  }

  // -- RViz diagnostics ------------------------------------------------------
  static visualization_msgs::msg::Marker MakeMarker(
      const std::string& ns, int id, int type, float r, float g, float b, float a) {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = "map";
    m.ns = ns;
    m.id = id;
    m.type = type;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.color.r = r;
    m.color.g = g;
    m.color.b = b;
    m.color.a = a;
    m.lifetime = rclcpp::Duration(0, 200000000);  // 0.2 s persistence
    return m;
  }

  void publishMarkers(const FilterDiagnostics& diag, double mu_hat) {
    visualization_msgs::msg::MarkerArray arr;
    const double psi = x_(kPsi);
    // Ego footprint (cube), green->red by worst barrier margin.
    {
      auto m = MakeMarker("ego_footprint", 0, visualization_msgs::msg::Marker::CUBE, 0.2f, 0.8f,
                          0.2f, 0.7f);
      const double worst = std::min({diag.h_obs_min, diag.h_road, diag.h_drift});
      if (worst < 1.0) {  // blend toward red as margins shrink
        const float t = static_cast<float>(std::clamp(1.0 - worst, 0.0, 1.0));
        m.color.r = 0.2f + 0.8f * t;
        m.color.g = 0.8f - 0.6f * t;
      }
      m.pose.position.x = x_(kPx);
      m.pose.position.y = x_(kPy);
      m.pose.position.z = 0.5;
      m.pose.orientation.z = std::sin(psi * 0.5);
      m.pose.orientation.w = std::cos(psi * 0.5);
      m.scale.x = 4.5f;
      m.scale.y = 1.9f;
      m.scale.z = 1.4f;
      arr.markers.push_back(m);
    }
    // Obstacle ellipses: filled cylinder + contour outline, colored by h.
    for (int i = 0; i < num_obs_; ++i) {
      const ObstacleState& o = obstacles_[static_cast<size_t>(i)];
      ObstacleCbf probe;
      const double h = probe.value(x_, o);
      const float t = static_cast<float>(std::clamp(1.0 - h * 0.25, 0.0, 1.0));
      auto fill = MakeMarker("obstacles", i, visualization_msgs::msg::Marker::CYLINDER,
                             0.9f, 0.2f + 0.6f * (1.0f - t), 0.2f, 0.45f);
      fill.pose.position.x = o.x;
      fill.pose.position.y = o.y;
      fill.pose.position.z = 0.5;
      fill.scale.x = 2.0f * static_cast<float>(o.a);
      fill.scale.y = 2.0f * static_cast<float>(o.b);
      fill.scale.z = 1.2f;
      arr.markers.push_back(fill);
      auto edge = MakeMarker("barrier_contours", i, visualization_msgs::msg::Marker::LINE_STRIP,
                             0.9f, 0.9f, 0.2f, 0.9f);
      edge.pose.orientation.w = 1.0f;
      edge.scale.x = 0.12f;
      for (int k = 0; k <= 32; ++k) {
        const double a = 2.0 * 3.141592653589793 * k / 32.0;
        geometry_msgs::msg::Point pt;
        pt.x = o.x + o.a * std::cos(a);
        pt.y = o.y + o.b * std::sin(a);
        pt.z = 0.1;
        edge.points.push_back(pt);
      }
      arr.markers.push_back(edge);
    }
    // Road margins: two line strips at y = center +- margin.
    {
      auto lines = MakeMarker("road_margins", 0, visualization_msgs::msg::Marker::LINE_LIST,
                              0.3f, 0.6f, 1.0f, 0.8f);
      lines.pose.orientation.w = 1.0f;
      lines.scale.x = 0.12f;
      const double x0 = x_(kPx) - 30.0, x1 = x_(kPx) + 60.0;
      for (const double yc : {bc_.road.lane_center_y - bc_.road.lane_margin,
                              bc_.road.lane_center_y + bc_.road.lane_margin}) {
        geometry_msgs::msg::Point p0, p1;
        p0.x = x0;
        p0.y = yc;
        p1.x = x1;
        p1.y = yc;
        lines.points.push_back(p0);
        lines.points.push_back(p1);
      }
      arr.markers.push_back(lines);
    }
    // Drift/friction status text.
    {
      auto txt = MakeMarker("drift_status", 0,
                            visualization_msgs::msg::Marker::TEXT_VIEW_FACING, 1.0f, 1.0f,
                            1.0f, 0.9f);
      txt.pose.position.x = x_(kPx);
      txt.pose.position.y = x_(kPy) + 3.0;
      txt.pose.position.z = 2.0;
      txt.scale.z = 0.8f;
      char buf[128];
      std::snprintf(buf, sizeof(buf), "mu=%.2f h_drift=%.1f h_gap=%.1f slack=%.2f", mu_hat,
                    diag.h_drift, diag.h_gap_min, diag.slack_max);
      txt.text = buf;
      arr.markers.push_back(txt);
    }
    markers_pub_->publish(arr);
  }

  double mu_{0.9}, vx_ref_{15.0}, y_ref_{0.0}, dt_{0.01};
  std::string controller_{"lqr"};
  bool use_ukf_{true};
  VehicleParams vp_{};
  LqrConfig lqr_cfg_{};
  MpcConfig mpc_cfg_{};
  BarrierConfig bc_{};
  estimation::UkfConfig ukf_cfg_{};
  qp::EnumerationQpSolver backend_{};
  std::unique_ptr<DynamicBicycleModel> model_;
  std::unique_ptr<LqrTracker> tracker_;
  std::unique_ptr<MpcTracker> mpc_;
  std::unique_ptr<estimation::FrictionUkf> ukf_;
  std::unique_ptr<CbfQpFilter> filter_;
  StateVector x_{StateVector::Zero()};
  ControlVector u_applied_{ControlVector::Zero()};
  double imu_ax_{0.0}, imu_ay_{0.0}, imu_r_{0.0};
  std::array<ObstacleState, kMaxFilterObstacles> obstacles_{};
  int num_obs_{0};
  bool have_odom_{false};
  bool have_imu_{false};
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr obs_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr cmd_pub_, diag_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace ros
}  // namespace av_safety

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<av_safety::ros::CbfSafetyNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

#endif  // USE_ROS2
