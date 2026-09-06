// Python bindings for the safety-critical control stack (SIL / benchmarking).
//
// Uses only pybind11 + STL containers (no numpy dependency at build time):
// Eigen vectors map to fixed-size Python lists. The hot C++ loop stays
// allocation-free; conversions allocate on the Python side only.
//
// Lifetime: CbfQpFilter keeps its backends alive via keep_alive, so scripts
// only need to hold the filter.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>

#include "av_safety/barriers/barrier_params.hpp"
#include "av_safety/common/vehicle_params.hpp"
#include "av_safety/controllers/cbf_qp_filter.hpp"
#include "av_safety/controllers/lqr_tracker.hpp"
#include "av_safety/controllers/mpc_tracker.hpp"
#include "av_safety/dynamics/dynamic_bicycle_model.hpp"
#include "av_safety/dynamics/pacejka_tire.hpp"
#include "av_safety/estimation/friction_ukf.hpp"
#include "av_safety/qp/enumeration_qp_solver.hpp"
#ifdef HAVE_OSQP
#include "av_safety/qp/mpc_osqp_solver.hpp"
#include "av_safety/qp/osqp_solver.hpp"
#endif

namespace py = pybind11;
using namespace av_safety;

namespace {

using Arr6 = std::array<double, 6>;
using Arr4 = std::array<double, 4>;
using Arr2 = std::array<double, 2>;
using Mat4x4 = std::array<std::array<double, 4>, 4>;

StateVector ToState(const Arr6& a) {
  StateVector x;
  for (int i = 0; i < 6; ++i) x(i) = a[static_cast<size_t>(i)];
  return x;
}
Arr6 FromState(const StateVector& x) {
  Arr6 a{};
  for (int i = 0; i < 6; ++i) a[static_cast<size_t>(i)] = x(i);
  return a;
}
ControlVector ToControl(const Arr2& a) {
  ControlVector u;
  u(0) = a[0];
  u(1) = a[1];
  return u;
}
Arr2 FromControl(const ControlVector& u) {
  Arr2 a{};
  a[0] = u(0);
  a[1] = u(1);
  return a;
}
LateralState ToLateral(const Arr4& a) {
  LateralState e;
  for (int i = 0; i < 4; ++i) e(i) = a[static_cast<size_t>(i)];
  return e;
}
Mat4x4 FromMat4(const Eigen::Matrix<double, 4, 4>& M) {
  Mat4x4 o{};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) o[static_cast<size_t>(i)][static_cast<size_t>(j)] = M(i, j);
  return o;
}
Eigen::Matrix<double, 4, 4> ToMat4(const Mat4x4& o) {
  Eigen::Matrix<double, 4, 4> M;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) M(i, j) = o[static_cast<size_t>(i)][static_cast<size_t>(j)];
  return M;
}

py::dict DiagToDict(const FilterDiagnostics& d) {
  py::dict o;
  o["solve_us"] = d.solve_us;
  o["iters"] = d.iters;
  o["optimal"] = d.optimal;
  o["feasible"] = d.feasible;
  o["intervened"] = d.intervened;
  o["softened"] = d.softened;
  o["used_fallback"] = d.used_fallback;
  o["slack_max"] = d.slack_max;
  o["slack_norm"] = d.slack_norm;
  o["h_road"] = d.h_road;
  o["h_drift"] = d.h_drift;
  o["h_obs_min"] = d.h_obs_min;
  o["h_gap_min"] = d.h_gap_min;
  o["num_rows"] = d.num_rows;
  o["num_soft"] = d.num_soft;
  return o;
}

}  // namespace

PYBIND11_MODULE(av_safety_py, m) {
  m.doc() = "Safety-critical CBF-QP vehicle control stack (C++ core, Python SIL)";
  m.attr("kStateDim") = kStateDim;
  m.attr("kMaxFilterObstacles") = kMaxFilterObstacles;

  // -- Params / configs ------------------------------------------------------
  py::class_<PacejkaParams>(m, "PacejkaParams")
      .def(py::init<>())
      .def_readwrite("B", &PacejkaParams::B)
      .def_readwrite("C", &PacejkaParams::C)
      .def_readwrite("D", &PacejkaParams::D)
      .def_readwrite("E", &PacejkaParams::E);

  py::class_<VehicleParams>(m, "VehicleParams")
      .def(py::init<>())
      .def_readwrite("mass", &VehicleParams::mass)
      .def_readwrite("yaw_inertia", &VehicleParams::yaw_inertia)
      .def_readwrite("lf", &VehicleParams::lf)
      .def_readwrite("lr", &VehicleParams::lr)
      .def_readwrite("track_width", &VehicleParams::track_width)
      .def_readwrite("wheel_radius", &VehicleParams::wheel_radius)
      .def_readwrite("gravity", &VehicleParams::gravity)
      .def_readwrite("pacejka", &VehicleParams::pacejka)
      .def_readwrite("mu_nominal", &VehicleParams::mu_nominal)
      .def_readwrite("drive_bias_front", &VehicleParams::drive_bias_front)
      .def_readwrite("steer_max", &VehicleParams::steer_max)
      .def_readwrite("steer_rate_max", &VehicleParams::steer_rate_max)
      .def_readwrite("accel_max", &VehicleParams::accel_max)
      .def_readwrite("accel_min", &VehicleParams::accel_min)
      .def_readwrite("accel_rate_max", &VehicleParams::accel_rate_max)
      .def_readwrite("vx_epsilon", &VehicleParams::vx_epsilon)
      .def_readwrite("operating_speed", &VehicleParams::operating_speed);
  m.def("load_vehicle_params", &LoadVehicleParams, py::arg("yaml_path"));

  py::class_<LqrConfig>(m, "LqrConfig")
      .def(py::init<>())
      .def_property(
          "Q", [](const LqrConfig& c) { return FromMat4(c.Q); },
          [](LqrConfig& c, const Mat4x4& v) { c.Q = ToMat4(v); })
      .def_readwrite("R", &LqrConfig::R)
      .def_readwrite("target_speed", &LqrConfig::target_speed)
      .def_readwrite("kp_lon", &LqrConfig::kp_lon)
      .def_readwrite("ki_lon", &LqrConfig::ki_lon)
      .def_readwrite("integrator_max", &LqrConfig::integrator_max)
      .def_readwrite("use_feedforward", &LqrConfig::use_feedforward);
  m.def("load_lqr_config", &LoadLqrConfig, py::arg("yaml_path"));

  py::class_<MpcConfig>(m, "MpcConfig")
      .def(py::init<>())
      .def_property(
          "Q", [](const MpcConfig& c) { return FromMat4(c.Q); },
          [](MpcConfig& c, const Mat4x4& v) { c.Q = ToMat4(v); })
      .def_readwrite("R", &MpcConfig::R)
      .def_readwrite("R_delta", &MpcConfig::R_delta)
      .def_readwrite("dt", &MpcConfig::dt)
      .def_readwrite("target_speed", &MpcConfig::target_speed)
      .def_readwrite("kp_lon", &MpcConfig::kp_lon)
      .def_readwrite("ki_lon", &MpcConfig::ki_lon);
  m.def("load_mpc_config", &LoadMpcConfig, py::arg("yaml_path"));

  py::class_<RoadBarrierParams>(m, "RoadBarrierParams")
      .def(py::init<>())
      .def_readwrite("lane_margin", &RoadBarrierParams::lane_margin)
      .def_readwrite("lane_center_y", &RoadBarrierParams::lane_center_y)
      .def_readwrite("kappa", &RoadBarrierParams::kappa);
  py::class_<ObstacleBarrierParams>(m, "ObstacleBarrierParams")
      .def(py::init<>())
      .def_readwrite("kappa", &ObstacleBarrierParams::kappa);
  py::class_<DriftBarrierParams>(m, "DriftBarrierParams")
      .def(py::init<>())
      .def_readwrite("kappa", &DriftBarrierParams::kappa);
  py::class_<GapBarrierParams>(m, "GapBarrierParams")
      .def(py::init<>())
      .def_readwrite("t_headway", &GapBarrierParams::t_headway)
      .def_readwrite("d_min", &GapBarrierParams::d_min)
      .def_readwrite("kappa", &GapBarrierParams::kappa)
      .def_readwrite("lateral_gate", &GapBarrierParams::lateral_gate);
  py::class_<BarrierConfig>(m, "BarrierConfig")
      .def(py::init<>())
      .def_readwrite("road", &BarrierConfig::road)
      .def_readwrite("obstacle", &BarrierConfig::obstacle)
      .def_readwrite("drift", &BarrierConfig::drift)
      .def_readwrite("gap", &BarrierConfig::gap);
  m.def("load_barrier_config", &LoadBarrierConfig, py::arg("yaml_path"));

  py::class_<CbfQpConfig>(m, "CbfQpConfig")
      .def(py::init<>())
      .def_readwrite("H_delta", &CbfQpConfig::H_delta)
      .def_readwrite("H_ax", &CbfQpConfig::H_ax)
      .def_readwrite("use_road", &CbfQpConfig::use_road)
      .def_readwrite("use_drift", &CbfQpConfig::use_drift)
      .def_readwrite("use_gap", &CbfQpConfig::use_gap)
      .def_readwrite("slack_weight", &CbfQpConfig::slack_weight)
      .def_readwrite("use_slack", &CbfQpConfig::use_slack)
      .def_readwrite("use_slew", &CbfQpConfig::use_slew)
      .def_readwrite("dt", &CbfQpConfig::dt);
  m.def("load_cbf_qp_config", &LoadCbfQpConfig, py::arg("yaml_path"));

  py::class_<estimation::UkfConfig>(m, "UkfConfig")
      .def(py::init<>())
      .def_readwrite("alpha", &estimation::UkfConfig::alpha)
      .def_readwrite("beta", &estimation::UkfConfig::beta)
      .def_readwrite("kappa", &estimation::UkfConfig::kappa)
      .def_readwrite("mu_init", &estimation::UkfConfig::mu_init)
      .def_readwrite("mu_min", &estimation::UkfConfig::mu_min)
      .def_readwrite("mu_max", &estimation::UkfConfig::mu_max)
      .def_readwrite("R_prior", &estimation::UkfConfig::R_prior);
  m.def("load_ukf_config", &estimation::LoadUkfConfig, py::arg("yaml_path"));

  py::class_<ObstacleState>(m, "ObstacleState")
      .def(py::init<>())
      .def_readwrite("x", &ObstacleState::x)
      .def_readwrite("y", &ObstacleState::y)
      .def_readwrite("vx", &ObstacleState::vx)
      .def_readwrite("vy", &ObstacleState::vy)
      .def_readwrite("a", &ObstacleState::a)
      .def_readwrite("b", &ObstacleState::b);

  // -- Dynamics ---------------------------------------------------------------
  py::class_<PacejkaTire>(m, "PacejkaTire")
      .def(py::init<const PacejkaParams&, double>(), py::arg("params") = PacejkaParams{},
           py::arg("mu_nominal") = 0.9)
      .def("lateral_force", &PacejkaTire::lateralForce)
      .def("lateral_stiffness", &PacejkaTire::lateralStiffness)
      .def("cornering_stiffness", &PacejkaTire::corneringStiffness)
      .def_static("front_slip_angle", &PacejkaTire::frontSlipAngle)
      .def_static("rear_slip_angle", &PacejkaTire::rearSlipAngle);

  py::class_<DynamicBicycleModel>(m, "DynamicBicycleModel")
      .def(py::init<const VehicleParams&>())
      .def("continuous_dynamics",
           [](const DynamicBicycleModel& self, const Arr6& x, const Arr2& u, double mu) {
             StateDerivative xd;
             self.continuousDynamics(ToState(x), ToControl(u), mu, xd);
             return FromState(xd);
           })
      .def("step_rk4",
           [](const DynamicBicycleModel& self, const Arr6& x, const Arr2& u, double mu,
              double dt) {
             StateVector xn;
             self.stepRK4(ToState(x), ToControl(u), mu, dt, xn);
             return FromState(xn);
           })
      .def("tire_forces", [](const DynamicBicycleModel& self, const Arr6& x, double steer,
                              double mu) {
        double fyf = 0.0, fyr = 0.0;
        self.tireForces(ToState(x), steer, mu, fyf, fyr);
        return std::make_pair(fyf, fyr);
      });

  // -- Controllers ------------------------------------------------------------
  py::class_<LqrTracker>(m, "LqrTracker")
      .def(py::init<const VehicleParams&, const LqrConfig&>())
      .def("compute_gains", &LqrTracker::computeGains)
      .def("update",
           [](LqrTracker& self, const Arr4& err, double vx, double kappa, double vx_ref,
              double dt) {
             ControlVector u;
             self.update(ToLateral(err), vx, kappa, vx_ref, dt, u);
             return FromControl(u);
           })
      .def("reset_integrator", &LqrTracker::resetIntegrator)
      .def("gains_computed", &LqrTracker::gainsComputed);

  py::class_<MpcTracker>(m, "MpcTracker")
      .def(py::init<const VehicleParams&, const MpcConfig&>())
      .def("compute_gains", &MpcTracker::computeGains)
      .def("update",
           [](MpcTracker& self, const Arr4& err, double vx, double kappa, double vx_ref,
              double dt) {
             ControlVector u;
             self.update(ToLateral(err), vx, kappa, vx_ref, dt, u);
             return FromControl(u);
           })
      .def("set_curvature_preview",
           [](MpcTracker& self, const std::vector<double>& k) {
             self.setCurvaturePreview(k.data(), static_cast<int>(k.size()));
           })
      .def("clear_curvature_preview", &MpcTracker::clearCurvaturePreview)
      .def("reset", &MpcTracker::reset)
      .def("mpc_ready", &MpcTracker::mpcReady)
      .def("gains_computed", &MpcTracker::gainsComputed);

  py::class_<qp::QpSolverBackend>(m, "QpSolverBackend");  // abstract, non-constructible

  py::class_<qp::EnumerationQpSolver, qp::QpSolverBackend>(m, "EnumerationQpSolver")
      .def(py::init<>());
#ifdef HAVE_OSQP
  py::class_<qp::OsqpSolver, qp::QpSolverBackend>(m, "OsqpSolver")
      .def(py::init([](const std::array<double, kControlDim>& H) {
        return new qp::OsqpSolver(H.data());
      }))
      .def("ready", &qp::OsqpSolver::ready);
#endif

  py::class_<CbfQpFilter>(m, "CbfQpFilter")
      .def(py::init<const VehicleParams&, const BarrierConfig&, const CbfQpConfig&,
                    qp::QpSolverBackend*>(),
           py::keep_alive<1, 5>())
      .def("set_fallback_backend", &CbfQpFilter::setFallbackBackend, py::keep_alive<1, 2>())
      .def("set_obstacles", [](CbfQpFilter& self, const std::vector<ObstacleState>& obs) {
        self.setObstacles(obs.data(), static_cast<int>(obs.size()));
      });
  m.def(
      "cbf_filter_step",
      [](CbfQpFilter& f, const Arr6& x, const Arr2& u_nom, double mu) {
        ControlVector safe;
        FilterDiagnostics d;
        f.filter(ToState(x), ToControl(u_nom), mu, safe, d);
        return std::make_pair(FromControl(safe), DiagToDict(d));
      },
      py::arg("filter"), py::arg("x"), py::arg("u_nom"), py::arg("mu"));

  // -- Estimation ---------------------------------------------------------------
  py::class_<estimation::FrictionUkf>(m, "FrictionUkf")
      .def(py::init<const VehicleParams&, const estimation::UkfConfig&>())
      .def("reset", &estimation::FrictionUkf::reset)
      .def("predict", [](estimation::FrictionUkf& self, const Arr2& u, double dt) {
        self.predict(ToControl(u), dt);
      })
      .def("update", [](estimation::FrictionUkf& self, const Arr4& z, const Arr2& u) {
        estimation::UkfMeas zm;
        for (int i = 0; i < 4; ++i) zm(i) = z[static_cast<size_t>(i)];
        self.update(zm, ToControl(u));
      });
  m.def(
      "ukf_step",
      [](estimation::FrictionUkf& ukf, const Arr2& u, const Arr4& z, double dt) {
        estimation::UkfMeas zm;
        for (int i = 0; i < 4; ++i) zm(i) = z[static_cast<size_t>(i)];
        ukf.step(ToControl(u), zm, dt);
        return ukf.mu();
      },
      py::arg("ukf"), py::arg("u"), py::arg("z"), py::arg("dt"));
  m.def(
      "ukf_state",
      [](const estimation::FrictionUkf& ukf) {
        Arr4 a{};
        for (int i = 0; i < 4; ++i) a[static_cast<size_t>(i)] = ukf.state()(i);
        return a;
      },
      py::arg("ukf"));
}
