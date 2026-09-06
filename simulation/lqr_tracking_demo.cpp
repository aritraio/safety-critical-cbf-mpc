// Standalone Phase-1 verification: LQR sine-road tracking with the nonlinear
// Pacejka bicycle model. No ROS 2 required.
//
// Usage:
//   lqr_tracking_demo [vehicle_yaml] [lqr_yaml]
// Defaults to ../config/*.yaml relative to CWD. Prints CSV to stdout:
//   t, x, y, psi, vx, vy, r, ey, epsi, delta, ax
// Summary (RMS/max lateral error) goes to stderr.

#include <cmath>
#include <cstdio>
#include <string>

#include "av_safety/common/vehicle_params.hpp"
#include "av_safety/controllers/lqr_tracker.hpp"
#include "av_safety/dynamics/dynamic_bicycle_model.hpp"

namespace {
/// Sinusoidal centerline: y_ref(s) = A sin(2 pi s / L). Curvature via analytic
/// second derivative (small-slope approx kappa ~= y'').
struct SineRoad {
  double amplitude{2.0};
  double wavelength{200.0};
  double y(double s) const { return amplitude * std::sin(2.0 * M_PI * s / wavelength); }
  double heading(double s) const {
    const double dyds = amplitude * 2.0 * M_PI / wavelength * std::cos(2.0 * M_PI * s / wavelength);
    return std::atan(dyds);
  }
  double curvature(double s) const {
    const double k = 2.0 * M_PI / wavelength;
    return -amplitude * k * k * std::sin(k * s);  // exact for y(s) param approx
  }
};
}  // namespace

int main(int argc, char** argv) {
  const std::string vehicle_yaml = argc > 1 ? argv[1] : "config/vehicle_params.yaml";
  const std::string lqr_yaml = argc > 2 ? argv[2] : "config/lqr_tuning.yaml";

  av_safety::VehicleParams vp;
  av_safety::LqrConfig cfg;
  try {
    vp = av_safety::LoadVehicleParams(vehicle_yaml);
    cfg = av_safety::LoadLqrConfig(lqr_yaml);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "config error: %s\n", e.what());
    return 1;
  }

  av_safety::DynamicBicycleModel model(vp);
  av_safety::LqrTracker tracker(vp, cfg);
  const double mu = vp.mu_nominal;
  if (!tracker.computeGains(cfg.target_speed, mu)) {
    std::fprintf(stderr, "CARE solve failed at vx=%.1f\n", cfg.target_speed);
    return 1;
  }
  std::fprintf(stderr, "K = [%.4f %.4f %.4f %.4f]\n", tracker.gain()(0, 0), tracker.gain()(0, 1),
               tracker.gain()(0, 2), tracker.gain()(0, 3));

  SineRoad road;
  const double dt = 0.01;  // 100 Hz
  const double T = 30.0;
  av_safety::StateVector x = av_safety::StateVector::Zero();
  x(av_safety::kVx) = cfg.target_speed;

  double sum_sq = 0.0, max_ey = 0.0;
  int n = 0;
  std::printf("t,x,y,psi,vx,vy,r,ey,epsi,delta,ax\n");
  for (double t = 0.0; t < T; t += dt) {
    // Closest-point approx by arc length s ~= x (gentle road, small heading).
    const double s = x(av_safety::kPx);
    const double y_ref = road.y(s);
    const double psi_ref = road.heading(s);
    double epsi = x(av_safety::kPsi) - psi_ref;
    while (epsi > M_PI) epsi -= 2 * M_PI;
    while (epsi < -M_PI) epsi += 2 * M_PI;
    const double ey = (x(av_safety::kPy) - y_ref) * std::cos(psi_ref);

    av_safety::LateralState err;
    err << ey, epsi, x(av_safety::kVy), x(av_safety::kYawRate);
    av_safety::ControlVector u;
    tracker.update(err, x(av_safety::kVx), road.curvature(s), cfg.target_speed, dt, u);
    av_safety::StateVector xn;
    model.stepRK4(x, u, mu, dt, xn);
    x = xn;

    sum_sq += ey * ey;
    max_ey = std::max(max_ey, std::abs(ey));
    ++n;
    std::printf("%.3f,%.4f,%.4f,%.5f,%.3f,%.4f,%.5f,%.4f,%.5f,%.5f,%.3f\n", t, x(0), x(1), x(2),
                x(3), x(4), x(5), ey, epsi, u(0), u(1));
  }
  std::fprintf(stderr, "RMS lateral error: %.4f m | max: %.4f m | N=%d\n", std::sqrt(sum_sq / n),
               max_ey, n);
  return 0;
}
