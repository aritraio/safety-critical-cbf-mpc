"""World abstraction for SIL benchmarks: MockWorld (deterministic, CI-friendly,
driven by the TRUE C++ Pacejka model through bindings) and CarlaWorld (real
CARLA server, Town04 highway; requires the `carla` package + a running
server with the desired map).

Both expose an identical interface so run_benchmarks.py is world-agnostic:
    world.reset(scenario)          # scenario: dict from scenarios.py
    world.step(delta, ax, dt)      # advance one control step
    world.ego_state                # dict(x,y,psi,vx,vy,r)
    world.obstacles                # list of dicts(x,y,vx,vy,a,b)
    world.imu                      # dict(ax,ay,r,vx) latest measurement
    world.true_mu(x, y)            # plant friction at a position
    world.collision                # bool latch (CARLA sensor / ellipse contact)
    world.time
    world.close()
"""

import math

try:
    import carla  # type: ignore

    _HAVE_CARLA = True
except Exception:  # pragma: no cover - import guard
    carla = None
    _HAVE_CARLA = False


def make_world(kind, av, vehicle_params, **kwargs):
    """Factory: kind in {"mock", "carla"}."""
    if kind == "mock":
        return MockWorld(av, vehicle_params, **kwargs)
    if kind == "carla":
        return CarlaWorld(vehicle_params, **kwargs)
    raise ValueError(f"unknown world kind: {kind}")


def scripted_rates(o, t):
    """Piecewise obstacle script shared by both worlds.

    segments = [[t_end, vx, vy], ...]; before/during/after fall back to the
    obstacle's constant (vx, vy). Advances o['_seg_idx'] in place.
    """
    segs = o.get("segments")
    if segs:
        while o["_seg_idx"] < len(segs) and t >= segs[o["_seg_idx"]][0]:
            o["_seg_idx"] += 1
        if o["_seg_idx"] < len(segs):
            return segs[o["_seg_idx"]][1], segs[o["_seg_idx"]][2]
        last = segs[-1]
        return last[1], last[2]
    return o.get("vx", 0.0), o.get("vy", 0.0)


class MockWorld:
    """Deterministic kinematic shell around the TRUE C++ dynamics.

    Scripted obstacles follow constant-velocity segments; plant friction comes
    from scenario['mu_field'](x, y, t). IMU is synthesized from the truth
    state with caller-supplied noise (see run_benchmarks.py).
    """

    def __init__(self, av, vehicle_params, **kwargs):
        self._av = av
        self._model = av.DynamicBicycleModel(vehicle_params)
        self._ego = [0.0, 0.0, 0.0, 10.0, 0.0, 0.0]
        self._obs = []
        self._t = 0.0
        self._mu_field = lambda x, y, t: 0.9
        self._collision = False
        # Discrete-time contact tolerance: at 100 Hz / 12 m/s one step covers
        # 0.12 m, so sub-step boundary grazes are ripple, not contact. Matches
        # the repo's h_obs >= -0.05 demo gate (here tighter: -0.02).
        self.contact_margin = -0.02

    def reset(self, scenario):
        e = scenario["ego_init"]
        self._ego = [e["x"], e["y"], e["psi"], e["vx"], 0.0, 0.0]
        self._obs = [dict(o) for o in scenario.get("obstacles", [])]
        for o in self._obs:
            o.setdefault("segments", None)
            o["_seg_idx"] = 0
        self._mu_field = scenario.get("mu_field", lambda x, y, t: 0.9)
        self._t = 0.0
        self._collision = False

    def _obstacle_rates(self, o):
        return scripted_rates(o, self._t)

    def step(self, delta, ax, dt):
        mu = self._mu_field(self._ego[0], self._ego[1], self._t)
        self._ego = list(
            self._model.step_rk4(tuple(self._ego), (delta, ax), mu, dt)
        )
        for o in self._obs:
            vx, vy = self._obstacle_rates(o)
            o["x"] += vx * dt
            o["y"] += vy * dt
            o["vx"], o["vy"] = vx, vy  # publish scripted rates (filters need them)
        self._t += dt
        for o in self._obs:  # ellipse contact latch (mock collision proxy)
            dx = (self._ego[0] - o["x"]) / max(o.get("a", 4.0), 1e-6)
            dy = (self._ego[1] - o["y"]) / max(o.get("b", 2.0), 1e-6)
            if dx * dx + dy * dy - 1.0 < self.contact_margin:
                self._collision = True

    @property
    def ego_state(self):
        x = self._ego
        return {"x": x[0], "y": x[1], "psi": x[2], "vx": x[3], "vy": x[4], "r": x[5]}

    @property
    def obstacles(self):
        return [dict(o) for o in self._obs]

    @property
    def imu(self):
        # Mock has no sensors; run_benchmarks synthesizes IMU from truth.
        s = self.ego_state
        return {"ax": 0.0, "ay": 0.0, "r": s["r"], "vx": s["vx"]}

    def true_mu(self, x, y):
        return self._mu_field(x, y, self._t)

    @property
    def collision(self):
        return self._collision

    @property
    def time(self):
        return self._t

    def close(self):
        pass


class CarlaWorld:
    """Live CARLA bridge (Town04 highway by default).

    Ego: spawned sedan, VehicleControl from (delta, ax). Obstacles: spawned
    vehicles repositioned every tick from the scripted scenario motion
    (deterministic cut-ins, matching MockWorld semantics). IMU + collision
    sensors attached to ego. Requires a running server: e.g.
    `SDL_VIDEODRIVER=offscreen ./CarlaUE4.sh -RenderOffScreen -carla-port=2000`
    with Town04 loaded.
    """

    STEER_GAIN = 0.6  # [rad] full-lock mapping for steer [-1, 1]; calibrate per car

    def __init__(self, vehicle_params, host="localhost", port=2000, dt=0.01,
                 town="Town04", ego_bp="vehicle.tesla.model3", obs_bp="vehicle.audi.a2",
                 timeout=20.0, **kwargs):
        if not _HAVE_CARLA:
            raise RuntimeError(
                "carla package not importable. Install the CARLA PythonAPI "
                "(matching server version) or use --world mock."
            )
        self._dt = dt
        self._client = carla.Client(host, port)
        self._client.set_timeout(timeout)
        if self._client.get_world().get_map().name != town:
            self._client.load_world(town)
        self._world = self._client.get_world()
        self._bp_lib = self._world.get_blueprint_library()
        self._ego_bp_name = ego_bp
        self._obs_bp_name = obs_bp
        self._ego = None
        self._actors = []
        self._imu = None
        self._collision_flag = False
        self._t = 0.0
        self._mu_nominal = 0.9
        # Synchronous fixed-step mode for determinism.
        settings = self._world.get_settings()
        self._old_settings = settings
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = dt
        self._world.apply_settings(settings)

    # -- lifecycle -----------------------------------------------------------
    def reset(self, scenario):
        self._destroy_actors()
        self._t = 0.0
        self._collision_flag = False
        self._mu_nominal = scenario.get("mu_nominal", 0.9)
        e = scenario["ego_init"]
        spawn = self._scenario_spawn(scenario, e)
        ego_bp = self._bp_lib.find(self._ego_bp_name)
        self._ego = self._world.spawn_actor(ego_bp, spawn)
        self._actors.append(self._ego)
        self._ego.set_target_velocity(carla.Vector3D(e.get("vx", 10.0), 0, 0))
        # IMU + collision sensors.
        imu_bp = self._bp_lib.find("sensor.other.imu")
        imu = self._world.spawn_actor(
            imu_bp, carla.Transform(), attach_to=self._ego)
        imu.listen(self._on_imu)
        self._actors.append(imu)
        col_bp = self._bp_lib.find("sensor.other.collision")
        col = self._world.spawn_actor(
            col_bp, carla.Transform(), attach_to=self._ego)
        col.listen(self._on_collision)
        self._actors.append(col)
        # Scripted traffic.
        self._obs = [dict(o) for o in scenario.get("obstacles", [])]
        for o in self._obs:
            o.setdefault("segments", None)
            o["_seg_idx"] = 0
        obs_bp = self._bp_lib.find(self._obs_bp_name)
        for o in self._obs:
            t = carla.Transform(
                carla.Location(x=o["x"], y=-o["y"], z=0.5),
                carla.Rotation(yaw=-math.degrees(o.get("psi", 0.0))),
            )
            a = self._world.spawn_actor(obs_bp, t)
            a.set_autopilot(False)
            self._actors.append(a)
            o["_actor"] = a
        self._world.tick()

    def _scripted_rates(self, o):
        # Piecewise script shared with MockWorld (see scripted_rates).
        return scripted_rates(o, self._t)

    def _scenario_spawn(self, scenario, e):
        if "spawn" in scenario:
            s = scenario["spawn"]
            return carla.Transform(
                carla.Location(x=s["x"], y=-s.get("y", 0.0), z=0.5),
                carla.Rotation(yaw=-math.degrees(s.get("yaw", 0.0))),
            )
        # Fallback: map spawn point closest to the requested ego pose.
        best, best_d = None, 1e18
        for sp in self._world.get_map().get_spawn_points():
            d = (sp.location.x - e["x"]) ** 2 + (sp.location.y + e["y"]) ** 2
            if d < best_d:
                best, best_d = sp, d
        return best

    def step(self, delta, ax, dt):
        ctrl = carla.VehicleControl()
        ctrl.steer = max(-1.0, min(1.0, delta / self.STEER_GAIN))
        ctrl.throttle = max(0.0, min(1.0, ax / 3.0)) if ax >= 0 else 0.0
        ctrl.brake = max(0.0, min(1.0, -ax / 8.0)) if ax < 0 else 0.0
        ctrl.manual_gear_shift = False
        self._ego.apply_control(ctrl)
        # Scripted adversaries follow the scenario motion exactly.
        for o in self._obs:
            vx, vy = self._scripted_rates(o)
            o["x"] += vx * dt
            o["y"] += vy * dt
            o["vx"], o["vy"] = vx, vy
            a = o.get("_actor")
            if a is not None and a.is_alive:
                a.set_transform(
                    carla.Transform(
                        carla.Location(x=o["x"], y=-o["y"], z=0.5),
                        carla.Rotation(yaw=-math.degrees(o.get("psi", 0.0))),
                    )
                )
                a.set_target_velocity(
                    carla.Vector3D(o.get("vx", 0.0), -o.get("vy", 0.0), 0)
                )
        self._world.tick()
        self._t += dt

    # -- sensors ---------------------------------------------------------------
    def _on_imu(self, meas):
        # Vehicle-frame specific force (flat road: gravity only in z).
        self._imu = {
            "ax": meas.accelerometer.x,
            "ay": meas.accelerometer.y,
            "r": meas.gyroscope.z,
            "vx": None,  # filled from odometry below
        }

    def _on_collision(self, event):
        self._collision_flag = True

    @property
    def ego_state(self):
        t = self._ego.get_transform()
        v = self._ego.get_velocity()
        yaw = -math.radians(t.rotation.yaw)
        # World velocity -> body frame.
        vx = math.cos(yaw) * v.x - math.sin(yaw) * v.y
        vy = math.sin(yaw) * v.x + math.cos(yaw) * v.y
        av = self._ego.get_angular_velocity()
        return {
            "x": t.location.x,
            "y": -t.location.y,
            "psi": yaw,
            "vx": vx,
            "vy": vy,
            "r": -math.radians(av.z),
        }

    @property
    def obstacles(self):
        out = []
        for o in self._obs:
            a = o.get("_actor")
            if a is not None and a.is_alive:
                t = a.get_transform()
                v = a.get_velocity()
                out.append({
                    "x": t.location.x, "y": -t.location.y,
                    "vx": v.x, "vy": -v.y,
                    "a": o.get("a", 4.0), "b": o.get("b", 2.0),
                })
            else:
                out.append({k: o.get(k, 0.0) for k in ("x", "y", "vx", "vy", "a", "b")})
        return out

    @property
    def imu(self):
        m = dict(self._imu) if self._imu else {"ax": 0.0, "ay": 0.0, "r": 0.0, "vx": None}
        if m["vx"] is None:
            m["vx"] = self.ego_state["vx"]  # wheel-speed proxy when IMU lags a tick
        return m

    def true_mu(self, x, y):
        return self._mu_nominal  # CARLA tire friction is baked into the build

    @property
    def collision(self):
        return self._collision_flag

    @property
    def time(self):
        return self._t

    def _destroy_actors(self):
        for a in self._actors:
            try:
                if a.is_alive:
                    a.destroy()
            except Exception:
                pass
        self._actors = []
        self._ego = None

    def close(self):
        self._destroy_actors()
        try:
            self._world.apply_settings(self._old_settings)
        except Exception:
            pass
