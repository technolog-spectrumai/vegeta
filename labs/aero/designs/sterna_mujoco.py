"""STERNA in MuJoCo (notebook 51): the lab's fixed-wing ``Sim`` and ``Autopilot`` (``arigo1_mujoco``) on STERNA's
coefficients, one simple mission and its video.

The aerodynamics are ``arigo1_model.aero`` on ``sterna_model.coefficients`` (lattice + hand estimates, or the CFD
fit): MuJoCo integrates the rigid body and the ground contact, it does not compute any flow. The autopilot's gains are
the lab's hand gains, scaled with speed; nothing here is a tuned controller.

The mission: a hand launch (released level at ``RELEASE_M_S``, full throttle), a climb to the survey height, a
straight leg, one orbit, a power-off glide with the propeller stopped, a flare and a belly landing on the rails.

In the video the propeller is the lab's bladed propeller design as two blades on the spinner, turning at the real
speed: the rpm the Boreas drive needs for the thrust and the airspeed of each moment (about 9500 rpm at full
throttle, about 5500 in cruise). The thrust in the simulation is the same map; the blades themselves are a picture. Each frame is
exposed with motion blur (``Recorder``, ``Sim.set_prop``). When the throttle closes the propeller stops and the blades fold back.
"""
from __future__ import annotations

import math
import subprocess
from pathlib import Path

import mujoco
import numpy as np

import arigo1_model as am
import arigo1_mujoco as amj
import sterna as st

PROP_STATION_MM = 26.0       # the propeller's plane from the spinner tip
HINGE_MM = 12.0              # each blade's fold hinge from the thrust line
GHOSTS = 16                  # the instants of one frame's exposure the blades are drawn at
BLADE_RGB = "0.80 0.80 0.78"  # light grey nylon blades
FOLD_DEG = 74.0              # folded: the blades lie back along the nose, clear of the cowl
RELEASE_M_S = 8.0            # HAND: an overarm throw of a 2 kg aircraft (+-2 m/s); the lab's ARIGO value is 12
RELEASE_HEIGHT_M = 1.9
SURVEY_HEIGHT_M = 40.0
CRUISE_M_S = 14.0
ORBIT_BANK_DEG = 30.0


class Sim(amj.Sim):
    """The lab's ``Sim`` on a given coefficient set (not on an ARIGO variant)."""

    def __init__(self, C: dict, mesh: str | None = None, blade: str | None = None):
        self.C = C
        xml = amj.mjcf(C, mesh).replace('rgba="0.75 0.72 0.62 1"', 'rgba="0.93 0.94 0.95 1"')
        self.hub = np.array([C["x_cg_m"] - PROP_STATION_MM / 1000.0, 0.0, -C["z_cg_m"]])      # body frame: x forward from the CG
        if blade:                                 # GHOSTS exposures of the two blades, each a massless body posed by ``set_prop``
            xml = xml.replace("<asset>", f'<asset><mesh name="blade" file="{blade}" scale="0.001 0.001 0.001"/>', 1)
            k = xml.index("</body>")
            xml = xml[:k] + "".join(f'<body name="blade{g}_{i}" pos="{self.hub[0]:.4f} 0 0"><geom name="blade{g}_{i}" type="mesh" mesh="blade" '
                                    f'density="0" contype="0" conaffinity="0" rgba="{BLADE_RGB} 1"/></body>' for g in range(GHOSTS) for i in (0, 1)) + xml[k:]
        self.m = mujoco.MjModel.from_xml_string(xml)
        ids = lambda kind, n: mujoco.mj_name2id(self.m, kind, n)
        self.blades = [[(ids(mujoco.mjtObj.mjOBJ_BODY, f"blade{g}_{i}"), ids(mujoco.mjtObj.mjOBJ_GEOM, f"blade{g}_{i}")) for i in (0, 1)]
                       for g in range(GHOSTS)] if blade else []
        self.prop = {"angle": 0.0, "rate": 0.0, "fold": math.radians(FOLD_DEG)}
        self.d = mujoco.MjData(self.m)
        self.bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "arigo")
        self.controls = np.array([0.0, 0.0, 0.0, 0.5])
        self.wind = np.zeros(3)
        self.info = {}

    def start(self, V=CRUISE_M_S, h=SURVEY_HEIGHT_M, heading_deg=0.0):
        t = am.trim(self.C, V)
        al = math.radians(t["alpha_deg"])
        mujoco.mj_resetData(self.m, self.d)
        ps = math.radians(heading_deg)
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, np.array([math.cos(ps / 2), 0, 0, math.sin(ps / 2)]), np.array([math.cos(-al / 2), 0, math.sin(-al / 2), 0]))
        self.d.qpos[:3] = (0, 0, h)
        self.d.qpos[3:7] = q
        self.d.qvel[:3] = (V * math.cos(ps), V * math.sin(ps), 0.0)
        self.controls[:] = (t["elevator_deg"], 0, 0, t["throttle"])
        self.trim = t
        mujoco.mj_forward(self.m, self.d)
        return t


    # ------------------------------------------------------------------------------------------ the propeller (looks)
    def set_prop(self, angle: float, fold: float, sweep: float = 0.0):
        """Pose the blades for one frame: ``angle`` about the thrust line at the shutter's opening, ``fold`` back about
        each blade's root hinge, ``sweep`` the angle turned while the shutter is open [rad].

        A frame's exposure is drawn in one render: the blade pair at GHOSTS instants of the exposure, each 1/GHOSTS
        opaque, which composites to the blur a camera records. With two blades, any sweep of half a turn or more covers
        the whole disc, so the instants are spread over half a turn. Under 6 degrees of sweep one solid pair is drawn.
        The blade's own frame is the lab propeller's: axis z, blade along x."""
        from scipy.spatial.transform import Rotation as Rot
        R0 = Rot.from_matrix([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])          # blade x -> body z, axis z -> body x
        h = np.array([0.0, 0.0, HINGE_MM / 1000.0])
        Rf = Rot.from_euler("y", -fold)
        blurred = abs(sweep) > math.radians(6.0)
        span = math.copysign(min(abs(sweep), math.pi), sweep)
        for g, pair in enumerate(self.blades):
            a = angle + (span * (g + 0.5) / GHOSTS if blurred else 0.0)
            for i, (b, geom) in enumerate(pair):
                Rs = Rot.from_euler("x", a + i * math.pi)
                self.m.body_pos[b] = self.hub + Rs.apply(h - Rf.apply(h))
                self.m.body_quat[b] = (Rs * Rf * R0).as_quat()[[3, 0, 1, 2]]
                self.m.geom_rgba[geom, 3] = 1.0 / GHOSTS if blurred else (1.0 if g == 0 else 0.0)

    def prop_rpm_target(self) -> float:
        """The propeller's speed [rpm] for the thrust the flight model is producing at this airspeed: the Boreas
        drive's map carried in the coefficients (``thrust['rpm_grid']``) read backwards, thrust -> rpm. Without a map,
        9500 rpm x throttle. Zero with the throttle closed (the propeller is stopped and folds)."""
        thr = float(self.controls[3])
        if thr <= 0.02:
            return 0.0
        g = self.C["thrust"].get("rpm_grid")
        if not g:
            return 9500.0 * thr
        V = float(np.clip(self.state()["V"], g["V"][0], g["V"][-1]))
        T = am.max_thrust(self.C, V) * min(thr, 1.0)
        rpm = [float(np.interp(V, g["V"], row)) for row in g["rpm"]]                 # over the drive's throttle, at this airspeed
        thrust = [float(np.interp(V, g["V"], row)) for row in g["thrust_N"]]
        return float(np.interp(T, thrust, rpm))

    def advance_prop(self, dt: float):
        """The propeller's state over ``dt``: its speed follows the target (the drive's inertia: 0.3 s), the angle
        turns at that speed; stopped, the blades fold back along the nose. Returns the speed [rad/s] during ``dt``."""
        q = self.prop
        q["rpm"] = q.get("rpm", 0.0) + (self.prop_rpm_target() - q.get("rpm", 0.0)) * min(1.0, dt / 0.3)
        w = q["rpm"] * 2 * math.pi / 60.0
        want = math.radians(FOLD_DEG) if q["rpm"] < 200.0 else 0.0
        step = math.radians(240.0) * dt
        q["fold"] += max(-step, min(step, want - q["fold"]))
        q["angle0"] = q["angle"]
        q["angle"] = (q["angle"] + w * dt) % (2 * math.pi)
        return w


class Autopilot(amj.Autopilot):
    """The lab's autopilot with its speed scaling referred to STERNA's cruise (its gains were set at 18 m/s)."""


def export_blade(folder) -> str:
    """One blade of the 11x7 folding propeller with its root, from the lab's propeller design (``dedalus.examples``):
    the planform ``sterna_model.propulsion`` gives Boreas. Axis z, blade along x; mm."""
    from vegeta.dedalus.examples import Propeller
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "sterna_blade.stl"
    if not path.exists():
        Propeller().generate(diameter=280.0, pitch=7 * 25.4, blades=1, hub_diameter=16.0, hub_height=10.0, bore=0.0, chord_root=18.0,
                             chord_max=26.0, chord_tip=8.0, stations=14).export_stl(str(path), tolerance=0.1)
    return str(path)


def export_mesh(folder) -> str:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "sterna.stl"
    if not path.exists():
        st.Sterna().generate().export_stl(str(path), tolerance=0.5)
    return str(path)


class Recorder:
    """A chase camera into an mp4 (ffmpeg, H.264). Needs an OpenGL context (``MUJOCO_GL=egl`` or ``osmesa``).

    The propeller turns at its real speed. At 30 frames per second that is several turns per frame, so each frame is
    exposed like a camera's: the blades are drawn over the angle they turn while the shutter is open (``shutter`` of
    the frame time: 0.5 is a 180 degree shutter), see ``Sim.set_prop``. At speed they come out as the faint disc a
    camera sees; slowing down, as a smear; stopped, as blades."""

    def __init__(self, sim, path, size=(960, 540), fps=30, distance=2.6, shutter=0.5):
        self.sim, self.fps, self.size, self.shutter, self.rpm = sim, fps, size, shutter, []
        self.r = mujoco.Renderer(sim.m, size[1], size[0])
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.elevation = distance, -16.0
        self.every = max(1, round(1 / (fps * amj.DT * 5)))
        self.k, self.psi, self.frames, self.last = 0, None, 0, None
        self.ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
                                    "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(path)], stdin=subprocess.PIPE)

    def __call__(self, sim):
        self.k += 1
        if self.k % self.every:
            return
        s = sim.state()
        vh = s["v_world"][:2]
        psi = math.degrees(math.atan2(vh[1], vh[0])) if np.linalg.norm(vh) > 1 else (self.psi or 0.0)
        self.psi = psi if self.psi is None else self.psi + 0.15 * ((psi - self.psi + 180) % 360 - 180)
        self.cam.lookat[:] = s["pos"]
        self.cam.azimuth = self.psi + 32.0                     # behind and to one side, looking along the flight path
        if sim.blades:
            w = sim.advance_prop(1.0 / self.fps)
            self.rpm.append(sim.prop["rpm"])
            sim.set_prop(sim.prop["angle0"], sim.prop["fold"], sweep=w * self.shutter / self.fps)
            mujoco.mj_kinematics(sim.m, sim.d)
            self.r.update_scene(sim.d, camera=self.cam)
            self.last = self.r.render()
        else:
            self.r.update_scene(sim.d, camera=self.cam)
            self.last = self.r.render()
        self.ff.stdin.write(self.last.tobytes())
        self.frames += 1

    def close(self):
        self.ff.stdin.close()
        self.ff.wait()


def mission(sim: Sim, record=None, release=RELEASE_M_S, wind=(0.0, 0.0, 0.0)) -> tuple:
    """Fly the mission; returns (history, summary). The phases are switched on height, heading and time, by hand."""
    C = sim.C
    Vs = math.sqrt(C["mass"] * am.G / (0.5 * am.RHO * C["S"] * C["CL_MAX"]))
    sim.start(max(release, 1.15 * Vs), RELEASE_HEIGHT_M)
    sim.wind[:] = wind
    sim.d.qvel[:3] = (release, 0.0, 0.0)
    mujoco.mj_forward(sim.m, sim.d)
    ap = Autopilot(sim)
    st_ = {"phase": "launch", "t_phase": 0.0, "turned": 0.0, "psi": None, "t_safe": None, "touch": None, "log": []}

    def go(name, t):
        st_["log"].append((round(t, 1), name))
        st_["phase"], st_["t_phase"] = name, t

    def fn(t, s):
        sd = s.state()
        h, V, ph = sd["pos"][2], sd["V"], st_["phase"]
        if ph == "launch":
            ap(phi_c=0.0, theta_c=math.radians(10.0) if V > 1.2 * Vs else math.radians(2.0), throttle=1.0)
            if h > 10.0 and V > 1.3 * Vs:
                st_["t_safe"] = t
                go("climb", t)
        elif ph == "climb":
            ap(phi_c=0.0, theta_c=math.radians(10.0), throttle=1.0)
            if h > SURVEY_HEIGHT_M - 2.0:
                go("leg", t)
        elif ph == "leg":
            ap(phi_c=0.0, theta_c=math.radians(0.5) + 0.03 * (SURVEY_HEIGHT_M - h), V_c=CRUISE_M_S)
            if t - st_["t_phase"] > 6.0:
                st_["psi"] = sd["psi"]
                go("orbit", t)
        elif ph == "orbit":
            ap(phi_c=math.radians(ORBIT_BANK_DEG), theta_c=math.radians(1.5) + 0.03 * (SURVEY_HEIGHT_M - h), V_c=CRUISE_M_S)
            d = (sd["psi"] - st_["psi"] + math.pi) % (2 * math.pi) - math.pi
            st_["turned"] += abs(d)
            st_["psi"] = sd["psi"]
            if st_["turned"] > 2 * math.pi:
                go("glide", t)
        elif ph == "glide":                                        # propeller stopped: speed held on pitch
            ap(phi_c=0.0, theta_c=math.radians(-4.0) + 0.02 * (1.35 * Vs - V), throttle=0.0)
            if h < 2.5:
                go("flare", t)
        elif ph == "flare":
            ap(phi_c=0.0, theta_c=math.radians(3.0), throttle=0.0, protect=False)
            if h < 0.16 and st_["touch"] is None:
                st_["touch"] = {"t": t, "V": V, "sink": -float(sd["v_world"][2])}
                go("slide", t)
        else:
            s.controls[:] = (0.0, 0.0, 0.0, 0.0)

    hist = {k: [] for k in ("t", "V", "z", "x", "y", "phi", "theta", "alpha", "n", "throttle", "phase")}
    t0 = sim.d.time
    while sim.d.time - t0 < 150.0:
        t = sim.d.time - t0
        fn(t, sim)
        sim.step(5)
        s, i = sim.state(), sim.info
        for k, v in (("t", t), ("V", s["V"]), ("z", s["pos"][2]), ("x", s["pos"][0]), ("y", s["pos"][1]), ("phi", math.degrees(s["phi"])),
                     ("theta", math.degrees(s["theta"])), ("alpha", math.degrees(i["alpha"])), ("n", i["load_factor"]), ("throttle", sim.controls[3])):
            hist[k].append(float(v))
        hist["phase"].append(st_["phase"])
        if record is not None:
            record(sim)
        if st_["phase"] == "slide" and (t - st_["t_phase"] > 3.0 or float(np.linalg.norm(s["v_world"])) < 0.3):
            break
        if st_["phase"] in ("launch", "climb") and s["pos"][2] < 0.12:
            st_["log"].append((round(t, 1), "ground contact after launch"))
            break
    H = {k: (np.array(v) if k != "phase" else v) for k, v in hist.items()}
    launch = H["t"] < (st_["t_safe"] or 5.0)
    orbit = np.array([p == "orbit" for p in H["phase"]])
    summary = {"mass_kg": round(C["mass"], 3), "stall_m_s": round(Vs, 2), "release_m_s": release,
               "launch_min_height_m": round(float(H["z"][launch].min()), 2), "time_to_safe_s": None if st_["t_safe"] is None else round(st_["t_safe"], 1),
               "orbit_radius_m": round(float(np.mean(H["V"][orbit]) ** 2 / (am.G * math.tan(math.radians(ORBIT_BANK_DEG)))), 1) if orbit.any() else None,
               "orbit_load_factor": round(float(np.mean(H["n"][orbit])), 2) if orbit.any() else None,
               "touchdown": None if st_["touch"] is None else {k: round(v, 2) for k, v in st_["touch"].items()},
               "duration_s": round(float(H["t"][-1]), 1), "phases": st_["log"],
               "note": "aerodynamics from sterna_model (lattice, hand estimates, CFD fit when given); hand autopilot gains; still air unless wind is set"}
    return H, summary
