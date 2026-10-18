"""The MuJoCo run (not executed in this session: written against the scene files and the existing controller, with
the same referee and the same result format as the mock; run it yourself with ``python -m races run --course farm``).

The reference quad is the free body in the scene (``refquad.mjcf_body``); every control step the rotor hook puts the
four thrusts (first-order lag, the drive table's limit at the axial inflow, the propellers' drag torques) and the
frame's drag on it through ``data.xfrc_applied``; Drongo's geometric flight controller (``drongo_controller.Flight``:
position → attitude → mixer) follows the racing line's moving reference; the referee reads the contacts from
``data.contact`` and the position; a reset teleports the free joint to the checkpoint. Cameras: ``overview``,
``grandstand`` (scene), ``onboard``, ``chase`` (quad); ``render`` makes frames from a recorded run with
``mujoco.Renderer`` (needs an OpenGL context: EGL or GLFW)."""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path as _P

os.environ.setdefault("MUJOCO_GL", "egl")             # the renderer's backend, read when mujoco is first imported
from types import SimpleNamespace

import numpy as np

from . import courses, refquad
from .mock import schedule
from .path import carrot, resync
from .rules import Referee, Rules

G = 9.81
DESIGNS = _P(__file__).resolve().parents[1] / "notebooks" / "designs"


def _flight(prop, *, omega_pos: float = 2.5, omega_att: float = 18.0, tilt_max_deg: float = 60.0):
    """Drongo's ``Flight`` with this quad's layout (its reset reads the body's inertia from the model)."""
    if str(DESIGNS) not in sys.path:
        sys.path.insert(0, str(DESIGNS))
    import drongo_controller as dc

    class QuadFlight(dc.Flight):
        def reset(self, model, data, root_jnt):            # the same maths as dc.Flight.reset without a ChironLab
            body = int(model.jnt_bodyid[root_jnt])          # one free body: its mass and principal inertia come straight from the model (no mass matrix, no MuJoCo-version API)
            self.m0 = float(model.body_subtreemass[body])
            self.J = np.array(model.body_inertia[body], dtype=float).copy()
            w, z = self.omega_att, self.zeta_att
            self.kR = np.array([self.J[0] * w * w, self.J[1] * w * w, self.J[2] * self.omega_yaw ** 2])
            self.kW = np.array([2 * z * self.J[0] * w, 2 * z * self.J[1] * w, 2 * 0.9 * self.J[2] * self.omega_yaw])
            pos, spin = refquad.rotor_layout()
            self.A = np.vstack([np.ones(4), pos[:, 1], -pos[:, 0], spin * self.prop.torque_per_thrust_m])
            self.Ainv = np.linalg.inv(self.A)
            self.integral = np.zeros(3)
            self.last = {}
    return QuadFlight(prop, omega_pos=omega_pos, tilt_max_deg=tilt_max_deg, omega_att=omega_att)


def run_mujoco(course, *, rules: Rules | None = None, control_dt: float = 0.004, lookahead_s: float = 0.25, duration: float | None = None, scene_xml: str | None = None,
               record_every: int = 5, safety: float = 0.6, v_cap: float | None = None, omega_pos: float = 2.5, omega_att: float = 18.0, tilt_max_deg: float = 60.0) -> dict:
    """The MuJoCo lap. The defaults are the first real laps' lesson (the hangar, 2026-10-18): the geometric controller
    lags the racing line by metres where the mock lagged by 0.25 m, so the schedule runs at ``safety`` 0.6 of the
    thrust-vector corner speed, the look-ahead is short and the tilt is capped at 60°; raise them course by course
    (``wf R3 --safety 0.8 --tilt-max 70``) once a lap is clean."""
    import mujoco
    rules = rules or Rules()
    xml = scene_xml or courses.to_mjcf(course)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "quad_free")
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "quad")
    qa, va = int(model.jnt_qposadr[jnt]), int(model.jnt_dofadr[jnt])
    quad_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == body}
    prop = refquad.propulsion()
    flight = _flight(prop, omega_pos=omega_pos, omega_att=omega_att, tilt_max_deg=tilt_max_deg)
    path = schedule(course, safety=safety, v_cap=v_cap)
    path.zone_half = np.array([min(z.half) for z in course.zones])
    ref = Referee(course, rules)
    pos_r, spin = refquad.rotor_layout()
    start = np.array(course.zones[0].center, dtype=float)
    data.qpos[qa:qa + 3] = start
    data.qpos[qa + 3:qa + 7] = (1, 0, 0, 0)
    mujoco.mj_forward(model, data)
    flight.reset(model, data, jnt)
    import drongo_controller as dc
    thrust = np.zeros(4)
    alpha = 1.0 - math.exp(-control_dt / prop.spin_up_s)
    n_sub = max(int(round(control_dt / model.opt.timestep)), 1)
    s, t, hist, events, energy_J, hover_until = 0.0, 0.0, [], [], 0.0, 1.0
    duration = duration or min(1.5 * path.time_estimate() + 30.0, rules.timeout_s + 10.0)
    k = 0
    while data.time < duration and ref.status == "racing":
        p = data.qpos[qa:qa + 3].copy()
        v = data.qvel[va:va + 3].copy()
        if ref.reset_request is not None:                                  # a reset: the free joint to the checkpoint, at rest, level
            data.qpos[qa:qa + 3] = ref.reset_request[0]; data.qpos[qa + 3:qa + 7] = (1, 0, 0, 0); data.qvel[va:va + 6] = 0.0
            hover_until = ref.reset_request[1]; s = path.s[ref.last_checkpoint]; flight.reset_integral(); ref.reset_request = None
            mujoco.mj_forward(model, data); p, v = data.qpos[qa:qa + 3].copy(), np.zeros(3)
        if data.time < hover_until:
            r = dc.Ref(p=np.array(course.zones[ref.last_checkpoint].center if ref.resets else start, dtype=float), v=np.zeros(3), a=np.zeros(3), yaw=0.0)
        else:
            s = path.project(p, s)
            nz = min(ref.next, len(path.s) - 1)
            s, parked = resync(path, s, path.s[nz], nz >= len(path.s) - 1)
            p_ref, v_ref, a_ref = carrot(path, s, nz, lookahead_s=lookahead_s, parked=parked)
            yaw = math.atan2(v_ref[1], v_ref[0]) if np.linalg.norm(v_ref[:2]) > 0.5 else math.atan2(v[1], v[0]) if np.linalg.norm(v[:2]) > 0.5 else 0.0
            r = dc.Ref(p=p_ref, v=v_ref, a=a_ref, yaw=yaw)
        R = data.xmat[body].reshape(3, 3)
        obs = SimpleNamespace(base_pos=p, base_quat=data.qpos[qa + 3:qa + 7].copy(), body_linvel=[v], body_angvel=[data.qvel[va + 3:va + 6].copy()])
        cmd = flight.thrusts(obs, r, 0.0, control_dt)
        zb = R[:, 2]
        v_axial = max(float(np.dot(v, zb)), 0.0)
        cmd = np.clip(cmd, 0.0, prop.drive.t_max(v_axial))
        thrust += (cmd - thrust) * alpha
        arms = (pos_r - np.array([0.0, 0.0, 0.0])) @ R.T
        force = zb * thrust.sum() - 0.5 * refquad.RHO * refquad.CDA_FORWARD_M2 * np.linalg.norm(v) * v
        torque = np.cross(arms, np.outer(thrust, zb)).sum(axis=0) + zb * float(np.dot(spin, thrust)) * prop.torque_per_thrust_m
        data.xfrc_applied[body, :3] = force
        data.xfrc_applied[body, 3:] = torque
        for _ in range(n_sub):
            mujoco.mj_step(model, data)
        P = sum(prop.drive.power(float(tt), v_axial) for tt in thrust) + refquad.FPV_W
        energy_J += P * control_dt
        contact = any(int(data.contact[i].geom1) in quad_geoms or int(data.contact[i].geom2) in quad_geoms for i in range(data.ncon))
        events += ref.update(float(data.time), data.qpos[qa:qa + 3].copy(), contact, data.time > hover_until)
        if k % record_every == 0:                                             # + the attitude and the body rates (races.metrics reads them)
            hist.append([data.time, *data.qpos[qa:qa + 3], *data.qvel[va:va + 3], float(thrust.sum()), P, float(np.linalg.norm(data.qvel[va:va + 3])), *data.qpos[qa + 3:qa + 7], *data.qvel[va + 3:va + 6]])
        k += 1
    H = np.array(hist).reshape(-1, 17)
    st = ref.stats()
    st.update({"course": course.name, "aircraft": "reference 5-inch quad", "mode": "mujoco", "energy [Wh]": energy_J / 3600, "Wh per km": energy_J / 3600 / (course.length() / 1e3),
               "max speed [m/s]": float(H[:, 9].max()) if len(H) else float("nan"), "mean speed [m/s]": course.length() / st["lap time [s]"] if st["status"] == "finished" else float("nan"),
               "peak power [W]": float(H[:, 8].max()) if len(H) else float("nan"), "mean power [W]": float(H[:, 8].mean()) if len(H) else float("nan"), "course length [m]": course.length(),
               "schedule time [s]": path.time_estimate(), "racing time [s]": float(data.time),
               "controller": {"omega_pos": omega_pos, "omega_att": omega_att, "tilt_max_deg": tilt_max_deg, "lookahead_s": lookahead_s, "safety": safety, "v_cap": v_cap}})
    return {"stats": st, "history": H, "events": events, "gate times": ref.gate_times, "path": path, "referee": ref, "course": course, "xml": xml}


def render(course, result: dict, out_dir, *, camera: str = "chase", size=(960, 540), every: int = 4, max_frames: int = 400) -> list:
    """Frames of a recorded run from one of the scene's cameras (``overview``, ``grandstand``, ``onboard``, ``chase``):
    the quad's free joint is replayed along the recorded trajectory (position only; the attitude from the recorded
    acceleration is not kept, so the onboard view is a pointing approximation). Needs an OpenGL context."""
    import mujoco
    out_dir = _P(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_string(result.get("xml") or courses.to_mjcf(course))
    data = mujoco.MjData(model)
    jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "quad_free")
    qa = int(model.jnt_qposadr[jnt])
    renderer = mujoco.Renderer(model, size[1], size[0])
    paths = []
    H = result["history"]
    for i, row in enumerate(H[::every][:max_frames]):
        data.qpos[qa:qa + 3] = row[1:4]
        vx, vy = row[4], row[5]
        yaw = math.atan2(vy, vx) if math.hypot(vx, vy) > 0.5 else 0.0
        data.qpos[qa + 3:qa + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        img = renderer.render()
        from PIL import Image
        f = out_dir / f"{course.slug}_{camera}_{i:04d}.png"
        Image.fromarray(img).save(f)
        paths.append(f)
    return paths
