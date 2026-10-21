"""The MuJoCo run (not executed in this session: written against the scene files and the existing controller, with
the same referee and the same result format as the mock; run it yourself with ``python -m races run --course farm``).

The reference quad is the free body in the scene (``refquad.mjcf_body``); every control step the rotor hook puts the
four thrusts (first-order lag, the drive table's limit at the axial inflow, the propellers' drag torques) and the
frame's drag on it through ``data.xfrc_applied``; Drongo's geometric flight controller (``races.flight.Flight``, copied from Vegeta's ``drongo_controller``:
position → attitude → mixer) follows the racing line's moving reference; the referee reads the contacts from
``data.contact`` and the position; a reset teleports the free joint to the checkpoint. Cameras: ``overview``,
``grandstand`` (scene), ``onboard``, ``chase`` (quad); ``render`` makes frames from a recorded run with
``mujoco.Renderer`` (needs an OpenGL context: EGL or GLFW)."""
from __future__ import annotations

import math
import os
from pathlib import Path as _P

os.environ.setdefault("MUJOCO_GL", "egl")             # the renderer's backend, read when mujoco is first imported
from types import SimpleNamespace

import numpy as np

from . import courses, refquad
from .mock import schedule
from .path import Drops, carrot, resync
from .rules import STUCK, Referee, Rules

G = 9.81
PILOT_COLUMNS = ["t", "v_ref", "p_err", "tilt_cmd", "tilt", "thrust_cmd", "thrust_max", "saturated", "mode"]
PILOT_MODES = ["racing", "hover", "missed gate: back", "drop: approach", "drop: align", "drop: descend"]


def _flight(prop, *, omega_pos: float = 2.5, omega_att: float = 18.0, tilt_max_deg: float = 60.0):
    """Drongo's ``Flight`` with this quad's layout (its reset reads the body's inertia from the model)."""
    from . import flight as dc

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
               record_every: int = 5, safety: float = 0.6, v_cap: float | None = None, omega_pos: float = 2.5, omega_att: float = 18.0, tilt_max_deg: float = 60.0,
               scenario: dict | None = None) -> dict:
    """The MuJoCo lap. The defaults are the first real laps' lesson (the hangar, 2026-10-18): the geometric controller
    lags the racing line by metres where the mock lagged by 0.25 m, so the schedule runs at ``safety`` 0.6 of the
    thrust-vector corner speed, the look-ahead is short and the tilt is capped at 60°; raise them course by course
    (``wf R3 --safety 0.8 --tilt-max 70``) once a lap is clean.

    ``scenario`` (``races.validate``) changes the world, not the autopilot, which keeps believing the nominal quad:
    ``start_offset`` [m] moves the start, ``mass_scale`` the quad's mass and inertia, ``thrust_scale`` every rotor's
    thrust, ``wind`` [m/s] the air the frame's drag acts against. Without it the lap is the nominal one."""
    import mujoco
    from ogma import rotors
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
    drops = Drops(path)
    ref = Referee(course, rules)
    pos_r, spin = refquad.rotor_layout()
    sc = scenario or {}
    start = np.array(course.zones[0].center, dtype=float) + np.asarray(sc.get("start_offset", (0.0, 0.0, 0.0)), dtype=float)
    k_thrust, wind = float(sc.get("thrust_scale", 1.0)), np.asarray(sc.get("wind", (0.0, 0.0, 0.0)), dtype=float)
    data.qpos[qa:qa + 3] = start
    data.qpos[qa + 3:qa + 7] = (1, 0, 0, 0)
    mujoco.mj_forward(model, data)
    flight.reset(model, data, jnt)
    if sc.get("mass_scale", 1.0) != 1.0:                                   # after the reset: the controller keeps the nominal mass and inertia
        model.body_mass[body] *= sc["mass_scale"]; model.body_inertia[body] *= sc["mass_scale"]
        q, qd = data.qpos.copy(), data.qvel.copy()
        mujoco.mj_setConst(model, data)                                    # the derived constants follow (it uses data as scratch: the state is put back)
        data.qpos[:], data.qvel[:] = q, qd
        mujoco.mj_forward(model, data)
    from . import flight as dc
    thrust = np.zeros(4)
    alpha = 1.0 - math.exp(-control_dt / prop.spin_up_s)
    n_sub = max(int(round(control_dt / model.opt.timestep)), 1)
    s, t, hist, pilot, events, energy_J, hover_until = 0.0, 0.0, [], [], [], 0.0, 1.0
    duration = duration or min(1.5 * path.time_estimate() + 30.0, rules.timeout_s + 10.0)
    k = 0
    while data.time < duration and ref.status == "racing":
        p = data.qpos[qa:qa + 3].copy()
        v = data.qvel[va:va + 3].copy()
        if ref.reset_request is not None:                                  # a reset: the free joint to the checkpoint, at rest, level
            data.qpos[qa:qa + 3] = ref.reset_request[0]; data.qpos[qa + 3:qa + 7] = (1, 0, 0, 0); data.qvel[va:va + 6] = 0.0
            hover_until = ref.reset_request[1]; s = path.s[ref.last_checkpoint]; flight.reset_integral(); ref.reset_request = None
            mujoco.mj_forward(model, data); p, v = data.qpos[qa:qa + 3].copy(), np.zeros(3)
        parked = False
        if data.time < hover_until:
            drops.mode = None
            r = dc.Ref(p=np.array(course.zones[ref.last_checkpoint].center if ref.resets else start, dtype=float), v=np.zeros(3), a=np.zeros(3), yaw=0.0)
        else:
            s = path.project(p, s)
            nz = min(ref.next, len(path.s) - 1)
            s, parked = resync(path, s, path.s[nz], nz >= len(path.s) - 1)
            p_ref, v_ref, a_ref = drops.adjust(s, nz, p, v, carrot(path, s, nz, lookahead_s=lookahead_s, parked=parked))
            yaw = math.atan2(v_ref[1], v_ref[0]) if np.linalg.norm(v_ref[:2]) > 0.5 else math.atan2(v[1], v[0]) if np.linalg.norm(v[:2]) > 0.5 else 0.0
            r = dc.Ref(p=p_ref, v=v_ref, a=a_ref, yaw=yaw)
        R = data.xmat[body].reshape(3, 3)
        obs = SimpleNamespace(base_pos=p, base_quat=data.qpos[qa + 3:qa + 7].copy(), body_linvel=[v], body_angvel=[data.qvel[va + 3:va + 6].copy()])
        cmd = flight.thrusts(obs, r, 0.0, control_dt)
        zb = R[:, 2]
        v_axial = max(float(np.dot(v, zb)), 0.0)
        t_max = prop.drive.t_max(v_axial)
        cmd_raw = float(np.sum(cmd))
        cmd = np.clip(cmd, 0.0, t_max)
        thrust += (cmd - thrust) * alpha
        applied = thrust * k_thrust
        force, torque = rotors.wrench(applied, R, pos_r, spin, prop.torque_per_thrust_m,        # about the COM (xfrc_applied acts there)
                                      com_offset=data.xipos[body] - data.xpos[body])
        force = force + rotors.quadratic_drag(v - wind, refquad.CDA_FORWARD_M2, refquad.RHO)
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
            F = flight.last.get("F", np.array([0.0, 0.0, 1.0]))                 # what the autopilot asked for at this step (races.video_panel draws it)
            mode = 1 if data.time < hover_until else {"approach": 3, "align": 4, "descend": 5}.get(drops.mode, 2 if parked else 0)
            pilot.append([data.time, float(np.linalg.norm(r.v)), float(np.linalg.norm(p - r.p)), math.degrees(math.atan2(math.hypot(F[0], F[1]), F[2])),
                          float(flight.last.get("tilt_deg", 0.0)), cmd_raw, 4.0 * float(t_max), float(bool(flight.last.get("saturated")) or cmd_raw > 4.0 * float(t_max) + 1e-9), mode])
        k += 1
    H = np.array(hist).reshape(-1, 17)
    st = ref.stats()
    if st["status"] == "racing":                                          # the loop ran out of time with the race still open
        st["status"], st["reason"] = STUCK, f"no finish within the {duration:.0f} s limit; last zone '{course.zones[max(ref.passed)].name}'"
    st.update({"course": course.name, "aircraft": "reference 5-inch quad", "mode": "mujoco", "energy [Wh]": energy_J / 3600, "Wh per km": energy_J / 3600 / (course.length() / 1e3),
               "max speed [m/s]": float(H[:, 9].max()) if len(H) else float("nan"), "mean speed [m/s]": course.length() / st["lap time [s]"] if st["status"] == "finished" else float("nan"),
               "peak power [W]": float(H[:, 8].max()) if len(H) else float("nan"), "mean power [W]": float(H[:, 8].mean()) if len(H) else float("nan"), "course length [m]": course.length(),
               "schedule time [s]": path.time_estimate(), "racing time [s]": float(data.time),
               "controller": {"omega_pos": omega_pos, "omega_att": omega_att, "tilt_max_deg": tilt_max_deg, "lookahead_s": lookahead_s, "safety": safety, "v_cap": v_cap}})
    return {"stats": st, "history": H, "pilot": np.array(pilot).reshape(-1, len(PILOT_COLUMNS)), "events": events, "gate times": ref.gate_times, "path": path, "referee": ref, "course": course, "xml": xml}


def frame_rows(t, *, fps: float = 30.0, speed: float = 1.0, every: int | None = None, max_frames: int | None = None) -> np.ndarray:
    """The history rows a video shows: one per frame at ``fps`` with the lap played ``speed`` × (1 = real time,
    whatever the history's step), or — given ``every`` — every Nth row (the older presets)."""
    t = np.asarray(t, dtype=float)
    if every:
        idx = np.arange(0, len(t), int(every))
    else:
        idx = np.searchsorted(t, np.arange(t[0], t[-1] + 1e-9, speed / fps)).clip(0, len(t) - 1)
    return idx[:max_frames] if max_frames else idx


def attitude(row) -> tuple:
    """The quad's attitude quaternion (w, x, y, z) at a history row: recorded (17 columns), else level and heading
    along the velocity (older histories)."""
    if len(row) >= 14:
        q = np.asarray(row[10:14], dtype=float)
        n = float(np.linalg.norm(q))
        if n > 0.5:
            return tuple(q / n)
    yaw = math.atan2(row[5], row[4]) if math.hypot(row[4], row[5]) > 0.5 else 0.0
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def _hud(img, lines: list[str]):
    """A text strip at the top of a frame (PIL)."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img)
    h = max(12, im.height // 32)
    try:
        font = ImageFont.load_default(size=h)
    except TypeError:                                                   # Pillow < 10.1
        font = ImageFont.load_default()
    d = ImageDraw.Draw(im, "RGBA")
    d.rectangle([0, 0, im.width, (h + 4) * len(lines) + 6], fill=(0, 0, 0, 140))
    for k, s in enumerate(lines):
        s = s.replace("—", "-").replace("–", "-").replace("≤", "<=").replace("±", "+/-")         # glyphs the default font lacks
        d.text((8, 4 + (h + 4) * k), s, fill=(255, 255, 255, 255), font=font)
    return im


CAMERA_NAMES = {"fpv": "FPV (on-board)", "chase": "chase (3rd person)", "overview": "overview", "grandstand": "grandstand", "onboard": "FPV (on-board)"}
CHASE = {"distance": 2.2, "elevation": -14.0, "tau": 0.6, "min_distance": 0.5, "fovy": 60.0}     # the third-person camera: m behind, ° down, heading smoothing [s]


def chase_headings(H, rows, *, tau: float = CHASE["tau"]) -> np.ndarray:
    """The chase camera's heading [°] at each rendered row: the direction of flight (held while the quad is slow),
    unwrapped and smoothed with a first-order lag of ``tau`` seconds so the view does not swing with every twitch."""
    H = np.asarray(H)
    raw, last = [], 0.0
    for r in rows:
        vx, vy = H[r, 4], H[r, 5]
        if math.hypot(vx, vy) > 0.8:
            last = math.atan2(vy, vx)
        raw.append(last)
    yaw = np.unwrap(np.array(raw))
    out, y = [], yaw[0] if len(yaw) else 0.0
    for k, r in enumerate(rows):
        dt = float(H[r, 0] - H[rows[k - 1], 0]) if k else 0.0
        y += (yaw[k] - y) * (1.0 - math.exp(-dt / tau)) if dt > 0 else 0.0
        out.append(math.degrees(y))
    return np.array(out)


def _clear_distance(mujoco, model, data, origin, direction, limit: float, exclude_body: int) -> float:
    """How far the camera may sit from ``origin`` along ``direction`` before a colliding geom: rays pass through the
    non-colliding ones (the zones' transparent boxes, the pads)."""
    p, gone = np.array(origin, dtype=float), 0.0
    gid = np.zeros(1, dtype=np.int32)
    for _ in range(8):
        d = mujoco.mj_ray(model, data, p, direction, None, 1, exclude_body, gid)
        if d < 0 or gone + d >= limit:
            return limit
        if model.geom_contype[gid[0]] or model.geom_conaffinity[gid[0]]:
            return gone + d
        gone += d + 1e-3
        p = p + (d + 1e-3) * direction
    return limit


def _badge(path, height: int):
    """The logo as an RGBA badge ``height`` pixels tall, rounded corners, slightly transparent."""
    from PIL import Image, ImageDraw
    im = Image.open(path).convert("RGBA")
    im = im.resize((max(1, round(im.width * height / im.height)), height), Image.LANCZOS)
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, im.width - 1, im.height - 1], radius=max(2, height // 10), fill=int(255 * 0.88))
    im.putalpha(mask)
    return im


def render(course, result: dict, out_dir, *, camera: str = "chase", size=(960, 540), every: int | None = 4, max_frames: int | None = 400,
           fps: float = 30.0, speed: float = 1.0, hud: bool = False, label: str = "", panel=None, logo=None) -> list:
    """Frames of a recorded run from a camera: ``fpv`` (the quad's on-board camera, alias ``onboard``), ``chase`` (a
    third-person camera behind the quad along its smoothed heading, pulled in front of walls: ``chase_headings``,
    ``CHASE``), or the scene's ``overview`` and ``grandstand``:
    the quad's free joint is replayed along the recorded trajectory with its recorded attitude (``attitude``). Frames
    are every Nth row (``every``) or, with ``every=None``, picked by time (``frame_rows``: ``fps``, ``speed``).
    ``hud`` writes the course, ``label``, time, speed, altitude, zones and collisions on every frame; ``logo`` (a
    path) puts that image as a badge in the bottom right corner; ``panel`` (``races.video_panel.SidePanel``) is
    drawn to the right of the view (the frame is then the view and the panel side by side). Needs an OpenGL
    context."""
    import mujoco
    from PIL import Image
    out_dir = _P(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_string(result.get("xml") or courses.to_mjcf(course))
    data = mujoco.MjData(model)
    jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "quad_free")
    qa = int(model.jnt_qposadr[jnt])
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(size[0]))      # the offscreen framebuffer must hold the frame
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(size[1]))
    model.vis.map.znear = 0.02 / model.stat.extent                           # MuJoCo scales the clip planes by the extent: a 2 cm near plane, not 3–10 m
    model.vis.map.zfar = 4.0 * max(model.stat.extent, 1.0) / model.stat.extent
    model.vis.global_.fovy = CHASE["fovy"]                                   # the free (chase) camera's field of view
    renderer = mujoco.Renderer(model, size[1], size[0])
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "quad")
    H = result["history"]
    ev = [(float(e[0]), str(e[1])) for e in result.get("events") or []]
    zone_t = [te for te, s in ev if s.startswith("zone")]
    hit_t = [te for te, s in ev if s.startswith("collision")]
    badge = _badge(logo, max(24, int(size[1] * 0.16))) if logo else None
    rows = frame_rows(H[:, 0], fps=fps, speed=speed, every=every, max_frames=max_frames)
    view = "onboard" if camera in ("fpv", "onboard") else camera
    if camera == "chase":
        heads = chase_headings(H, rows)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.elevation = CHASE["elevation"]
    paths = []
    for i, r in enumerate(rows):
        row = H[r]
        data.qpos[qa:qa + 3] = row[1:4]
        data.qpos[qa + 3:qa + 7] = attitude(row)
        mujoco.mj_forward(model, data)
        if camera == "chase":
            az, el = math.radians(heads[i]), math.radians(CHASE["elevation"])
            back = -np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
            room = _clear_distance(mujoco, model, data, row[1:4], back, CHASE["distance"] + 0.3, body)
            cam.lookat[:] = row[1:4]
            cam.azimuth = heads[i]
            cam.distance = max(CHASE["min_distance"], min(CHASE["distance"], room - 0.3))
            renderer.update_scene(data, camera=cam)
        else:
            renderer.update_scene(data, camera=view)
        img = renderer.render()
        im = Image.fromarray(img)
        if hud:
            tt = float(row[0])
            im = _hud(img, [f"{course.name} | {CAMERA_NAMES.get(camera, camera)}{' | ' + label if label else ''}",
                            f"t {tt:5.1f} s   v {row[9]:4.1f} m/s   alt {row[3]:4.1f} m   zones {sum(z <= tt for z in zone_t)}/{len(course.zones) - 1}"
                            f"   collisions {sum(c <= tt for c in hit_t)}"])
        if badge is not None:
            m = max(4, size[1] // 60)
            im = im.convert("RGBA")
            im.alpha_composite(badge, (im.width - badge.width - m, im.height - badge.height - m))
            im = im.convert("RGB")
        if panel is not None:
            side = Image.fromarray(panel.frame(float(row[0])))
            w = im.width + side.width
            both = Image.new("RGB", (w + w % 2, max(im.height, side.height) + max(im.height, side.height) % 2), (16, 20, 24))
            both.paste(im, (0, 0))
            both.paste(side, (im.width, 0))
            im = both
        f = out_dir / f"{course.slug}_{camera}_{i:05d}.png"
        im.save(f)
        paths.append(f)
    return paths
