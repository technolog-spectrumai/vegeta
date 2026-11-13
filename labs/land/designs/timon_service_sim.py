"""Timon changes Pumba's tool — a MuJoCo **dynamics** simulation of both machines in one world
(``sims/timon_tool_change_movie.py --physics``).

Both robots are their Chiron models (``pumba_robot`` with a bare quick-coupler plate on the main arm,
``timon_robot``) compiled together (``MjSpec.attach``): tracks as roller rows on the ground, bogies, gravity, every
joint driven by its servo — PD clipped to the motor's torque-speed line, as Chiron's servos (``_Servos``). The two
tools are free bodies with their masses (the grab 7.2 kg, the 2 kW chainsaw 8.0 kg) and a lifting lug each.

What holds what is a weld that switches on only when the parts are where they should be:

* ``latch_*`` — the tool on Pumba's coupler: on when the tool's coupler face is within ``LATCH_TOL`` (5 mm, 3 deg) of
  Pumba's flange;
* ``grip_*`` — the tool in Timon's grab: on when Timon's jaws are closed on the lug (the grasp point within 8 mm of
  the lug's centre);
Nothing holds a tool in Timon's **basket** (two open-topped slots on its rear, fixed to the hull): the tools stand
in it on their tips, on contact with its floor and walls. The chainsaw starts there; Timon lowers the grab over its
slot, opens its jaws and the grab drops the last 8 cm onto the slot's floor.

The plan is geometric: Timon docks so that its main arm approaches along the line Pumba's coupler faces; one grasp
of the lug (``G_rel``: the tool in Timon's hand) serves both tools at Pumba and in the basket, so every hand-over is
the same pose. The arms move between IK targets taken in each hull's frame at the moment the move starts (the hulls
settle and pitch under the loads) and finish with a closed-loop alignment on the measured positions (Timon's TCP
against the lug or the coupler, as a camera at the wrist would). Each step waits until its condition holds or
times out (then the run reports the failure — nothing is snapped into place).

    import timon_service_sim as tss
    sim = tss.ServiceSim()
    log = sim.run()                 # frames for the movie, events, the alignment errors at each latch
"""
from __future__ import annotations

import math

import numpy as np

import pekari_arms as pa
import pekari_family as pf
import pumba_robot as pu
import timon_robot as tr

__all__ = ["ServiceSim", "LATCH_TOL"]

LATCH_TOL = (0.005, 3.0)          # m, deg
GRIP_TOL = 0.008                  # m
G = 9.81
GEOM_KIND = {2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box"}


def _rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _T(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def _ang(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _quat(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


class _Servos:
    """Chiron's servo law for every servoed joint of the robots (names with their prefixes)."""

    def __init__(self, m, robots):
        import mujoco
        jid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(m.njnt)}
        rows = []
        for prefix, robot in robots:
            for j in robot.joints():
                if j.servo is None:
                    continue
                n = prefix + j.name
                k = jid[n]
                s = j.servo
                rows.append((n, int(m.jnt_qposadr[k]), int(m.jnt_dofadr[k]), s.kp, s.kd, s.stall_torque,
                             s.no_load_speed, bool(s.four_quadrant)))
        self.names = [r[0] for r in rows]
        self.index = {n: i for i, n in enumerate(self.names)}
        a = np.array([r[1:] for r in rows], dtype=float)
        self.qadr, self.dadr = a[:, 0].astype(int), a[:, 1].astype(int)
        self.kp, self.kd, self.stall, self.w0, self.fq = a[:, 2], a[:, 3], a[:, 4], a[:, 5], a[:, 6].astype(bool)
        self.qt = np.zeros(len(rows))
        self.qdt = np.zeros(len(rows))

    def apply(self, d):
        q, qd = d.qpos[self.qadr], d.qvel[self.dadr]
        tau = self.kp * (self.qt - q) + self.kd * (self.qdt - qd)
        line = self.stall * np.clip(1 - np.abs(qd) / self.w0, 0.0, 1.0)
        braking = tau * qd < 0
        lim = np.where(braking & self.fq, self.stall, line)
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[self.dadr] = np.clip(tau, -lim, lim)

    def set(self, targets: dict, vel: dict | None = None):
        for n, v in targets.items():
            if n in self.index:
                self.qt[self.index[n]] = v
        for n, v in (vel or {}).items():
            if n in self.index:
                self.qdt[self.index[n]] = v


class ServiceSim:
    """The two robots, the two tools, the welds and the service plan (see the module)."""

    LUG = (0.10, 0.0, 0.085)       # the lug's centre in the tool frame: 0.10 m down the tool, toward Timon
    PRESENT_R = 1.0                # Pumba presents its flange this far (horizontally) from its shoulder [m]
    PRESENT_Z = 0.95               # ... at this height [m]
    DOCK_R = 1.15                  # Timon's main shoulder this far from Pumba's flange, on the coupler's line [m]

    def __init__(self, fps: int = 25):
        import mujoco
        from vegeta import chiron as ch

        self.mj = mujoco
        self.fps = fps
        self.pl_p = pf.with_tool(pu.PUMBA, "main", "quick coupler")
        self.rp = pf.robot(self.pl_p, payload_kg=0.0, stabilizers_deployed=True)
        self.rt = tr.timon(payload_kg=0.0)
        self.parm = self.pl_p.mount("main").arm
        self.tarm = tr.TIMON.mount("main").arm
        self.aarm = tr.TIMON.mount("aux").arm
        self._plan_geometry()
        sp = mujoco.MjSpec.from_string(self.rp.to_mjcf(ch.Flat()))
        st = mujoco.MjSpec.from_string(self.rt.to_mjcf(ch.Flat()))
        fr = sp.worldbody.add_frame()
        self._add_basket(st)
        fr.attach_body(st.body("hull"), "t_", "")
        self._add_tools(sp)
        self.m = sp.compile()
        self.d = mujoco.MjData(self.m)
        self.servos = _Servos(self.m, [("", self.rp), ("t_", self.rt)])
        self.bid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)        # noqa: E731
        self.eid = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_EQUALITY, e): e for e in range(self.m.neq)}
        self.jadr = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j): int(self.m.jnt_qposadr[j])
                     for j in range(self.m.njnt)}
        self.events, self.frames, self.errors = [], [], {}
        self.G_held = {}
        self.trace = []
        self.render_geoms = [g for g in range(self.m.ngeom) if int(self.m.geom_type[g]) in GEOM_KIND
                             and self.m.geom_bodyid[g] != 0]

    # ------------------------------------------------------------------------------------------- geometry
    def _plan_geometry(self):
        """Pumba's presenting pose, the coupler's line, Timon's dock pose, the grasp, the holster."""
        from timon_tool_change import KinematicModel

        shP = np.array(pf.shoulder(self.pl_p, "main"))
        yawP = math.radians(80.0)                                  # the arm swung to the left, a little ahead
        self.flange_target = shP + np.array([self.PRESENT_R * math.cos(yawP), self.PRESENT_R * math.sin(yawP), 0.0])
        self.flange_target[2] = self.PRESENT_Z
        tcp = self.flange_target + np.array([0, 0, -(self.parm.tool - pa.flange(self.parm))])
        qP = pa.ik(self.parm, tuple(tcp), -math.pi / 2, tuple(shP))
        assert qP is not None
        self.qP_present = pa.qpos(self.parm, "main_", qP)
        kp = KinematicModel(self.rp)
        kp.set((0, 0), 0.0, {**self.rp.nominal_qpos, **self.qP_present}, pf.z_hull(pu.PUMBA))
        Th = kp.body_T("main_hand")
        # the tool, mated: its frame on Pumba's flange face (the coupler plate's far face)
        plate = self.parm.tool - pa.flange(self.parm)
        self.T_tool_mated_planned = Th @ _T(np.eye(3), np.array([plate, 0.0, 0.0]))
        # the coupler's line: horizontal, from the flange toward the side Timon comes from (the arm's yaw)
        u = np.array([math.cos(yawP), math.sin(yawP), 0.0])
        self.u = u
        # Timon: its main shoulder DOCK_R out along u, facing back along -u
        yawT = math.atan2(-u[1], -u[0])
        self.yawT = yawT
        shT_v = np.array(pf.shoulder(tr.TIMON, "main"))             # in Timon's vehicle frame
        shT_w = self.flange_target + self.DOCK_R * u
        self.dockT = shT_w[:2] - (_rotz(yawT) @ np.array([shT_v[0], shT_v[1], 0.0]))[:2]
        self.startT = self.dockT + 3.2 * u[:2]
        # Timon's grasp at the mating site: its gripper level, pointing along -u at the lug
        kt = KinematicModel(self.rt)
        lug_w = (self.T_tool_mated_planned @ np.array([*self.LUG, 1.0]))[:3]
        self.qT_grasp = self._timon_ik(lug_w, base=(self.dockT, yawT), elev=0.0, km=kt)
        strip = lambda q: {k[2:]: v for k, v in q.items()}         # noqa: E731  ("t_" names are the combined world's)
        kt.set(self.dockT, yawT, {**self.rt.nominal_qpos, **strip(self.qT_grasp)}, pf.z_hull(tr.TIMON))
        self.G_rel = np.linalg.inv(kt.body_T("main_hand")) @ self.T_tool_mated_planned
        # the holster: behind Timon's shoulder on the deck; the tool hangs in it as the grasp would hold it there
        Rt = _rotz(yawT)
        hol_v = np.array([-0.66, -0.12, pf.roof_z(tr.TIMON) + 0.35])   # on the rear, behind the hull: the tools hang clear
        hol_w = np.array([*self.dockT, 0.0]) + Rt @ hol_v
        self.qT_holster = self._timon_ik(hol_w, base=(self.dockT, yawT), elev=0.0, km=kt)
        kt.set(self.dockT, yawT, {**self.rt.nominal_qpos, **strip(self.qT_holster)}, pf.z_hull(tr.TIMON))
        T_tool_hol = kt.body_T("main_hand") @ self.G_rel
        Tbase = _T(Rt, np.array([*self.dockT, pf.z_hull(tr.TIMON)]))
        self.H_holster = np.linalg.inv(Tbase) @ T_tool_hol          # in Timon's hull frame (planned): the saw's slot
        self.H_grab = self.H_holster.copy()
        self.H_grab[1, 3] -= self.GRAB_DY                          # the grab's slot, across the basket

    # ------------------------------------------------------------------------------------------- the basket
    BASKET_HALF = {"saw": 0.085, "grab": 0.10}   # the slots' inner half-widths: snug on the saw, roomy to drop into [m]
    WALL = 0.012                   # wall half-thickness [m]
    GRAB_DY = 0.23                 # the grab's slot this far across from the saw's [m]

    def _add_basket(self, st):
        """The tool basket on Timon's rear: two open-topped slots, walls and floor fixed to the hull. The tools stand
        in it on their tips, held up only by contact with its floor and walls (no weld)."""
        import mujoco
        hull = st.body("hull")
        z_top = self.H_holster[2, 3] - 0.20                       # the walls stop below the lug, clear of the jaws
        floors = {"saw": self.H_holster[2, 3] - 0.725 - 0.003,      # the saw's tip just clear of its floor
                  "grab": self.H_grab[2, 3] - 0.325 - 0.08}         # the grab drops 8 cm onto a raised floor
        self.basket = {"z_top": z_top, "floors": floors}
        rgba = [0.3, 0.42, 0.3, 1]
        w = self.WALL
        for tool, H in (("saw", self.H_holster), ("grab", self.H_grab)):
            b = self.BASKET_HALF[tool]
            cx, cy = H[0, 3], H[1, 3]
            zf = floors[tool]
            zc, hz = (z_top + zf) / 2, (z_top - zf) / 2

            def geom(name, half, pos):
                g = hull.add_geom(name=f"basket_{tool}_{name}", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half),
                                  pos=list(pos), mass=0.4, rgba=rgba)
                g.contype, g.conaffinity = 0, 4                    # touches only the tools (contype bit 4)
            geom("floor", (b + w, b + w, w), (cx, cy, zf - w))
            for sx in (-1, 1):
                geom(f"wx{sx}", (w, b + 2 * w, hz), (cx + sx * (b + w), cy, zc))
                geom(f"wy{sx}", (b, w, hz), (cx, cy + sx * (b + w), zc))

    def in_basket(self, tool):
        """Where the tool rests in Timon's hull frame: (inside its slot?, its lowest point above the slot floor [m])."""
        Tb = self._timon_base()
        H = np.linalg.inv(Tb) @ self._tool_T(tool)
        slot = self.H_holster if tool == "saw" else self.H_grab
        dxy = float(np.hypot(H[0, 3] - slot[0, 3], H[1, 3] - slot[1, 3]))
        Ti, lowest = np.linalg.inv(Tb), np.inf
        corners = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
        for g in range(self.m.ngeom):
            if self.m.geom_bodyid[g] == self.bid(f"tool_{tool}"):
                pts = self.d.geom_xpos[g] + (corners * self.m.geom_size[g]) @ self.d.geom_xmat[g].reshape(3, 3).T
                lowest = min(lowest, float((Ti[:3, :3] @ pts.T).T[:, 2].min() + Ti[2, 3]))
        return dxy < self.BASKET_HALF[tool], lowest - self.basket["floors"][tool]

    def _timon_ik(self, p_world, base, elev, km=None, mount="main"):
        """Timon's joints that put its TCP at ``p_world`` with the tool at ``elev`` (planned base pose)."""
        arm = tr.TIMON.mount(mount).arm
        (bx, by), yaw = base
        p_v = _rotz(yaw).T @ (np.asarray(p_world, float) - np.array([bx, by, 0.0]))
        sh = pf.shoulder(tr.TIMON, mount)
        for e in (elev, elev - 0.15, elev + 0.15):
            q = pa.ik(arm, tuple(p_v), e, sh)
            if q is not None:
                return {f"t_{k}": v for k, v in pa.qpos(arm, f"{mount}_", q).items()}
        raise ValueError(f"Timon cannot reach {p_world}")

    # ------------------------------------------------------------------------------------------- the tools
    def _add_tools(self, sp):
        import mujoco
        lug = self.LUG
        steel, dark, orange, grey = [0.55, 0.57, 0.6, 1], [0.2, 0.2, 0.22, 1], [0.85, 0.35, 0.08, 1], [0.7, 0.7, 0.72, 1]

        def body(name, pos):
            b = sp.worldbody.add_body(name=name, pos=pos)
            b.add_freejoint(name=name + ":free")
            return b

        def box(b, name, half, pos, mass, rgba):
            g = b.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half), pos=list(pos), mass=mass,
                           rgba=rgba)
            g.contype, g.conaffinity = 2 | 4, 1                    # the ground and the basket
            return g
        gb = body("tool_grab", [0, 0, -5])
        box(gb, "grab_plate", (0.01, 0.07, 0.07), (0.01, 0, 0), 1.2, dark)
        box(gb, "grab_body", (0.06, 0.06, 0.06), (0.08, 0, 0), 3.0, dark)
        box(gb, "grab_lug", (0.01, 0.025, 0.03), (lug[0], 0, lug[2]), 0.3, steel)
        for s, n in ((1, "a"), (-1, "b")):
            box(gb, f"grab_jaw_{n}", (0.09, 0.025, 0.012), (0.23, 0, s * 0.04), 1.35, steel)
        gs = body("tool_saw", [0, 0, -6])
        box(gs, "saw_plate", (0.01, 0.07, 0.07), (0.01, 0, 0), 1.2, dark)
        box(gs, "saw_body", (0.15, 0.06, 0.08), (0.17, 0, 0), 5.3, orange)
        box(gs, "saw_lug", (0.01, 0.025, 0.03), (lug[0], 0, lug[2]), 0.3, steel)
        box(gs, "saw_bar", (0.20, 0.0025, 0.035), (0.52, 0, 0), 1.2, grey)
        for tool in ("grab", "saw"):
            for kind, other in (("latch", "main_hand"), ("grip", "t_main_hand")):
                e = sp.add_equality(name=f"{kind}_{tool}", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                                    name1=f"tool_{tool}", name2=other)
                e.active = False
                e.solref = [0.004, 1.0]

    # ------------------------------------------------------------------------------------------- state access
    def body_T(self, name):
        b = self.bid(name)
        return _T(self.d.xmat[b].reshape(3, 3).copy(), self.d.xpos[b].copy())

    def _place_tool(self, tool, T):
        a = self.jadr[f"tool_{tool}:free"]
        self.d.qpos[a:a + 3] = T[:3, 3]
        self.d.qpos[a + 3:a + 7] = _quat(T[:3, :3])
        self.d.qvel[self.m.jnt_dofadr[self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_JOINT, f"tool_{tool}:free")]:][:6] = 0

    def _weld(self, name, on: bool):
        e = self.eid[name]
        if on:                                                     # weld in the relative pose the parts have now
            b1, b2 = self.m.eq_obj1id[e], self.m.eq_obj2id[e]
            T1 = _T(self.d.xmat[b1].reshape(3, 3), self.d.xpos[b1])
            T2 = _T(self.d.xmat[b2].reshape(3, 3), self.d.xpos[b2])
            rel = np.linalg.inv(T1) @ T2
            self.m.eq_data[e, 0:3] = 0.0
            self.m.eq_data[e, 3:6] = rel[:3, 3]
            self.m.eq_data[e, 6:10] = _quat(rel[:3, :3])
        self.d.eq_active[e] = 1 if on else 0
        if on and name.startswith("grip_"):                       # the grasp as it is, for the moves that follow
            tool = name.split("_", 1)[1]
            self.G_held[tool] = np.linalg.inv(self.body_T("t_main_hand")) @ self._tool_T(tool)
        self.events.append((round(self.d.time, 2), f"{'on' if on else 'off'}: {name}"))

    def _tool_T(self, tool):
        return self.body_T(f"tool_{tool}")

    def _flange_T(self):
        plate = self.parm.tool - pa.flange(self.parm)
        return self.body_T("main_hand") @ _T(np.eye(3), np.array([plate, 0.0, 0.0]))

    def _timon_base(self):
        T = self.body_T("t_hull")
        return T

    # ------------------------------------------------------------------------------------------- control
    def _timon_joints_for_hand(self, T_hand_w, mount="main"):
        """IK for Timon's hand at ``T_hand_w`` (world) in Timon's current vehicle frame: TCP position, the tool's
        elevation and roll from the hand's x and z axes."""
        arm = tr.TIMON.mount(mount).arm
        reach_t = arm.tool - pa.flange(arm)
        tcp = (T_hand_w @ np.array([reach_t, 0, 0, 1.0]))[:3]
        Tb = self._timon_base()
        Rb = Tb[:3, :3]
        z_h = pf.z_hull(tr.TIMON)
        origin = Tb[:3, 3] - Rb @ np.array([0, 0, z_h])
        p_v = Rb.T @ (tcp - origin)
        x_h = Rb.T @ T_hand_w[:3, 0]
        elev = math.asin(max(-1.0, min(1.0, x_h[2])))
        sh = pf.shoulder(tr.TIMON, mount)
        q = pa.ik(arm, tuple(p_v), elev, sh)
        if q is None:
            raise ValueError(f"Timon cannot reach {tcp}")
        return {f"t_{k}": v for k, v in pa.qpos(arm, f"{mount}_", q).items()}

    def align_orientation(self, R_hand_target, label, passes: int = 4):
        """Turn Timon's hand to ``R_hand_target`` on its wrist (pitch about the hand's y, yaw about its z, roll about
        its x — the 6-DOF arm's three wrist joints), measuring the hand each pass; returns the error [deg]."""
        for _ in range(passes):
            Ra = self.body_T("t_main_hand")[:3, :3]
            Re = Ra.T @ np.asarray(R_hand_target)                  # the rotation still to make, in the hand's frame
            c = max(-1.0, min(1.0, (np.trace(Re) - 1) / 2))
            ang = math.acos(c)
            if ang < math.radians(0.2):
                break
            w = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]]) / (2 * math.sin(ang)) * ang
            cur = {n: self.servos.qt[self.servos.index[n]] for n in ("t_main_wrist_pitch", "t_main_wrist_yaw", "t_main_wrist_roll")}
            self.move({"t_main_wrist_pitch": cur["t_main_wrist_pitch"] + w[1], "t_main_wrist_yaw": cur["t_main_wrist_yaw"] + w[2],
                       "t_main_wrist_roll": cur["t_main_wrist_roll"] + w[0]}, 0.6, label)
        return _ang(self.body_T("t_main_hand")[:3, :3], R_hand_target)

    def place_tool(self, tool, T_target_fn, label, tol=GRIP_TOL, rounds=10):
        """Bring the held ``tool`` to ``T_target_fn()`` (re-evaluated each round: the target sits on a hull that
        moves): the hand's orientation on the wrist, then the tool's measured position, closed loop."""
        off = np.zeros(3)                                          # the accumulated position correction (integral)
        for _ in range(rounds):
            T_t = T_target_fn()
            H = self._hand_for_tool(T_t, tool)
            self.align_orientation(H[:3, :3], label)
            e_v = T_t[:3, 3] - self._tool_T(tool)[:3, 3]
            if np.linalg.norm(e_v) <= tol and _ang(self._tool_T(tool)[:3, :3], T_t[:3, :3]) <= 2.0:
                break
            off = off + 0.6 * e_v
            if np.linalg.norm(off) > 0.12:
                off *= 0.12 / np.linalg.norm(off)
            self.trace.append((round(self.d.time, 1), label, round(float(np.linalg.norm(e_v)) * 1000, 1)))
            try:
                q = self._timon_joints_for_hand(_T(np.eye(3), off) @ H)
            except ValueError:
                off *= 0.5
                continue
            keep = {n: self.servos.qt[self.servos.index[n]] for n in ("t_main_wrist_yaw", "t_main_wrist_roll")}
            self.move({**q, **keep}, 0.8, label)
            self.align_orientation(H[:3, :3], label, passes=2)
        T_t = T_target_fn()
        return float(np.linalg.norm(T_t[:3, 3] - self._tool_T(tool)[:3, 3])), _ang(self._tool_T(tool)[:3, :3], T_t[:3, :3])

    def _hand_for_tool(self, T_tool_w, held: str | None = None):
        """Timon's hand pose that puts the tool at ``T_tool_w``: with the planned grasp (``G_rel``) to take a tool,
        with the grasp it actually has (measured when the jaws closed) for a tool it ``held``."""
        G = self.G_held[held] if held else self.G_rel
        return T_tool_w @ np.linalg.inv(G)

    def step(self, n=1):
        for _ in range(n):
            self.servos.apply(self.d)
            self.mj.mj_step(self.m, self.d)
            if self.d.time + 1e-9 >= len(self.frames) / self.fps:
                self._record()

    def _record(self):
        g = self.render_geoms
        self.frames.append((self.d.time, self.label, [(GEOM_KIND[int(self.m.geom_type[i])], tuple(self.m.geom_size[i]),
                                                     tuple(self.m.geom_rgba[i]), self.d.geom_xpos[i].copy(),
                                                     self.d.geom_xmat[i].reshape(3, 3).copy()) for i in g]))

    def move(self, targets: dict, T: float, label: str):
        """Joint-space move to ``targets`` over ``T`` s (smooth), then hold."""
        self.label = label
        names = list(targets)
        q0 = {n: self.servos.qt[self.servos.index[n]] for n in names if n in self.servos.index}
        steps = int(T / self.m.opt.timestep)
        for k in range(steps):
            s = (k + 1) / steps
            s = s * s * (3 - 2 * s)
            self.servos.set({n: q0[n] + (targets[n] - q0[n]) * s for n in q0})
            self.step()
        # then wait until the joints are there (the arms are slow under load: a 46 kg arm on its motors' lines)
        idx = [self.servos.index[n] for n in q0]
        for _ in range(int(6.0 / (10 * self.m.opt.timestep))):
            err = np.abs(self.d.qpos[self.servos.qadr[idx]] - self.servos.qt[idx])
            vel = np.abs(self.d.qvel[self.servos.dadr[idx]])
            if err.max() < math.radians(0.3) and vel.max() < 0.01:
                break
            self.step(10)

    def hold(self, T: float, label=None):
        if label:
            self.label = label
        self.step(int(T / self.m.opt.timestep))

    def move_hand(self, T_hand_target, T: float, label: str, align: int = 2):
        """Timon's main hand to ``T_hand_target`` (world), then ``align`` closed-loop corrections on the measured
        hand position (the wrist camera)."""
        self.move(self._timon_joints_for_hand(T_hand_target), T, label)
        tgt = T_hand_target.copy()
        for _ in range(align):
            self.hold(0.4)
            err = T_hand_target[:3, 3] - self.body_T("t_main_hand")[:3, 3]
            tgt[:3, 3] = tgt[:3, 3] + err
            self.move(self._timon_joints_for_hand(tgt), 0.6, label)
        self.hold(0.3)
        return float(np.linalg.norm(T_hand_target[:3, 3] - self.body_T("t_main_hand")[:3, 3]))

    def drive_timon(self, goal_xy, v=0.5, label="Timon drives"):
        """Timon's tracks at belt speed ``v`` straight toward ``goal_xy`` (heading held), stopping on it."""
        self.label = label
        radius = {f"t_{s}_{n}": r for s in ("L", "R") for n, _, _, r, *_ in self.rt.rollers}
        vcur = 0.0
        while True:
            Tb = self._timon_base()
            d = np.asarray(goal_xy) - Tb[:2, 3]
            fwd = Tb[:2, 0]
            along = float(d @ fwd)
            if along < 0.01:
                break
            psi = math.atan2(fwd[1], fwd[0])
            err = (math.atan2(d[1], d[0]) - psi + math.pi) % (2 * math.pi) - math.pi if np.linalg.norm(d) > 0.2 else 0.0
            vt = math.copysign(min(abs(v), 0.8 * abs(along) + 0.05), v)
            vcur += max(-0.002, min(0.002, vt - vcur))
            w = 2.0 * err * (1 if v > 0 else 0)
            B = self.rt.geometry["B"]
            vl, vr = vcur - w * B / 2, vcur + w * B / 2
            self.servos.set({}, {n: (vl if "_L_" in n else vr) / r for n, r in radius.items()})
            self.step(2)
            if self.d.time > 200:
                break
        self.servos.set({}, {n: 0.0 for n in radius})
        self.hold(0.8)

    def drive_timon_back(self, dist, v=0.5, label="Timon backs away"):
        self.label = label
        radius = {f"t_{s}_{n}": r for s in ("L", "R") for n, _, _, r, *_ in self.rt.rollers}
        x0 = self._timon_base()[:2, 3].copy()
        while np.linalg.norm(self._timon_base()[:2, 3] - x0) < dist:
            self.servos.set({}, {n: -v / r for n, r in radius.items()})
            self.step(2)
        self.servos.set({}, {n: 0.0 for n in radius})
        self.hold(1.0)

    # ------------------------------------------------------------------------------------------- the service
    def check_latch(self, tool):
        Tt, Tf = self._tool_T(tool), self._flange_T()
        return float(np.linalg.norm(Tt[:3, 3] - Tf[:3, 3])), _ang(Tt[:3, :3], Tf[:3, :3])

    def run(self) -> dict:
        mj, d, m = self.mj, self.d, self.m
        self.label = "Pumba parks and lowers its stabilizers"
        # the start: Pumba at the origin, Timon away, its arms folded; nominal joint targets
        mj.mj_resetData(m, d)
        a = self.jadr["hull:free"]
        d.qpos[a:a + 3] = (0, 0, pf.z_hull(pu.PUMBA) + 0.03)
        d.qpos[a + 3:a + 7] = (1, 0, 0, 0)
        a = self.jadr["t_hull:free"]
        d.qpos[a:a + 3] = (*self.startT, pf.z_hull(tr.TIMON) + 0.03)
        d.qpos[a + 3:a + 7] = _quat(_rotz(self.yawT))
        nom = {**self.rp.nominal_qpos, **{f"t_{k}": v for k, v in self.rt.nominal_qpos.items()}}
        for n, v in nom.items():
            if n in self.jadr:
                d.qpos[self.jadr[n]] = v
        self.servos.set(nom)
        stab = {n: v for n, v in self.rp.nominal_qpos.items() if n.startswith("stab_")}
        self.servos.set({n: 0.0 for n in stab})
        mj.mj_forward(m, d)
        # the grab on Pumba's coupler (latched), the saw in Timon's holster
        self._place_tool("grab", self._flange_T())
        self._place_tool("saw", self._timon_base() @ self.H_holster)
        mj.mj_forward(m, d)
        self._weld("latch_grab", True)
        self.hold(1.5, "the chainsaw stands in Timon's basket")
        self.move(stab, 2.0, "Pumba parks and lowers its stabilizers")
        self.move(self.qP_present, 4.0, "Pumba presents its quick coupler")
        self.hold(1.0)
        self.drive_timon(self.dockT, label="Timon drives up and docks")
        self.move({f"t_{k}": v for k, v in pa.qpos(self.aarm, "aux_", pa.pose(self.aarm, "deployed")).items()}, 2.0,
                  "Timon's second hand: wrist camera on the coupler")
        # 1. take the grab off Pumba
        T_tool = self._tool_T("grab")
        approach = _T(np.eye(3), -0.18 * self.u) @ self._hand_for_tool(T_tool)        # 18 cm back along the line
        self.move({"t_main_grip_a": math.radians(-40), "t_main_grip_b": math.radians(-40)}, 0.8, "Timon opens its grab")
        self.move_hand(approach, 3.5, "Timon's main arm lines up with the lug")
        e = self.move_hand(self._hand_for_tool(self._tool_T("grab")), 2.0, "closes in on the grab's lug")
        self.move({"t_main_grip_a": math.radians(20), "t_main_grip_b": math.radians(20)}, 1.0, "grips the lug")
        self.errors["grip_grab"] = e
        if e > GRIP_TOL:
            self.events.append((round(d.time, 2), f"FAILED: grip on the grab off by {e * 1000:.1f} mm"))
            return self.result()
        self._weld("grip_grab", True)
        self.hold(0.5, "the coupler unlatches")
        self._weld("latch_grab", False)
        down = _T(np.eye(3), np.array([0, 0, -0.06]))
        self.move_hand(down @ self.body_T("t_main_hand"), 1.5, "slides the grab off the coupler", align=1)
        self.move_hand(_T(np.eye(3), 0.25 * self.u) @ self.body_T("t_main_hand"), 2.0, "backs off with the grab", align=0)
        # 2. over its slot in Timon's basket, lowered in, and dropped
        H_grab = self.H_grab
        T_hol = self._timon_base() @ H_grab
        self.move_hand(_T(np.eye(3), np.array([0, 0, 0.22])) @ self._hand_for_tool(T_hol, "grab"), 4.0, "swings it over the basket", align=1)
        err_h, ang_h = self.place_tool("grab", lambda: self._timon_base() @ H_grab, "lowers it into the basket", tol=0.02)
        self.errors["rack_grab"] = (err_h, ang_h)
        self._weld("grip_grab", False)
        self.move({"t_main_grip_a": math.radians(-40), "t_main_grip_b": math.radians(-40)}, 0.8, "drops it into the basket")
        self.hold(1.5, "the grab falls into its slot")
        self.move_hand(_T(np.eye(3), np.array([0, 0, 0.12])) @ self.body_T("t_main_hand"), 1.0, "drops it into the basket", align=0)
        inside, gap = self.in_basket("grab")
        self.errors["basket_grab"] = (inside, gap)
        if not inside or gap > 0.01:
            self.events.append((round(d.time, 2), f"FAILED: the grab is not resting in the basket {inside, gap}"))
            return self.result()
        # 3. the saw out of the holster
        T_saw = self._tool_T("saw")
        self.move_hand(_T(np.eye(3), np.array([0, 0, 0.12])) @ self._hand_for_tool(T_saw), 2.0, "to the chainsaw in the basket", align=1)
        e = self.move_hand(self._hand_for_tool(self._tool_T("saw")), 1.5, "onto the saw's lug")
        self.move({"t_main_grip_a": math.radians(20), "t_main_grip_b": math.radians(20)}, 1.0, "grips the saw")
        self.errors["grip_saw"] = e
        self._weld("grip_saw", True)
        self.move_hand(_T(np.eye(3), np.array([0, 0, 0.62])) @ self.body_T("t_main_hand"), 3.0, "lifts the saw out of the basket", align=0)
        # 4. onto Pumba's coupler: below it on the axis, aligned, then pushed up the axis
        T_f = self._flange_T()
        below = _T(np.eye(3), np.array([0, 0, -0.06])) @ T_f
        self.move_hand(_T(np.eye(3), 0.25 * self.u) @ self._hand_for_tool(below, "saw"), 4.0, "brings the saw to Pumba", align=1)
        self.place_tool("saw", lambda: _T(np.eye(3), np.array([0, 0, -0.06])) @ self._flange_T(),
                        "lines it up under the coupler", tol=0.004)
        self.place_tool("saw", self._flange_T, "pushes it up onto the coupler", tol=0.003)
        de, ae = self.check_latch("saw")
        self.errors["latch_saw"] = (de, ae)
        if de > LATCH_TOL[0] or ae > LATCH_TOL[1]:
            self.events.append((round(d.time, 2), f"FAILED: coupler misaligned {de * 1000:.1f} mm, {ae:.1f} deg"))
            return self.result()
        self._weld("latch_saw", True)
        self.hold(0.5, "the coupler latches")
        self._weld("grip_saw", False)
        self.move({"t_main_grip_a": math.radians(-40), "t_main_grip_b": math.radians(-40)}, 0.8, "Timon lets go")
        self.move_hand(_T(np.eye(3), 0.3 * self.u) @ self.body_T("t_main_hand"), 1.5, "Timon lets go", align=0)
        fold = {f"t_{k}": v for k, v in {**pa.qpos(self.tarm, "main_", pa.pose(self.tarm, "folded")),
                                         **pa.qpos(self.aarm, "aux_", pa.pose(self.aarm, "folded"))}.items()}
        self.move(fold, 3.0, "Timon folds its arms")
        self.drive_timon_back(2.0)
        cut = pa.qpos(self.parm, "main_", pa.pose(pa.with_tool(self.parm, "chainsaw 2 kW"), "cut"))
        self.move({**cut, **{n: 0.0 for n in stab}}, 4.0, "Pumba lifts its stabilizers and the saw to cut")
        self.hold(2.0, "Pumba: the chainsaw on its main arm")
        return self.result()

    def result(self):
        return {"events": self.events, "errors": self.errors, "frames": self.frames, "t_end": self.d.time}
