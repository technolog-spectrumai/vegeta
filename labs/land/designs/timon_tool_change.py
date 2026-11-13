"""Timon changes Pumba's tool — a kinematic animation of the service (``sims/timon_tool_change_movie.py``).

Pumba parks and presents its main arm to the left, the grab hanging over the ground beside it. Timon drives up
nose-on, its second hand (Pekari-2M's arm) steadies Pumba's wrist, its main arm takes the grab off Pumba's quick
coupler, sets it in the rack on its deck, picks up the 2 kW chainsaw from the rack and brings it to Pumba's coupler;
the coupler latches, Timon lets go and folds, Pumba lifts the saw to its cutting pose, Timon backs away.

What it is: both machines are their Chiron models (``timon_robot.timon``, ``pumba_robot.pumba``) moved through
joint-space trajectories between IK solutions (``pekari_arms.ik``) — MuJoCo forward kinematics, **not** a dynamics
run (Chiron simulates one robot per world; the coupler's latch, the contact of the tools and the robots' masses
during the handover are not simulated). The tools are the CAD-consistent geoms of Pumba's grab and saw models,
carried as rigid bodies: on Pumba's hand, in Timon's hand (in the pose they had when it took them), in the rack.

    import timon_tool_change as ttc
    scene = ttc.Scene()
    frames = scene.frames(fps=25)        # [(t, phase, [(kind, size, rgba, pos, mat)])]
"""
from __future__ import annotations

import math

import numpy as np

import pekari_arms as pa
import pekari_family as pf
import pumba_robot as pu
import timon_robot as tr

__all__ = ["Scene", "KinematicModel"]

GEOM_KIND = {2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box"}


def _rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _T(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


class KinematicModel:
    """A Chiron robot compiled alone in MuJoCo: place its base, set its joints, read every geom's world pose."""

    def __init__(self, robot):
        import mujoco
        from vegeta import chiron as ch

        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_string(robot.to_mjcf(ch.Flat()))
        self.d = mujoco.MjData(self.m)
        self.robot = robot
        self.jadr = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j): int(self.m.jnt_qposadr[j])
                     for j in range(self.m.njnt)}
        self.body = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b): b for b in range(self.m.nbody)}
        self.geoms = [g for g in range(self.m.ngeom) if int(self.m.geom_type[g]) in GEOM_KIND and self.m.geom_bodyid[g] != 0]
        self.z0 = float(robot.nominal_base_height) if getattr(robot, "nominal_base_height", None) else None

    def set(self, base_xy, yaw, joints: dict, z: float):
        q = self.d.qpos
        q[:] = self.m.qpos0
        a = self.jadr["hull:free"] if "hull:free" in self.jadr else 0
        q[a:a + 3] = (base_xy[0], base_xy[1], z)
        q[a + 3:a + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        for j, v in joints.items():
            if j in self.jadr:
                q[self.jadr[j]] = v
        self.mj.mj_kinematics(self.m, self.d)

    def body_T(self, name):
        b = self.body[name]
        return _T(self.d.xmat[b].reshape(3, 3).copy(), self.d.xpos[b].copy())

    def geom_list(self, skip_bodies=()):
        sb = {self.body[n] for n in skip_bodies if n in self.body}
        out = []
        for g in self.geoms:
            if self.m.geom_bodyid[g] in sb:
                continue
            out.append((GEOM_KIND[int(self.m.geom_type[g])], tuple(self.m.geom_size[g]), tuple(self.m.geom_rgba[g]),
                        self.d.geom_xpos[g].copy(), self.d.geom_xmat[g].reshape(3, 3).copy()))
        return out

    def subtree_local(self, root: str):
        """The geoms of body ``root`` and its descendants in ``root``'s frame (at the current pose)."""
        rb = self.body[root]
        Ti = np.linalg.inv(self.body_T(root))
        ids = set()
        for b in range(self.m.nbody):
            p = b
            while p not in (0, rb):
                p = int(self.m.body_parentid[p])
            if p == rb:
                ids.add(b)
        out = []
        for g in self.geoms:
            if self.m.geom_bodyid[g] in ids:
                Tg = _T(self.d.geom_xmat[g].reshape(3, 3), self.d.geom_xpos[g])
                out.append((GEOM_KIND[int(self.m.geom_type[g])], tuple(self.m.geom_size[g]), tuple(self.m.geom_rgba[g]),
                            Ti @ Tg))
        return out, ids


def _smooth(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


class Scene:
    """The service as keyframes (see the module)."""

    PUMBA_HAND = (0.35, 1.25, 0.80)               # where Pumba presents its coupler (world) [m]
    TIMON_DOCK = (0.30, 2.35)                     # Timon's hull centre when docked, facing -y
    TIMON_START = (0.30, 4.6)

    def __init__(self):
        self.timon_r = tr.timon(payload_kg=0.0)
        self.pg = pu.pumba(payload_kg=0.0)                       # Pumba with the grab
        self.ps = pu.pumba(tool="chainsaw 2 kW", payload_kg=0.0)
        self.T = KinematicModel(self.timon_r)
        self.P = KinematicModel(self.pg)
        self.S = KinematicModel(self.ps)
        self.zT = pf.z_hull(tr.TIMON) + 0.02
        self.zP = pf.z_hull(pu.PUMBA) + 0.02
        self.parm = pu.PUMBA.mount("main").arm
        self.tmain = tr.TIMON.mount("main").arm
        self.taux = tr.TIMON.mount("aux").arm
        # the tools in their hand frames
        self.P.set((0, 0), 0.0, dict(self.pg.nominal_qpos), self.zP)
        self.grab_local, self.grab_bodies = self.P.subtree_local("main_hand")
        self.S.set((0, 0), 0.0, dict(self.ps.nominal_qpos), self.zP)
        self.saw_local, _ = self.S.subtree_local("main_hand")
        self._plan()

    # ---- joint targets
    def _pumba_q(self, world, elev=-math.pi / 2):
        sh = pf.shoulder(pu.PUMBA, "main")
        q = pa.ik(self.parm, world, elev, sh)                     # Pumba at the origin, facing +x
        assert q is not None, world
        return pa.qpos(self.parm, "main_", q)

    def _timon_q(self, mount, world, base_xy, elevs=(-math.pi / 2, -math.pi / 4, 0.0)):
        arm = tr.TIMON.mount(mount).arm
        R = _rotz(-math.pi / 2)
        sh_h = np.array(pf.shoulder(tr.TIMON, mount))
        base = np.array([base_xy[0], base_xy[1], 0.0])
        p_h = R.T @ (np.asarray(world, float) - base)            # world -> Timon's frame (z above the ground)
        for e in elevs:
            q = pa.ik(arm, tuple(p_h), e, tuple(sh_h))
            if q is not None:
                return pa.qpos(arm, f"{mount}_", q)
        raise ValueError(f"Timon's {mount} cannot reach {world}")

    def _plan(self):
        dock = self.TIMON_DOCK
        R = _rotz(-math.pi / 2)
        hand = np.array(self.PUMBA_HAND)
        roof_T = pf.roof_z(tr.TIMON)
        deck_w = lambda xh, yh, z: tuple(R @ np.array([xh, yh, 0.0]) + np.array([dock[0], dock[1], z]))  # noqa: E731
        self.rack_grab = deck_w(-0.10, 0.09, roof_T + 0.30)
        self.rack_saw = deck_w(-0.50, -0.09, roof_T + 0.13)
        g = {k: v for k, v in pa.qpos(self.tmain, "main_", {}).items()}
        fold_T = {**pa.qpos(self.tmain, "main_", pa.pose(self.tmain, "folded")),
                  **pa.qpos(self.taux, "aux_", pa.pose(self.taux, "folded"))}
        fold_P = pa.qpos(self.parm, "main_", pa.pose(self.parm, "folded"))
        present = self._pumba_q(tuple(hand))
        above_hand = tuple(hand + np.array([0.0, 0.0, 0.35]))
        take = tuple(hand + np.array([0.0, 0.0, 0.20]))          # Timon's grip on the tool's body
        steady = tuple(hand + np.array([0.0, 0.18, 0.10]))        # the second hand on Pumba's wrist housing
        aux_steady = self._timon_q("aux", steady, dock, (0.0, -math.pi / 4, -math.pi / 2))
        cut = pa.qpos(self.parm, "main_", pa.pose(pa.with_tool(self.parm, "chainsaw 2 kW"), "cut"))
        K = []                                                   # (t, label, timon base y, timon joints, pumba joints)

        def key(t, label, ty, tq, pq, ev=None):
            K.append((t, label, ty, dict(tq), dict(pq), ev))
        ty0, ty1 = self.TIMON_START[1], dock[1]
        key(0.0, "Pumba parks; Timon on its way", ty0, fold_T, fold_P)
        key(4.0, "Pumba presents its quick coupler", ty0 - 0.8, fold_T, present)
        key(8.0, "Timon docks beside Pumba", ty1, fold_T, present)
        key(10.5, "the second hand steadies Pumba's wrist", ty1, {**fold_T, **aux_steady}, present)
        key(13.0, "Timon's main arm over the grab", ty1, {**fold_T, **aux_steady, **self._timon_q("main", tuple(hand + [0, 0, 0.55]), dock)}, present)
        key(15.0, "takes hold of the grab", ty1, {**fold_T, **aux_steady, **self._timon_q("main", take, dock)}, present, "grip_grab")
        key(16.5, "the coupler unlatches", ty1, {**fold_T, **aux_steady, **self._timon_q("main", take, dock)}, present, "take_grab")
        key(18.5, "lifts the grab off", ty1, {**fold_T, **aux_steady, **self._timon_q("main", tuple(hand + [0, 0, 0.55]), dock)}, present)
        key(22.0, "into the rack on Timon's deck", ty1, {**fold_T, **aux_steady, **self._timon_q("main", self.rack_grab, dock)}, present, "rack_grab")
        key(24.5, "to the chainsaw", ty1, {**fold_T, **aux_steady, **self._timon_q("main", tuple(np.array(self.rack_saw) + [0, 0, 0.10]), dock)}, present, "take_saw")
        key(27.0, "lifts the chainsaw", ty1, {**fold_T, **aux_steady, **self._timon_q("main", tuple(hand + [0, 0, 0.60]), dock)}, present)
        key(30.0, "onto Pumba's coupler", ty1, {**fold_T, **aux_steady, **self._timon_q("main", tuple(hand + [0, 0, 0.30]), dock)}, present, "latch_saw")
        key(31.5, "the coupler latches", ty1, {**fold_T, **aux_steady, **self._timon_q("main", tuple(hand + [0, 0, 0.30]), dock)}, present)
        key(34.0, "Timon lets go and folds", ty1, fold_T, present)
        key(38.0, "Pumba lifts the saw to cut; Timon backs away", ty1 + 1.4, fold_T, cut)
        key(41.0, "Pumba: chainsaw on the main arm", ty1 + 2.2, fold_T, cut)
        self.keys = K
        self.duration = K[-1][0]

    # ---- the timeline
    def state(self, t):
        K = self.keys
        i = max(k for k in range(len(K)) if K[k][0] <= t)
        a, b = K[i], K[min(i + 1, len(K) - 1)]
        s = 0.0 if b[0] == a[0] else _smooth((t - a[0]) / (b[0] - a[0]))
        mix = lambda qa, qb: {k: qa.get(k, 0.0) + (qb.get(k, qa.get(k, 0.0)) - qa.get(k, 0.0)) * s for k in set(qa) | set(qb)}  # noqa: E731
        return a[1], a[2] + (b[2] - a[2]) * s, mix(a[3], b[3]), mix(a[4], b[4])

    def frames(self, fps: int = 25):
        """[(t, label, geoms)] at ``fps``; geoms (kind, size, rgba, pos, mat) in the world."""
        out = []
        grab_on, saw_on = "pumba", "rack"                       # where each tool is
        grab_rel = saw_rel = None
        done = set()
        n = int(self.duration * fps) + 1
        R = _rotz(-math.pi / 2)
        for k in range(n):
            t = k / fps
            label, ty, tq, pq = self.state(t)
            base = (self.TIMON_DOCK[0], ty)
            self.T.set(base, -math.pi / 2, {**self.timon_r.nominal_qpos, **tq}, self.zT)
            self.P.set((0.0, 0.0), 0.0, {**self.pg.nominal_qpos, **pq}, self.zP)
            Th = self.T.body_T("main_hand")
            Ph = self.P.body_T("main_hand")
            Tb = _T(R, np.array([base[0], base[1], 0.0]))
            for kt, _, _, _, _, ev in self.keys:              # events fire once when their key is reached
                if ev and t >= kt and ev not in done:
                    done.add(ev)
                    if ev == "take_grab":
                        grab_on, grab_rel = "timon", np.linalg.inv(Th) @ Ph
                    elif ev == "rack_grab":
                        grab_on, grab_rel = "rack", np.linalg.inv(Tb) @ (Th @ grab_rel)
                    elif ev == "take_saw":
                        saw_T = Tb @ self._saw_rack_rel()
                        saw_on, saw_rel = "timon", np.linalg.inv(Th) @ saw_T
                    elif ev == "latch_saw":
                        saw_on = "pumba"
            geoms = self.T.geom_list() + self.P.geom_list(skip_bodies=("main_hand", "main_grip_a_link", "main_grip_b_link"))
            # Timon's own gripper stays; Pumba's hand subtree is drawn as the tool wherever it is
            for which, on, rel, local in (("grab", grab_on, grab_rel, self.grab_local), ("saw", saw_on, saw_rel, self.saw_local)):
                if which == "grab":
                    Tt = Ph if on == "pumba" else (Th @ rel if on == "timon" else Tb @ rel)
                else:
                    Tt = Ph if on == "pumba" else (Th @ rel if on == "timon" else Tb @ self._saw_rack_rel())
                for kind, size, rgba, Tl in local:
                    Tw = Tt @ Tl
                    geoms.append((kind, size, rgba, Tw[:3, 3], Tw[:3, :3]))
            out.append((t, label, geoms))
        return out

    def _saw_rack_rel(self):
        """The saw lying in the rack, in Timon's frame: the bar along Timon's -x, on the deck."""
        roof = pf.roof_z(tr.TIMON)
        return _T(_rotz(math.pi), np.array([0.20, -0.09, roof + 0.13]))
