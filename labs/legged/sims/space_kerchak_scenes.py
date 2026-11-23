#!/usr/bin/env python3
"""Belisario in its scenarios (``kerchak_scenarios``): the pre-programmed sequence of a scene played in MuJoCo -
the climbing wall, the tree, the ship's hull, the rope, the window, the obstacles, the door.

    python space_kerchak_scenes.py --scene wall                       # the keyframes played (default), the movie and stills
    python space_kerchak_scenes.py --scene wall --mode physics        # dynamics with the assist (a climb needs grips the
                                                                      #   assist does not give: expect the report to say so)
    python space_kerchak_scenes.py --scene tree --stills 8,14,20

Kinematic mode drives the pelvis along the motions' tracks (x, y, height, pitch per frame) and the joints along the
keyframes; the door's lever and hinge follow the hand, the rope stays put. Writes ``sims/output/space_kerchak_<scene>.mp4``,
``.json`` and ``space_kerchak_<scene>_<t>s.png`` stills, drawn by ``belisario_render`` (the CAD on the bodies: the
skeleton, the modules, the struts, the bays, the plates) unless ``--render boxes``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "labs" / "legged" / "designs"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gibbon_scenarios  # noqa: E402,F401  (registers the gibbon's scenes)
import space_unit_specs as su  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import kerchak_motions as gm  # noqa: E402
import kerchak_scenarios as gs  # noqa: E402
import motion_kit as mk  # noqa: E402
import space_squad_march as march_sim  # noqa: E402
import space_unit_robots as sr  # noqa: E402
from vegeta.chiron import SimOptions  # noqa: E402

OUT = march_sim.OUT
VIEWS = {"fire": (14.0, 225.0), "heist": (14.0, 225.0), "heist_jump": (14.0, 225.0), "tree_spider": (10.0, 235.0), "chimney": (10.0, 240.0), "course": (18.0, 232.0), "brachiate": (16.0, 228.0), "brachiate_up": (14.0, 230.0), "wall": (12.0, 265.0), "tree": (12.0, 265.0), "hull": (12.0, 265.0), "rope": (12.0, 265.0), "window": (14.0, 230.0),
         "window_hammer": (14.0, 230.0), "obstacles": (12.0, 265.0), "door": (14.0, 215.0)}


class ScenePlayer:
    def __init__(self, scene, mode, params=None, casing="none", machine="kerchak"):
        self.scene, self.mode, self.machine = scene, mode, machine
        if machine in ("humanoid", "gibbon", "spider"):                             # the upright frames: they stand, and walk
            import humanoid_motions as hum
            self.robot = {"humanoid": sr.humanoid, "gibbon": sr.gibbon, "spider": sr.spider}[machine](casing=casing)
            self.kin = mk.Kin(self.robot)
            self.q_quad = hum.stand_pose(self.kin)
            if machine in ("gibbon", "spider"):                            # the long arms held up: they would drag on the ground hanging
                for side in ("L", "R"):
                    self.q_quad[self.kin.index[f"{side}_shoulder_pitch"]], self.q_quad[self.kin.index[f"{side}_elbow"]] = -2.2, -0.9
            gm.set_stance(hum.base_height(self.kin, self.q_quad), 0.0, upright=True, frame=machine)
            S = gm.scale()
            if machine == "humanoid":
                self.walk = hum.walk(self.kin, speed=1.0, stance=0.6 * S, duty=0.6, step=0.08 * S, crouch=0.06, arm=0.4)
            else:                                                          # the gibbon's bipedal walk: short quick steps, the long arms held up and out
                self.walk = hum.walk(self.kin, speed=0.6, stance=0.16, duty=0.6, step=0.04, crouch=0.03, arm=0.15, arm_offset=-2.2, elbow=-0.9)
            self.walk_h = hum.gait_height(self.kin, self.walk, self.q_quad)   # the walk's pelvis height: the soles on the ground, not in it
        else:
            self.robot = sr.bare(casing=casing)
            self.kin = mk.Kin(self.robot)
            self.q_quad = gm.quad_pose(self.kin)
            gm.set_stance(gm.PELVIS_H, gm.PELVIS_PITCH, upright=False)
            self.walk = gm.knuckle_walk(self.kin, self.q_quad)
            self.walk_h = gm.STANCE["h"]
        self.v = float(self.walk.base["velocity_m_s"][0])
        gs.set_hand({"gibbon": su.GIBBON, "spider": su.SPIDER, "humanoid": su.HUMANOID}.get(machine, su.MARINE))   # the grip from this frame's own hand
        opts = SimOptions()
        opts.course_extent = (-2.0, 14.0, -3.0, 3.0)
        xml = self.robot.to_mjcf(options=opts, props=list(gs.props(scene, params)))
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.qadr = self.kin.qadr
        import kerchak_tasks as gt
        import solidity
        self.solid = solidity.Solid(self.model, self.kin, self.qadr)       # the limbs solid on the props: the IK re-solves penetrating frames
        gt.SOLID = self.solid
        try:
            self.segments = gs.sequence(scene, self.kin, self.q_quad, params)
        finally:
            gt.SOLID = None
        t = 0.0
        self.starts = []
        for seg in self.segments:
            self.starts.append(t)
            t += seg[1] if seg[0] == "gait" else seg[1].period
        self.total = t
        m = self.model
        self.act_joint = [int(m.actuator_trnid[i][0]) for i in range(m.nu)]
        self.act_col = [self.kin.hinges.index(j) for j in self.act_joint]
        servos = {j.name: j.servo for j in self.robot.joints() if j.servo is not None}
        names = [m.joint(j).name for j in self.act_joint]
        self.kp = np.array([servos[n].kp for n in names])
        self.lim = np.array([servos[n].stall_torque for n in names])
        self.x_kin, self.fell, self.tilt_max = 0.0, None, 0.0
        self.strike_t = None                                               # when the pane goes (window_hammer)
        self._hammer = None
        self._pack = None
        if "backpack" in [self.model.body(i).name for i in range(self.model.nbody)]:   # the heist: a backpack on the thorax's back
            m = self.model
            freep = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and m.body(m.jnt_bodyid[j]).name == "backpack"]
            self._pack = {"adr": m.jnt_qposadr[freep[0]], "thorax": m.body("thorax").id}
        self.impact = None
        if scene in ("window_hammer", "heist", "heist_jump"):
            m = self.model
            free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and m.body(m.jnt_bodyid[j]).name == "hammer"]
            by_name = {seg[1].name: (s0, seg[1]) for s0, seg in zip(self.starts, self.segments) if seg[0] == "motion"}
            pick, smash, drop = by_name["pick_up_the_hammer"], by_name["smash_the_glass"], by_name["drop_the_hammer"]
            R_h2m, p_h2m = (np.array(v) for v in pick[1].params["hammer_T"])
            rest_p, rest_R = (np.array(v) for v in pick[1].params.get("hammer_rest", ([0.0, 0.0, 0.0], np.eye(3).tolist())))
            self._hammer = {"adr": m.jnt_qposadr[free[0]], "R_h2m": R_h2m, "p_h2m": p_h2m, "rest": (rest_p, rest_R),
                            "t_grasp": pick[0] + pick[1].period, "t_release": drop[0] + drop[1].params["release_u"] * drop[1].period,
                            "released": None, "hand": m.body(pick[1].params.get("hand", "R_hand")).id}
            if "hammer_carry" in pick[1].params:                           # carried in the pack on a body until the grasp
                bn, Rc, pc = pick[1].params["hammer_carry"]
                self._hammer["carry"] = (m.body(bn).id, np.array(Rc), np.array(pc))
            self.impact = smash[1].params.get("impact")
            self.strike_t = smash[0] + smash[1].params["strike_u"] * smash[1].period
        self._fire = None
        if scene == "fire":
            m = self.model
            fj = lambda name: m.jnt_qposadr[[j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and m.body(m.jnt_bodyid[j]).name == name][0]]
            by_name = {seg[1].name: (s0, seg[1]) for s0, seg in zip(self.starts, self.segments) if seg[0] == "motion"}
            rd, tk, dr, sp = by_name["ride_in_on_chuck"], by_name["take_the_hose"], by_name["draw_the_nozzle"], by_name["open_the_valve_and_spray"]
            T_ = lambda key, src: tuple(np.array(v) for v in src[1].params[key])
            v0, v1 = sp[1].params["valve_u"]
            u0, u1 = sp[1].params["spray_u"]
            at = lambda src, u: src[0] + u * src[1].period
            self._fire = {"hat": fj("fire_hat"), "nozzle": fj("nozzle"), "chuck": fj("chuck"), "pack": m.body("backpack").id,
                          "thorax": m.body("thorax").id, "hand": m.body("R_hand").id,
                          "quad": rd[1].params["quad"], "ride": (rd[0], rd[0] + rd[1].period),
                          "hook": np.array(tk[1].params["hook"]), "T_take": T_("nozzle_T", tk), "t_take": at(tk, tk[1].params["take_u"]), "t_clip": at(tk, tk[1].params["clip_u"]),
                          "T_draw": T_("nozzle_T", dr), "T_aim": T_("nozzle_T_aim", dr), "t_grasp": at(dr, dr[1].params["grasp_u"]),
                          "t_aim": at(sp, v0), "t_valve": (at(sp, v0), at(sp, v1)), "t_on": at(sp, u1 * 0 + u0), "t_off": at(sp, u1),
                          "t_out": at(sp, sp[1].params["out_u"]), "fire": np.array(sp[1].params["fire"]), "trail": [], "trail_t": -1.0}
        self.reset()

    def segment(self, t):
        for s, seg in zip(reversed(self.starts), reversed(self.segments)):
            if t >= s:
                return seg, t - s
        return self.segments[0], 0.0

    def _gait_q(self, t):
        g = self.walk
        Q, keys = np.asarray(g.Q), len(g.t)
        s = (t % g.period) / g.period * keys
        i, f = int(s) % keys, s - int(s)
        q = self.q_quad.copy()
        q[[self.kin.index[j] for j in g.joints]] = (1 - f) * Q[i] + f * Q[(i + 1) % keys]
        return q

    def phase(self, t):
        """(q, base x or None, y, speed, height, pitch, what)."""
        seg, tl = self.segment(t)
        if seg[0] == "gait":
            return self._gait_q(tl), None, seg[2], self.v, self.walk_h + (seg[3] if len(seg) > 3 else 0.0), gm.STANCE["pitch"], "walking on four" if self.machine == "kerchak" else "walking"
        m = seg[1]
        Q, n = np.asarray(m.Q), len(m.t) - 1
        s = min(max(tl / m.period, 0.0), 1.0) * n
        i, f = min(int(s), n - 1), min(s - int(s), 1.0)
        q = self.kin.q0.copy()                                             # a motion's joints are offsets from the design pose
        q[[self.kin.index[j] for j in m.joints]] = (1 - f) * Q[i] + f * Q[i + 1]
        b = m.base
        lerp = lambda key: (1 - f) * b[key][i] + f * b[key][i + 1]
        return q, lerp("x_m"), lerp("y_m"), 0.0, lerp("height_m"), lerp("pitch_rad"), m.name

    def reset(self):
        d = self.data
        d.qpos[:] = self.model.qpos0
        d.qpos[0:3] = [0.0, 0.0, gm.STANCE["h"] + (0.01 if self.mode == "physics" else 0.0)]
        d.qpos[3:7] = gm.base_quat(gm.STANCE["pitch"])
        d.qpos[self.qadr] = self.phase(0.0)[0]
        d.qvel[:] = 0.0
        mujoco.mj_forward(self.model, d)

    def step_kinematic(self, t, dt):
        q, x, y, v, h, p, _ = self.phase(t)
        if x is None:
            self.x_kin += v * dt
        else:
            self.x_kin = x
        d = self.data
        d.qpos[0:3] = [self.x_kin, y, h]
        d.qpos[3:7] = gm.base_quat(p)
        d.qpos[self.qadr] = q
        self._scene_joints(t)
        mujoco.mj_kinematics(self.model, d)
        if self._pack is not None:                                         # the backpack on the thorax's back
            import gibbon_scenarios as gb
            Rt = d.xmat[self._pack["thorax"]].reshape(3, 3)
            qq = np.zeros(4)
            mujoco.mju_mat2Quat(qq, Rt.reshape(9))
            a = self._pack["adr"]
            d.qpos[a:a + 3], d.qpos[a + 3:a + 7] = d.xpos[self._pack["thorax"]] + Rt @ np.array(gb.BACKPACK["on_thorax"]), qq
            mujoco.mj_kinematics(self.model, d)
        if self._hammer is not None:
            self._hammer_pose(t)
            mujoco.mj_kinematics(self.model, d)
        if self._fire is not None:
            self._fire_props(t)
            mujoco.mj_kinematics(self.model, d)

    def shards(self, t):
        """The pane's shards after the blow: a jittered grid of glass triangles thrown from the impact point into the
        room, spinning and falling, lying on the floor where they land. (triangles, colours) for the renderer."""
        if self.strike_t is None or t < self.strike_t:
            return None
        if not hasattr(self, "_shards"):
            import kerchak_scenarios as gs
            m = self.model
            g = m.geom("glass").id
            c, hx, hy, hz = self.data.geom_xpos[g].copy(), *m.geom_size[g]
            rng = np.random.default_rng(7)
            ny, nz = 7, 6
            ys = np.linspace(-hy, hy, ny + 1)
            zs = np.linspace(-hz, hz, nz + 1)
            Y, Z = np.meshgrid(ys, zs, indexing="ij")
            Y[1:-1, 1:-1] += rng.uniform(-0.3, 0.3, (ny - 1, nz - 1)) * (2 * hy / ny)
            Z[1:-1, 1:-1] += rng.uniform(-0.3, 0.3, (ny - 1, nz - 1)) * (2 * hz / nz)
            impact = np.array(self.impact) if self.impact is not None else np.array([c[0], -0.15, gs.WINDOW["sill"] + gs.WINDOW["height"] * 0.45])
            tris, vel, spin = [], [], []
            for i in range(ny):
                for j in range(nz):
                    q = [np.array([c[0], c[1] + Y[a, b], c[2] + Z[a, b]]) for a, b in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))]
                    for tri in ((q[0], q[1], q[2]), (q[0], q[2], q[3])):
                        tri = np.array(tri)
                        cen = tri.mean(axis=0)
                        out = cen - impact
                        d = np.linalg.norm(out[1:])
                        v = np.array([2.2 + rng.uniform(0.0, 1.5), 0.0, 0.0]) + np.array([0.0, out[1], out[2]]) / max(d, 0.05) * (1.8 - min(d, 0.6) * 1.5)
                        tris.append(tri); vel.append(v); spin.append(rng.normal(0.0, 1.0, 3) * 6.0)
            self._shards = {"tris": np.array(tris), "vel": np.array(vel), "spin": np.array(spin), "landed": [None] * len(tris)}
        S = self._shards
        tau = t - self.strike_t
        out = []
        for k, tri in enumerate(S["tris"]):
            if S["landed"][k] is not None:
                out.append(S["landed"][k]); continue
            cen = tri.mean(axis=0)
            p = cen + S["vel"][k] * tau + np.array([0.0, 0.0, -0.5 * 9.81 * tau * tau])
            w = S["spin"][k]
            ang = np.linalg.norm(w) * tau
            axis = w / max(np.linalg.norm(w), 1e-9)
            K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
            R = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K
            shard = (tri - cen) @ R.T + p
            if p[2] < 0.02:                                                # on the floor: flat where it fell, for good
                flat = shard.copy(); flat[:, 2] = 0.008
                S["landed"][k] = flat
                shard = flat
            out.append(shard)
        tris = np.array(out)
        cols = np.repeat([[0.72, 0.86, 0.92]], len(tris), axis=0)
        return tris, cols

    def _fire_props(self, t):
        """Chuck driving up and parked; the helmet on the head; the nozzle on Chuck's hook, in the hand, in the holster
        on the pack, in the hand again for the fire (turned to point along the jet as it is aimed)."""
        import gibbon_scenarios as gb
        import kerchak_scenarios as gs
        F, d = self._fire, self.data

        def put(adr, p, R):
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).reshape(9))
            d.qpos[adr:adr + 3], d.qpos[adr + 3:adr + 7] = p, q
        Q = F["quad"]
        r0, r1 = F["ride"]
        u = min(1.0, max(0.0, (t - r0) / (r1 - r0)))
        put(F["chuck"], np.array([Q["x0"] + (Q["x1"] - Q["x0"]) * mk.smooth(u), Q["y"], 0.0]), np.eye(3))
        Rt, pt_ = d.xmat[F["thorax"]].reshape(3, 3), d.xpos[F["thorax"]]
        put(F["hat"], pt_ + Rt @ np.array(gb.HAT_ON_THORAX), Rt)
        Rh, ph = d.xmat[F["hand"]].reshape(3, 3), d.xpos[F["hand"]]
        if t < F["t_take"]:
            cpos = d.qpos[F["chuck"]:F["chuck"] + 3]
            put(F["nozzle"], cpos + np.array(gb.CHUCK["hook"]) - np.array([0.0, 0.0, 0.12]), np.eye(3))
        elif t < F["t_clip"]:
            Rr, pr = F["T_take"]
            put(F["nozzle"], ph + Rh @ pr, Rh @ Rr)
        elif t < F["t_grasp"]:
            put(F["nozzle"], pt_ + Rt @ gb.HOLSTER_ON_THORAX[0], Rt)
        else:
            (R0, p0), (R1, p1) = F["T_draw"], F["T_aim"]
            f = mk.smooth(min(1.0, max(0.0, (t - F["t_grasp"]) / max(F["t_aim"] - F["t_grasp"], 0.1))))
            put(F["nozzle"], ph + Rh @ ((1 - f) * p0 + f * p1), Rh @ gs._slerp(R0, R1, f))

    def extras(self, t):
        """Everything the renderer adds this frame: the pane's shards, the hose, the water jet, the flames."""
        parts = [x for x in (self.shards(t), self._fire_extras(t) if self._fire is not None else None) if x is not None and len(x[0])]
        if not parts:
            return None
        return np.concatenate([p_[0] for p_ in parts]), np.concatenate([p_[1] for p_ in parts])

    def _fire_extras(self, t):
        import gibbon_scenarios as gb
        F, d, m = self._fire, self.data, self.model
        tris, cols = [], []

        def tube(pts, r, col, k=8):
            for a, b in zip(pts[:-1], pts[1:]):
                ax = b - a
                n = np.linalg.norm(ax)
                if n < 1e-6:
                    continue
                ax = ax / n
                u = np.cross(ax, [0.0, 0.0, 1.0] if abs(ax[2]) < 0.9 else [1.0, 0.0, 0.0]); u /= np.linalg.norm(u)
                v = np.cross(ax, u)
                for i in range(k):
                    a0, a1 = 2 * math.pi * i / k, 2 * math.pi * (i + 1) / k
                    o0, o1 = r * (math.cos(a0) * u + math.sin(a0) * v), r * (math.cos(a1) * u + math.sin(a1) * v)
                    tris.extend([[a + o0, b + o0, b + o1], [a + o0, b + o1, a + o1]]); cols.extend([col, col])
        # the hose: off Chuck's reel, along the ground and up the way the monkey went (the pack's trail), to the nozzle
        import gibbon_scenarios as gb
        Rt, pt_ = d.xmat[F["thorax"]].reshape(3, 3), d.xpos[F["thorax"]]
        nid = m.body("nozzle").id
        Rn, pn = d.xmat[nid].reshape(3, 3), d.xpos[nid]
        root = pn
        cid = m.body("chuck").id
        reel = d.xpos[cid] + np.array(gb.CHUCK["reel"]) + np.array([0.0, -0.17, -0.1])
        P = gb.FIRE
        X, F_, xl = P["wall_x"], P["floor"], P["wall_x"] - P["wall_t"] / 2 - P["rung_out"]
        pack = pt_ + Rt @ np.array([-0.12, -0.08, -0.12])
        if t >= F["t_clip"] and t > F["trail_t"] + 0.2:                    # where the pack has been: the hose lies there
            F["trail"].append(pack.copy()); F["trail_t"] = t
        pts = [reel, reel + np.array([0.0, -0.05, -0.3])]
        if t < F["t_take"]:
            pts += [pn]
        else:
            way = []
            for q_ in (F["trail"] if t >= F["t_clip"] else []):
                q_ = q_.copy()
                if q_[0] < xl + 0.05:                                      # outside: on the ground, or against the ladder's face while climbing
                    if q_[2] < 0.9:
                        q_[2] = 0.03
                    else:
                        q_[0] = xl - 0.04
                        q_[2] -= 0.15
                elif q_[0] < X + P["wall_t"] / 2 + 0.05:                   # over the sill
                    q_[2] = gb._fire_sill(P) + 0.08
                else:                                                      # on the first floor
                    q_[2] = F_ + 0.03
                way.append(q_)
            if way and way[0][2] < 0.1:
                pts.append(np.array([pts[-1][0], pts[-1][1], 0.03]))
            pts += way
            pts.append(pack)
        pts.append(root)
        P_ = np.array(pts)
        if len(P_) > 3:                                                    # a smooth line through them (Catmull-Rom)
            sm = []
            for i in range(len(P_) - 1):
                p0_, p1_, p2_, p3_ = P_[max(i - 1, 0)], P_[i], P_[i + 1], P_[min(i + 2, len(P_) - 1)]
                for s_ in np.linspace(0, 1, 4, endpoint=False):
                    sm.append(0.5 * ((2 * p1_) + (-p0_ + p2_) * s_ + (2 * p0_ - 5 * p1_ + 4 * p2_ - p3_) * s_ ** 2 + (-p0_ + 3 * p1_ - 3 * p2_ + p3_) * s_ ** 3))
            sm.append(P_[-1])
            P_ = np.array(sm)
        tube(P_, 0.016, (0.72, 0.12, 0.1))
        # the nozzle's valve: its bail lever forward (shut), pulled back upright (open) by the left hand
        va, vb = F["t_valve"]
        open_ = mk.smooth(min(1.0, max(0.0, (t - va) / max(vb - va, 0.05))))
        if t > F["t_grasp"] and t > F["t_off"] + 0.3:
            open_ = 0.0
        piv = pn - Rn[:, 2] * 0.1 + Rn[:, 1] * 0.022                   # the shut-off valve at the hose's coupling behind the nozzle
        ang = (1 - open_) * 0.2 + open_ * 1.4
        lever = piv + 0.07 * (math.cos(ang) * Rn[:, 2] + math.sin(ang) * Rn[:, 1])
        tube(np.array([piv, lever]), 0.006, (0.95, 0.75, 0.1), k=6)
        # the flames: tongues on the table, dying from the moment the water reaches them
        fc = F["fire"]
        life = 1.0 if t < F["t_on"] + 0.3 else max(0.0, 1.0 - mk.smooth((t - F["t_on"] - 0.3) / max(F["t_out"] - F["t_on"] - 0.3, 0.1)))
        rng = np.random.default_rng(int(t * 30))
        if life > 0.0:
            for i in range(26):
                ang = 2 * math.pi * i / 26
                rr = 0.2 * ((i * 7) % 5) / 5
                b = fc + np.array([rr * math.cos(ang), rr * math.sin(ang) * 1.3, 0.0])
                h = life * (0.25 + 0.2 * rng.random()) * (1.0 - 0.6 * rr / 0.2)
                w_ = 0.05 + 0.03 * rng.random()
                sway = np.array([0.03 * rng.normal(), 0.03 * rng.normal(), 0.0])
                for k_, (c_, f_) in enumerate((((1.0, 0.45, 0.05), 1.0), ((1.0, 0.85, 0.2), 0.55))):
                    hh, ww = h * f_, w_ * f_
                    for dirv in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
                        tris.append([b - dirv * ww + np.array([0, 0, 0.002 * k_]), b + dirv * ww + np.array([0, 0, 0.002 * k_]), b + sway + np.array([0.0, 0.0, hh])]); cols.append(c_)
        if t > F["t_on"] + 0.3:                                            # the char where it burnt
            for i in range(10):
                a0, a1 = 2 * math.pi * i / 10, 2 * math.pi * (i + 1) / 10
                tris.append([fc + [0, 0, 0.003], fc + [0.24 * math.cos(a0), 0.3 * math.sin(a0), 0.003], fc + [0.24 * math.cos(a1), 0.3 * math.sin(a1), 0.003]])
                cols.append((0.12, 0.1, 0.09))
        # the jet: drops leaving the tip at 7 m/s along the nozzle, falling, stopped by the table top or the floor
        if F["t_on"] <= t <= F["t_off"] + 0.5:
            tip = pn + Rn[:, 2] * gb.NOZZLE["length"]
            v0 = 7.0
            for k in range(150):
                age = k * 0.006
                if t - age < F["t_on"] or t - age > F["t_off"]:
                    continue
                q = tip + Rn[:, 2] * v0 * age + np.array([0.0, 0.0, -0.5 * 9.81 * age * age])
                q = q + np.random.default_rng(k + int(t * 50)).normal(0.0, 0.004 + 0.02 * age, 3)
                if q[2] < fc[2] + 0.01 and abs(q[0] - fc[0]) < 0.4 and abs(q[1] - fc[1]) < 0.6:
                    q[2] = fc[2] + 0.015
                if q[2] < fc[2] - 0.8:
                    continue
                sz = 0.012 + 0.02 * age
                tris.append([q + [-sz, 0, 0], q + [sz, 0, 0], q + [0, 0, sz * 1.5]]); cols.append((0.55, 0.75, 0.95))
                tris.append([q + [0, -sz, 0], q + [0, sz, 0], q + [0, 0, -sz * 1.2]]); cols.append((0.75, 0.88, 1.0))
            if t > F["t_out"] - 0.5 and t < F["t_off"] + 1.5:              # steam off the doused fire
                for k in range(24):                                        # puffs: soft octagons rising and growing
                    g_ = np.random.default_rng(k)
                    rise = ((t * 0.5 + k / 24.0) % 1.0)
                    c = fc + np.array([g_.normal(0, 0.1), g_.normal(0, 0.12), 0.08 + 0.7 * rise])
                    sz = 0.03 + 0.07 * rise
                    for i in range(8):
                        a0, a1 = 2 * math.pi * i / 8, 2 * math.pi * (i + 1) / 8
                        tris.append([c, c + sz * np.array([math.cos(a0) * 0.7, math.cos(a0) * 0.7, math.sin(a0)]), c + sz * np.array([math.cos(a1) * 0.7, math.cos(a1) * 0.7, math.sin(a1)])])
                        cols.append((0.9, 0.9, 0.92))
        return np.array(tris, dtype=float), np.array(cols, dtype=float)

    def _hammer_pose(self, t):
        """The hammer through the sequence: lying where the hand finds it, in the hand from the grasp to the release,
        then falling (no tumble) and lying where it lands."""
        H, d, m = self._hammer, self.data, self.model
        adr = H["adr"]

        def put(p, R):
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).reshape(9))
            d.qpos[adr:adr + 3], d.qpos[adr + 3:adr + 7] = p, q

        if t < H["t_grasp"] and "carry" in H:                              # in the backpack, wherever the body is
            bid, Rc, pc = H["carry"]
            Rb = d.xmat[bid].reshape(3, 3)
            put(d.xpos[bid] + Rb @ pc, Rb @ Rc)
        elif t < H["t_grasp"]:
            put(*H["rest"])
        elif t < H["t_release"]:
            Rh, ph = d.xmat[H["hand"]].reshape(3, 3), d.xpos[H["hand"]]
            put(ph + Rh @ H["p_h2m"], Rh @ H["R_h2m"])
            H["released"] = (ph + Rh @ H["p_h2m"], Rh @ H["R_h2m"])
        else:
            if H["released"] is None:
                H["released"] = H["rest"]
            p0, R0 = H["released"]
            import kerchak_scenarios as gs
            Lh, rz = gs.HAMMER["handle"], gs.HAMMER["rest_z"]
            low = min(p0[2] - 0.018, p0[2] + Lh * R0[2, 2] - 0.03)          # the hammer's lowest point at the release
            tau = t - H["t_release"]
            t_land = math.sqrt(2.0 * max(low, 0.0) / 9.81)                 # when its lowest point meets the ground
            ax = R0[:, 2].copy(); ax[2] = 0.0
            ax = ax / max(np.linalg.norm(ax), 1e-6)
            z = np.array([0.0, 0.0, 1.0])
            R_flat = np.stack([np.cross(z, ax), z, ax], axis=1)
            p_flat = np.array([p0[0], p0[1], rz])
            if tau < t_land:
                put(p0 - np.array([0.0, 0.0, 0.5 * 9.81 * tau * tau]), R0)
            else:                                                          # landed: it tips flat over a quarter second where it fell, then stays
                f = mk.smooth(min(1.0, (tau - t_land) / 0.25))
                p_land = p0 - np.array([0.0, 0.0, low])
                put((1 - f) * p_land + f * p_flat, gs._slerp(R0, R_flat, f))

    def _scene_joints(self, t):
        """The scene's own joints along the sequence (kinematic playback: the door's lever and hinge follow the hand)."""
        if self.scene in ("heist", "heist_jump"):                                        # the sash: shoved open by the hammer, then it swings on
            m = self.model
            adr = m.jnt_qposadr[m.joint("sash_hinge").id]
            seg = [(s0, sg[1]) for s0, sg in zip(self.starts, self.segments) if sg[0] == "motion" and sg[1].name == "push_the_window"]
            if seg:
                s0, mo = seg[0]
                a, b = mo.params["shove_u"]
                ta, tb = s0 + a * mo.period, s0 + b * mo.period
                sh, op = mo.params["shove_rad"], mo.params["open_rad"]
                if t < ta:
                    phi = 0.0
                elif t < tb:
                    phi = sh * mk.smooth((t - ta) / (tb - ta))
                else:
                    phi = sh + (op - sh) * mk.smooth(min(1.0, (t - tb) / 2.0))
                self.data.qpos[adr] = phi
            return
        if self.scene not in ("door", "course"):
            return
        import kerchak_tasks as gt
        m = self.model
        lever, hinge = m.jnt_qposadr[m.joint("handle_hinge").id], m.jnt_qposadr[m.joint("door_hinge").id]
        seg, tl = self.segment(t)
        name = seg[1].name if seg[0] == "motion" else ""
        u = min(max(tl / seg[1].period, 0.0), 1.0) if seg[0] == "motion" else 0.0
        done = [s[1].name for s in self.segments[:self.segments.index(seg)] if s[0] == "motion"]
        turned = gt.DOOR["turn_rad"] * (mk.smooth(u) if name == "turn" else 1.0 if "turn" in done else 0.0)
        if name == "release":
            turned *= 1.0 - mk.smooth(min(1.0, u / 0.25))
        elif "release" in done:
            turned = 0.0
        opened = gt.door_open() * (mk.smooth(u) if name == "push" else 1.0 if "push" in done else 0.0)
        if name == "release":                                              # the shoved door swings on as the hand lets go
            opened = gt.door_open() + (gt.DOOR["swung_rad"] - gt.door_open()) * mk.smooth(u)
        elif "release" in done:
            opened = gt.DOOR["swung_rad"]
        self.data.qpos[lever], self.data.qpos[hinge] = turned, opened

    def step_physics(self, t, dt, assist):
        m, d = self.model, self.data
        n = max(1, int(round(dt / m.opt.timestep)))
        for k in range(n):
            q_t, x, y, v, h, p, _ = self.phase(t + k * m.opt.timestep)
            q = d.qpos[self.qadr]
            tau = np.clip(self.kp * (q_t[self.act_col] - q[self.act_col]), -self.lim, self.lim)
            d.ctrl[:] = tau
            d.xfrc_applied[1][:] = 0.0
            if assist > 0.0:
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, d.qpos[3:7])
                R = R.reshape(3, 3)
                err = np.cross(R[:, 2], gm.base_rot(p)[:, 2])
                w = d.qvel[3:6]
                yaw = math.atan2(R[1, 0], R[0, 0])
                torque = assist * (6000.0 * err - 400.0 * w) + np.array([0.0, 0.0, -assist * (3000.0 * yaw + 300.0 * w[2])])
                fx = -assist * (4000.0 * (d.qpos[0] - x) + 2000.0 * d.qvel[0]) if x is not None else -assist * march_sim.V_HOLD_C * (d.qvel[0] - v)
                force = np.array([fx, -assist * (2000.0 * (d.qpos[1] - y) + 400.0 * d.qvel[1]), 0.0])
                d.xfrc_applied[1][0:3] = force
                d.xfrc_applied[1][3:6] = torque
            mujoco.mj_step(m, d)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(R.reshape(3, 3)[:, 2] @ gm.base_rot(self.phase(t)[5])[:, 2])))))
        self.tilt_max = max(self.tilt_max, tilt)
        if self.fell is None and (tilt > 60.0 or d.qpos[2] < 0.2):
            self.fell = round(t, 2)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Belisario in a scenario")
    ap.add_argument("--scene", choices=sorted(gs.SCENARIOS), default="wall")
    ap.add_argument("--mode", choices=("kinematic", "physics"), default="kinematic")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--assist", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--stills", default="", help="times [s], comma-separated, for still images")
    ap.add_argument("--no-movie", action="store_true")
    ap.add_argument("--param", action="append", default=[], help="scene parameter key=value")
    ap.add_argument("--render", choices=("mesh", "boxes"), default="mesh", help="the CAD on the bodies (belisario_render), or the collision shapes")
    ap.add_argument("--casing", choices=("none", "plastic"), default="plastic")
    ap.add_argument("--machine", choices=("kerchak", "humanoid", "gibbon", "spider"), default="kerchak", help="the frame: Kerchak on four, the upright humanoid, the gibbon or the spider monkey")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--chunk", default="", help="i/n: draw only every n-th frame from i into --frames-dir (parallel renders); no movie is assembled")
    ap.add_argument("--from-s", type=float, default=0.0, help="draw only the frames from this time on (re-drawing part of a --frames-dir)")
    ap.add_argument("--frames-dir", default="", help="where the frames go (default: a temporary directory)")
    a = ap.parse_args(argv)
    if os.environ.get("VEGETA_DEBUG") == "1" and argv is None and len(sys.argv) == 1:
        a.seconds, a.no_movie = 0.5, True
    params = {}
    for kv in a.param:
        k, v = kv.split("=", 1)
        params[k] = float(v)
    OUT.mkdir(parents=True, exist_ok=True)
    g = ScenePlayer(a.scene, a.mode, params, casing=a.casing, machine=a.machine)
    tag = f"space_{a.machine}"
    seconds = a.seconds if a.seconds is not None else g.total
    elev, azim = VIEWS.get(a.scene, (12.0, 265.0))
    span = 3.2
    stills = [float(x) for x in a.stills.split(",") if x.strip()]
    tmp = a.frames_dir or tempfile.mkdtemp()
    os.makedirs(tmp, exist_ok=True)
    ci, cn = (int(v) for v in a.chunk.split("/")) if a.chunk else (0, 1)
    if a.render == "mesh":
        import belisario_mesh as bm
        import belisario_render as br
        w, h = (int(v) for v in a.size.split("x"))
        import space_unit_specs as su
        renderer = br.Renderer(g.model, bm.load(a.casing, P={"humanoid": su.HUMANOID, "gibbon": su.GIBBON, "spider": su.SPIDER}.get(a.machine)), w, h)

    def draw(path, t, what):
        c = np.array(g.data.qpos[0:3])
        if a.render == "mesh":
            az = azim
            if a.scene in ("window", "window_hammer"):                     # once through the wall the camera comes round to the inside
                f = mk.smooth(min(1.0, max(0.0, (c[0] - (gs.WINDOW["x"] - 0.1)) / 0.9)))
                az = azim + 60.0 * f
            renderer.hidden = {"glass"} if (g.strike_t is not None and t >= g.strike_t) else set()
            if a.scene in ("heist", "heist_jump", "fire"):                                       # through the window: the walls stay, the camera comes round into the room
                P = gs.params_for(a.scene, params)
                wx = P["wall_x"] if "wall_x" in P else P["x"] + P["wall_gap"]
                f = mk.smooth(min(1.0, max(0.0, (c[0] - (wx - 0.2)) / 0.6)))
                az = azim + 110.0 * f
                if a.scene == "fire" and f > 0.2:                          # the camera beyond the room: the walls and the roof between it and the monkey drawn away
                    renderer.hidden |= {"back_wall", "side_left", "side_right", "roof"}
            if a.scene == "course":                                        # inside the house: the walls behind it drawn away, the camera keeps its side
                P = gs.params_for("course", params)
                if c[0] > P["window_x"] - 0.1:
                    renderer.hidden |= {"wall_above", "wall_below", "wall_left", "wall_right", "sill_plate"}
                if c[0] > P["door_x"] - 0.05:
                    renderer.hidden |= {"door_wall_left", "door_wall_right", "post_left", "post_right", "lintel", "door_panel", "handle_shaft", "handle_lever"}
            renderer.extra = g.extras(t)
            cam = br.orbit((c[0], c[1], max(c[2], 0.8)), elev, az, 4.6)
            renderer.render(g.data, cam, title=f"{ {'kerchak': 'Kerchak', 'humanoid': 'the humanoid', 'gibbon': 'the gibbon', 'spider': 'the spider monkey'}[a.machine]}: {a.scene}   -   t = {t:4.1f} s   -   {what}  ({a.mode})").save(path)
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        geoms, centre = march_sim.snapshot([g])
        fig = plt.figure(figsize=(12.8, 7.2))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        cc = np.array([centre[0], centre[1], max(centre[2], 1.0)])
        march_sim.draw_frame(ax, geoms, cc, t, what, span, elev=elev, azim=azim)
        ax.set_zlim(max(0.0, cc[2] - 0.6 * span), cc[2] + 0.6 * span)
        ax.set_title(f"Belisario: {a.scene}   -   t = {t:4.1f} s   -   {what}  ({a.mode})", fontsize=11, loc="left", y=0.96)
        fig.subplots_adjust(left=0, right=1, top=1.0, bottom=0)
        fig.savefig(path, dpi=72)
        plt.close(fig)

    t, dt, k = 0.0, 1.0 / a.fps, 0
    while t <= seconds + 1e-9:
        if a.mode == "kinematic":
            g.step_kinematic(t, dt)
        else:
            g.step_physics(t, dt, a.assist)
        what = g.phase(t)[6]
        if not a.no_movie and k % cn == ci and t >= a.from_s - 1e-9:
            draw(os.path.join(tmp, f"f{k:04d}.png"), t, what)
        for ts in (stills if ci == 0 else []):
            if abs(t - ts) < dt / 2:
                draw(str(OUT / f"{tag}_{a.scene}_{ts:g}s.png"), t, what)
        t += dt
        k += 1
    report = {"scene": a.scene, "mode": a.mode, "seconds": seconds, "total_s": round(g.total, 2), "x_m": round(float(g.data.qpos[0]), 3),
              "z_m": round(float(g.data.qpos[2]), 3), "fell_at_s": g.fell, "tilt_off_phase_max_deg": round(g.tilt_max, 1),
              "segments": [(s[0], round(s[1] if s[0] == "gait" else s[1].period, 2), s[2] if s[0] == "gait" else s[1].name) for s in g.segments],
              "ik_error_max_m": {s[1].name: s[1].check.get("ik_error_max_m") for s in g.segments if s[0] == "motion"}}
    (OUT / f"{tag}_{a.scene}.json").write_text(json.dumps(report, indent=1))
    if not a.no_movie and not a.chunk:
        out = str(OUT / f"{tag}_{a.scene}.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(tmp, "f%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", out], check=True)
        report["movie"] = out
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
