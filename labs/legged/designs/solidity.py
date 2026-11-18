"""Solid limbs on solid obstacles: the scene's own collision check, fed back to the IK.

``Solid`` wraps the scene's MuJoCo model (the frame with its props: the wall and its rungs, the sill, the door) and
reports every penetration of a limb into a prop or into another limb for a pose, as push-out targets
(body, local point, world goal) that move the penetrating point out along the contact normal. ``kerchak_tasks
.contact_motion`` re-solves each frame with them until the frame is clear, and the players' JSON carries the worst
depth left. The hands are left out (the fingers close round the bars on purpose); a foot standing on its hold is
allowed the sole's tolerance."""
import numpy as np

__all__ = ["Solid", "SKIP_BODIES", "TOL"]

SKIP_BODIES = ("hand", "finger", "thumb")      #: body-name fragments whose geoms are not checked: the hands close round the bars (the forearms are checked)
TOL = 0.012                                    #: a penetration deeper than this [m] counts (a sole on its bar sits at ~0)
MARGIN = 0.02                                  #: the push-out puts the point this far clear of the surface [m]


class Solid:
    def __init__(self, model, kin, qadr, machine_bodies=None):
        import copy
        import mujoco
        model = copy.deepcopy(model)                                       # its own copy: every piece of the frame made to collide
        self.mj, self.m, self.d = mujoco, model, mujoco.MjData(model)
        self.kin, self.qadr = kin, np.asarray(qadr)
        self.scene = None                                                  # fn(u) -> {joint name: value}: the door's hinge, etc.
        self.ignore = set()                                                # geom names not checked for now (the pane once broken)
        kin_names = {kin.model.body(i).name for i in range(kin.model.nbody)} - {"world"}
        names = set(machine_bodies) if machine_bodies is not None else kin_names
        self.robot_bodies = {i for i in range(model.nbody) if model.body(i).name in names}
        self.skip = {i for i in self.robot_bodies if any(k in model.body(i).name for k in SKIP_BODIES)}
        for g in range(model.ngeom):                                       # the casing's covers and shells too (drawn, so solid), not the masses
            if int(model.geom_bodyid[g]) in self.robot_bodies and not model.geom(g).name.startswith("pm:"):
                model.geom_contype[g], model.geom_conaffinity[g] = 2, 1
        self.jadr = {model.joint(i).name: model.jnt_qposadr[i] for i in range(model.njnt)}

    def set(self, q, base, pitch, u=None):
        import kerchak_motions as gm
        d = self.d
        d.qpos[:] = self.m.qpos0
        d.qpos[0:3] = base
        d.qpos[3:7] = gm.base_quat(pitch)
        d.qpos[self.qadr] = q
        if self.scene is not None and u is not None:
            for j, v in self.scene(u).items():
                d.qpos[self.jadr[j]] = v
        self.mj.mj_kinematics(self.m, d)
        self.mj.mj_collision(self.m, d)

    def penetrations(self, q, base, pitch, u=None, tol=TOL):
        """[(body name, local point, world goal, depth)] for every contact deeper than ``tol``: a limb's point and
        where it must go to be ``MARGIN`` clear. A limb against another limb gives one for each."""
        self.set(q, base, pitch, u)
        m, d, out = self.m, self.d, []
        for k in range(d.ncon):
            c = d.contact[k]
            depth = -float(c.dist)
            if depth <= tol or (self.ignore and (m.geom(c.geom1).name in self.ignore or m.geom(c.geom2).name in self.ignore)):
                continue
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            n = np.array(c.frame[0:3])                                     # from geom1 towards geom2
            for b, sgn in ((b1, -1.0), (b2, 1.0)):
                if b not in self.robot_bodies or b in self.skip:
                    continue
                other = b2 if b == b1 else b1
                share = 0.5 if other in self.robot_bodies and other not in self.skip else 1.0
                p = np.array(c.pos)
                local = d.xmat[b].reshape(3, 3).T @ (p - d.xpos[b])
                out.append((m.body(b).name, local, p + sgn * n * (share * depth + MARGIN), depth, m.geom(c.geom2 if b == b1 else c.geom1).name))
        return out

    def worst(self, q, base, pitch, u=None):
        pens = self.penetrations(q, base, pitch, u, tol=0.0)
        return max((p[3] for p in pens), default=0.0), sorted({f"{p[0]}>{p[4]}" for p in pens if p[3] > TOL})

    def worst_qpos(self, qpos):
        """(worst depth, the pairs in) for a whole scene state (a player's qpos: the base, the frame, the scene's joints)."""
        d = self.d
        d.qpos[:] = qpos
        self.mj.mj_kinematics(self.m, d)
        self.mj.mj_collision(self.m, d)
        m, worst, pairs = self.m, 0.0, set()
        for k in range(d.ncon):
            c = d.contact[k]
            depth = -float(c.dist)
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if self.ignore and (m.geom(c.geom1).name in self.ignore or m.geom(c.geom2).name in self.ignore):
                continue
            if not ((b1 in self.robot_bodies and b1 not in self.skip) or (b2 in self.robot_bodies and b2 not in self.skip)):
                continue
            worst = max(worst, depth)
            if depth > TOL:
                pairs.add(f"{m.geom(c.geom1).name}>{m.geom(c.geom2).name}")
        return worst, sorted(pairs)
