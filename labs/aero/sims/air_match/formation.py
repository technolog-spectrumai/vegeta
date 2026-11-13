"""The formation chase: seven Émerillons chase a hare round the field in formation, through both hoops, in MuJoCo.

The **hare** is an eighth aircraft flying a racetrack course: along the field's axis through hoop A and hoop B, round
the end, back down the far straight, round the other end. The **formation** hangs on a reference point that follows
the hare's own track a few metres behind it (``swarm.PathReference``): on the straights the seven fly a **wedge**
(a V, the first member at the apex); for the turns and the hoops they change to a **column** that follows the
reference's path like a train, single file through the ring, and open into the wedge again behind it. Every change
re-assigns the slots (the Hungarian algorithm: the least sum of squared moves) and moves them over on a smoothstep.
A member's command is its slot's velocity plus a gain on its slot error, with separation from its neighbours as a
reflex on top, then the match's steering round balloons, the ground and the hoops. The swarm algorithms are Nomios's
(``swarm.py``).

What is counted: each member's slot error, the closest pair, the gap to the hare, the hoops flown through, the
contacts (a member that touches something goes back to its slot from above). There is no score.
"""
from __future__ import annotations

import math

import numpy as np

from . import arena as ar
from . import match as mt
from . import swarm as sw

HARE_SPEED = 8.0                   # m/s round the course
FOLLOW = 5.0                       # the reference flies this far behind the hare, on the hare's track [m]
SPACING = 2.4                      # between members [m]
TURN_R = 7.0


def course(a: ar.Arena) -> tuple:
    """The racetrack as a closed polyline with, per point, whether the formation flies there in column (the hoops and
    the turns) or may open into the wedge (the straights)."""
    zc = float(a.hoops["A"][2])
    hx, far, zb = a.hoop_x, 2 * TURN_R, zc + 2.6
    pts = []

    def add(x, y, z, column):
        pts.append((x, y, z, column))

    for x in np.arange(-hx - 6.0, hx + 6.01, 2.0):                  # the axis: through A, the middle, through B
        add(x, 0.0, zc, x < -hx + 16.0 or x > hx - 27.0)               # in column until the tail is through A, and again in time for B
    for ang in np.radians(np.arange(-80.0, 81.0, 10.0)):            # round the +x end
        add(hx + 6.0 + TURN_R * math.cos(ang), TURN_R + TURN_R * math.sin(ang), zc + (zb - zc) * (ang + 1.4) / 2.8, True)
    for x in np.arange(hx + 6.0, -hx - 6.01, -2.0):                 # the far straight
        add(x, far, zb, x > hx - 4.0 or x < -hx + 6.0)
    for ang in np.radians(np.arange(100.0, 261.0, 10.0)):           # round the -x end
        add(-hx - 6.0 + TURN_R * math.cos(ang), TURN_R + TURN_R * math.sin(ang), zb - (zb - zc) * (ang - 1.745) / 2.8, True)
    arr = np.array(pts)
    return arr[:, :3], arr[:, 3] > 0.5


class Formation(mt.Match):
    def __init__(self, n: int = 7, seed: int = 1, duration: float = 60.0, variant: str = "ER2", n_balloons: int = 6):
        hoop_x = 32.0                                                # a longer field than the match's: room for the wedge between the hoops
        a = ar.Arena(length=2 * (hoop_x + 6.0 + TURN_R + 16.0), width=2 * (2 * TURN_R + 10.0), hoop_x=hoop_x, start_x=hoop_x - 8.0,
                     n_balloons=n_balloons, seed=seed)
        self.way, self.col = course(a)
        zc = float(a.hoops["A"][2])
        start = self.way[0].copy()
        hare = mt.Player("hare", variant, "H", mt.Pilot("the hare", v_max=HARE_SPEED, caution=0.9), home=start, color="#555555")
        self.offsets = {"wedge": sw.wedge_offsets(n, SPACING, climb=0.25), "column": sw.column_offsets(n, SPACING)}
        homes = [start + np.array([-FOLLOW - o[0] * -1.0 if False else -FOLLOW + o[0], o[1], o[2]]) for o in self.offsets["column"]]
        members = [mt.Player(f"f{i + 1}", variant, "F", mt.Pilot(f"wing {i + 1}", v_max=14.0, caution=0.7), home=homes[i], color=ar.PALETTE[i % len(ar.PALETTE)])
                   for i in range(n)]
        self.members, self.hare = members, hare
        super().__init__(a, [hare] + members, mt.Rules(duration=duration, possession=False, kickoff_s=0.6, penalty_s=0.0), seed=seed)
        self.ref = sw.PathReference(start + np.array([-FOLLOW, 0.0, 0.0]), 0.0, keep_m=FOLLOW + n * SPACING + 6.0)
        self.track = sw.PathReference(start, 0.0, keep_m=FOLLOW + 4.0)   # the hare's own track: the reference rides it
        self.layout, self.slot_of = "column", list(range(n))           # member i flies slot slot_of[i]
        self.morph, self.old = None, None                              # a change in progress: (Morph, the old layout and slots)
        self.i_way, self.changes, self.frames = 1, [], []

    # ------------------------------------------------------------------------------------------------ the slots
    def slot(self, layout: str, k: int) -> tuple:
        o = self.offsets[layout][k]
        return self.ref.path_slot(-o[0]) if layout == "column" else self.ref.rigid_slot(o)

    def target(self, i: int, t: float) -> tuple:
        """Member ``i``'s slot now: its place in the current layout, or on its way there from the old one."""
        p1, v1 = self.slot(self.layout, self.slot_of[i])
        if self.morph is None:
            return p1, v1
        layout0, slots0 = self.old
        p0, v0 = self.slot(layout0, slots0[i])
        w, rate = self.morph.weight(t)
        return p0 + (p1 - p0) * w, v0 + (v1 - v0) * w + (p1 - p0) * rate

    def change(self, layout: str, t: float) -> None:
        """Take a new layout: the members get the slots that cost the least squared moves, and the slots move over."""
        new = [self.slot(layout, k)[0] for k in range(len(self.members))]
        here = [self.target(i, t)[0] for i in range(len(self.members))]
        m = sw.Morph(here, new, self.members[0].pilot.v_max, t)
        self.old = (self.layout, list(self.slot_of))
        if self.morph is not None:                                   # a change on top of a change: start from where the slots are
            self.offsets["_frozen"] = np.array([self._local(h) for h in here])
            self.old = ("_frozen", list(range(len(self.members))))
        self.layout, self.slot_of, self.morph = layout, list(m.assign), m
        self.changes.append({"t": round(t, 2), "to": layout, "seconds": round(m.T, 2), "crossings": m.crossings,
                             "kept_numbers_crossings": sw.Morph.count_crossings(here, new, list(range(len(here))))})
        self.events.append({"t": round(t, 2), "what": "formation", "to": layout, "seconds": round(m.T, 2), "crossings": m.crossings})

    def _local(self, world) -> np.ndarray:
        d = np.asarray(world, float) - self.ref.pos
        c, s = math.cos(-self.ref.heading), math.sin(-self.ref.heading)
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])

    # ------------------------------------------------------------------------------------------------ the pilots
    def decide(self, me: mt.Player, t: float) -> tuple:
        a = self.a
        p = self.pos(me)
        if me is self.hare:
            while float(np.linalg.norm(self.way[self.i_way][:2] - p[:2])) < 3.0:
                self.i_way = (self.i_way + 1) % len(self.way)
            to = self.way[self.i_way] - p
            v_want = to / max(float(np.linalg.norm(to)), 1e-6) * HARE_SPEED
            hoop = min(a.hoops.values(), key=lambda hc: abs(hc[0] - p[0]) + 5.0 * abs(p[1]))
            me.role = "attack"
            return self.steer(me, v_want, {q.name: 0.8 for q in self.members}, hoop if abs(p[1]) < 2.0 else None, cap=HARE_SPEED * 1.2), "attack"
        i = self.members.index(me)
        slot_p, slot_v = self.target(i, t)
        v_want = sw.slot_command(p, slot_p, slot_v, me.pilot.v_max)
        mates = [self.pos(q) for q in self.members if q is not me] + [self.pos(self.hare)]
        v_want = v_want + sw.separation(p, mates, 2 * me.n["reach"])
        hoop = min(a.hoops.values(), key=lambda hc: abs(hc[0] - p[0]) + 5.0 * abs(p[1]))
        in_line = self.layout == "column" and abs(p[1]) < 2.0 and abs(hoop[0] - p[0]) < 12.0
        me.role = "mark" if self.layout == "wedge" else "chase"
        rooms = {q.name: me.n["reach"] + q.n["reach"] + 0.25 for q in self.players if q is not me}   # separation does the rest
        return self.steer(me, v_want, rooms, hoop if in_line else None, cap=me.pilot.v_max), me.role

    def restart(self, p: mt.Player, t: float, hold: float):
        """A member that touched something comes back above its slot, flying with it."""
        d = self.data
        if p is self.hare:
            pos, vel = self.way[self.i_way] + np.array([0.0, 0.0, 1.5]), np.zeros(3)
        else:
            pos, vel = self.target(self.members.index(p), t)
            pos = pos + np.array([0.0, 0.0, 1.6])
        d.qpos[p.qa:p.qa + 3] = pos
        d.qpos[p.qa + 3:p.qa + 7] = (1, 0, 0, 0)
        d.qvel[p.va:p.va + 3] = vel
        d.qvel[p.va + 3:p.va + 6] = 0.0
        p.thrust[:] = p.ctl.m0 * mt.G / 4
        p.dv[:] = 0.0
        p.ctl.reset_integral()
        p.frozen_until = t + 0.3                                     # a moment's grace from the referee, not a hover at home

    def crossed(self, p: mt.Player, hoop: str, good: bool, t: float) -> bool:
        if good:
            p.passes += 1
        elif abs(self.pos(p)[1]) < 3.0:                              # at the hoop, not through it (the far straight crosses the same plane)
            self.events.append({"t": round(t, 2), "what": "missed", "who": p.name, "hoop": hoop})
        return False

    # --------------------------------------------------------------------------------------------------- run
    def run(self, duration: float | None = None, record_dt: float = 0.04) -> dict:
        mj, m, d = self.mj, self.model, self.data
        duration = self.rules.duration if duration is None else duration
        next_rec = 0.0
        while d.time < duration:
            t = d.time
            ph, vh = self.pos(self.hare), self.vel(self.hare)
            self.track.update(ph, vh, self.control_dt)
            rp, rv = self.track.path_slot(FOLLOW)                    # the reference: on the hare's track, FOLLOW behind
            self.ref.update(rp, rv, self.control_dt)
            j = int(np.argmin(np.linalg.norm(self.way[:, :2] - self.ref.pos[:2], axis=1)))
            ahead = [(j + k) % len(self.way) for k in range(0, 3)]   # the course a few points ahead decides the shape
            want = "column" if any(self.col[k] for k in ahead) else "wedge"
            if self.morph is not None and self.morph.done(t):
                self.morph, self.old = None, None
            if want != self.layout and (self.morph is None or want == "column") and t > 1.0:
                self.change(want, t)
            before = {p.name: self.pos(p) for p in self.players}
            for p in self.players:
                if t < self.rules.kickoff_s:
                    self.actuate(p, None, hover_at=p.home)
                else:
                    self.actuate(p, self.decide(p, t)[0])
            self.drift()
            for _ in range(self.n_sub):
                mj.mj_step(m, d)
            self.referee(d.time, before)
            if d.time >= next_rec:
                next_rec += record_dt
                P = np.array([self.pos(q) for q in self.members])
                err = [float(np.linalg.norm(self.target(i, d.time)[0] - P[i])) for i in range(len(self.members))]
                dd = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2) + np.eye(len(P)) * 99.0
                self.frames.append((d.time, float(np.mean(err)), float(np.max(err)), float(dd.min()), float(np.linalg.norm(self.pos(self.hare) - P[self.slot_of.index(0)])),
                                    1.0 if self.layout == "wedge" else 0.0, 0.0 if self.morph is None else 1.0))
                for p in self.players:
                    R = d.xmat[p.body].reshape(3, 3)
                    p.rows.append((d.time, *self.pos(p), *self.vel(p), *R[:, 2], {"attack": 0, "mark": 1, "restart": 2, "chase": 3, "race": 4}[p.role], p.wh, 0.0))
                self.balloon_rows.append(np.array([d.xpos[b].copy() for b in self.balloons]))
                self.qpos_rows.append(d.qpos.copy())
                self.token_rows.append((self.token.copy(), "none"))
        r = self.result()
        f = np.array(self.frames)
        fly = f[:, 0] > 4.0
        wedge, column, moving = fly & (f[:, 5] > 0.5) & (f[:, 6] < 0.5), fly & (f[:, 5] < 0.5) & (f[:, 6] < 0.5), fly & (f[:, 6] > 0.5)
        laps = min(q.passes for q in self.members) // 2
        r["formation"] = {"t": f[:, 0], "err_mean": f[:, 1], "err_max": f[:, 2], "closest": f[:, 3], "gap_to_hare": f[:, 4], "wedge": f[:, 5], "morphing": f[:, 6],
                          "changes": self.changes, "course": self.way, "spacing": SPACING,
                          "summary": {"members": len(self.members), "slot_error_wedge_m": float(f[wedge, 1].mean()) if wedge.any() else None,
                                      "slot_error_column_m": float(f[column, 1].mean()) if column.any() else None,
                                      "slot_error_changing_m": float(f[moving, 1].mean()) if moving.any() else None,
                                      "worst_slot_error_m": float(f[fly, 2].max()), "closest_pair_m": float(f[fly, 3].min()),
                                      "touching_at_m": 2 * self.members[0].n["reach"], "gap_to_hare_m": float(f[fly, 4].mean()),
                                      "changes": len(self.changes), "hoops_flown": {q.name: q.passes for q in self.players},
                                      "missed_hoops": sum(1 for e in self.events if e["what"] == "missed"), "laps": laps,
                                      "contacts": sum(q.fouls + q.crashes for q in self.players)}}
        return r


def fly(n: int = 7, duration: float = 60.0, seed: int = 1) -> dict:
    return Formation(n, seed, duration).run()
