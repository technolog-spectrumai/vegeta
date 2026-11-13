"""The team chase: three reds flee, three blues hunt them, in MuJoCo. The reds get a head start.

The reds start in the middle of the field; the blues wait at one end, hovering, for ``head_start_s`` before they may go.
A blue **tags** a red by flying within ``tag_radius`` of it for ``tag_dwell_s`` without touching it: the red is out
(it climbs off the field and waits). The blues win if every red is out before the clock stops; otherwise the reds win,
and the reds' survival seconds are the score. Touching anything (another aircraft, a balloon, a hoop, the ground) or
leaving the field sends that aircraft back to its start for ``penalty_s`` — a blue that rams a red does not tag it.

**Red strategies** (``RED``): ``scatter`` — flee the nearest blues and spread away from the other reds, so the blues
must split up; ``flock`` — flee together, held toward the reds' centroid; ``jink`` — scatter, and when a blue is close
break hard sideways and up or down instead of running straight.

**Blue strategies** (``BLUE``): ``assigned`` — one blue per red, re-assigned every second by the Hungarian algorithm on
the distances (``swarm.min_cost_assignment``); ``nearest`` — every blue goes for the red nearest to it; ``pincer`` —
assigned, but a second blue that is free (its red is out) comes at the same red from the other side.

What it is not: rules, not learners; no aerodynamic moments; the hoops stand on the field as obstacles only.
"""
from __future__ import annotations

import math

import numpy as np

from . import arena as ar
from . import match as mt
from . import swarm as sw

RED = ("scatter", "flock", "jink")
BLUE = ("assigned", "nearest", "pincer")
RED_COLORS = ("#d62728", "#ff7f0e", "#e377c2")
BLUE_COLORS = ("#1f77b4", "#17becf", "#9467bd")


class TeamChase(mt.Match):
    def __init__(self, red: str = "scatter", blue: str = "assigned", n: int = 3, seed: int = 1, duration: float = 60.0, head_start_s: float = 3.0,
                 tag_radius: float = 2.0, tag_dwell_s: float = 0.35, v_red: float = 14.0, v_blue: float = 15.0, variant: str = "ER2", n_balloons: int = 8):
        a = ar.Arena(length=72.0, width=40.0, n_balloons=n_balloons, seed=seed)
        self.red_tactic, self.blue_tactic = red, blue
        self.head_start_s, self.tag_radius, self.tag_dwell_s = head_start_s, tag_radius, tag_dwell_s
        reds = [mt.Player(f"red{i + 1}", variant, "red", mt.Pilot(f"red {i + 1}", v_max=v_red, caution=1.0), home=np.array([0.0, (i - (n - 1) / 2) * 6.0, 4.0 + i % 2]),
                          color=RED_COLORS[i % 3]) for i in range(n)]
        blues = [mt.Player(f"blue{i + 1}", variant, "blue", mt.Pilot(f"blue {i + 1}", v_max=v_blue, caution=0.9),
                           home=np.array([-a.length / 2 + 6.0, (i - (n - 1) / 2) * 6.0, 4.0 + i % 2]), color=BLUE_COLORS[i % 3]) for i in range(n)]
        self.reds, self.blues = reds, blues
        super().__init__(a, reds + blues, mt.Rules(duration=duration, possession=False, kickoff_s=0.5, penalty_s=2.0), seed=seed)
        for b in blues:
            b.frozen_until = head_start_s
        self.caught = {}                                              # red name -> time
        self.dwell = {r.name: 0.0 for r in reds}
        self.assign, self.next_assign = {}, 0.0
        self.parked = 0

    # ---------------------------------------------------------------------------------------------- who is in play
    def live(self, team, t):
        return [p for p in (self.reds if team == "red" else self.blues) if p.name not in self.caught and t >= p.frozen_until]

    def in_game_reds(self):
        return [r for r in self.reds if r.name not in self.caught]

    # ------------------------------------------------------------------------------------------------ the blues
    def targets(self, t: float) -> dict:
        """Which red each blue hunts."""
        reds = self.in_game_reds()
        if not reds:
            return {}
        blues = self.blues
        if self.blue_tactic == "nearest":
            return {b.name: min(reds, key=lambda r: float(np.linalg.norm(self.pos(r) - self.pos(b)))) for b in blues}
        if t >= self.next_assign or any(self.assign.get(b.name) not in reds for b in blues):
            self.next_assign = t + 1.0
            k = len(reds)
            cols = [reds[j % k] for j in range(len(blues))]          # more blues than reds: some reds get two
            cost = [[float(np.linalg.norm(self.pos(b) - self.pos(c))) ** 2 for c in cols] for b in blues]
            a = sw.min_cost_assignment(cost)
            self.assign = {b.name: cols[a[i]] for i, b in enumerate(blues)}
        return self.assign

    def decide_blue(self, me: mt.Player, t: float):
        quarry = self.targets(t).get(me.name)
        if quarry is None:
            return self.steer(me, -self.vel(me)), "chase"
        me.role = "chase"
        _, v_want = self.chase_point(me, quarry)
        if self.blue_tactic == "pincer":                              # a second hunter on the same red comes from its other side
            mates = [b for b in self.blues if b is not me and self.targets(t).get(b.name) is quarry]
            if mates and self.blues.index(me) > self.blues.index(mates[0]):
                pq, vq = self.pos(quarry), self.vel(quarry)
                other = pq + (pq - self.pos(mates[0])) / max(float(np.linalg.norm(pq - self.pos(mates[0]))), 1e-6) * 1.4
                v_want = vq + 2.5 * (other - self.pos(me))
                n_ = float(np.linalg.norm(v_want))
                v_want *= min(1.0, me.pilot.v_max / max(n_, 1e-6))
        rooms = {quarry.name: me.n["reach"] + quarry.n["reach"] + 0.35}
        return self.steer(me, v_want, rooms), "chase"

    # ------------------------------------------------------------------------------------------------- the reds
    def decide_red(self, me: mt.Player, t: float):
        a, pl = self.a, me.pilot
        p, v = self.pos(me), self.vel(me)
        away = np.zeros(3)
        blues = [b for b in self.blues if t >= b.frozen_until - 1.0]
        near = 99.0
        for b in blues:
            d = p - self.pos(b) - self.vel(b) * 0.4                   # from where it will be in a moment
            dn = max(float(np.linalg.norm(d)), 0.5)
            near = min(near, dn)
            away += d / dn ** 2
        away[2] *= 0.4
        mates = [r for r in self.in_game_reds() if r is not me]
        if self.red_tactic in ("scatter", "jink"):
            for r in mates:
                d = p - self.pos(r)
                away += 0.5 * d / max(float(np.dot(d, d)), 1.0)
        elif mates:                                                   # flock: toward the reds' centroid
            c = np.mean([self.pos(r) for r in mates + [me]], axis=0)
            away += 0.03 * (c - p)
        # the field: away from the walls and the corners, before they trap it
        for k, lim in ((0, a.length / 2), (1, a.width / 2)):
            room = lim - abs(p[k])
            if room < 12.0:
                away[k] -= math.copysign(0.02 * (12.0 - room), p[k])
        away[2] += 0.01 * (5.0 - p[2])
        if self.red_tactic == "jink" and near < 6.0:                  # break across the chaser's line
            b = min(blues, key=lambda q: float(np.linalg.norm(self.pos(q) - p)))
            rel = p - self.pos(b)
            side = np.cross(rel, np.array([0.0, 0.0, 1.0]))
            side = side / max(float(np.linalg.norm(side)), 1e-6) * (1 if int(t * 1.5) % 2 else -1)
            away = side * float(np.linalg.norm(away)) * 1.5 + away + np.array([0.0, 0.0, 0.15 * (1 if p[2] < 6 else -1)])
        n_ = float(np.linalg.norm(away))
        v_want = away / n_ * pl.v_max if n_ > 1e-9 else -v
        me.role = "attack"
        rooms = {b.name: self.tag_radius + 0.6 for b in self.blues}  # keep out of the tag
        return self.steer(me, v_want, rooms), "attack"

    def decide(self, me, t):
        return self.decide_red(me, t) if me.team == "red" else self.decide_blue(me, t)

    # ----------------------------------------------------------------------------------------------- the referee
    def crossed(self, p, hoop, good, t):
        return False

    def restart(self, p: mt.Player, t: float, hold: float):
        p.place(self.data, t, hold)

    def tags(self, t: float):
        for r in self.live("red", t):
            close = [b for b in self.live("blue", t) if float(np.linalg.norm(self.pos(b) - self.pos(r))) < self.tag_radius]
            self.dwell[r.name] = self.dwell[r.name] + self.control_dt if close else 0.0
            if close and self.dwell[r.name] >= self.tag_dwell_s:
                self.caught[r.name] = t
                b = min(close, key=lambda q: float(np.linalg.norm(self.pos(q) - self.pos(r))))
                b.goals += 1
                self.events.append({"t": round(t, 2), "what": "tag", "who": b.name, "from": r.name, "pos": [round(float(x), 1) for x in self.pos(r)]})
                r.home = np.array([-20.0 + 8.0 * self.parked, self.a.width / 2 + 6.0, 10.0])   # out: off the side of the field, waiting
                self.parked += 1
                r.place(self.data, t, 1e9)

    def run(self, duration=None, record_dt: float = 0.04) -> dict:
        mj, m, d = self.mj, self.model, self.data
        duration = self.rules.duration if duration is None else duration
        next_rec, k = 0.0, 0
        while d.time < duration and len(self.caught) < len(self.reds):
            t = d.time
            before = {p.name: self.pos(p) for p in self.players}
            for p in self.players:
                if t < p.frozen_until:
                    T, v_ax = self.actuate(p, None, hover_at=p.home)
                    if p.name not in self.caught:
                        p.role = "restart"
                else:
                    T, v_ax = self.actuate(p, self.decide(p, t)[0])
                if k % 5 == 0:
                    u = mt.mi.throttle_for(p.env, 0.0, max(v_ax, 0.0), T)
                    p.wh += mt.mi.thrust_at(p.env, 0.0, max(v_ax, 0.0), u)[1] * 5 * self.control_dt / 3600.0
            self.drift()
            for _ in range(self.n_sub):
                mj.mj_step(m, d)
            self.referee(d.time, before)
            self.tags(d.time)
            if d.time >= next_rec:
                next_rec += record_dt
                for p in self.players:
                    R = d.xmat[p.body].reshape(3, 3)
                    p.rows.append((d.time, *self.pos(p), *self.vel(p), *R[:, 2], {"attack": 0, "restart": 2, "chase": 3}.get(p.role, 2), p.wh,
                                   1.0 if p.name in self.caught else 0.0))
                self.balloon_rows.append(np.array([d.xpos[b].copy() for b in self.balloons]))
                self.qpos_rows.append(d.qpos.copy())
                self.token_rows.append((self.token.copy(), "none"))
            k += 1
        r = self.result()
        end = float(d.time)
        surv = {x.name: round(self.caught.get(x.name, end), 2) for x in self.reds}
        blues_won = len(self.caught) == len(self.reds)
        r["teamchase"] = {"red": self.red_tactic, "blue": self.blue_tactic, "head_start_s": self.head_start_s, "caught": dict(self.caught), "survival": surv,
                          "all_caught_at": round(end, 2) if blues_won else None, "winner": "blue" if blues_won else "red",
                          "red_seconds": round(sum(surv.values()), 1), "tags": {b.name: b.goals for b in self.blues},
                          "fouls": {p.name: p.fouls + p.crashes for p in self.players}, "duration": round(end, 2)}
        r["score"] = {"red": len(self.reds) - len(self.caught), "blue": len(self.caught)}
        r["winner"] = r["teamchase"]["winner"]
        return r


def play(red="scatter", blue="assigned", seed=1, duration=60.0, head_start_s=3.0, **kw) -> dict:
    return TeamChase(red, blue, seed=seed, duration=duration, head_start_s=head_start_s, **kw).run()


def tournament(seeds: int = 3, seed: int = 1, duration: float = 60.0, head_start_s: float = 3.0, reds=RED, blues=BLUE, progress=None) -> dict:
    """Every red tactic against every blue tactic over ``seeds`` seeds: blue wins, mean time to catch all, red survival."""
    rows = []
    for rt in reds:
        for bt in blues:
            res = []
            for s in range(seed, seed + seeds):
                r = play(rt, bt, s, duration, head_start_s)["teamchase"]
                res.append(r)
                if progress:
                    progress(rt, bt, s, r)
            caught = [x["all_caught_at"] for x in res if x["all_caught_at"] is not None]
            rows.append({"red": rt, "blue": bt, "blue wins": f"{len(caught)}/{seeds}", "mean time to catch all [s]": round(float(np.mean(caught)), 1) if caught else None,
                         "reds caught per game": round(float(np.mean([len(x["caught"]) for x in res])), 2),
                         "red survival [s, of 3 x the game]": round(float(np.mean([x["red_seconds"] for x in res])), 1),
                         "faults per game": round(float(np.mean([sum(x["fouls"].values()) for x in res])), 2)})
    return {"table": rows, "seeds": seeds, "duration": duration, "head_start_s": head_start_s}
