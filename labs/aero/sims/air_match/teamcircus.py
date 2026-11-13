"""The team chase round the circus: N reds flee round the Circus Maximus, N blues hunt them (3v3, 5v5, 7v7), in MuJoCo.

The track is the circus race's (``circus.Stadium``: the spina and its two turning posts, anticlockwise) with **boxes**
standing on it (fixed obstacles: crates on the straights and in the turns, ``BoxStadium``) and **balloons** that drift
with a **wind that changes during the race** (``circus.CHANGING``: its speed and direction wander, gust fronts pass).
There is no token: the teams are the game. The reds start on the south straight; the blues start on the north
straight, half a lap behind, and hover there for ``head_start_s`` before they may go.

A blue **tags** a red by flying within ``tag_radius`` of it for ``tag_dwell_s`` without touching: the red is out. A
contact with anything (an aircraft, a balloon, a box, the spina, a post, the ground) or leaving the field sends that
aircraft back to the last gate it passed, standing for 2 s. Every aircraft's **laps** are counted (a lap is the four
gates in order, as in the race). The blues win if every red is out before the clock stops; otherwise the reds win.

**Reds** are individual agents: each flies one of the circus race's strategies (``RED_STRATEGIES``, cycled), and breaks
away from any blue that closes within ``EVADE_M`` — sideways and up or down off its line. **Blues** are assigned one
red each (more reds than blues cannot happen here; equal numbers), by the Hungarian algorithm on how far ahead along
the track each red is, re-assigned every second; a blue flies the hot racing line until its red is within
``ATTACK_M`` on the same side of the spina, then closes for the tag (``blue_tactic="track"``), or — the default,
``"cut"`` — flies straight at where its red will be, over the 2.5 m spina when it is in the way (the posts are higher).
Flying the same aircraft on the same line, a blue never closes half a lap on a red: the track game has no catch.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from . import arena as ar
from . import circus as cc
from . import match as mt
from . import swarm as sw

RED_STRATEGIES = ("hot", "inside", "high", "windwise", "middle", "adaptive", "tempered")
BLUE_STRATEGY = "hot"
EVADE_M = 9.0
ATTACK_M = 24.0
RED_COLORS = ("#d62728", "#ff7f0e", "#e377c2", "#8c564b", "#bcbd22", "#ff9896", "#c49c94")
BLUE_COLORS = ("#1f77b4", "#17becf", "#9467bd", "#aec7e8", "#393b79", "#6baed6", "#3182bd")


@dataclass
class BoxStadium(cc.Stadium):
    """The circus with fixed boxes on the track: (x, y, half size x, half size y, height)."""
    boxes: tuple = (
        (-35.0, -10.0, 1.2, 1.2, 4.0), (30.0, -14.5, 1.5, 1.0, 3.0), (55.0, -6.5, 1.0, 1.0, 5.5),       # the south straight
        (-50.0, 11.0, 1.0, 1.5, 3.5), (15.0, 7.5, 1.2, 1.2, 4.5), (40.0, 15.0, 1.0, 1.0, 6.0),          # the north straight
        (80.0, 0.0, 1.5, 1.5, 3.0), (-78.0, -4.0, 1.2, 1.2, 5.0), (-70.0, 12.0, 1.0, 1.0, 4.0),         # the turns
    )

    def extra_xml(self) -> str:
        out = [super().extra_xml()]
        for i, (x, y, hx, hy, h) in enumerate(self.boxes):
            out.append(f'<geom name="wallbox{i}" type="box" pos="{x:.2f} {y:.2f} {h / 2:.2f}" size="{hx:.2f} {hy:.2f} {h / 2:.2f}" rgba="0.55 0.40 0.25 1"/>')
        return "\n".join(out)

    def balloons(self) -> np.ndarray:
        pts = super().balloons()
        return np.array([p for p in pts if all(abs(p[0] - x) > hx + 2.5 or abs(p[1] - y) > hy + 2.5 for x, y, hx, hy, _ in self.boxes)] or pts[:1])


class TeamCircus(cc.Race):
    def __init__(self, n: int = 3, seed: int = 1, duration: float = 90.0, head_start_s: float = 3.0, wind: dict | None = None, n_balloons: int = 14,
                 tag_radius: float = 3.0, tag_dwell_s: float = 0.2, slipstream: bool = True, blue_tactic: str = "cut"):
        reds = [dataclasses.replace(cc.BY_NAME[RED_STRATEGIES[i % len(RED_STRATEGIES)]], name=f"red{i + 1}") for i in range(n)]
        blues = [dataclasses.replace(cc.BY_NAME[BLUE_STRATEGY], name=f"blue{i + 1}") for i in range(n)]
        self.blue_tactic = blue_tactic
        self.n, self.head_start_s, self.tag_radius, self.tag_dwell_s = n, head_start_s, tag_radius, tag_dwell_s
        super().__init__(reds + blues, seed, 999, duration, wind=cc.CHANGING if wind is None else wind, n_balloons=n_balloons, slipstream=slipstream,
                         stadium=BoxStadium)
        a = self.a
        self.reds = [p for p in self.players if p.name.startswith("red")]
        self.blues = [p for p in self.players if p.name.startswith("blue")]
        self.red_kind = {p.name: RED_STRATEGIES[int(p.name[3:]) - 1] if False else RED_STRATEGIES[(int(p.name[3:]) - 1) % len(RED_STRATEGIES)] for p in self.reds}
        lane = np.linspace(0.10, 0.40, n) * a.width
        for k, p in enumerate(sorted(self.reds, key=lambda q: q.name)):
            p.team, p.color = "red", RED_COLORS[k % len(RED_COLORS)]
            p.home = np.array([2.0 + 2.6 * (k % 4), -lane[k], 3.0 + 0.5 * (k % 3)])
            p.place(self.data, 0.0, self.rules.kickoff_s)
        for k, p in enumerate(sorted(self.blues, key=lambda q: q.name)):
            p.team, p.color = "blue", BLUE_COLORS[k % len(BLUE_COLORS)]
            p.home = np.array([-2.0 - 2.6 * (k % 4), lane[k], 3.0 + 0.5 * (k % 3)])
            p.place(self.data, 0.0, head_start_s)
            self.gate[p.name] = 3                                    # half a lap round: the west turn next, then the start line
        self.lap_start.update({p.name: head_start_s for p in self.blues})
        self.mj.mj_forward(self.model, self.data)
        self.caught, self.caught_by = {}, {}
        self.dwell = {p.name: 0.0 for p in self.reds}
        self.assign, self.next_assign = {}, 0.0
        self.tags = {p.name: 0 for p in self.blues}

    # ----------------------------------------------------------------------------------------------- the boxes
    def extra_obstacles(self, me):
        out = super().extra_obstacles(me)
        p, reach = self.pos(me), me.n["reach"]
        for x, y, hx, hy, h in self.a.boxes:
            if p[2] < h + reach + 0.8 and abs(p[0] - x) < 25.0 and abs(p[1] - y) < 25.0:
                # round its centre: the way past is the side the aircraft is already on (a point on its face gives no side)
                q = np.array([x, y, min(float(p[2]), h)])
                out.append((q, np.zeros(3), math.hypot(hx, hy) + reach + 1.6, 3.0))
        return out

    # ------------------------------------------------------------------------------------------------- the reds
    def lap_m(self) -> float:
        return 4 * self.a.half + 2 * math.pi * 9.0

    def gap_ahead(self, chaser, quarry) -> float:
        """How far along the track ``quarry`` is ahead of ``chaser`` [m]."""
        return ((self.progress(quarry) - self.progress(chaser)) % 1.0) * self.lap_m()

    def decide_red(self, me, t):
        v_want, role = super().decide(me, t)
        p = self.pos(me)
        push = np.zeros(3)
        for b in self.blues:
            if t < b.frozen_until or b.name in self.finish:
                continue
            d = p - self.pos(b) - self.vel(b) * 0.3
            dn = float(np.linalg.norm(d))
            if dn < EVADE_M:
                vb = self.vel(b)
                side = np.cross(vb, np.array([0.0, 0.0, 1.0]))
                side /= max(float(np.linalg.norm(side)), 1e-6)
                side *= 1.0 if float(side @ d) >= 0 else -1.0             # break to the side it is already on
                up = np.array([0.0, 0.0, 1.0 if p[2] < 7.0 else -1.0])
                push += (side * 0.8 + up * 0.4) * (EVADE_M - dn) / EVADE_M * me.pilot.v_max
        if push.any():
            v_want = self.steer(me, v_want + push, {b.name: self.tag_radius + 0.6 for b in self.blues})
        return v_want, role

    # ------------------------------------------------------------------------------------------------ the blues
    def targets(self, t):
        reds = [r for r in self.reds if r.name not in self.caught]
        if not reds:
            return {}
        if t >= self.next_assign or any(self.assign.get(b.name) not in reds for b in self.blues):
            self.next_assign = t + 1.0
            cols = [reds[j % len(reds)] for j in range(len(self.blues))]
            cost = [[self.gap_ahead(b, c) ** 2 for c in cols] for b in self.blues]
            a = sw.min_cost_assignment(cost)
            self.assign = {b.name: cols[a[i]] for i, b in enumerate(self.blues)}
        return self.assign

    def decide_blue(self, me, t):
        q = self.targets(t).get(me.name)
        if q is None:
            return super().decide(me, t)
        a, p, pq = self.a, self.pos(me), self.pos(q)
        across = abs(p[0]) < a.half and abs(pq[0]) < a.half and p[1] * pq[1] < 0      # the spina between them
        dist = float(np.linalg.norm(pq - p))
        if self.blue_tactic == "track" and (across or dist > ATTACK_M):
            return super().decide(me, t)                              # the racing line round to it
        me.role = "attack"
        vq = self.vel(q)
        aim = pq + vq * 0.25 - p                                      # close: match its velocity and close the gap to it
        v_want = vq + 3.0 * aim
        n_ = float(np.linalg.norm(v_want))
        v_want *= min(1.0, me.pilot.v_max / max(n_, 1e-6))
        if self.blue_tactic == "cut" and dist > 10.0:                  # straight at where it will be, over the spina if it is in the way
            lead = pq + self.vel(q) * min(dist / max(me.pilot.v_max, 1.0), 2.5)
            to = lead - p
            v_want = to / max(float(np.linalg.norm(to)), 1e-6) * me.pilot.v_max
            if across or abs(p[1]) < a.spina_w + 3.0 and abs(p[0]) < a.half + 2.0:
                v_want[2] = max(v_want[2], 2.0 * (a.spina_h + 2.0 - p[2]))
        return self.steer(me, v_want, {q.name: me.n["reach"] + q.n["reach"] + 0.35}, cap=me.pilot.v_max * 1.1), "attack"

    def line(self, st, p):
        """The strategy's line, moved aside (or over) round a box that stands on it within the next 30 m."""
        point, tan, to_turn, R = super().line(st, p)
        for x, y, hx, hy, h in self.a.boxes:
            rel = np.array([x, y, 0.0]) - point
            along = float(rel[:2] @ tan[:2])
            if -3.0 < along < 30.0:
                side = np.array([-tan[1], tan[0], 0.0])
                lat = float(rel[:2] @ side[:2])
                need = max(hx, hy) + 2.6
                if abs(lat) < need:
                    point = point + side * (-math.copysign(need - abs(lat), lat if abs(lat) > 1e-6 else 1.0))
        return point, tan, to_turn, R

    def decide(self, me, t):
        return self.decide_red(me, t) if me.team == "red" else self.decide_blue(me, t)

    # ----------------------------------------------------------------------------------------------- the referee
    def gates(self, t, before):
        super().gates(t, before)
        live_b = [b for b in self.blues if t >= b.frozen_until and b.name not in self.finish]
        for r in self.reds:
            if r.name in self.finish or t < r.frozen_until:
                continue
            close = [b for b in live_b if float(np.linalg.norm(self.pos(b) - self.pos(r))) < self.tag_radius]
            self.dwell[r.name] = self.dwell[r.name] + self.control_dt if close else 0.0
            if close and self.dwell[r.name] >= self.tag_dwell_s:
                b = min(close, key=lambda q: float(np.linalg.norm(self.pos(q) - self.pos(r))))
                self.caught[r.name], self.caught_by[r.name] = t, b.name
                self.tags[b.name] += 1
                self.events.append({"t": round(t, 2), "what": "tag", "who": b.name, "from": r.name, "laps": self.lap[r.name],
                                    "pos": [round(float(x), 1) for x in self.pos(r)]})
                self.finish[r.name] = t
                r.home = np.array([-30.0 + 4.0 * len(self.caught), -self.a.width / 2 + 2.0, 13.0])
                r.place(self.data, t, 1e9)
        if len(self.caught) == len(self.reds):                       # all out: the game is over
            for b in self.blues:
                self.finish.setdefault(b.name, t)

    def run(self, duration=None, record_dt: float = 0.04) -> dict:
        r = super().run(duration, record_dt)
        end = float(self.data.time)
        laps = {p.name: self.lap[p.name] for p in self.players}
        frac = {p.name: round(self.lap[p.name] + (self.progress(p) % 1.0 if p.name not in self.caught else 0.0), 2) for p in self.players}
        reds = [{"red": x.name, "strategy": self.red_kind[x.name], "laps": laps[x.name], "caught_at": round(self.caught[x.name], 2) if x.name in self.caught else None,
                 "by": self.caught_by.get(x.name), "faults": x.fouls + x.crashes, "lap_times": self.lap_times[x.name]} for x in self.reds]
        blues = [{"blue": x.name, "laps": laps[x.name], "tags": self.tags[x.name], "faults": x.fouls + x.crashes, "lap_times": self.lap_times[x.name]} for x in self.blues]
        won = len(self.caught) == len(self.reds)
        r["teamchase"] = {"n": self.n, "winner": "blue" if won else "red", "all_caught_at": round(end, 2) if won else None, "duration": round(end, 2),
                          "head_start_s": self.head_start_s, "reds": reds, "blues": blues, "caught": {k: round(v, 2) for k, v in self.caught.items()},
                          "red_laps": sum(x["laps"] for x in reds), "blue_laps": sum(x["laps"] for x in blues),
                          "red_seconds": round(sum(self.caught.get(x.name, end) for x in self.reds), 1), "laps_with_fraction": frac,
                          "boxes": self.a.boxes, "gust_fronts": len(r["race"]["gust_fronts"])}
        r["score"] = {"red": len(self.reds) - len(self.caught), "blue": len(self.caught)}
        r["winner"] = r["teamchase"]["winner"]
        return r


def tournament(sizes=(3, 5, 7), seeds: int = 3, seed: int = 1, duration: float = 90.0, head_start_s: float = 3.0, blue_tactics=("cut", "track"), progress=None) -> dict:
    """Every team size against every blue tactic over ``seeds`` seeds: blue wins, reds out, laps flown, faults."""
    rows = []
    for n in sizes:
        for bt in blue_tactics:
            res = []
            for s in range(seed, seed + seeds):
                x = play(n, s, duration, head_start_s, blue_tactic=bt)["teamchase"]
                res.append(x)
                if progress:
                    progress(n, bt, s, x)
            rows.append({"size": f"{n}v{n}", "blue": bt, "blue wins": f"{sum(x['winner'] == 'blue' for x in res)}/{seeds}",
                         "reds out per game": round(float(np.mean([len(x["caught"]) for x in res])), 2),
                         "first tag [s]": round(float(np.mean([min(x["caught"].values()) for x in res if x["caught"]])), 1) if any(x["caught"] for x in res) else None,
                         "red laps per red": round(float(np.mean([x["red_laps"] / n for x in res])), 2),
                         "most laps by a red": max(max(r["laps"] for r in x["reds"]) for x in res),
                         "blue laps per blue": round(float(np.mean([x["blue_laps"] / n for x in res])), 2),
                         "faults per aircraft": round(float(np.mean([sum(r["faults"] for r in x["reds"] + x["blues"]) / (2 * n) for x in res])), 2),
                         "laps by red strategy": {k: round(float(np.mean([r["laps"] for x in res for r in x["reds"] if r["strategy"] == k])), 1)
                                                  for k in RED_STRATEGIES[:n]}})
    return {"table": rows, "seeds": seeds, "duration": duration, "head_start_s": head_start_s}


def play(n: int = 3, seed: int = 1, duration: float = 90.0, head_start_s: float = 3.0, **kw) -> dict:
    return TeamCircus(n, seed, duration, head_start_s, **kw).run()


def summary(result: dict, path=None):
    """The team chase round the circus on one sheet: the tracks, laps against time with the tags, and the tables."""
    import matplotlib.pyplot as plt
    from . import replay
    a, x = result["arena"], result["teamchase"]
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], width_ratios=[2.2, 1])
    top, prog, txt = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
    replay._field(top, a)
    for bx, by, hx, hy, _ in a.boxes:
        top.add_patch(plt.Rectangle((bx - hx, by - hy), 2 * hx, 2 * hy, color="#8c6d31", zorder=3))
    bal = result["balloons"]
    for i in range(bal.shape[1] if bal.ndim == 3 else 0):
        top.plot(bal[:, i, 0], bal[:, i, 1], color="#f0a500", lw=0.5, alpha=0.5)
    for name, p in result["players"].items():
        h = p["hist"]
        fly = h["role"] != 2
        top.plot(np.where(fly, h["x"], np.nan), np.where(fly, h["y"], np.nan), color=p["color"], lw=0.7, alpha=0.75)
        lt = [0.0] + [e["t"] for e in result["events"] if e["what"] == "lap" and e["who"] == name]
        prog.step(lt, range(len(lt)), where="post", color=p["color"], lw=2.0 if name.startswith("red") else 1.0, ls="-" if name.startswith("red") else "--")
    for e in result["events"]:
        if e["what"] == "tag":
            prog.axvline(e["t"], color="k", lw=0.6, ls=":")
            pr = result["players"][e["from"]]["hist"]
            k = min(max(int(np.searchsorted(pr["t"], e["t"])) - 2, 0), len(pr["t"]) - 1)
            top.scatter([pr["x"][k]], [pr["y"][k]], marker="X", s=90, color=result["players"][e["from"]]["color"], edgecolors="k", zorder=6)
    top.set_title(f"{x['n']}v{x['n']} round the circus — {x['winner']} wins" + (f" (all reds out at {x['all_caught_at']:.0f} s)" if x["all_caught_at"] else "")
                  + "   ✕ a tag   brown: boxes   yellow: balloons in a changing wind")
    prog.set(xlabel="s", ylabel="laps (reds solid, blues dashed)", title="laps against time (dotted: a tag)")
    prog.grid(alpha=0.3)
    txt.axis("off")
    lines = ["red     strategy   laps  out at   by     faults"]
    lines += [f"{r['red']:7s} {r['strategy']:9s} {r['laps']:>4}  {(str(r['caught_at']) + ' s') if r['caught_at'] else 'free':>8}  {r['by'] or '':6s} {r['faults']:>3}" for r in x["reds"]]
    lines += ["", "blue     laps  tags  faults"] + [f"{b['blue']:8s} {b['laps']:>4}  {b['tags']:>4}  {b['faults']:>5}" for b in x["blues"]]
    lines += ["", f"red laps {x['red_laps']}, blue laps {x['blue_laps']}, red seconds {x['red_seconds']}", f"gust fronts {x['gust_fronts']}, head start {x['head_start_s']:g} s"]
    txt.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
    return fig
