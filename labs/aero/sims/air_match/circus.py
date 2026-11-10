"""The circus race: seven Émerillons, each its own agent, race three laps round a stadium in MuJoCo.

The stadium is the Circus Maximus's plan at a reduced scale: a long field with a wall down the middle (the *spina*)
and a turning post (a *meta*) at each end; the racers go round it anticlockwise, as the chariots did. The real circus
was about 621 m long and 118 m wide with a spina of about 335 m (figures as commonly quoted, not checked here); this
one is ``SCALE`` of that. Balloons hang over the track and **move with the wind** — it blows down the field, they
drift with it and with the gusts, and one that leaves comes back in at the upwind end. Nobody cooperates: every
aircraft flies its own **strategy** and steers round the others as round any obstacle.

**The race** (``Race``): a lap is the four gates in order (the start line on the south straight, the east turn, the
north straight, the west turn); the first to complete ``laps`` wins, and the race runs on until all have finished or
the clock stops. A contact with a balloon, another aircraft, the spina, a post or the ground, or leaving the field,
sends the aircraft back to the last gate it passed, standing for ``penalty_s``.

**A strategy** (``Strategy``) is a racing line and how hard it is flown:

- the line is a stadium curve round the spina at ``radius`` (close in: the shortest way, the slowest turns; wide: longer,
  faster turns), at ``height`` — or two heights, one per straight;
- the speed is an **airspeed** cap ``v_air`` (so a tailwind is free ground speed and a headwind costs it), held down in
  the turns to what ``a_lat`` allows on the line's radius, with braking at ``a_brake`` into them;
- ``caution`` scales the room left round balloons and other aircraft.

``STRATEGIES`` holds seven; ``tournament`` races them against each other over many seeds, the grid turned each time,
and counts wins, places, lap times and faults.

What it is not: no slipstream (an aircraft behind another gains nothing), no aerodynamic moments, the wind does not
bend round the walls, and the pilots are rules, not learners.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from ogma import wind as og_wind

from . import arena as ar
from . import match as mt

SCALE = 0.40                       # of the Circus Maximus's plan
REAL = {"length": 621.0, "width": 118.0, "spina": 335.0}          # m, as commonly quoted (not checked)


@dataclass
class Stadium(ar.Arena):
    """The circus: ``half`` is the spina's half length (the turning posts stand there), ``spina_w`` and ``spina_h`` its
    section, ``meta_h`` the posts' height."""
    length: float = REAL["length"] * SCALE
    width: float = REAL["width"] * SCALE * 1.45                     # the stands' inner edge, a little more than to scale: room to pass
    ceiling: float = 16.0
    half: float = REAL["spina"] * SCALE / 2
    spina_w: float = 2.4
    spina_h: float = 2.5
    meta_h: float = 6.0
    meta_r: float = 0.9
    n_balloons: int = 14
    balloon_radius: float = 0.55

    def __post_init__(self):
        self.hoops = {}

    def extra_xml(self) -> str:
        h, w = self.half, self.spina_w / 2
        out = [f'<geom name="spina" type="box" pos="0 0 {self.spina_h / 2:.2f}" size="{h:.2f} {w:.2f} {self.spina_h / 2:.2f}" rgba="0.75 0.70 0.60 1"/>']
        for i, x in enumerate((-h, h)):
            out.append(f'<geom name="meta{i}" type="cylinder" pos="{x:.2f} 0 {self.meta_h / 2:.2f}" size="{self.meta_r:.2f} {self.meta_h / 2:.2f}" rgba="0.85 0.65 0.20 1"/>')
        out.append(f'<geom name="start_line" type="box" pos="0 {-self.width / 4:.2f} 0.003" size="0.15 {self.width / 4:.2f} 0.003" rgba="1 1 1 0.9" contype="0" conaffinity="0"/>')
        return "\n".join(out)

    def balloons(self) -> np.ndarray:
        """Over the track on both straights and in the turns, between 2 and 9 m up."""
        rng = np.random.default_rng(self.seed)
        pts = []
        while len(pts) < self.n_balloons:
            p = np.array([rng.uniform(-0.46, 0.46) * self.length, rng.uniform(-0.45, 0.45) * self.width, rng.uniform(2.0, 9.0)])
            if abs(p[1]) > self.spina_w / 2 + 2.0 or abs(p[0]) > self.half + 3.0:
                if all(np.linalg.norm(p - q) > 5.0 for q in pts) and not (abs(p[0]) < 22.0 and p[1] < 0):   # the grid is clear
                    pts.append(p)
        return np.array(pts)


@dataclass
class Strategy:
    name: str
    what: str
    radius: float = 9.0            # the line's distance from the spina's axis [m]
    height: float = 4.0            # on the south straight (downwind) and in the turns [m]
    height_upwind: float | None = None   # on the north straight (upwind); None: the same
    v_air: float = 30.0            # airspeed cap [m/s]
    a_lat: float = 12.5            # what the turns are flown at [m/s2]
    a_brake: float = 11.0          # braking into the turns [m/s2]
    caution: float = 1.0           # room round balloons and aircraft (x)
    look_s: float = 1.3            # how far ahead it looks for a conflict [s]


STRATEGIES = [
    Strategy("inside", "hugs the spina: the shortest way round, the slowest turns", radius=6.5),
    Strategy("wide", "a wide line: more metres, faster turns", radius=15.0),
    Strategy("middle", "between the two", radius=10.5),
    Strategy("hot", "the inside line at the limit: more airspeed, harder turns, later brakes, less room", radius=7.0, v_air=34.0, a_lat=14.5, a_brake=13.5, caution=0.6, look_s=1.0),
    Strategy("careful", "a middle line with margins: less airspeed, softer turns, more room", radius=10.0, v_air=27.0, a_lat=10.0, a_brake=9.0, caution=1.5, look_s=1.8),
    Strategy("high", "over the balloons: the inside line at 11.5 m, where the wind is stronger", radius=7.0, height=11.5),
    Strategy("windwise", "high with the wind behind, low against it", radius=8.0, height=10.5, height_upwind=2.6),
]
BY_NAME = {s.name: s for s in STRATEGIES}
WIND = {"speed": 5.0, "gust": 1.2, "shear": 0.18}                   # m/s at 10 m along +x (down the south straight), gust sigma, power-law exponent


class Race(mt.Match):
    def __init__(self, strategies=None, seed: int = 1, laps: int = 3, duration: float = 150.0, variant: str = "ER2", grid=None, wind: dict | None = None,
                 penalty_s: float = 2.0, n_balloons: int = 14):
        self.strats = list(strategies or STRATEGIES)
        self.laps, self.windp = laps, dict(WIND if wind is None else wind)
        a = Stadium(seed=seed, n_balloons=n_balloons)
        n = len(self.strats)
        order = list(grid) if grid is not None else list(range(n))                   # order[k]: the strategy on grid place k
        lane = np.linspace(-0.40, -0.10, n) * a.width                                 # across the south straight
        players = []
        for k, si in enumerate(order):
            st = self.strats[si]
            home = np.array([2.0 + 2.2 * (k % 4), lane[k], 3.0 + 0.4 * (k % 2)])
            players.append(mt.Player(st.name, variant, f"R{si + 1}", mt.Pilot(st.name, v_max=st.v_air + self.windp["speed"] + 4.0, caution=st.caution, look_s=st.look_s),
                                     home=home, color=ar.PALETTE[si % len(ar.PALETTE)]))
        self.strategy = {p.name: self.strats[si] for p, si in zip(players, order)}
        super().__init__(a, players, mt.Rules(duration=duration, possession=False, kickoff_s=1.0, penalty_s=penalty_s, margin=1.0), seed=seed)
        self.gate = {p.name: 1 for p in self.players}                # the next gate to pass (0: the start line)
        self.lap = {p.name: 0 for p in self.players}
        self.lap_times = {p.name: [] for p in self.players}
        self.lap_start = {p.name: self.rules.kickoff_s for p in self.players}
        self.finish = {}
        self.balloon_qadr = [int(self.model.jnt_qposadr[self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_JOINT, f"balloon{i}_free")]) for i in range(a.n_balloons)]
        self.gustv = np.zeros(3)
        self.respawns = 0

    # ---------------------------------------------------------------------------------------------- the air
    def wind_at(self, p) -> np.ndarray:
        w = self.windp
        if w["speed"] <= 0:
            return np.zeros(3)
        return np.array([og_wind.power_law(w["speed"], max(float(p[2]), 1.0), 10.0, w["shear"]), 0.0, 0.0]) + self.gustv

    def drift(self):
        """The balloons in the wind: pulled to the air's velocity, held near their height; one that leaves the field
        comes back in at the upwind end."""
        a, d, w = self.a, self.data, self.windp
        self.gustv = og_wind.ou_euler_step(self.gustv, self.control_dt, w["gust"], 5.0, self.rng) if w["gust"] > 0 else self.gustv
        self.gust += (-self.gust / 3.0) * self.control_dt + 0.5 * math.sqrt(2 * self.control_dt / 3.0) * self.rng.standard_normal(self.gust.shape)
        for i, b in enumerate(self.balloons):
            x = d.xpos[b]
            v = d.cvel[b][3:]
            air = self.wind_at(x) * 0.5 + self.gust[i] * (1.0, 1.0, 0.3)   # a balloon on a long tether drifts at half the wind
            f = a.balloon_mass * (air - v) / 0.8
            f[2] += 0.004 * (5.0 - x[2])
            d.xfrc_applied[b, :3] = f
            if abs(x[0]) > a.length / 2 - 2.0 or abs(x[1]) > a.width / 2 - 1.0:
                q = self.balloon_qadr[i]
                side = -1.0 if w["speed"] >= 0 else 1.0
                y = self.rng.uniform(-0.42, 0.42) * a.width
                d.qpos[q:q + 3] = (side * (a.length / 2 - 4.0), y, self.rng.uniform(2.0, 9.0))
                dof = self.balloon_dof[i]
                d.qvel[dof:dof + 6] = 0.0
                self.respawns += 1

    # ------------------------------------------------------------------------------------------- the line
    def line(self, st: Strategy, p) -> tuple:
        """On the strategy's stadium curve nearest ``p``: (the point, the tangent, the distance to the next turn's
        entry or 0 in a turn, the turn's radius or inf)."""
        h, R = self.a.half, st.radius
        if abs(p[0]) <= h:
            south = p[1] < 0
            point = np.array([p[0], -R if south else R, 0.0])
            tan = np.array([1.0, 0.0, 0.0]) if south else np.array([-1.0, 0.0, 0.0])
            return point, tan, (h - p[0]) if south else (p[0] + h), math.inf
        c = np.array([math.copysign(h, p[0]), 0.0, 0.0])
        r = np.array([p[0] - c[0], p[1], 0.0])
        rn = float(np.linalg.norm(r))
        r = r / rn if rn > 1e-6 else np.array([math.copysign(1.0, p[0]), 0.0, 0.0])
        return c + r * R, np.array([-r[1], r[0], 0.0]), 0.0, R

    def decide(self, me: mt.Player, t: float) -> tuple:
        st = self.strategy[me.name]
        a = self.a
        p, v = self.pos(me), self.vel(me)
        point, tan, to_turn, R = self.line(st, p)
        wind = self.wind_at(p)
        v_cap = st.v_air + float(wind @ tan)                         # airspeed is what the drive gives: the wind adds or takes ground speed
        v_turn = math.sqrt(st.a_lat * st.radius)
        v_line = min(v_cap, v_turn) if R < math.inf else min(v_cap, math.sqrt(v_turn ** 2 + 2 * st.a_brake * max(to_turn - 2.0, 0.0)))
        if R < math.inf:                                              # in a turn: aim along the arc a little ahead (the velocity loop lags)
            lead = float(np.linalg.norm(v[:2])) * 0.22 / R
            c, s = math.cos(lead), math.sin(lead)
            tan = np.array([c * tan[0] - s * tan[1], s * tan[0] + c * tan[1], 0.0])
        north = abs(p[0]) <= a.half and p[1] > 0
        z_line = st.height_upwind if (north and st.height_upwind is not None) else st.height
        off = point - p
        v_want = tan * v_line + 1.6 * np.array([off[0], off[1], 0.0])
        v_want[2] = float(np.clip(1.5 * (z_line - p[2]), -5.0, 5.0))
        me.role = "attack"
        return self.steer(me, v_want, cap=max(v_cap, 8.0) * 1.1, near=1.1), "attack"

    def extra_obstacles(self, me: mt.Player) -> list:
        """The spina (its nearest point at the aircraft's height, while it is below the wall's top) and the posts."""
        a, p = self.a, self.pos(me)
        out = []
        reach = me.n["reach"]
        if p[2] < a.spina_h + reach + 0.8:
            out.append((np.array([float(np.clip(p[0], -a.half, a.half)), 0.0, min(p[2], a.spina_h)]), np.zeros(3), a.spina_w / 2 + reach + 1.0, 2.5))
        for x in (-a.half, a.half):
            if p[2] < a.meta_h + reach + 0.8:
                out.append((np.array([x, 0.0, min(p[2], a.meta_h)]), np.zeros(3), a.meta_r + reach + 1.0, 2.5))
        return out

    # ------------------------------------------------------------------------------------------ the referee
    def gate_point(self, g: int, slot: int = 0) -> np.ndarray:
        a = self.a
        mid = 0.5 * (a.spina_w / 2 + a.width / 2)
        lat = mid + (slot - 3) * 1.6
        return [np.array([0.0, -lat, 4.0]), np.array([a.half + lat, 0.0, 4.0]), np.array([0.0, lat, 4.0]), np.array([-a.half - lat, 0.0, 4.0])][g]

    def restart(self, p: mt.Player, t: float, hold: float):
        """Back to the last gate passed, standing there for the penalty."""
        last = (self.gate[p.name] - 1) % 4
        p.home = self.gate_point(last, self.players.index(p))
        p.place(self.data, t, hold)

    def gates(self, t: float, before: dict):
        a = self.a
        for p in self.players:
            if p.name in self.finish or t < p.frozen_until:
                continue
            x0, x = before[p.name], self.pos(p)
            g = self.gate[p.name]
            hit = [x0[0] < 0 <= x[0] and x[1] < 0, x0[1] < 0 <= x[1] and x[0] > a.half * 0.5, x0[0] > 0 >= x[0] and x[1] > 0, x0[1] > 0 >= x[1] and x[0] < -a.half * 0.5][g]
            if not hit:
                continue
            self.gate[p.name] = (g + 1) % 4
            if g == 0:
                self.lap[p.name] += 1
                self.lap_times[p.name].append(round(t - self.lap_start[p.name], 2))
                self.lap_start[p.name] = t
                self.events.append({"t": round(t, 2), "what": "lap", "who": p.name, "lap": self.lap[p.name], "time": self.lap_times[p.name][-1]})
                if self.lap[p.name] >= self.laps:
                    self.finish[p.name] = t
                    self.events.append({"t": round(t, 2), "what": "finish", "who": p.name, "place": len(self.finish)})
                    p.home = np.array([20.0 + 3.0 * len(self.finish), -a.width / 2 + 2.0, 12.0 + 0.6 * len(self.finish)])   # out of the way, above the stands' edge
                    p.place(self.data, t, 1e9)

    def progress(self, p: mt.Player) -> float:
        """Laps and fractions of a lap, for the order of those still racing."""
        x, a = self.pos(p), self.a
        g = (self.gate[p.name] - 1) % 4                               # the last gate passed
        ang = math.atan2(x[1], x[0])
        frac = {0: (x[0] / a.half) * 0.5, 1: (ang / math.pi) if ang > 0 else 0.0, 2: (-x[0] / a.half) * 0.5, 3: ((ang + math.pi) / math.pi) if ang < 0 else 0.0}[g]
        return self.lap[p.name] + (g + min(max(frac, 0.0), 1.0)) / 4.0

    def run(self, duration: float | None = None, record_dt: float = 0.04) -> dict:
        mj, m, d = self.mj, self.model, self.data
        duration = self.rules.duration if duration is None else duration
        next_rec, k = 0.0, 0
        while d.time < duration and len(self.finish) < len(self.players):
            t = d.time
            before = {p.name: self.pos(p) for p in self.players}
            for p in self.players:
                if t < p.frozen_until:
                    T, v_ax = self.actuate(p, None, hover_at=p.home)
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
            self.gates(d.time, before)
            if d.time >= next_rec:
                next_rec += record_dt
                for p in self.players:
                    R = d.xmat[p.body].reshape(3, 3)
                    p.rows.append((d.time, *self.pos(p), *self.vel(p), *R[:, 2], {"attack": 0, "restart": 2}[p.role], p.wh, 0.0))
                self.balloon_rows.append(np.array([d.xpos[b].copy() for b in self.balloons]))
                self.qpos_rows.append(d.qpos.copy())
                self.token_rows.append((self.token.copy(), "none"))
            k += 1
        r = self.result()
        order = sorted(self.players, key=lambda p: (self.finish.get(p.name, 1e9), -self.progress(p)))
        standings = []
        for place, p in enumerate(order, 1):
            st = self.strategy[p.name]
            h = r["players"][p.name]["hist"]
            flying = h["role"] == 0
            standings.append({"place": place, "strategy": p.name, "finished": p.name in self.finish, "time_s": round(self.finish[p.name], 2) if p.name in self.finish else None,
                              "laps": self.lap[p.name], "progress": round(self.progress(p), 2) if p.name not in self.finish else float(self.laps),
                              "lap_times": self.lap_times[p.name], "best_lap_s": min(self.lap_times[p.name][1:] or self.lap_times[p.name] or [None]),
                              "faults": p.fouls + p.crashes, "fouls": p.fouls, "crashes": p.crashes, "energy_wh": round(p.wh, 2),
                              "mean_speed": float(np.hypot(np.hypot(h["vx"], h["vy"]), h["vz"])[flying].mean()) if flying.any() else 0.0,
                              "max_speed": r["players"][p.name]["max_speed"], "grid": self.players.index(p) + 1, "what": st.what})
        r["race"] = {"standings": standings, "winner": standings[0]["strategy"] if standings[0]["finished"] else None, "laps": self.laps, "wind": self.windp,
                     "balloon_respawns": self.respawns, "lines": {p.name: self.strategy[p.name] for p in self.players},
                     "faults": [e for e in self.events if e["what"] in ("foul", "crash")]}
        r["score"] = {s["strategy"]: s["laps"] for s in standings}
        r["winner"] = r["race"]["winner"]
        return r


def race(seed: int = 1, laps: int = 3, strategies=None, grid=None, wind: dict | None = None, duration: float = 150.0) -> dict:
    return Race(strategies, seed, laps, duration, grid=grid, wind=wind).run()


def solo(strategy: Strategy, laps: int = 3, wind: dict | None = None, balloons: int = 0, seed: int = 1) -> dict:
    """One strategy alone on the track (no balloons unless asked): its lap time with nobody in the way."""
    return Race([strategy], seed, laps, 120.0, wind=wind, n_balloons=balloons).run()


def tournament(n: int = 7, seed: int = 1, laps: int = 3, strategies=None, wind: dict | None = None, progress=None) -> dict:
    """``n`` races of all the strategies against each other, the grid turned one place each race: per strategy the
    wins, the mean place, the races finished, the mean race time, the best lap and the faults per race."""
    strats = list(strategies or STRATEGIES)
    k = len(strats)
    rows = {s.name: {"strategy": s.name, "wins": 0, "podiums": 0, "places": [], "finished": 0, "times": [], "best_lap": [], "faults": 0, "what": s.what} for s in strats}
    races = []
    for i in range(n):
        grid = [(j + i) % k for j in range(k)]
        r = Race(strats, seed + i, laps, grid=grid, wind=wind).run()
        races.append({"seed": seed + i, "order": [s["strategy"] for s in r["race"]["standings"]], "winner": r["race"]["winner"],
                      "time": r["race"]["standings"][0]["time_s"], "faults": sum(s["faults"] for s in r["race"]["standings"])})
        for s in r["race"]["standings"]:
            row = rows[s["strategy"]]
            row["places"].append(s["place"])
            row["wins"] += s["place"] == 1 and s["finished"]
            row["podiums"] += s["place"] <= 3
            row["finished"] += s["finished"]
            row["faults"] += s["faults"]
            if s["finished"]:
                row["times"].append(s["time_s"])
            if s["best_lap_s"]:
                row["best_lap"].append(s["best_lap_s"])
        if progress:
            progress(i, races[-1])
    table = []
    for row in rows.values():
        table.append({"strategy": row["strategy"], "wins": int(row["wins"]), "podiums": int(row["podiums"]), "mean place": round(float(np.mean(row["places"])), 2),
                      "finished": f"{row['finished']}/{n}", "mean time [s]": round(float(np.mean(row["times"])), 1) if row["times"] else None,
                      "best lap [s]": round(float(min(row["best_lap"])), 2) if row["best_lap"] else None, "faults per race": round(row["faults"] / n, 2), "what": row["what"]})
    table.sort(key=lambda t: (-t["wins"], t["mean place"]))
    return {"table": table, "races": races, "n": n, "laps": laps}
