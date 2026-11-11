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

``STRATEGIES`` holds seven fixed ones. ``ADAPTIVE`` holds three that read the race (``Race.adapt``): the **drafter** sits
in the wake of whoever is ahead and pulls out to pass on a straight; the **adaptive** one flies the hot line while the
way is clear and goes up and over when balloons lie ahead on it; the **tempered** one starts hot and gets more careful
with every fault, a little bolder with every clean lap. ``tournament`` races a field (``FIELD``: four fixed, three
adaptive) over many seeds, the grid turned each time, and counts wins, places, lap times and faults.

**The slipstream** (``WAKE``, ``Race.wake_at``): behind an aircraft the air moves with it, in a cone that fades with
distance, so whoever flies there meets less air — more ground speed for the same airspeed. Its numbers are assumed.

**A changing wind** (``CHANGING``, ``Race.blow``): the mean wind's speed and direction wander during the race and gust
fronts of 4–8 m/s come through for a few seconds from off the wind; the balloons go where it takes them (across the
track when it veers) and come back in at whichever edge is upwind. The strategies that read the wind (``windwise``, and
every airspeed cap) read it as it is at that moment.

What it is not: no aerodynamic moments, the wind does not bend round the walls, the wake does not disturb the
aircraft that flies in it, and the pilots are rules, not learners.
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
                if all(np.linalg.norm(p - q) > (5.0 if self.n_balloons <= 20 else 3.0) for q in pts) and not (abs(p[0]) < 22.0 and p[1] < 0):   # the grid is clear
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
    kind: str = "fixed"            # "fixed", or adaptive: "draft", "adapt", "temper" (``Race.adapt``)


STRATEGIES = [
    Strategy("inside", "hugs the spina: the shortest way round, the slowest turns", radius=6.5),
    Strategy("wide", "a wide line: more metres, faster turns", radius=15.0),
    Strategy("middle", "between the two", radius=10.5),
    Strategy("hot", "the inside line at the limit: more airspeed, harder turns, later brakes, less room", radius=7.0, v_air=34.0, a_lat=14.5, a_brake=13.5, caution=0.6, look_s=1.0),
    Strategy("careful", "a middle line with margins: less airspeed, softer turns, more room", radius=10.0, v_air=27.0, a_lat=10.0, a_brake=9.0, caution=1.5, look_s=1.8),
    Strategy("high", "over the balloons: the inside line at 11.5 m, where the wind is stronger", radius=7.0, height=11.5),
    Strategy("windwise", "high with the wind behind, low against it", radius=8.0, height=10.5, height_upwind=2.6),
]
# the adaptive ones: they read the race and change what they fly
ADAPTIVE = [
    Strategy("drafter", "sits in the wake of whoever is ahead, then pulls out and passes on a straight", radius=7.5, kind="draft"),
    Strategy("adaptive", "the hot line while the way is clear, up and over when balloons lie ahead on it", radius=7.0, v_air=34.0, a_lat=14.0, a_brake=13.0,
             caution=0.7, look_s=1.1, kind="adapt"),
    Strategy("tempered", "starts hot; every fault makes it more careful, every clean lap a little bolder", radius=7.0, v_air=34.0, a_lat=14.5, a_brake=13.5,
             caution=0.6, look_s=1.0, kind="temper"),
]
BY_NAME = {s.name: s for s in STRATEGIES + ADAPTIVE}
FIELD = [BY_NAME[n] for n in ("hot", "high", "inside", "windwise", "drafter", "adaptive", "tempered")]   # the second tournament's seven
RACE_TILT = 72.0                   # degrees the racers may lean (the match flies at 58): at 58 every strategy topped out near 32 m/s
                                   # whatever its airspeed cap; at 72 the cap, the wind and the wake decide the straights
WAKE = {"gain": 0.25, "length": 8.0, "radius": 1.0, "spread": 0.08, "from": 1.0, "to": 20.0}
# the slipstream: behind an aircraft the air moves with it — ``gain`` x its speed just behind, fading over ``length`` metres, in a
# cone of ``radius`` + ``spread`` x distance. Assumed numbers (no wake was measured or computed): the effect is there to be raced with.
WIND = {"speed": 5.0, "gust": 1.2, "shear": 0.18}                   # m/s at 10 m along +x (down the south straight), gust sigma, power-law exponent
# a wind that changes during the race: its speed and its direction wander (sigma and time constant), and gust fronts come through
# — every ``front_every_s`` on average one of ``front_ms`` for ``front_s`` seconds, from up to ``front_veer`` degrees off the wind
CHANGING = dict(WIND, changing=True, speed_sigma=2.0, speed_tau=8.0, veer_sigma=50.0, veer_tau=12.0, front_every_s=9.0, front_ms=(4.0, 8.0),
                front_s=(2.0, 4.0), front_veer=40.0)


class Race(mt.Match):
    def __init__(self, strategies=None, seed: int = 1, laps: int = 3, duration: float = 150.0, variant: str = "ER2", grid=None, wind: dict | None = None,
                 penalty_s: float = 2.0, n_balloons: int = 14, slipstream: bool = True, stadium=None):
        self.strats = list(strategies or STRATEGIES)
        self.laps, self.windp = laps, dict(WIND if wind is None else wind)
        a = (stadium or Stadium)(seed=seed, n_balloons=n_balloons)
        n = len(self.strats)
        order = list(grid) if grid is not None else list(range(n))                   # order[k]: the strategy on grid place k
        lane = np.linspace(-0.40, -0.10, n) * a.width                                 # across the south straight
        players = []
        for k, si in enumerate(order):
            st = self.strats[si]
            home = np.array([2.0 + 2.2 * (k % 4), lane[k], 3.0 + 0.4 * (k % 2)])
            players.append(mt.Player(st.name, variant, f"R{si + 1}", mt.Pilot(st.name, v_max=st.v_air + self.windp["speed"] + 4.0, caution=st.caution, look_s=st.look_s),
                                     home=home, color=ar.PALETTE[si % len(ar.PALETTE)]))
        for p in players:
            p.tilt_max_deg = RACE_TILT
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
        self.wrng = np.random.default_rng(seed + 1000)               # the changing wind's own dice
        self.w_speed, self.w_dir, self.fronts, self.wind_rows = float(self.windp["speed"]), 0.0, [], []
        self.slipstream = slipstream
        self.state = {p.name: {"faults": 0, "bold": 0, "high_until": 0.0, "pass_until": 0.0, "draft_s": 0.0, "high_s": 0.0, "passes": 0} for p in self.players}
        self.wake_now = {p.name: 0.0 for p in self.players}

    # ---------------------------------------------------------------------------------------------- the air
    def wind_now(self) -> np.ndarray:
        """The wind at 10 m right now: the mean (steady, or wandering in speed and direction) plus the gust fronts passing."""
        w = np.array([math.cos(self.w_dir), math.sin(self.w_dir), 0.0]) * self.w_speed
        t = self.data.time
        for f in self.fronts:
            x = (t - f["t0"]) / f["T"]
            if 0.0 < x < 1.0:
                w = w + np.array([math.cos(f["dir"]), math.sin(f["dir"]), 0.0]) * f["amp"] * math.sin(math.pi * x) ** 2
        return w

    def blow(self):
        """Advance a changing wind by one control step."""
        w, dt = self.windp, self.control_dt
        if not w.get("changing"):
            return
        r = self.wrng
        self.w_speed = max(self.w_speed + (w["speed"] - self.w_speed) / w["speed_tau"] * dt + w["speed_sigma"] * math.sqrt(2 * dt / w["speed_tau"]) * r.standard_normal(), 0.0)
        sig = math.radians(w["veer_sigma"])
        self.w_dir += -self.w_dir / w["veer_tau"] * dt + sig * math.sqrt(2 * dt / w["veer_tau"]) * r.standard_normal()
        if r.random() < dt / w["front_every_s"]:
            self.fronts.append({"t0": self.data.time, "T": float(r.uniform(*w["front_s"])), "amp": float(r.uniform(*w["front_ms"])),
                                "dir": self.w_dir + math.radians(float(r.uniform(-w["front_veer"], w["front_veer"])))})
            self.events.append({"t": round(self.data.time, 2), "what": "gust front", "ms": round(self.fronts[-1]["amp"], 1),
                                "from_deg": round(math.degrees(self.fronts[-1]["dir"])), "seconds": round(self.fronts[-1]["T"], 1)})
        self.fronts = [f for f in self.fronts if self.data.time < f["t0"] + f["T"]]

    def wind_at(self, p) -> np.ndarray:
        w = self.windp
        if w["speed"] <= 0 and not w.get("changing"):
            return np.zeros(3)
        return self.wind_now() * og_wind.power_law(1.0, max(float(p[2]), 1.0), 10.0, w["shear"]) + self.gustv

    def wake_at(self, me: mt.Player, p) -> np.ndarray:
        """The air the others drag along where ``me`` flies: behind each one, along its velocity, a fading cone."""
        out = np.zeros(3)
        if not self.slipstream:
            return out
        for q in self.players:
            if q is me or q.name in self.finish:
                continue
            vq = self.vel(q)
            sq = float(np.linalg.norm(vq))
            if sq < 6.0:
                continue
            rel = np.asarray(p, float) - self.pos(q)
            back = -float(rel @ vq) / sq                              # how far behind it
            if not (WAKE["from"] < back < WAKE["to"]):
                continue
            off = float(np.linalg.norm(rel + vq / sq * back))
            rw = WAKE["radius"] + WAKE["spread"] * back
            out += vq * WAKE["gain"] * math.exp(-back / WAKE["length"]) * math.exp(-(off / rw) ** 2)
        return out

    def air_for(self, me: mt.Player, p) -> np.ndarray:
        wake = self.wake_at(me, p)
        self.wake_now[me.name] = float(np.linalg.norm(wake))
        return self.wind_at(p) + wake

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
                wn = self.wind_now()
                if abs(wn[0]) * a.width >= abs(wn[1]) * a.length or abs(wn[1]) < 0.5:      # in at the upwind edge: an end, or a side
                    spot = (-math.copysign(a.length / 2 - 4.0, wn[0] if abs(wn[0]) > 1e-6 else 1.0), self.rng.uniform(-0.42, 0.42) * a.width)
                else:
                    spot = (self.rng.uniform(-0.45, 0.45) * a.length, -math.copysign(a.width / 2 - 3.0, wn[1]))
                d.qpos[q:q + 3] = (*spot, self.rng.uniform(2.0, 9.0))
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

    def ahead_of(self, me: mt.Player):
        """The nearest aircraft in front of ``me`` on the track, and how far along it [m] (None when nobody is within 30 m)."""
        mine = self.progress(me)
        lap_m = 4 * self.a.half + 2 * math.pi * 9.0
        best, gap = None, 30.0
        for q in self.players:
            if q is me or q.name in self.finish or self.data.time < q.frozen_until:
                continue
            d = ((self.progress(q) - mine) % 1.0) * lap_m
            if 0.5 < d < gap and float(np.linalg.norm(self.pos(q) - self.pos(me))) < 35.0:
                best, gap = q, d
        return best, gap

    def adapt(self, me: mt.Player, st: Strategy, t: float, to_turn: float, R: float) -> Strategy:
        """What an adaptive strategy flies right now: the base strategy with the line, the height and the limits its rule
        sets from what it sees."""
        import dataclasses
        if st.kind == "fixed":
            return st
        sx, p = self.state[me.name], self.pos(me)
        if st.kind == "temper":                                       # each fault costs boldness, each clean lap gives a little back
            k = max(sx["faults"] - 0.5 * sx["bold"], 0.0)
            return dataclasses.replace(st, a_lat=max(st.a_lat - 1.3 * k, 10.0), a_brake=max(st.a_brake - 1.2 * k, 9.0), v_air=max(st.v_air - 1.5 * k, 27.0),
                                       caution=min(st.caution + 0.3 * k, 1.5), look_s=min(st.look_s + 0.2 * k, 1.8))
        if st.kind == "adapt":                                        # balloons on my line in the next 2 s: over them, and stay up a moment
            _, tan, _, _ = self.line(st, p)
            v = max(float(np.linalg.norm(self.vel(me)[:2])), 10.0)
            for b in self.balloons:
                rel = self.data.xpos[b] - p
                along = float(rel[:2] @ tan[:2])
                if 0.0 < along < 2.2 * v and abs(float(rel[0] * -tan[1] + rel[1] * tan[0])) < 4.0 and self.data.xpos[b][2] < 9.5:
                    sx["high_until"] = t + 1.2
                    break
            return dataclasses.replace(st, height=11.5) if t < sx["high_until"] else st
        # draft: into the wake of whoever is ahead; out and past on a straight when it has closed up
        lead, gap = self.ahead_of(me)
        if lead is None:
            return st
        pl = self.pos(lead)
        closing = st.v_air + float((self.wind_at(p) + self.wake_at(me, p)) @ self.line(st, p)[1]) - float(np.linalg.norm(self.vel(lead)))
        if t < sx["pass_until"] or (gap < 4.5 and R == math.inf and to_turn > 55.0 and closing > 1.0):
            if t >= sx["pass_until"]:
                sx["pass_until"], sx["passes"] = t + 2.5, sx["passes"] + 1
            return dataclasses.replace(st, radius=min(max(self.radial(pl) + 2.4, 6.5), 0.5 * self.a.width - 4.0), height=float(np.clip(pl[2], 2.5, 12.0)))
        r_lead = float(np.clip(self.radial(pl), 6.0, 0.5 * self.a.width - 4.0))
        return dataclasses.replace(st, radius=r_lead, height=float(np.clip(pl[2], 2.5, 12.0)))

    def radial(self, p) -> float:
        """A point's distance from the spina's axis line (the radius of the stadium curve through it)."""
        h = self.a.half
        return abs(float(p[1])) if abs(p[0]) <= h else float(math.hypot(abs(p[0]) - h, p[1]))

    def decide(self, me: mt.Player, t: float) -> tuple:
        base = self.strategy[me.name]
        a = self.a
        p, v = self.pos(me), self.vel(me)
        _, _, to_turn0, R0 = self.line(base, p)
        st = self.adapt(me, base, t, to_turn0, R0)
        point, tan, to_turn, R = self.line(st, p)
        air = self.wind_at(p) + self.wake_at(me, p)
        v_cap = st.v_air + float(air @ tan)                          # airspeed is what the drive gives: wind and wake add or take ground speed
        v_turn = math.sqrt(st.a_lat * st.radius)
        v_line = min(v_cap, v_turn) if R < math.inf else min(v_cap, math.sqrt(v_turn ** 2 + 2 * st.a_brake * max(to_turn - 2.0, 0.0)))
        if st.kind == "draft" and t >= self.state[me.name]["pass_until"]:
            lead, gap = self.ahead_of(me)
            if lead is not None and gap < 4.0:                        # tucked in: no faster than the one ahead
                v_line = min(v_line, float(np.linalg.norm(self.vel(lead))) + 1.5 * (gap - 3.0))
        if R < math.inf:                                              # in a turn: aim along the arc a little ahead (the velocity loop lags)
            lead_a = float(np.linalg.norm(v[:2])) * 0.22 / R
            c, s_ = math.cos(lead_a), math.sin(lead_a)
            tan = np.array([c * tan[0] - s_ * tan[1], s_ * tan[0] + c * tan[1], 0.0])
        against = float(self.wind_at(p) @ tan) < 0.0                  # the wind in the face on this stretch
        z_line = st.height_upwind if (against and st.height_upwind is not None and R == math.inf) else st.height
        off = point - p
        v_want = tan * v_line + 1.6 * np.array([off[0], off[1], 0.0])
        v_want[2] = float(np.clip(1.5 * (z_line - p[2]), -5.0, 5.0))
        me.role = "attack"
        me.pilot.caution, me.pilot.look_s = st.caution, st.look_s
        rooms = None
        if st.kind == "draft":                                        # close behind the one it follows; the usual room from the rest
            lead, gap = self.ahead_of(me)
            rooms = {lead.name: me.n["reach"] + lead.n["reach"] + 0.5} if lead is not None else None
            if lead is not None and gap < 14.0 and t >= self.state[me.name]["pass_until"]:
                vl = self.vel(lead)                                   # fly to the point 3.5 m behind it, in its wake, at its speed
                sl = max(float(np.linalg.norm(vl)), 1.0)
                spot = self.pos(lead) - vl / sl * 3.5
                v_want = vl + 2.2 * (spot - p)
                n_ = float(np.linalg.norm(v_want))
                if n_ > v_cap * 1.08:
                    v_want *= v_cap * 1.08 / n_
        return self.steer(me, v_want, rooms, cap=max(v_cap, 8.0) * 1.1, near=1.1), "attack"

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
        self.state[p.name]["faults"] += 1
        self.state[p.name]["lap_faults"] = self.state[p.name].get("lap_faults", 0) + 1
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
                sx = self.state[p.name]
                sx["bold"] += 1 if sx.get("lap_faults", 0) == 0 else 0
                sx["lap_faults"] = 0
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
                if p.role == "attack" and p.name not in self.finish:
                    self.state[p.name]["draft_s"] += self.control_dt if self.wake_now[p.name] > 1.0 else 0.0
                    self.state[p.name]["high_s"] += self.control_dt if self.pos(p)[2] > 9.0 else 0.0
            self.blow()
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
                wn = self.wind_now()
                self.wind_rows.append((d.time, float(np.hypot(wn[0], wn[1])), math.degrees(math.atan2(wn[1], wn[0])), wn[0], wn[1]))
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
                              "max_speed": r["players"][p.name]["max_speed"], "grid": self.players.index(p) + 1, "what": st.what,
                              "in_a_wake_s": round(self.state[p.name]["draft_s"], 1), "up_high_s": round(self.state[p.name]["high_s"], 1),
                              "passes_tried": self.state[p.name]["passes"]})
        r["race"] = {"standings": standings, "winner": standings[0]["strategy"] if standings[0]["finished"] else None, "laps": self.laps, "wind": self.windp,
                     "balloon_respawns": self.respawns, "wind_hist": np.array(self.wind_rows),
                     "gust_fronts": [e for e in self.events if e["what"] == "gust front"], "lines": {p.name: self.strategy[p.name] for p in self.players},
                     "faults": [e for e in self.events if e["what"] in ("foul", "crash")]}
        r["score"] = {s["strategy"]: s["laps"] for s in standings}
        r["winner"] = r["race"]["winner"]
        return r


def race(seed: int = 1, laps: int = 3, strategies=None, grid=None, wind: dict | None = None, duration: float = 150.0, n_balloons: int = 14,
         slipstream: bool = True) -> dict:
    return Race(strategies, seed, laps, duration, grid=grid, wind=wind, n_balloons=n_balloons, slipstream=slipstream).run()


def solo(strategy: Strategy, laps: int = 3, wind: dict | None = None, balloons: int = 0, seed: int = 1) -> dict:
    """One strategy alone on the track (no balloons unless asked): its lap time with nobody in the way."""
    return Race([strategy], seed, laps, 120.0, wind=wind, n_balloons=balloons).run()


def tournament(n: int = 7, seed: int = 1, laps: int = 3, strategies=None, wind: dict | None = None, progress=None, n_balloons: int = 14,
               slipstream: bool = True) -> dict:
    """``n`` races of all the strategies against each other, the grid turned one place each race: per strategy the
    wins, the mean place, the races finished, the mean race time, the best lap and the faults per race."""
    strats = list(strategies or FIELD)
    k = len(strats)
    rows = {s.name: {"strategy": s.name, "wins": 0, "podiums": 0, "places": [], "finished": 0, "times": [], "best_lap": [], "faults": 0, "what": s.what,
                     "wake": 0.0, "high": 0.0} for s in strats}
    races = []
    for i in range(n):
        grid = [(j + i) % k for j in range(k)]
        r = Race(strats, seed + i, laps, grid=grid, wind=wind, n_balloons=n_balloons, slipstream=slipstream).run()
        races.append({"seed": seed + i, "order": [s["strategy"] for s in r["race"]["standings"]], "winner": r["race"]["winner"],
                      "time": r["race"]["standings"][0]["time_s"], "faults": sum(s["faults"] for s in r["race"]["standings"])})
        for s in r["race"]["standings"]:
            row = rows[s["strategy"]]
            row["places"].append(s["place"])
            row["wins"] += s["place"] == 1 and s["finished"]
            row["podiums"] += s["place"] <= 3
            row["finished"] += s["finished"]
            row["faults"] += s["faults"]
            row["wake"] += s["in_a_wake_s"]
            row["high"] += s["up_high_s"]
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
                      "best lap [s]": round(float(min(row["best_lap"])), 2) if row["best_lap"] else None, "faults per race": round(row["faults"] / n, 2), "in a wake [s]": round(row["wake"] / n, 1), "up high [s]": round(row["high"] / n, 1),
                      "what": row["what"]})
    table.sort(key=lambda t: (-t["wins"], t["mean place"]))
    return {"table": table, "races": races, "n": n, "laps": laps}
