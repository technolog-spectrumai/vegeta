"""The match: MuJoCo flies the Émerillons, a pilot per aircraft decides where to go, the referee counts.

Each control step (250 Hz): the pilot gives a wanted velocity; the lab's geometric quad controller
(``races.flight.Flight``: force → the body's nose axis onto it → four rotor thrusts through the mixer) follows it; the
rotor hook puts the four thrusts (first-order lag, the drive's thrust limit at the axial inflow from
``emerillon_mission.envelope``, the propellers' reaction torques) and the airframe's drag and normal force on the body
through ``data.xfrc_applied``; MuJoCo integrates and finds the contacts; the referee reads them.

**Rules** (``Rules``): a point for flying through the other side's hoop, from either face; after a point both
aircraft go back to their starts. With ``possession`` (the default — it is what makes them chase each other; switch it
off for the plain race to the hoops) a **token** waits at the centre of the field: the first aircraft through it
carries it, **only the carrier scores**, and the other takes the token by a **tag**: flying within ``tag_radius`` of
the carrier for ``tag_dwell_s`` without touching it. A carrier that fouls hands the token over; one that crashes
sends it back to the centre. A **foul** is a contact with the other aircraft (charged to the one closing faster,
to both when it is even) or with a balloon; a **crash** is a contact with the ground or a hoop, or leaving the
field. A foul or a crash sends the aircraft back to its start for a penalty of a few seconds. The winner has more
points; on equal points, fewer fouls and crashes.

**Pilots** (``Pilot``): no learning — a rule-based player. *Attack* (the carrier): line up on the hoop's axis, fly
through, keep away from the chaser until the last metres. *Chase*: fly to where the carrier will be and sit beside it
inside the tag radius, a metre off. *Race*: to the free token. Without possession, *mark*: when the opponent is the
nearer to scoring and close to my hoop, get between it and the hoop (for a few seconds: then go for theirs). Always: steer round whatever the next second and a half brings too close — balloons, the other
aircraft — and keep off the ground and inside the field. Two pilots differ by their numbers (speed, caution, patience).

What it is not: the aircraft's aerodynamic moments, the propellers at an angle to the flow and the balloons' air are
not modelled; a balloon is a light free body with its weight taken off that drifts on a little random force.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

import emerillon_flight as ef
import emerillon_mission as mi
from races import flight as dc

from . import arena as ar

G = 9.81
DV_MAX = 3.2                       # the largest velocity error handed to the controller [m/s]: about 1.7 g asked for
DV_TAU = 0.12                      # and how fast that error may turn [s] — a reversed command must not ask for a flip
SPIN_UP = 0.04                     # rotor thrust lag [s] (assumed: a 7-inch propeller on a 2807)


@dataclass
class Rules:
    duration: float = 60.0
    penalty_s: float = 3.0
    kickoff_s: float = 1.2
    margin: float = 3.0            # how far outside the field is out
    score: str = "through"         # "through" the hoop, or "over" it (within ``over_height`` above the ring)
    over_height: float = 3.0
    possession: bool = True        # the chase: a token at the centre, only its carrier scores, the other takes it by a tag
    token_radius: float = 1.5      # flying within this of the free token picks it up [m]
    tag_radius: float = 2.0        # the chaser this close to the carrier, without touching ...
    tag_dwell_s: float = 0.35      # ... for this long, takes the token
    tag_immunity_s: float = 1.5    # the one who lost it cannot tag back at once
    mode: str = "hoops"            # "hoops": points through the other side's hoop; "keepaway": every aircraft for itself,
                                   # a point per second carrying the token and ``hoop_bonus`` for a hoop flown through with it
    hoop_bonus: float = 5.0
    hoop_cooldown_s: float = 6.0


@dataclass
class Pilot:
    """A rule-based player's numbers."""
    name: str = "pilot"
    v_max: float = 14.0            # cruising speed in the field [m/s]
    v_hoop: float = 8.0            # through the hoop
    caution: float = 1.0           # scales the room it leaves round balloons and the other aircraft
    look_s: float = 1.5            # how far ahead it looks for a conflict
    mark_within: float = 14.0      # it marks an opponent this close to its hoop
    patience_s: float = 5.0        # for this long, then it attacks
    z_cruise: float = 4.5


def _closest(rel_p, rel_v, horizon):
    """Time and miss vector of the closest approach within ``horizon`` for a relative position and velocity."""
    vv = float(rel_v @ rel_v)
    t = 0.0 if vv < 1e-9 else min(max(-float(rel_p @ rel_v) / vv, 0.0), horizon)
    return t, rel_p + rel_v * t


class Player:
    """One aircraft in the match: its body in the model, its controller, its pilot, its record."""

    def __init__(self, name: str, variant: str, team: str, pilot: Pilot, slot: int = 0, home=None, color: str | None = None):
        self.name, self.variant, self.team, self.pilot, self.slot = name, variant, team, pilot, slot
        self.home_given = None if home is None else np.asarray(home, float)
        self.color = color or {"A": ar.PALETTE[0], "B": ar.PALETTE[1]}.get(team, ar.PALETTE[2])
        self.hold_s = self.bonus = 0.0
        self.passes = 0
        self.n = ar.body_numbers(variant)
        self.env = mi.envelope(variant)
        st = ef.drive(self.n["config"], 0.5)
        self.kappa = float(st["shaft_w"] / (st["rpm"] * 2 * math.pi / 60) / st["thrust"])      # reaction torque per newton of thrust [m]
        self.prop = SimpleNamespace(max_thrust_N=float(st["thrust"]) / 4, torque_per_thrust_m=self.kappa, spin_up_s=SPIN_UP)
        self.goals = self.fouls = self.crashes = 0
        self.frozen_until = 0.0
        self.role, self.mark_since, self.wh = "attack", None, 0.0
        self.thrust = np.zeros(4)
        self.dv = np.zeros(3)
        self.rows = []

    def bind(self, model, data, a: ar.Arena):
        import mujoco
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.name)
        jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{self.name}_free")
        self.qa, self.va = int(model.jnt_qposadr[jnt]), int(model.jnt_dofadr[jnt])
        self.home = self.home_given if self.home_given is not None else a.start(self.team, self.slot)
        n = self.n

        class Ctl(dc.Flight):
            def reset(ctl):
                ctl.m0 = float(model.body_subtreemass[self.body])
                ctl.J = np.array(model.body_inertia[self.body], dtype=float).copy()
                w, z = ctl.omega_att, ctl.zeta_att
                ctl.kR = np.array([ctl.J[0] * w * w, ctl.J[1] * w * w, ctl.J[2] * ctl.omega_yaw ** 2])
                ctl.kW = np.array([2 * z * ctl.J[0] * w, 2 * z * ctl.J[1] * w, 2 * 0.9 * ctl.J[2] * ctl.omega_yaw])
                ctl.A = np.vstack([np.ones(4), n["rotor_pos"][:, 1], -n["rotor_pos"][:, 0], n["spin"] * self.kappa])
                ctl.Ainv = np.linalg.inv(ctl.A)
                ctl.integral = np.zeros(3)
                ctl.last = {}
            def mix(ctl, T, M):                                     # keep thrust on the rotors when the nose is far from where it should
                return super().mix(max(float(T), 0.6 * ctl.m0 * G), M)   # point: without it there is no torque to turn it back (it tumbled)
        self.ctl = Ctl(self.prop, omega_pos=2.6, omega_att=20.0, tilt_max_deg=58.0)
        self.ctl.reset()

    def place(self, data, t: float, hold: float):
        """Back to the start, upright and at rest; frozen (hovering there) until ``t + hold``."""
        data.qpos[self.qa:self.qa + 3] = self.home
        data.qpos[self.qa + 3:self.qa + 7] = (1, 0, 0, 0)
        data.qvel[self.va:self.va + 6] = 0.0
        self.thrust[:] = self.ctl.m0 * G / 4
        self.dv[:] = 0.0
        self.ctl.reset_integral()
        self.frozen_until, self.mark_since, self.role = t + hold, None, "attack"


class Match:
    def __init__(self, a: ar.Arena | None = None, players=None, rules: Rules | None = None, control_dt: float = 0.004, seed: int = 1):
        import mujoco
        self.mj = mujoco
        self.a = a or ar.Arena()
        self.rules = rules or Rules()
        self.players = players or [Player("a1", "ER2", "A", Pilot("Ada", v_max=14.5, v_hoop=9.0, caution=1.0)),
                                   Player("b1", "ER2", "B", Pilot("Bruno", v_max=15.0, v_hoop=7.5, caution=0.8, patience_s=3.5))]
        xml = ar.to_mjcf(self.a, [(p.name, p.variant, p.color, p.home_given if p.home_given is not None else self.a.start(p.team, p.slot)) for p in self.players])
        self.xml = xml
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.control_dt = control_dt
        self.n_sub = max(int(round(control_dt / self.model.opt.timestep)), 1)
        self.rng = np.random.default_rng(seed)
        m = self.model
        self.balloons = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"balloon{i}") for i in range(self.a.n_balloons)]
        self.balloon_dof = [int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"balloon{i}_free")]) for i in range(self.a.n_balloons)]
        self.gust = np.zeros((self.a.n_balloons, 3))
        for p in self.players:
            p.bind(m, self.data, self.a)
        self.owner = {}                                            # geom -> ("drone", player) | ("balloon", i) | ("world", name)
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            pl = next((p for p in self.players if p.body == b), None)
            self.owner[g] = ("drone", pl) if pl else (("balloon", self.balloons.index(b)) if b in self.balloons else ("world", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)))
        for i, dof in enumerate(self.balloon_dof):                 # a slow start
            self.data.qvel[dof:dof + 3] = self.rng.normal(0.0, 0.25, 3) * (1, 1, 0.3)
        self.events, self.balloon_rows, self.qpos_rows, self.token_rows = [], [], [], []
        self.holder, self.token = None, np.array([0.0, 0.0, 4.0])
        self.new_token()
        self.tag_time, self.immune, self.hoop_free = 0.0, {}, {}
        mujoco.mj_forward(m, self.data)
        for p in self.players:
            p.place(self.data, 0.0, self.rules.kickoff_s)

    # ------------------------------------------------------------------------------------------------- state
    def new_token(self):
        """The free token's place for the next kick-off: somewhere in the middle third, clear of the balloons."""
        a = self.a
        for _ in range(50):
            c = np.array([self.rng.uniform(-0.15, 0.15) * a.length, self.rng.uniform(-0.3, 0.3) * a.width, self.rng.uniform(3.0, 7.0)])
            if all(float(np.linalg.norm(self.data.xpos[b] - c)) > a.balloon_radius + 2.5 for b in self.balloons):
                break
        self.token = c

    def pos(self, p: Player) -> np.ndarray:
        return self.data.qpos[p.qa:p.qa + 3].copy()

    def vel(self, p: Player) -> np.ndarray:
        return self.data.qvel[p.va:p.va + 3].copy()

    def opponent(self, p: Player) -> Player:
        return min((q for q in self.players if q.team != p.team), key=lambda q: float(np.linalg.norm(self.pos(q) - self.pos(p))))

    # ------------------------------------------------------------------------------------------------- pilot
    def restart(self, p: Player, t: float, hold: float):
        """Where a faulted aircraft goes (the formation puts it back in its slot instead)."""
        p.place(self.data, t, hold)

    def others(self, me: Player) -> list:
        return [q for q in self.players if q is not me]

    def run_at_hoop(self, me: Player, H, speed_far: float, speed_near: float) -> tuple:
        """Where to aim and how fast to fly through the hoop at ``H`` from the side ``me`` is on; True when it is lined
        up for the run (then the hoop is not something to steer round)."""
        p = self.pos(me)
        s = 1.0 if p[0] <= H[0] else -1.0
        along = s * (H[0] - p[0])
        off = float(np.hypot(p[1] - H[1], p[2] - H[2]))
        if off < 0.2 + 0.08 * along:
            return H + np.array([s * 3.0, 0.0, 0.0]), (speed_near if along < 8.0 else speed_far), True
        return H - np.array([s * min(6.0, max(along, 3.0)), 0.0, 0.0]), speed_far, off < 0.7 + 0.08 * along

    def steer(self, me: Player, v_want, rooms: dict | None = None, hoop_run=None, cap: float | None = None, near: float = 1.3):
        """A wanted velocity made safe: round whatever the next moments bring too close — balloons, the other aircraft
        (``rooms``: the room to keep from a given aircraft, else ``near`` beyond touching), the hoops (unless
        ``hoop_run`` is the hoop it is lined up on) — off the ground, under the ceiling, inside the field."""
        a, pl = self.a, me.pilot
        p, v = self.pos(me), self.vel(me)
        reach = me.n["reach"]
        # (where, its velocity, the room to keep, how much it counts): what cannot move counts most
        obstacles = [(self.data.xpos[b].copy(), self.data.cvel[b][3:].copy(), a.balloon_radius + reach + 0.9 * pl.caution, 1.5) for b in self.balloons]
        for q in self.others(me):
            room = (rooms or {}).get(q.name, reach + q.n["reach"] + near * pl.caution)
            obstacles.append((self.pos(q), self.vel(q), room, 1.0))
        for hc in a.hoops.values():
            if hoop_run is None or hc is not hoop_run:
                obstacles.append((hc, np.zeros(3), a.hoop_radius + reach + 0.9, 2.5))
                obstacles.append((np.array([hc[0], hc[1], 0.5 * a.hoop_stick]), np.zeros(3), reach + 0.9, 2.5))
        push = np.zeros(3)
        for q_p, q_v, room, weight in obstacles:
            tc, miss = _closest(p - q_p, v - q_v, pl.look_s)
            d = float(np.linalg.norm(miss))
            if d < room:
                away = miss / d if d > 1e-3 else np.array([0.0, 1.0, 0.0])
                push += away * (room - d) / room * pl.v_max * (1.6 if tc < 0.6 else 1.0) * weight
        push[2] *= 0.4                                              # sideways first: the ground is near
        pn_ = float(np.linalg.norm(push))
        if pn_ > 1.2 * pl.v_max:
            push *= 1.2 * pl.v_max / pn_
        v_want = np.asarray(v_want, float) + push
        me.urgent = {}                                              # the axes where safety speaks: they get the controller's full authority
        z_soon = p[2] + 0.7 * min(v[2], 0.0)                        # where a descent puts it shortly
        if z_soon < 1.9:
            v_want[2] = max(v_want[2], 3.0 * (1.9 - z_soon))
            me.urgent[2] = v_want[2]
        if p[2] > a.ceiling - 1.0:
            v_want[2] = min(v_want[2], -1.0)
        for k, lim in ((0, a.length / 2), (1, a.width / 2)):       # the field's edge: turn back with the room it takes to stop
            room = lim - 1.0 - abs(p[k])
            out = v[k] * math.copysign(1.0, p[k])                   # speed toward that edge
            if room < max(out, 0.0) * 0.5 + max(out, 0.0) ** 2 / (2 * 9.0) + 0.5 and (p[k] * v_want[k] > 0 or out > 0):   # half a second to turn the nose, then the brake
                v_want[k] = -math.copysign(max(2.0, 0.6 * abs(out)), p[k])
                me.urgent[k] = v_want[k]
        cap = pl.v_max * 1.15 if cap is None else cap
        n = float(np.linalg.norm(v_want))
        if n > cap:
            v_want = v_want * cap / n
            me.urgent = {k: x * cap / n for k, x in me.urgent.items()}
        return v_want

    def chase_point(self, me: Player, quarry: Player) -> tuple:
        """Beside where the quarry will be, inside the tag radius, a metre off: (the point, the velocity to hold there)."""
        p, pl = self.pos(me), me.pilot
        ph, vh = self.pos(quarry), self.vel(quarry)
        lead = ph + vh * min(float(np.linalg.norm(ph - p)) / max(pl.v_max, 1.0), 1.2)
        side = p - lead
        side[2] *= 0.3
        sn = float(np.linalg.norm(side))
        goal = lead + (side / sn if sn > 1e-3 else np.array([0.0, 1.0, 0.0])) * 1.25
        goal[2] = max(goal[2], 1.9)
        v_want = vh + (goal - p) * 2.5
        n_ = float(np.linalg.norm(v_want))
        return goal, v_want * min(1.0, pl.v_max / max(n_, 1e-6))

    def decide(self, me: Player, t: float) -> tuple:
        """The wanted velocity of ``me`` and what it is doing."""
        if self.rules.mode == "keepaway":
            return self.decide_keepaway(me, t)
        a, pl, r = self.a, me.pilot, self.rules
        p = self.pos(me)
        opp = self.opponent(me)
        po, vo = self.pos(opp), self.vel(opp)
        s = 1.0 if me.team == "A" else -1.0                         # my attack goes along s * x
        H, O = a.hoops["B" if me.team == "A" else "A"], a.hoops[me.team]
        d_me, d_opp = float(np.linalg.norm(H - p)), float(np.linalg.norm(O - po))
        threat = d_opp < d_me - 2.0 and d_opp < pl.mark_within and (po[0] - O[0]) * s > 0.5
        rooms, hoop_run = {}, None
        if r.possession and self.holder is None:                    # the free token: first there has it
            me.role = "race"
            to = self.token - p
            v_want = to / max(float(np.linalg.norm(to)), 1e-6) * pl.v_max
        elif r.possession and self.holder is not me:                # chase the carrier: beside it, inside the tag radius, not touching
            me.role = "chase"
            _, v_want = self.chase_point(me, self.holder)
            rooms[self.holder.name] = me.n["reach"] + self.holder.n["reach"] + 0.35
        elif not r.possession and threat and (me.mark_since is None or t - me.mark_since < pl.patience_s):
            me.mark_since = t if me.mark_since is None else me.mark_since
            me.role = "mark"
            gap = po - O
            dist = float(np.linalg.norm(gap))
            to = O + gap / max(dist, 1e-6) * min(3.0, max(dist - 2.0, 1.0)) - p   # between the opponent and my hoop
            dn = float(np.linalg.norm(to))
            v_want = to / max(dn, 1e-6) * min(pl.v_max, 2.2 * dn + 1.0) + (0.6 * vo if dn < 2.5 else 0.0)
        else:
            if not threat:
                me.mark_since = None
            me.role = "attack"
            along = s * (H[0] - p[0])
            if along > -1.0:
                goal, speed, lined = self.run_at_hoop(me, H, pl.v_max, pl.v_hoop)
                hoop_run = H if lined else None
            else:                                                   # behind their hoop: round the side and line up again
                goal, speed = np.array([H[0] - s * 5.0, (1.0 if p[1] >= 0 else -1.0) * 4.0, H[2] + 1.0]), 0.7 * pl.v_max
            to = goal - p
            dn = float(np.linalg.norm(to))
            v_want = to / max(dn, 1e-6) * min(speed, 2.2 * dn + 1.0)
            if r.possession and not (hoop_run is not None and along < 8.0 and float(np.hypot(p[1] - H[1], p[2] - H[2])) < 1.5):
                rooms = {q.name: r.tag_radius + 0.5 * pl.caution for q in self.others(me)}     # the carrier keeps out of the tag
        return self.steer(me, v_want, rooms, hoop_run), me.role

    def decide_keepaway(self, me: Player, t: float) -> tuple:
        """Every aircraft for itself: race to the free token; with it, run — away from the chasers, and through a hoop
        for the bonus when they are far enough; without it, chase the carrier."""
        a, pl, r = self.a, me.pilot, self.rules
        p = self.pos(me)
        rooms, hoop_run = {}, None
        if self.holder is None:
            me.role = "race"
            to = self.token - p
            v_want = to / max(float(np.linalg.norm(to)), 1e-6) * pl.v_max
        elif self.holder is not me:
            me.role = "chase"
            _, v_want = self.chase_point(me, self.holder)
            rooms[self.holder.name] = me.n["reach"] + self.holder.n["reach"] + 0.35
        else:
            me.role = "attack"
            chasers = [q for q in self.others(me) if t >= q.frozen_until]
            gaps = [(float(np.linalg.norm(p - self.pos(q))), q) for q in chasers]
            nearest = min((g for g, _ in gaps), default=99.0)
            free = [hc for k, hc in a.hoops.items() if t >= self.hoop_free.get((me.name, k), 0.0)]
            target = min(free, key=lambda hc: float(np.linalg.norm(hc - p))) if free else None
            if target is not None and (nearest > 7.0 or float(np.linalg.norm(target - p)) < 6.0):
                goal, speed, lined = self.run_at_hoop(me, target, pl.v_max, pl.v_hoop + 2.0)
                hoop_run = target if lined else None
                to = goal - p
                dn = float(np.linalg.norm(to))
                v_want = to / max(dn, 1e-6) * min(speed, 2.2 * dn + 1.0)
            else:                                                   # run: away from each chaser, the nearer the stronger; not into a corner
                away = np.zeros(3)
                for g, q in gaps:
                    away += (p - self.pos(q)) / max(g, 0.5) ** 2
                away[2] *= 0.3
                away[:2] -= 0.004 * p[:2] * np.array([1.0, 2.0])
                away[2] += 0.02 * (pl.z_cruise - p[2])
                v_want = away / max(float(np.linalg.norm(away)), 1e-6) * pl.v_max
            rooms = {q.name: r.tag_radius + 0.6 * pl.caution for q in self.others(me)}
        return self.steer(me, v_want, rooms, hoop_run), me.role

    # ----------------------------------------------------------------------------------------------- physics
    def actuate(self, me: Player, v_want, hover_at=None):
        d, dt = self.data, self.control_dt
        p, v = self.pos(me), self.vel(me)
        R = d.xmat[me.body].reshape(3, 3)
        omega = R.T @ d.qvel[me.va + 3:me.va + 6] if False else d.qvel[me.va + 3:me.va + 6].copy()     # a free joint's angular velocity is in the body frame
        b = R[:, 2]
        v_ax = float(v @ b)
        me.prop.max_thrust_N = max(mi.thrust_at(me.env, 0.0, max(v_ax, 0.0), 1.0)[0] / 4, 0.5)
        obs = SimpleNamespace(base_pos=p, base_quat=d.qpos[me.qa + 3:me.qa + 7].copy(), body_linvel=[v], body_angvel=[omega])
        yaw = 0.0
        if hover_at is not None:
            ref = dc.Ref(p=np.asarray(hover_at, float), v=np.zeros(3), a=np.zeros(3), yaw=yaw)
        else:                                                       # a velocity loop: the error bounded, and turned slowly
            e = np.asarray(v_want, float) - v
            en = float(np.linalg.norm(e))
            if en > DV_MAX:
                e *= DV_MAX / en
            for k, vt in getattr(me, "urgent", {}).items():         # the ground and the field's edge are not traded against the chase
                e[k] = min(max(vt - v[k], -DV_MAX), DV_MAX)
            me.dv += (1.0 - math.exp(-dt / DV_TAU)) * (e - me.dv)
            ref = dc.Ref(p=p, v=v + me.dv, a=np.zeros(3), yaw=yaw)
        cmd = np.asarray(me.ctl.thrusts(obs, ref, 0.0, dt), float)
        me.thrust += (1.0 - math.exp(-dt / SPIN_UP)) * (cmd - me.thrust)
        T = me.thrust
        # the air on the airframe: axial drag and a normal force, as emerillon_mission
        V = float(np.linalg.norm(v))
        aero = np.zeros(3)
        if V > 0.5:
            vhat = v / V
            ca = float(b @ vhat)
            sa = math.sqrt(max(1.0 - ca * ca, 0.0))
            q = 0.5 * ef.RHO * V * V
            env = me.env
            cda = float(np.interp(V, env["V"], env["cda"]))
            cna = (2 * env["frontal"] + 2.5 * env["arms"]) * sa * abs(ca) + 1.2 * env["plan"] * sa * sa
            perp = -(vhat - ca * b)
            pn = float(np.linalg.norm(perp))
            aero = -q * cda * ca * abs(ca) * (1.0 if ca >= 0 else mi.BACKWARD_DRAG) * b + (q * cna * perp / pn if pn > 1e-6 else 0.0)
        rp = me.n["rotor_pos"]
        M_body = np.array([float(rp[:, 1] @ T), -float(rp[:, 0] @ T), float(me.n["spin"] @ T) * me.kappa])
        d.xfrc_applied[me.body, :3] = b * float(T.sum()) + aero
        d.xfrc_applied[me.body, 3:] = R @ M_body
        return float(T.sum()), v_ax

    def drift(self):
        """The balloons' air: a slow random push, a pull back toward the middle heights and the field."""
        a, d = self.a, self.data
        self.gust += (-self.gust / 4.0) * self.control_dt + 0.02 * math.sqrt(2 * self.control_dt / 4.0) * self.rng.standard_normal(self.gust.shape)
        for i, b in enumerate(self.balloons):
            x = d.xpos[b]
            f = self.gust[i].copy() * a.balloon_mass * 30.0
            f[2] += 0.004 * (0.45 * a.ceiling - x[2])
            f[0] += -0.002 * x[0] if abs(x[0]) > 0.6 * a.hoop_x else 0.0
            f[1] += -0.002 * x[1] if abs(x[1]) > 0.35 * a.width else 0.0
            d.xfrc_applied[b, :3] = f

    # ----------------------------------------------------------------------------------------------- referee
    def referee(self, t: float, before: dict):
        a, d, r = self.a, self.data, self.rules
        hit = {}                                                   # player -> what it touched
        for i in range(d.ncon):
            c = d.contact[i]
            o1, o2 = self.owner[int(c.geom1)], self.owner[int(c.geom2)]
            for mine, other in ((o1, o2), (o2, o1)):
                if mine[0] == "drone" and t >= mine[1].frozen_until:
                    kind = "aircraft" if other[0] == "drone" else ("balloon" if other[0] == "balloon" else "ground" if other[1] == "ground" else "hoop")
                    if kind != "aircraft" or hit.get(mine[1], ("", None))[0] != "balloon":
                        hit[mine[1]] = (kind, other[1])
        pair = [p for p, (k, _) in hit.items() if k == "aircraft"]
        if len(pair) >= 2:                                         # who ran into whom: the one closing faster
            p1, p2 = pair[0], pair[1]
            n12 = self.pos(p2) - self.pos(p1)
            n12 /= max(float(np.linalg.norm(n12)), 1e-6)
            c1, c2 = float(self.vel(p1) @ n12), -float(self.vel(p2) @ n12)
            guilty = [p1, p2] if abs(c1 - c2) < 0.25 * max(abs(c1), abs(c2), 1.0) else [p1 if c1 > c2 else p2]
            for p in pair:
                if p not in guilty:
                    del hit[p]
                    self.events.append({"t": round(t, 2), "what": "fouled", "who": p.name, "by": guilty[0].name})
                    self.restart(p, t, r.kickoff_s)                 # the one run into restarts without a penalty
        for p, (kind, what) in hit.items():
            if kind in ("aircraft", "balloon"):
                p.fouls += 1
            else:
                p.crashes += 1
            if r.possession and self.holder is p:                  # the carrier's fault: the token changes hands, or goes back
                self.holder = self.opponent(p) if kind in ("aircraft", "balloon") else None
                self.tag_time = 0.0
                if self.holder is None:
                    self.new_token()
                self.events.append({"t": round(t, 2), "what": "token", "to": self.holder.name if self.holder else "centre", "why": f"{p.name}'s {kind}"})
            self.events.append({"t": round(t, 2), "what": "foul" if kind in ("aircraft", "balloon") else "crash", "who": p.name, "on": kind,
                                "pos": [round(float(x), 2) for x in self.pos(p)], "speed": round(float(np.linalg.norm(self.vel(p))), 1)})
            self.restart(p, t, r.penalty_s)
        for p in self.players:
            if p in hit or t < p.frozen_until:
                continue
            x = self.pos(p)
            if abs(x[0]) > a.length / 2 + r.margin or abs(x[1]) > a.width / 2 + r.margin or x[2] > a.ceiling + r.margin:
                p.crashes += 1
                self.events.append({"t": round(t, 2), "what": "crash", "who": p.name, "on": "out of the field", "pos": [round(float(v), 2) for v in x]})
                self.restart(p, t, r.penalty_s)
                continue
            x0 = before[p.name]
            stop = False
            for hoop, H in a.hoops.items():
                if (x0[0] - H[0]) * (x[0] - H[0]) < 0:             # it crossed this hoop's plane this step
                    k = (H[0] - x0[0]) / (x[0] - x0[0])
                    y, z = x0[1] + k * (x[1] - x0[1]), x0[2] + k * (x[2] - x0[2])
                    through = math.hypot(y - H[1], z - H[2]) < a.hoop_radius - a.hoop_tube
                    over = abs(y - H[1]) < a.hoop_radius and 0.0 < z - (H[2] + a.hoop_radius) < r.over_height
                    stop = self.crossed(p, hoop, (r.score == "through" and through) or (r.score == "over" and over), t) or stop
            if stop:
                break
        if r.possession:
            live = [p for p in self.players if t >= p.frozen_until]
            if self.holder is None:
                near = [p for p in live if float(np.linalg.norm(self.pos(p) - self.token)) < r.token_radius]
                if near:
                    self.holder = min(near, key=lambda p: float(np.linalg.norm(self.pos(p) - self.token)))
                    self.immune = {}
                    self.events.append({"t": round(t, 2), "what": "token", "to": self.holder.name, "why": "picked up", "pos": [round(float(x), 1) for x in self.token]})
            elif t >= self.holder.frozen_until:
                tagger = next((p for p in live if p.team != self.holder.team and t >= self.immune.get(p.name, 0.0)
                               and float(np.linalg.norm(self.pos(p) - self.pos(self.holder))) < r.tag_radius), None)
                self.tag_time = self.tag_time + self.control_dt if tagger else 0.0
                if tagger and self.tag_time >= r.tag_dwell_s:
                    self.immune = {self.holder.name: t + r.tag_immunity_s}
                    self.events.append({"t": round(t, 2), "what": "tag", "who": tagger.name, "from": self.holder.name,
                                        "pos": [round(float(x), 1) for x in self.pos(tagger)]})
                    self.holder, self.tag_time = tagger, 0.0
        if hit:
            self.mj.mj_forward(self.model, d)

    def crossed(self, p: Player, hoop: str, good: bool, t: float) -> bool:
        """An aircraft crossed a hoop's plane (``good``: through the ring, or over it under the 'over' rule). True when
        the play restarts (a goal)."""
        r, d = self.rules, self.data
        if not good:
            return False
        p.passes += 1
        if r.mode == "keepaway":
            if self.holder is p and t >= self.hoop_free.get((p.name, hoop), 0.0):
                p.bonus += r.hoop_bonus
                self.hoop_free[(p.name, hoop)] = t + r.hoop_cooldown_s
                self.events.append({"t": round(t, 2), "what": "hoop", "who": p.name, "hoop": hoop, "points": r.hoop_bonus})
            return False
        if hoop == p.team or (r.possession and self.holder is not p):
            return False
        p.goals += 1
        self.holder, self.tag_time = None, 0.0
        self.new_token()
        self.events.append({"t": round(t, 2), "what": "goal", "who": p.name, "team": p.team, "speed": round(float(np.linalg.norm(self.vel(p))), 1),
                            "score": {q.team: sum(w.goals for w in self.players if w.team == q.team) for q in self.players}})
        for q in self.players:
            q.place(d, t, r.kickoff_s)
        return True

    # ---------------------------------------------------------------------------------------------------- run
    def run(self, duration: float | None = None, record_dt: float = 0.04) -> dict:
        mj, m, d = self.mj, self.model, self.data
        duration = self.rules.duration if duration is None else duration
        next_rec, k = 0.0, 0
        while d.time < duration:
            t = d.time
            before = {p.name: self.pos(p) for p in self.players}
            for p in self.players:
                if t < p.frozen_until:
                    T, v_ax = self.actuate(p, None, hover_at=p.home)
                    p.role = "restart"
                else:
                    v_want, _ = self.decide(p, t)
                    T, v_ax = self.actuate(p, v_want)
                if k % 5 == 0:
                    u = mi.throttle_for(p.env, 0.0, max(v_ax, 0.0), T)
                    p.wh += mi.thrust_at(p.env, 0.0, max(v_ax, 0.0), u)[1] * 5 * self.control_dt / 3600.0
            if self.holder is not None and t >= self.holder.frozen_until:
                self.holder.hold_s += self.control_dt
            self.drift()
            for _ in range(self.n_sub):
                mj.mj_step(m, d)
            self.referee(d.time, before)
            if d.time >= next_rec:
                next_rec += record_dt
                for p in self.players:
                    R = d.xmat[p.body].reshape(3, 3)
                    p.rows.append((d.time, *self.pos(p), *self.vel(p), *R[:, 2], {"attack": 0, "mark": 1, "restart": 2, "chase": 3, "race": 4}[p.role], p.wh, 1.0 if self.holder is p else 0.0))
                self.balloon_rows.append(np.array([d.xpos[b].copy() for b in self.balloons]))
                self.qpos_rows.append(d.qpos.copy())
                self.token_rows.append((self.token.copy(), self.holder.name if self.holder else None))
            k += 1
        return self.result()

    def result(self) -> dict:
        teams = sorted({p.team for p in self.players})
        score = {tm: sum(p.goals for p in self.players if p.team == tm) for tm in teams}
        if self.rules.mode == "keepaway":
            score = {tm: round(sum(p.hold_s + p.bonus for p in self.players if p.team == tm), 1) for tm in teams}
        faults = {tm: sum(p.fouls + p.crashes for p in self.players if p.team == tm) for tm in teams}
        best = max(score.values())
        lead = [tm for tm in teams if score[tm] == best]
        if len(lead) > 1:
            fewest = min(faults[tm] for tm in lead)
            lead = [tm for tm in lead if faults[tm] == fewest]
        cols = ("t", "x", "y", "z", "vx", "vy", "vz", "bx", "by", "bz", "role", "wh", "token")
        return {"duration": float(self.data.time), "score": score, "faults": faults, "winner": lead[0] if len(lead) == 1 else None, "events": self.events,
                "players": {p.name: {"team": p.team, "variant": p.variant, "pilot": p.pilot.name, "goals": p.goals, "fouls": p.fouls, "crashes": p.crashes,
                                     "color": p.color, "hold_s": round(p.hold_s, 2), "bonus": p.bonus, "hoops": p.passes,
                                     "energy_wh": round(p.wh, 2), "pack_wh": p.env["wh"],
                                     "max_speed": float(max(math.sqrt(r[4] ** 2 + r[5] ** 2 + r[6] ** 2) for r in p.rows)) if p.rows else 0.0,
                                     "hist": {c: np.array([r[i] for r in p.rows]) for i, c in enumerate(cols)}} for p in self.players},
                "balloons": np.array(self.balloon_rows), "arena": self.a, "rules": self.rules, "qpos": np.array(self.qpos_rows), "xml": self.xml,
                "token_hist": [(t.tolist(), h) for t, h in self.token_rows],
                "counts": {k: sum(1 for e in self.events if e["what"] == k) for k in ("goal", "tag", "foul", "crash", "hoop")}}


def play(duration: float = 60.0, seed: int = 1, variants=("ER2", "ER2"), score: str = "through", n_balloons: int = 8, possession: bool = True) -> dict:
    """A 1 v 1 match with the two house pilots."""
    a = ar.Arena(n_balloons=n_balloons, seed=seed)
    players = [Player("a1", variants[0], "A", Pilot("Ada", v_max=14.5, v_hoop=9.0, caution=1.0)),
               Player("b1", variants[1], "B", Pilot("Bruno", v_max=15.0, v_hoop=7.5, caution=0.8, patience_s=3.5))]
    return Match(a, players, Rules(duration=duration, score=score, possession=possession), seed=seed).run()


def chase(n: int = 3, duration: float = 60.0, seed: int = 1, variant: str = "ER2", n_balloons: int = 8) -> dict:
    """A group token chase, every aircraft for itself (``n`` of them, 3 or 7): a point per second carrying the token,
    ``Rules.hoop_bonus`` for a hoop flown through with it; a tag takes the token; fouls and crashes as in a match."""
    rng = np.random.default_rng(seed)
    a = ar.Arena(n_balloons=n_balloons, seed=seed, length=60.0 if n <= 4 else 72.0, width=30.0 if n <= 4 else 40.0)
    names = ("Ada", "Bruno", "Chiara", "Dmitri", "Eleni", "Farid", "Greta", "Hugo")
    players = [Player(f"p{i + 1}", variant, f"P{i + 1}", Pilot(names[i % len(names)], v_max=float(rng.uniform(13.5, 15.5)), v_hoop=float(rng.uniform(7.5, 9.5)),
                                                             caution=float(rng.uniform(0.8, 1.1))), home=a.ring_start(i, n), color=ar.PALETTE[i % len(ar.PALETTE)]) for i in range(n)]
    return Match(a, players, Rules(duration=duration, mode="keepaway", possession=True), seed=seed).run()

