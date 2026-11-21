"""ASTOR's net drop in MuJoCo: ASTOR-V (or ASTOR-BOOST) overtakes a bandit, flies a little ahead of it and above
its track, pushes a net out of its tail; the net stops in the air, opens and sinks; the bandit flies into it — or
does not. Prepared here, RUN BY YOU (``sims/astor_netdrop.sh``).

**The bandits.**

- ``armageddon``: Armageddon's bandit quadcopter (``armageddon/chase.py``, ported here unchanged in behaviour): it
  wanders over a field at 6 m/s between 6 and 22 m up; with the hunter within THREAT_M it runs at 7 m/s away from
  it, jinking 60 degrees left and right every 2 s and changing height; it turns at most 1.6 rad/s, climbs 2 m/s, and
  stays within 55 m of the field's middle. A propeller aircraft is heard farther than Armageddon's quad (30 m):
  ``threat_m`` 60 here (HAND; Armageddon's loud hexacopter: 120).
- ``fast``: the target ASTOR is sized for: a fixed-wing drone at 50 m/s on a gentle weave (``astor_mujoco``'s).

**The attack** (``Attack``): fly at an aim point ``lead`` ahead of the bandit along its velocity and ``over`` above it
(proportional navigation on that point, the simulator's), at a speed that overtakes it; release when ASTOR is ahead
of the bandit by the distance the bandit covers while the net opens and sinks to its height, laterally inside the
net's radius; then climb away, turn back and come again with the next net. The bandit is seen as a camera sees it:
``latency_s`` late, ``noise_m`` of position noise.

**The net** (``Net``, HAND values): a 110 g cartridge that leaves at ASTOR's velocity less the ejector's 5 m/s aft;
packed for ``t_open`` (small drag), then a disc of ``radius`` with the drag of a 10 % solid mesh: it loses its
forward speed within a metre or two and sinks at ~2 m/s. A bandit is **caught** when it is inside the open net's
radius (less a margin) within ``thickness`` of the net's height. Not modelled: how a net entangles rotors or a
wing (a catch is contact with the open net), the net's shape as it falls, ASTOR's wake on it.
"""
from __future__ import annotations

import math
import random

import numpy as np

import arigo1_model as am
import arigo1_mujoco as amj
import astor_v_flight as VF

# ---- Armageddon's bandit (armageddon/chase.py @ dev_egui): its numbers
SPEED, RUN_SPEED, TURN, CLIMB = 6.0, 7.0, 1.6, 2.0
HEIGHTS, JINK_S, FIELD_M = (6.0, 22.0), 2.0, 55.0
THREAT_M = 60.0                    # HAND: how far it hears ASTOR (Armageddon: 30 m for a quad, 120 m for the hexacopter)
FLOOR_M = 6.0                      # ASTOR does not go below this (the bandit's lowest height)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class ArmageddonBandit:
    """Armageddon's ``Bandit``: wander, run and jink (its hit and gust responses are not needed here)."""

    def __init__(self, centre=(0.0, 0.0), seed=1, threat_m=THREAT_M):
        self.rng = random.Random(seed)
        self.centre = centre
        self.p = [centre[0] + 25.0, centre[1] - 10.0, 14.0]
        self.heading, self.speed, self.v = 1.0, SPEED, (0.0, 0.0, 0.0)
        self.threat, self.field = threat_m, FIELD_M
        self.goal = self._waypoint()
        self.want_z = self.p[2]
        self.jink, self.jink_at = 1.0, 0.0
        self.caught = None

    def _waypoint(self):
        a, r = self.rng.uniform(0, 2 * math.pi), self.rng.uniform(10.0, self.field * 0.8)
        return (self.centre[0] + r * math.cos(a), self.centre[1] + r * math.sin(a))

    def step(self, dt, t, hunter=None):
        if self.caught is not None:                              # in the net: it comes down
            self.p[2] = max(0.05, self.p[2] - 4.0 * dt)
            self.v = (0.0, 0.0, -4.0 if self.p[2] > 0.05 else 0.0)
            return
        x, y, z = self.p
        near = hunter is not None and math.dist(hunter, self.p) < self.threat
        if near:
            if t - self.jink_at > JINK_S:
                self.jink, self.jink_at = -self.jink, t
                self.want_z = self.rng.uniform(*HEIGHTS)
            want = math.atan2(y - hunter[1], x - hunter[0]) + self.jink * math.radians(60.0)
            self.speed = RUN_SPEED
        else:
            if math.dist((x, y), self.goal) < 6.0:
                self.goal = self._waypoint()
                self.want_z = self.rng.uniform(*HEIGHTS)
            want = math.atan2(self.goal[1] - y, self.goal[0] - x)
            self.speed = SPEED
        out = math.dist((x, y), self.centre) - self.field
        if out > 0:
            back = math.atan2(self.centre[1] - y, self.centre[0] - x)
            k = min(1.0, out / 10.0 + 0.5)
            want = math.atan2((1 - k) * math.sin(want) + k * math.sin(back), (1 - k) * math.cos(want) + k * math.cos(back))
        self.heading += max(-TURN * dt, min(TURN * dt, _wrap(want - self.heading)))
        vz = max(-CLIMB, min(CLIMB, (self.want_z - z) * 0.8))
        self.v = (self.speed * math.cos(self.heading), self.speed * math.sin(self.heading), vz)
        for i in range(3):
            self.p[i] += self.v[i] * dt


class FastBandit:
    """A fixed-wing drone at 50 m/s on a gentle weave (+-20 deg, 20 s), level: ASTOR's sizing target."""

    def __init__(self, speed=50.0, height=120.0, ahead=300.0, seed=1):
        self.speed, self.t0 = speed, random.Random(seed).uniform(0.0, 20.0)
        self.p = [ahead, 0.0, height]
        self.v = (speed, 0.0, 0.0)
        self.caught = None

    def step(self, dt, t, hunter=None):
        if self.caught is not None:
            self.p[2] = max(0.05, self.p[2] - 15.0 * dt)
            self.v = (0.0, 0.0, -15.0)
            return
        psi = math.radians(20.0) * math.sin(2 * math.pi * (t + self.t0) / 20.0)
        self.v = (self.speed * math.cos(psi), self.speed * math.sin(psi), 0.0)
        for i in range(3):
            self.p[i] += self.v[i] * dt


class Net:
    """A net cartridge after release (see the module's notes). ``step`` integrates it; ``catches(bandit)``."""

    def __init__(self, pos, vel, radius=1.0, mass=0.110, t_open=0.15, solidity=0.10, thickness=0.5):
        self.p, self.v = np.array(pos, float), np.array(vel, float)
        self.R, self.m, self.t_open, self.thick = radius, mass, t_open, thickness
        self.cda_packed = 0.0025
        self.cda_open = 1.2 * math.pi * radius ** 2 * solidity          # flat, a 10 % solid mesh, Cd 1.2 on the solid area
        self.t = 0.0
        self.on_ground = False

    @property
    def open(self):
        return self.t >= self.t_open

    def sink_rate(self):
        return math.sqrt(2 * self.m * am.G / (am.RHO * self.cda_open))

    def step(self, dt, wind=(0.0, 0.0, 0.0)):
        if self.on_ground:
            return
        rel = self.v - np.asarray(wind, float)
        s = float(np.linalg.norm(rel))
        cda = self.cda_open if self.open else self.cda_packed
        a = -0.5 * am.RHO * cda * s * rel / self.m + np.array([0.0, 0.0, -am.G])
        self.v = self.v + a * dt
        self.p = self.p + self.v * dt
        self.t += dt
        if self.p[2] <= 0.05:
            self.on_ground = True

    def catches(self, bandit, margin=0.85) -> bool:
        if not self.open or self.on_ground:
            return False
        d = np.asarray(bandit.p) - self.p
        return bool(math.hypot(d[0], d[1]) <= margin * self.R and abs(d[2]) <= self.thick)


class Camera:
    """The bandit as ASTOR's cameras give it: late and noisy; the velocity from the positions over 0.1 s."""

    def __init__(self, latency_s=0.10, noise_m=0.3, seed=0):
        self.tau, self.sig, self.rng, self.buf = latency_s, noise_m, np.random.default_rng(seed), []

    def see(self, t, pos, vel):
        self.buf.append((t, np.asarray(pos, float) + self.rng.normal(0.0, self.sig, 3)))
        while len(self.buf) > 2 and self.buf[1][0] <= t - self.tau - 0.1:
            self.buf.pop(0)
        old = [x for x in self.buf if x[0] <= t - self.tau]
        if len(old) < 2:
            return np.asarray(pos, float), np.asarray(vel, float)
        (t0, b0), (t1, b1) = old[0], old[-1]
        v = (b1 - b0) / max(t1 - t0, 1e-3)
        return b1 + v * self.tau, v                                 # the position brought forward by the latency


def attack(name="v", bandit_kind="armageddon", seed=0, net_radius=1.0, over_m=None, latency_s=0.10, noise_m=0.3,
           wind=(0.0, 0.0, 0.0), boost=False, duration_s=None, record=None) -> dict:
    """One sortie: ASTOR against one bandit with its nets. Returns what happened (see the keys)."""
    rng = np.random.default_rng(seed)
    fast = bandit_kind == "fast"
    sim = VF.SimV(name, bandit=True)
    Vs = VF.stall(VF.get(name, VF.SWEEPS[0]))
    if fast:
        bandit = FastBandit(seed=seed)
        V0, h0, over = 60.0, bandit.p[2], (3.0 if over_m is None else over_m)
        sim.start(V0, h0)
    else:
        bandit = ArmageddonBandit(centre=(250.0 + rng.uniform(-30, 30), rng.uniform(-40, 40)), seed=seed)
        V0, h0, over = 1.4 * Vs, 30.0, (1.8 if over_m is None else over_m)
        sim.start(V0, h0, heading_deg=float(rng.uniform(-15, 15)))
    T = duration_s or (60.0 if fast else 90.0)
    ap = amj.Autopilot(sim)
    cam = Camera(latency_s, noise_m, seed)
    net0 = Net((0, 0, 100), (0, 0, 0), radius=net_radius)
    t_fall = lambda dz: net0.t_open + max(dz, 0.2) / net0.sink_rate()
    nets, drops, phase, t_phase = [], [], "attack", 0.0
    closest, min_alt, caught = 1e9, 1e9, None
    sim.wind[:] = wind
    import mujoco
    bid = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_BODY, "bandit")
    mid = sim.m.body_mocapid[bid] if bid >= 0 else -1
    t0 = sim.d.time
    while sim.d.time - t0 < T:
        t = sim.d.time - t0
        s = sim.state()
        bandit.step(0.01, t, hunter=tuple(s["pos"]))
        if mid >= 0:
            sim.d.mocap_pos[mid] = bandit.p
        b, vb = cam.see(t, bandit.p, bandit.v)
        vbh = np.array([vb[0], vb[1], 0.0])
        sb = max(float(np.linalg.norm(vbh)), 0.5)
        ub = vbh / sb
        dz_rel = over
        lead = sb * t_fall(dz_rel) + 0.10 * s["V"]                       # the bandit's run while the net opens and sinks (+ the ejector's delay)
        aim = b + ub * lead + np.array([0.0, 0.0, over])
        r = aim - s["pos"]
        rng_h = math.hypot(r[0], r[1])
        V_c = max(1.3 * Vs, sb + min(18.0, 6.0 + 0.25 * rng_h)) if not fast else min(72.0, sb + min(20.0, 6.0 + 0.1 * rng_h))
        if boost and "boost" in sim.C:
            sim.boost = 1.0 if (fast or rng_h > 60.0) else 0.0
        if phase == "attack":
            phi, gam = amj.pn_command(sim, aim, vb)
            n_av = 0.5 * am.RHO * s["V"] ** 2 * sim.C["S"] * sim.C["CL_MAX"] / (sim.C["mass"] * am.G)
            phi_max = math.acos(min(1.0, 1.0 / max(0.85 * n_av, 1.0001)))
            low = s["pos"][2] + 2.0 * min(s["v_world"][2], 0.0) < FLOOR_M
            ap(phi_c=0.0 if low else max(-phi_max, min(phi_max, phi)), theta_c=math.radians(12.0) if low else gam, V_c=V_c)
            # release: ASTOR ahead of the bandit along its track by the lead, inside the net's radius across it, above it
            rel = s["pos"] - np.asarray(b)
            along, across = float(rel @ ub), float(rel[0] * -ub[1] + rel[1] * ub[0])
            if sim.nets > 0 and abs(along - lead) < 0.6 * net_radius + 0.02 * s["V"] and abs(across) < 0.7 * net_radius and 0.5 < rel[2] < over + 2.5:
                sim.drop_net()
                v_rel = s["v_world"] - 5.0 * (s["v_world"] / max(float(np.linalg.norm(s["v_world"])), 1e-6))     # the ejector pushes it aft
                nets.append(Net(s["pos"] - 0.5 * s["v_world"] / max(s["V"], 1.0), v_rel, radius=net_radius))
                drops.append({"t": round(t, 2), "along_m": round(along, 2), "across_m": round(across, 2), "above_m": round(float(rel[2]), 2),
                              "astor_speed_m_s": round(s["V"], 1), "bandit_speed_m_s": round(sb, 1)})
                phase, t_phase = "away", t
        else:                                                           # climb away, then come again
            ap(phi_c=math.radians(35.0), theta_c=math.radians(8.0), V_c=max(1.4 * Vs, sb + 8.0))
            if t - t_phase > (4.0 if not fast else 2.0):
                phase = "attack" if sim.nets > 0 else "home"
            if phase == "home":
                ap(phi_c=0.0, theta_c=math.radians(6.0), V_c=1.5 * Vs)
        sim.step(5)
        for nt in nets:
            for _ in range(5):
                nt.step(amj.DT, wind)
            if caught is None and nt.catches(bandit):
                caught = {"t": round(t, 2), "net": nets.index(nt) + 1, "height_m": round(bandit.p[2], 1)}
                bandit.caught = t
        s = sim.state()
        closest = min(closest, math.dist(s["pos"], bandit.p))
        min_alt = min(min_alt, float(s["pos"][2]))
        if record is not None:
            record(sim, bandit, nets, t)
        if s["pos"][2] < 1.0 or (caught is not None and t - caught["t"] > 2.0):
            break
        if sim.nets == 0 and all(n.on_ground or n.t > 8.0 for n in nets) and caught is None:
            break
    sim.wind[:] = 0.0
    return {"design": name, "bandit": bandit_kind, "seed": seed, "caught": caught is not None, "catch": caught, "drops": drops,
            "nets_used": len(drops), "closest_approach_m": round(closest, 2), "min_altitude_m": round(min_alt, 1),
            "crashed": bool(min_alt < 1.0), "duration_s": round(sim.d.time - t0, 1), "energy_Wh": round(sim.energy_J / 3600.0, 2),
            "net": {"radius_m": net_radius, "sink_m_s": round(net0.sink_rate(), 2), "t_open_s": net0.t_open}}


def campaign(name="v", bandit_kind="armageddon", n=20, seed0=0, **kw) -> dict:
    """``n`` sorties with different bandits and starts: the capture rate per sortie and per net, and why not."""
    runs = [attack(name, bandit_kind, seed=seed0 + k, **kw) for k in range(n)]
    caught = [r for r in runs if r["caught"]]
    nets = sum(r["nets_used"] for r in runs)
    return {"design": name, "bandit": bandit_kind, "sorties": n, "capture_rate_per_sortie": round(len(caught) / n, 3),
            "captures_per_net": round(len(caught) / max(nets, 1), 3), "nets_dropped": nets,
            "caught_with_first_net": sum(1 for r in caught if r["catch"]["net"] == 1),
            "no_drop_sorties": sum(1 for r in runs if r["nets_used"] == 0), "crashes": sum(1 for r in runs if r["crashed"]),
            "median_time_to_catch_s": float(np.median([r["catch"]["t"] for r in caught])) if caught else None,
            "median_closest_approach_m": float(np.median([r["closest_approach_m"] for r in runs])), "settings": kw, "runs": runs}
