"""Simulated birds: species presets, soaring and wandering behaviour, and the flight from the drone.

Species values are order-of-magnitude literature figures (assumptions): wingspans from field guides (Svensson et al.,
Collins Bird Guide), cruise and escape speeds after Pennycuick (1987, 2001) and Bruderer & Boldt (2001), the hard-turn
load factor of manoeuvring birds ~2-2.5 g. Fear distances (the "flight initiation distance" for an approaching drone)
vary widely with species and habituation; the values here are mid-range assumptions after Vas et al. 2015 and
Mulero-Pázmány et al. 2017 (birds tolerate a small drone to 4-30 m, raptors and corvids react earlier).

Behaviours: 'thermal' circles a centre that drifts with the wind (radius and sense per bird), climbing slowly in the
lift and gliding down when above the top; 'wander' flies between random points of an area at its height band (a
loose flock when several birds share a group: they follow the same points with offsets and keep apart). Within
``fear_m`` of the drone a bird escapes at its escape speed for ``fear_hold_s`` (4 s) after the drone was last that close:
away from the drone and sideways off its path (prey veers across the threat's line rather than running ahead of it),
diving or climbing; the direction follows the drone as it moves.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["SPECIES", "Species", "BirdGroup", "BirdField", "SimTruth"]

G = 9.81


@dataclass(frozen=True)
class Species:
    name: str
    wingspan_m: float
    cruise: float
    escape: float
    n_turn: float
    fear_m: float
    colour: tuple = (40, 40, 40)


SPECIES = {
    "black-headed gull": Species("black-headed gull", 1.0, 10.0, 14.0, 2.0, 18.0, (235, 235, 235)),
    "herring gull": Species("herring gull", 1.4, 11.0, 15.0, 2.0, 22.0, (220, 220, 225)),
    "carrion crow": Species("carrion crow", 0.95, 12.0, 16.0, 2.0, 28.0, (25, 25, 30)),
    "feral pigeon": Species("feral pigeon", 0.65, 17.0, 22.0, 2.5, 22.0, (120, 125, 140)),
    "common buzzard": Species("common buzzard", 1.2, 9.0, 14.0, 1.8, 35.0, (110, 80, 50)),
}


@dataclass
class BirdGroup:
    species: str
    n: int
    centre: tuple                    # (x, y) world
    behaviour: str = "thermal"       # 'thermal' or 'wander'
    height: tuple = (50.0, 90.0)     # band [m]
    radius: float = 35.0             # thermal radius, or the wander area's radius
    spread: float = 20.0             # initial scatter


@dataclass(frozen=True)
class SimTruth:
    t: float
    ids: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    size: np.ndarray
    species: tuple
    afraid: np.ndarray


class BirdField:
    def __init__(self, groups: list, wind=(0.0, 0.0, 0.0), seed: int = 0):
        self.groups = groups
        self.wind = np.asarray(wind, float)
        self.rng = np.random.default_rng(seed)
        rows = []
        for gi, g in enumerate(groups):
            sp = SPECIES[g.species]
            for _ in range(g.n):
                rows.append((gi, sp))
        self.n = len(rows)
        self.group = np.array([r[0] for r in rows], int)
        self.sp = [r[1] for r in rows]
        self.ids = np.arange(1, self.n + 1)
        self.size = np.array([s.wingspan_m * self.rng.uniform(0.92, 1.08) for s in self.sp])
        self.cruise = np.array([s.cruise for s in self.sp])
        self.escape = np.array([s.escape for s in self.sp])
        self.a_max = np.array([G * math.sqrt(s.n_turn ** 2 - 1) for s in self.sp])
        self.fear = np.array([s.fear_m for s in self.sp])
        self.pos = np.zeros((self.n, 3))
        self.vel = np.zeros((self.n, 3))
        self.sense = self.rng.choice([-1.0, 1.0], self.n)
        self.radius = np.zeros(self.n)
        self.centre = np.zeros((self.n, 2))
        self.goal = np.zeros((self.n, 3))
        self.climb = np.zeros(self.n, bool)
        self.afraid_until = np.full(self.n, -1.0)
        self.flee_dir = np.zeros((self.n, 3))
        self.dive = np.zeros(self.n)
        self.t = 0.0
        for i in range(self.n):
            g = groups[self.group[i]]
            c = np.asarray(g.centre, float)
            self.centre[i] = c
            self.radius[i] = g.radius * self.rng.uniform(0.8, 1.2) if g.behaviour == "thermal" else g.radius
            a = self.rng.uniform(0, 2 * math.pi)
            if g.behaviour == "thermal":
                p = c + self.radius[i] * np.array([math.cos(a), math.sin(a)])
                hd = a + self.sense[i] * math.pi / 2
            else:
                p = c + self.rng.normal(0, g.spread, 2)
                hd = self.rng.uniform(0, 2 * math.pi)
            h = self.rng.uniform(*g.height)
            self.pos[i] = [p[0], p[1], h]
            self.vel[i] = self.cruise[i] * np.array([math.cos(hd), math.sin(hd), 0.0])
            self.climb[i] = self.rng.random() < 0.5
        for gi, g in enumerate(groups):
            if g.behaviour == "wander":
                self._new_goal(gi)

    def _new_goal(self, gi):
        g = self.groups[gi]
        a, r = self.rng.uniform(0, 2 * math.pi), g.radius * math.sqrt(self.rng.random())
        p = np.asarray(g.centre, float) + r * np.array([math.cos(a), math.sin(a)])
        h = self.rng.uniform(*g.height)
        for i in np.where(self.group == gi)[0]:
            self.goal[i] = [p[0] + self.rng.normal(0, 6), p[1] + self.rng.normal(0, 6), h + self.rng.normal(0, 3)]

    def step(self, dt, drone_pos=None, drone_vel=None):
        self.t += dt
        v_des = np.zeros_like(self.vel)
        for i in range(self.n):
            g = self.groups[self.group[i]]
            p = self.pos[i]
            if g.behaviour == "thermal":
                self.centre[i] += self.wind[:2] * dt
                d = p[:2] - self.centre[i]
                r = max(np.linalg.norm(d), 1e-3)
                radial = d / r
                tang = self.sense[i] * np.array([-radial[1], radial[0]])
                vh = self.cruise[i] * (tang - 0.8 * np.tanh((r - self.radius[i]) / 15.0) * radial)
                vh = vh / max(np.linalg.norm(vh), 1e-6) * self.cruise[i]
                if p[2] > g.height[1]:
                    self.climb[i] = False
                elif p[2] < g.height[0]:
                    self.climb[i] = True
                vz = 0.8 if self.climb[i] else -0.6
                v_des[i] = [vh[0] + self.wind[0], vh[1] + self.wind[1], vz]
            else:
                d = self.goal[i] - p
                if np.linalg.norm(d[:2]) < 15.0:
                    self._new_goal(self.group[i])
                    d = self.goal[i] - p
                u = d / max(np.linalg.norm(d), 1e-6)
                v_des[i] = self.cruise[i] * u + self.wind
                v_des[i, 2] = float(np.clip(0.3 * d[2], -2.0, 2.0))
        # keep apart (flocks)
        if self.n > 1:
            dd = self.pos[:, None, :] - self.pos[None]
            r = np.linalg.norm(dd, axis=-1) + np.eye(self.n) * 1e6
            push = np.where((r < 4.0)[..., None], dd / r[..., None] ** 2, 0).sum(1)
            v_des += 8.0 * push
        # the drone
        afraid = np.zeros(self.n, bool)
        if drone_pos is not None:
            dd = self.pos - np.asarray(drone_pos, float)
            rd = np.linalg.norm(dd, axis=1)
            vd = np.zeros(3) if drone_vel is None else np.asarray(drone_vel, float)
            sd = float(np.linalg.norm(vd[:2]))
            near = rd < self.fear
            for i in np.where(near & (self.afraid_until < self.t))[0]:
                self.dive[i] = self.rng.choice([-0.5, 0.35])               # dive or climb: chosen once per fright
            self.afraid_until[near] = self.t + 4.0
            afraid = self.afraid_until >= self.t
            for i in np.where(afraid)[0]:
                away = dd[i, :2] / max(np.linalg.norm(dd[i, :2]), 1e-3)
                if sd > 1.0:
                    ud = vd[:2] / sd
                    lateral = away - (away @ ud) * ud                         # off the drone's line
                    n = np.linalg.norm(lateral)
                    lateral = lateral / n if n > 1e-3 else np.array([-ud[1], ud[0]])
                    h = 0.4 * away + 0.9 * lateral
                else:
                    h = away
                h = h / max(np.linalg.norm(h), 1e-6)
                u = np.array([h[0], h[1], self.dive[i]])
                self.flee_dir[i] = u / np.linalg.norm(u)
            v_des[afraid] = self.flee_dir[afraid] * self.escape[afraid, None]
        dv = v_des - self.vel
        a = np.linalg.norm(dv, axis=1) / dt
        lim = np.where(afraid, self.a_max * 1.2, self.a_max)
        scale = np.minimum(1.0, lim / np.maximum(a, 1e-9))[:, None]
        self.vel = self.vel + dv * scale
        sp = np.linalg.norm(self.vel, axis=1)
        vmax = np.where(afraid, self.escape, self.cruise * 1.25 + np.linalg.norm(self.wind))
        self.vel *= (np.clip(sp, 0.5 * self.cruise, vmax) / np.maximum(sp, 1e-6))[:, None]
        self.pos = self.pos + self.vel * dt
        self.pos[:, 2] = np.maximum(self.pos[:, 2], 5.0)
        self._afraid = afraid
        return self.truth()

    def truth(self) -> SimTruth:
        return SimTruth(self.t, self.ids.copy(), self.pos.copy(), self.vel.copy(), self.size.copy(), tuple(s.name for s in self.sp),
                        getattr(self, "_afraid", np.zeros(self.n, bool)).copy())
