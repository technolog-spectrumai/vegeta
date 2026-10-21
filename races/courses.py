"""The four courses, parameterised, as Python objects and as MJCF scenes.

A ``Course`` is a list of ``Obstacle`` s (collision geometry: boxes, cylinders, spheres), an ordered list of ``Zone`` s
(the key boxes the racer must pass through: start, gates, checkpoints, finish), a starting grid, landmarks, and
the notes on recovery. ``to_mjcf`` writes the scene with the reference quad in it; ``build_all`` saves the four
scene files under ``races/scenes/``. Dimensions are metres; z up; the ground is a plane at z = 0 (the quarry floor is
the ground, its benches rise around it).

Every route is flyable by the reference quad at the schedule ``races.path`` gives it, with at least a metre of
clearance to the nearest obstacle at every zone (``clearance_check``), and every zone is reachable on foot or by a
ladder for recovery (floors by the staircase, benches by their ramps, roofs not used).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import refquad

SCENES = Path(__file__).resolve().parent / "scenes"


@dataclass
class Obstacle:
    name: str
    kind: str                        # box | cylinder | sphere
    pos: tuple
    size: tuple                      # box: half sizes; cylinder: (r, half height); sphere: (r,)
    rgba: tuple = (0.6, 0.6, 0.62, 1.0)
    quat: tuple = (1.0, 0.0, 0.0, 0.0)
    landmark: bool = False

    def half_extents(self) -> np.ndarray:
        if self.kind == "box":
            return np.array(self.size)
        if self.kind == "cylinder":
            return np.array([self.size[0], self.size[0], self.size[1]])
        return np.array([self.size[0]] * 3)

    def distance(self, p: np.ndarray) -> float:
        """Distance from ``p`` to the obstacle's surface (a turned box is measured in its own frame)."""
        rel = np.asarray(p, dtype=float) - np.asarray(self.pos)
        if self.kind == "box" and tuple(self.quat) != (1.0, 0.0, 0.0, 0.0):
            w, x, y, z = self.quat
            R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            rel = R.T @ rel
        d = np.abs(rel) - self.half_extents()
        if self.kind == "sphere":
            return float(np.linalg.norm(np.asarray(p) - np.asarray(self.pos)) - self.size[0])
        if self.kind == "cylinder":
            r_xy = math.hypot(p[0] - self.pos[0], p[1] - self.pos[1]) - self.size[0]
            dz = abs(p[2] - self.pos[2]) - self.size[1]
            return float(math.hypot(max(r_xy, 0.0), max(dz, 0.0)) + min(max(r_xy, dz), 0.0))
        return float(np.linalg.norm(np.maximum(d, 0.0)) + min(d.max(), 0.0))


@dataclass
class Zone:
    name: str
    center: tuple
    half: tuple = (1.5, 1.5, 1.5)   # the box the racer must enter
    kind: str = "gate"              # start | gate | checkpoint | finish
    radius: float = 2.0             # the corner radius the opening allows (for the schedule)

    def contains(self, p) -> bool:
        return bool(np.all(np.abs(np.asarray(p) - np.asarray(self.center)) <= np.asarray(self.half)))


@dataclass
class Course:
    name: str
    slug: str
    kind: str                        # technical | open
    extent: tuple                    # x0, x1, y0, y1
    obstacles: list
    zones: list
    start_grid: list                 # ground positions of the starting pads
    landmarks: list = field(default_factory=list)
    description: str = ""
    recovery: str = ""
    params: dict = field(default_factory=dict)

    @property
    def points(self) -> np.ndarray:
        return np.array([z.center for z in self.zones], dtype=float)

    @property
    def radii(self) -> list:
        return [z.radius for z in self.zones]

    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())

    def nearest_obstacle(self, p) -> tuple:
        best, name = 1e9, ""
        for o in self.obstacles:
            d = o.distance(np.asarray(p, dtype=float))
            if d < best:
                best, name = d, o.name
        return best, name

    def clearance_check(self, margin: float = 1.0):
        """Every zone centre at least ``margin`` from any obstacle and above the ground."""
        import pandas as pd
        rows = []
        for z in self.zones:
            d, n = self.nearest_obstacle(z.center)
            need = min(margin, max(0.45, 0.7 * min(z.half)))            # a door or a window zone sits inside its opening: a smaller margin
            rows.append({"zone": z.name, "kind": z.kind, "nearest obstacle [m]": min(d, z.center[2]), "which": n if d < z.center[2] else "the ground", "needed [m]": need, "ok": min(d, z.center[2]) >= need})
        return pd.DataFrame(rows).set_index("zone")


def _box(name, c, h, rgba=(0.6, 0.6, 0.62, 1.0), landmark=False, quat=(1.0, 0.0, 0.0, 0.0)):
    return Obstacle(name, "box", tuple(c), tuple(h), rgba, quat, landmark)


# ================================================================================================= 1. the hangar
def hangar(length=40.0, width=18.0, height=12.0, floor_gap=4.0, gate_half=0.9) -> Course:
    """A hangar with three floors 4 m apart: the ground floor, two mezzanines with a stair opening at the west end and
    a floor hole at the east end, a straight staircase, open windows in the long walls, roof beams, a door in the
    west gable. Technical: the route climbs the stair opening, crosses each floor, drops through the hole, and leaves
    by a window."""
    L, Wd, H = length, width, height
    wall = 0.15
    grey, slab, beamc, stair = (0.62, 0.62, 0.66, 1.0), (0.5, 0.5, 0.55, 1.0), (0.45, 0.3, 0.2, 1.0), (0.55, 0.55, 0.6, 1.0)
    ob = []
    # long walls: a low band, a high band, and piers between 1.6 m windows at 1.6–3.2 m, every 8 m
    for side, y in (("s", -Wd / 2), ("n", Wd / 2)):
        ob.append(_box(f"wall_{side}_low", (0, y, 0.8), (L / 2, wall / 2, 0.8), grey))
        ob.append(_box(f"wall_{side}_high", (0, y, (H + 3.2) / 2), (L / 2, wall / 2, (H - 3.2) / 2), grey))
        xs = np.arange(-L / 2, L / 2 + 1e-6, 8.0)
        for i in range(len(xs) - 1):
            x0, x1 = xs[i], xs[i + 1]
            ob.append(_box(f"wall_{side}_pier{i}", ((x0 + x1 - 1.6) / 2, y, 2.4), ((x1 - x0 - 1.6) / 2, wall / 2, 0.8), grey))      # the window is the last 1.6 m of the bay
    # the west gable with a 4 × 3.2 m door in the middle, the east gable solid
    ob.append(_box("gable_w_s", (-L / 2, -(Wd / 4 + 1.0), H / 2), (wall / 2, Wd / 4 - 1.0, H / 2), grey))
    ob.append(_box("gable_w_n", (-L / 2, (Wd / 4 + 1.0), H / 2), (wall / 2, Wd / 4 - 1.0, H / 2), grey))
    ob.append(_box("gable_w_top", (-L / 2, 0.0, (H + 3.2) / 2), (wall / 2, 2.0, (H - 3.2) / 2), grey))
    ob.append(_box("gable_e", (L / 2, 0.0, H / 2), (wall / 2, Wd / 2, H / 2), grey))
    # two mezzanine floors: slabs leaving a 6 × 4 m stair opening at the south-west and a 4 × 4 m hole at the east end
    t = 0.12
    for k, z in enumerate((floor_gap, 2 * floor_gap)):
        ob.append(_box(f"floor{k}_main", (0.0, 2.0, z), (L / 2 - 6.0, Wd / 2 - 2.0, t / 2), slab))                      # x −14…14, y −5…9
        ob.append(_box(f"floor{k}_south", (-2.0, -Wd / 2 + 2.0, z), (L / 2 - 8.0, 2.0, t / 2), slab))                 # x −14…10, y −9…−5
        ob.append(_box(f"floor{k}_west_n", (-L / 2 + 3.0, 2.5, z), (3.0, Wd / 2 - 2.5, t / 2), slab))                # west end: y −4…9 (the stair opening is y −9…−4, x −20…−14)
        ob.append(_box(f"floor{k}_east_s", (L / 2 - 3.0, -Wd / 2 + 2.0, z), (3.0, 2.0, t / 2), slab))               # east end: y −9…−5
        ob.append(_box(f"floor{k}_east_n", (L / 2 - 3.0, 4.5, z), (3.0, 4.5, t / 2), slab))                          # east end: y 0…9 (the hole is y −5…0 → 4 m wide... x 14…20)
        ob.append(_box(f"floor{k}_east_strip", (L / 2 - 1.0 - 0.0, -2.5, z), (1.0, 2.5, t / 2), slab))               # the hole narrows to x 14…18
    # the staircase: a straight flight in the stair opening from the ground to the first floor, then on up
    for k in range(2):
        for i in range(8):
            ob.append(_box(f"stair{k}_{i}", (-L / 2 + 0.6 + 0.7 * i, -Wd / 2 + 1.0, floor_gap * k + 0.25 + 0.5 * i), (0.35, 0.9, 0.25), stair))
    # roof beams every 5 m, 0.6 m deep, and the roof
    for i, x in enumerate(np.arange(-L / 2 + 5.0, L / 2 - 1.0, 5.0)):
        ob.append(_box(f"beam{i}", (x, 0.0, H - 0.3), (0.15, Wd / 2, 0.3), beamc))
    ob.append(_box("roof", (0.0, 0.0, H + 0.1), (L / 2 + 0.3, Wd / 2 + 0.3, 0.1), (0.5, 0.5, 0.5, 0.25)))
    # landmarks: a red tank by the door, a yellow crane gantry on the ground floor
    ob.append(Obstacle("red_tank", "cylinder", (-L / 2 - 4.0, -8.0, 1.5), (1.0, 1.5), (0.8, 0.15, 0.1, 1.0), landmark=True))
    ob.append(_box("gantry_beam", (6.0, 0.0, 3.4), (0.2, Wd / 2 - 1.0, 0.2), (0.95, 0.8, 0.1, 1.0), landmark=True))
    ob.append(_box("gantry_leg_s", (6.0, -Wd / 2 + 1.0, 1.7), (0.15, 0.15, 1.7), (0.95, 0.8, 0.1, 1.0), landmark=True))
    ob.append(_box("gantry_leg_n", (6.0, Wd / 2 - 1.0, 1.7), (0.15, 0.15, 1.7), (0.95, 0.8, 0.1, 1.0), landmark=True))
    g = gate_half
    zones = [Zone("start", (-L / 2 - 10.0, 0.0, 1.5), (2.0, 2.0, 1.0), "start", 3.0),
             Zone("door", (-L / 2 - 0.0, 0.0, 1.6), (0.6, 1.6, 1.2), "gate", 1.5),
             Zone("ground floor", (-L / 2 + 8.0, 1.0, 2.0), (1.5, 1.5, 1.2), "gate", 2.5),
             Zone("under the gantry", (6.0, 0.0, 1.8), (1.0, 2.0, 1.0), "checkpoint", 2.0),
             Zone("back west", (-8.0, -6.5, 2.0), (1.5, 1.5, 1.2), "gate", 2.0),
             Zone("under the stair opening", (-L / 2 + 3.0, -Wd / 2 + 3.5, 2.0), (1.0, 1.0, 0.9), "gate", 1.0),
             Zone("stair opening up", (-L / 2 + 3.0, -Wd / 2 + 3.5, floor_gap + 1.6), (1.0, 1.0, 0.9), "gate", 1.0),
             Zone("first floor", (0.0, 2.0, floor_gap + 1.8), (1.5, 1.5, 1.0), "checkpoint", 2.5),
             Zone("under the east hole", (L / 2 - 4.0, -2.5, floor_gap + 1.8), (1.2, 1.2, 0.9), "gate", 1.0),
             Zone("east hole up", (L / 2 - 4.0, -2.5, 2 * floor_gap + 1.6), (1.2, 1.2, 0.9), "gate", 1.0),
             Zone("under the beams", (4.0, 2.0, 2 * floor_gap + 2.2), (1.5, 1.5, 1.0), "gate", 2.5),
             Zone("second floor west", (-10.0, 3.0, 2 * floor_gap + 2.0), (1.5, 1.5, 1.0), "checkpoint", 2.5),
             Zone("over the stair opening", (-L / 2 + 3.0, -Wd / 2 + 3.5, 2 * floor_gap + 1.6), (1.0, 1.0, 0.9), "gate", 1.0),
             Zone("stair opening down", (-L / 2 + 3.0, -Wd / 2 + 3.5, floor_gap + 1.6), (1.0, 1.0, 0.9), "gate", 1.0),
             Zone("ground again", (-L / 2 + 3.0, -Wd / 2 + 3.5, 1.8), (1.0, 1.0, 0.9), "gate", 1.0),
             Zone("to the window", (-L / 2 + 7.2, -Wd / 2 + 3.5, 2.4), (1.5, 1.5, 1.0), "gate", 1.8),        # square to the window (a diagonal approach clips its pier)
             Zone("window", (-L / 2 + 8.0 - 0.8, -Wd / 2, 2.4), (0.7, 0.6, 0.7), "gate", 0.6),
             Zone("finish", (-L / 2 + 7.2, -Wd / 2 - 8.0, 2.0), (2.0, 2.0, 1.5), "finish", 3.0)]
    zones[0].half = (2.0, 2.0, 0.75)
    grid = [(-L / 2 - 12.0 - 1.5 * i, 2.5 - 1.7 * j, 0.0) for i in range(2) for j in range(4)][:4]
    return Course("Hangar — three floors", "hangar", "technical", (-L / 2 - 16.0, L / 2 + 6.0, -Wd / 2 - 12.0, Wd / 2 + 6.0), ob, zones, grid,
                  [("red tank by the door", (-L / 2 - 4.0, -8.0, 1.5)), ("yellow gantry", (6.0, 0.0, 3.4))],
                  f"a {L:.0f} × {Wd:.0f} × {H:.0f} m hangar with two mezzanine floors {floor_gap:.0f} m apart: in by the west door, under the gantry, up the stair opening, "
                  "across the first floor, up through the east hole, under the roof beams, down both stair openings, out by a south window; 18 zones, about 165 m",
                  "every floor is reached by the staircase; the windows open at 1.6 m; a downed aircraft on a floor or on a beam is reachable from the mezzanine with a pole",
                  dict(length=length, width=width, height=height, floor_gap=floor_gap, gate_half=gate_half))


# ================================================================================================= 2. the industrial zone
def industrial(scale=1.0, gate_half=1.5) -> Course:
    """Outside the hangar: seven large buildings (20–35 m, 10–16 m tall) with 10–15 m gaps and two connecting
    passages (covered walkways between buildings, 4 m high, to fly under), three stacks as landmarks. Technical
    outdoors: the route threads the gaps, under both passages, round the far end and back."""
    s = scale
    ob = []
    conc, dark = (0.55, 0.5, 0.48, 1.0), (0.4, 0.4, 0.42, 1.0)
    buildings = [((0, -25, 7), (15, 8, 7)), ((45, -28, 6), (12, 10, 6)), ((90, -22, 8), (16, 7, 8)), ((10, 20, 5.5), (17, 9, 5.5)), ((55, 18, 7.5), (14, 8, 7.5)), ((100, 20, 6.5), (12, 10, 6.5)), ((140, 0, 8), (10, 20, 8))]
    for i, (c, h) in enumerate(buildings):
        ob.append(_box(f"building{i}", tuple(np.array(c) * (s, s, 1)), tuple(np.array(h) * (s, s, 1)), conc))
    # two covered passages between building pairs: a slab at 4–5 m on two piers, the route goes under
    for i, (x, y0, y1) in enumerate(((30.0, -17.0, 11.0), (75.0, -15.0, 10.0))):
        ob.append(_box(f"passage{i}_deck", (x * s, (y0 + y1) / 2 * s, 4.6), (1.5, (y1 - y0) / 2 * s, 0.4), (0.7, 0.68, 0.6, 1.0)))
        for j, y in enumerate(np.linspace(y0, y1, 4)[1:-1]):
            ob.append(_box(f"passage{i}_pier{j}", (x * s, y * s, 2.1), (0.3, 0.3, 2.1), conc))
    for i, (c, h, col) in enumerate([((38, -9, 2.5), (1.0, 1.0, 2.5), dark), ((66, 6, 3.0), (1.2, 1.2, 3.0), dark), ((120, -6, 6.0), (0.8, 0.8, 6.0), (0.8, 0.2, 0.15, 1.0))]):
        ob.append(_box(f"stack{i}", tuple(np.array(c) * (s, s, 1)), h, col, landmark=True))
    ob.append(Obstacle("water_tower", "cylinder", (130 * s, 36 * s, 10.0), (3.0, 2.0), (0.3, 0.5, 0.8, 1.0), landmark=True))
    ob.append(Obstacle("water_tower_leg", "cylinder", (130 * s, 36 * s, 4.0), (0.5, 4.0), dark))
    g = gate_half
    pts = [("start", (-30, 0, 1.5), "start", 4.0), ("gap 1", (-5, -3, 6), "gate", 5.0), ("under passage 1", (30, -3, 2.6), "gate", 1.8), ("between", (45, -10, 6), "checkpoint", 4.0),
           ("gap 2", (60, -6, 6), "gate", 5.0), ("under passage 2", (75, -2.5, 2.6), "gate", 1.8), ("gap 3", (95, 0, 6), "checkpoint", 5.0), ("east of the plant", (117, 2, 6), "gate", 4.0), ("the turn", (121, 22, 6), "gate", 4.0),
           ("water tower", (144, 32, 8), "gate", 5.0), ("round the last building", (170, 8, 8), "checkpoint", 6.0), ("back", (160, -30, 8), "gate", 6.0), ("south of the last building", (125, -30, 8), "gate", 6.0), ("over the red stack", (120, -6, 15), "gate", 4.0),
           ("gap 2 again", (70, -2, 8), "checkpoint", 4.0), ("gap 1 again", (30, 8, 8), "gate", 4.0), ("home straight", (-10, 2, 4), "gate", 5.0), ("finish", (-30, 0, 4.5), "finish", 5.0)]
    zones = [Zone(n, (c[0] * s, c[1] * s, c[2]), (g, g, g) if k != "start" else (2.5, 2.5, 0.75), k, r) for n, c, k, r in pts]
    for z in zones:
        if "passage" in z.name:
            z.half = (1.0, 2.5, 0.9)
    grid = [(-34 * s - 1.5 * i, 2.5 - 1.7 * j, 0.0) for i in range(1) for j in range(4)]
    return Course("Industrial zone", "industrial", "technical", (-45 * s, 185 * s, -45 * s, 45 * s), ob, zones, grid,
                  [("red stack", (120 * s, -6 * s, 6.0)), ("water tower", (130 * s, 36 * s, 10.0))],
                  "seven buildings with 10–15 m gaps and two covered passages to fly under, a 470 m lap at 6–8 m round the far end and back",
                  "all on the ground outside; the passages' decks are 4.2 m up (a ladder); roofs are not part of the route", dict(scale=scale, gate_half=gate_half))


# ================================================================================================= 3. the farm
def farm(gate_half=1.5) -> Course:
    """Open: a cottage with a pitched roof and a chimney, a barn open at both ends with a row of posts down its
    middle, a line of trees along the field edge, a copse, a hedge; fast stretches over the fields between."""
    ob = []
    grey, wood, trunk, crown = (0.8, 0.75, 0.65, 1.0), (0.55, 0.35, 0.2, 1.0), (0.35, 0.25, 0.15, 1.0), (0.2, 0.45, 0.2, 1.0)
    ob += [_box("cottage", (0, 15, 2.0), (5, 4, 2.0), grey), _box("cottage_roof", (0, 15, 4.8), (5.5, 4.5, 0.8), (0.6, 0.3, 0.25, 1.0)), _box("chimney", (3, 15, 6.3), (0.4, 0.4, 0.7), grey, landmark=True)]
    ob += [_box("barn_wall_s", (40, -5, 3.0), (10, 0.15, 3.0), wood), _box("barn_wall_n", (40, 5, 3.0), (10, 0.15, 3.0), wood), _box("barn_roof", (40, 0, 6.3), (10.5, 5.5, 0.3), (0.5, 0.25, 0.2, 1.0))]
    for i, x in enumerate(np.arange(32.0, 49.0, 4.0)):
        ob.append(_box(f"barn_post{i}", (x, 0.0, 3.0), (0.15, 0.15, 3.0), wood))
    ob.append(Obstacle("silo", "cylinder", (55, 12, 6.0), (2.5, 6.0), (0.75, 0.75, 0.78, 1.0), landmark=True))
    k = 0
    for x in np.arange(-10.0, 70.0, 8.0):
        ob.append(Obstacle(f"trunk{k}", "cylinder", (x, -20.0, 2.5), (0.25, 2.5), trunk)); ob.append(Obstacle(f"crown{k}", "sphere", (x, -20.0, 7.0), (2.5,), crown)); k += 1
    for (x, y) in ((82, 0), (88, 6), (80, 8), (92, -4), (96, 3)):
        ob.append(Obstacle(f"trunk{k}", "cylinder", (x, y, 3.0), (0.3, 3.0), trunk)); ob.append(Obstacle(f"crown{k}", "sphere", (x, y, 8.0), (4.0,), crown)); k += 1
    ob.append(_box("hedge", (120, -10, 1.0), (0.6, 30.0, 1.0), (0.25, 0.4, 0.2, 1.0)))
    g = gate_half
    pts = [("start", (-20, 0, 1.5), "start", 4.0), ("round the cottage", (-11, 15, 3), "gate", 3.0), ("behind it", (0, 25, 3), "gate", 3.0), ("and back", (11, 15, 3), "checkpoint", 3.0),
           ("to the barn", (20, 2.5, 3), "gate", 3.0), ("barn door", (29, 2.5, 3), "gate", 1.6), ("barn far door", (51, 2.5, 3), "checkpoint", 1.6), ("round the silo", (62, 12, 5), "gate", 3.0),
           ("the long field", (110, 20, 8), "gate", 8.0), ("over the hedge", (120, -10, 4), "checkpoint", 4.0), ("back over the field", (95, -25, 8), "gate", 8.0), ("down to the trees", (74, -22, 3.0), "gate", 4.0),
           ("tree 1", (62, -16, 3.0), "gate", 2.0), ("tree 2", (54, -24, 3.0), "gate", 2.0), ("tree 3", (46, -16, 3.0), "gate", 2.0), ("tree 4", (38, -24, 3.0), "checkpoint", 2.0),
           ("tree 5", (30, -16, 3.0), "gate", 2.0), ("tree 6", (22, -24, 3.0), "gate", 2.0), ("the yard", (0, -8, 4), "gate", 4.0), ("finish", (-20, 0, 4.0), "finish", 4.0)]
    zones = [Zone(n, c, (g, g, g) if k not in ("start",) else (2.5, 2.5, 0.75), k, r) for n, c, k, r in pts]
    for z in zones:
        if "door" in z.name:
            z.half = (1.0, 1.8, 1.2)
        if z.kind == "finish":
            z.half = (2.5, 2.5, 1.0)
    grid = [(-24.0, 2.5 - 1.7 * j, 0.0) for j in range(4)]
    return Course("Farm", "farm", "open", (-30, 135, -40, 35), ob, zones, grid, [("chimney", (3, 15, 6.3)), ("silo", (55, 12, 6.0))],
                  "round the cottage, through the barn beside its posts, round the silo, a 60 m straight over the long field, over the hedge, back over the field, a slalom down the tree line, home over the yard: about 520 m",
                  "open fields; the barn floor and the tree line are on foot; the roofs are not on the route", dict(gate_half=gate_half))


# ================================================================================================= 4. the quarry
def quarry(floor_r=45.0, bench_h=6.0, bench_w=8.0, n_benches=3, gate_half=2.0) -> Course:
    """An excavated pit: a broad floor, ``n_benches`` terraced benches rising around it (each ``bench_h`` tall,
    ``bench_w`` wide, an octagon of blocks), a haul ramp spiralling up the benches, spoil heaps on the floor that make
    two winding corridors, a crusher and a conveyor as landmarks. Open and fast on the floor and along the benches,
    technical in the corridors."""
    ob = []
    rock, spoil, steel = (0.62, 0.55, 0.45, 1.0), (0.5, 0.45, 0.38, 1.0), (0.35, 0.35, 0.4, 1.0)
    for b in range(n_benches):
        r_in = floor_r + b * bench_w
        z = (b + 1) * bench_h / 2
        for i in range(16):
            a = 2 * math.pi * (i + 0.5) / 16
            if b == 0 and i in (12, 13):               # the haul ramp's cut through the first bench on the south-west
                continue
            c = ((r_in + bench_w / 2) * math.cos(a), (r_in + bench_w / 2) * math.sin(a), z)
            seg = 2 * math.pi * (r_in + bench_w / 2) / 16 / 2 + 0.6
            ob.append(_box(f"bench{b}_{i}", c, (bench_w / 2, seg, z), rock, quat=(math.cos(a / 2), 0, 0, math.sin(a / 2))))
    # the haul ramp: a sloped run of blocks up the south-west cut
    for i in range(10):
        a = math.radians(225 + 3 * i)
        r = floor_r - 2.0 + i * 1.2
        ob.append(_box(f"ramp{i}", (r * math.cos(a), r * math.sin(a), 0.3 * (i + 1) / 2), (4.0, 3.0, 0.3 * (i + 1) / 2), spoil, quat=(math.cos(a / 2), 0, 0, math.sin(a / 2))))
    # spoil heaps on the floor: two rows making corridors 8 m wide
    heaps = [(-25, -2), (-10, -2), (5, -2), (20, -2), (-20, 12), (-5, 12), (10, 12), (25, 12), (-25, -14), (-10, -14), (5, -14), (20, -14)]
    for i, (x, y) in enumerate(heaps):
        ob.append(Obstacle(f"heap{i}", "sphere", (x, y, 0.0), (3.0,), spoil))
    ob.append(_box("crusher", (0, 28, 4.0), (4.0, 3.0, 4.0), steel, landmark=True))
    ob.append(_box("conveyor", (20, 28, 5.0), (16.0, 0.6, 0.3), steel, landmark=True, quat=(math.cos(math.radians(8)), 0, math.sin(math.radians(8)), 0)))
    ob.append(_box("conveyor_leg", (30, 28, 3.0), (0.3, 0.3, 3.0), steel))
    ob.append(Obstacle("excavator", "box", (-30, 25, 2.0), (3.0, 1.8, 2.0), (0.9, 0.6, 0.1, 1.0), landmark=True))
    g = gate_half
    R1 = floor_r + bench_w / 2
    pts = [("start", (-35, -20, 1.5), "start", 4.0), ("corridor in", (-32, 5, 2.5), "gate", 2.5), ("corridor 1a", (-20, 7.5, 2.5), "gate", 2.0), ("corridor 1b", (-10, 2.5, 2.5), "gate", 2.0),
           ("corridor 1c", (0, 7.5, 2.5), "gate", 2.0), ("corridor out", (12, 3, 2.5), "checkpoint", 2.0), ("floor east", (38, 0, 4), "gate", 6.0), ("up to bench 1", (R1 * math.cos(math.radians(20)), R1 * math.sin(math.radians(20)), bench_h + 3), "gate", 5.0),
           ("bench 1 north", (R1 * math.cos(math.radians(80)), R1 * math.sin(math.radians(80)), bench_h + 3), "checkpoint", 6.0),
           ("bench 2 north-west", ((R1 + bench_w) * math.cos(math.radians(130)), (R1 + bench_w) * math.sin(math.radians(130)), 2 * bench_h + 3), "gate", 6.0),
           ("rim west", ((R1 + 2 * bench_w) * math.cos(math.radians(175)), (R1 + 2 * bench_w) * math.sin(math.radians(175)), 3 * bench_h + 3), "gate", 6.0),
           ("down the ramp", (floor_r * math.cos(math.radians(228)), floor_r * math.sin(math.radians(228)), bench_h + 2), "checkpoint", 4.0),
           ("floor south", (0, -30, 4), "gate", 6.0), ("past the crusher", (8, 22, 3), "gate", 3.0), ("under the conveyor", (20, 28, 2.4), "gate", 2.0), ("floor east again", (36, 10, 4), "gate", 6.0),
           ("corridor 2 in", (25, -8, 2.5), "gate", 2.0), ("corridor 2a", (12, -10, 2.5), "gate", 2.0), ("corridor 2b", (-2, -6, 2.5), "checkpoint", 2.0), ("corridor 2c", (-17, -10, 2.5), "gate", 2.0),
           ("finish", (-35, -20, 4.5), "finish", 4.0)]
    zones = [Zone(n, c, (g, g, g) if k != "start" else (2.5, 2.5, 0.75), k, r) for n, c, k, r in pts]
    for z in zones:
        if z.name.startswith("corridor"):
            z.half = (1.5, 1.5, 1.2)
    grid = [(-40.0, -22.0 + 1.7 * j, 0.0) for j in range(4)]
    R = floor_r + n_benches * bench_w + 10
    return Course("Quarry", "quarry", "open", (-R, R, -R, R), ob, zones, grid, [("crusher", (0, 28, 4.0)), ("excavator", (-30, 25, 2.0))],
                  f"a pit with a {2 * floor_r:.0f} m floor and {n_benches} benches {bench_h:.0f} m tall: two winding corridors between the spoil heaps, the floor, up the benches to the rim, down the haul ramp, past the crusher and under the conveyor: about 600 m",
                  "the benches are reached by the haul ramp; the heaps are walkable; nothing on the route is above a bench's top", dict(floor_r=floor_r, bench_h=bench_h, bench_w=bench_w, n_benches=n_benches, gate_half=gate_half))


COURSES = {"hangar": hangar, "industrial": industrial, "farm": farm, "quarry": quarry}


def build(slug: str, **params) -> Course:
    return COURSES[slug](**params)


# ================================================================================================= MJCF
def to_mjcf(course: Course, *, quad_start=None, with_quad: bool = True, timestep: float = 0.002) -> str:
    """The scene: a ground plane, the obstacles as colliding geoms, the zones as transparent non-colliding boxes,
    the starting pads, an overview camera, and the reference quad with its onboard and chase cameras."""
    x0, x1, y0, y1 = course.extent
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    span = max(x1 - x0, y1 - y0)
    start = quad_start or (course.start_grid[0][0], course.start_grid[0][1], 0.06)
    out = [f'<mujoco model="race {course.slug}">', f'  <option timestep="{timestep}" integrator="implicitfast" gravity="0 0 -9.81" density="0" viscosity="0"/>',
           '  <visual><headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/><map znear="0.05" zfar="800"/><quality shadowsize="4096"/></visual>',
           '  <asset><texture name="grid" type="2d" builtin="checker" rgb1="0.75 0.78 0.7" rgb2="0.65 0.68 0.6" width="512" height="512"/>',
           '    <material name="ground" texture="grid" texrepeat="40 40" reflectance="0.05"/></asset>',
           '  <default><geom condim="3" friction="0.8 0.005 0.0001" solref="0.01 1"/></default>',
           '  <worldbody>', f'    <geom name="ground" type="plane" size="{span} {span} 0.1" material="ground"/>',
           f'    <light pos="{cx} {cy} {span}" dir="0 0 -1" directional="true"/>',
           f'    <camera name="overview" pos="{cx} {cy} {span * 1.1}" xyaxes="1 0 0 0 1 0" fovy="60"/>',
           f'    <camera name="grandstand" pos="{x0} {cy} {span * 0.25}" xyaxes="0 -1 0 0.35 0 0.94" fovy="70"/>']
    for o in course.obstacles:
        q = " ".join(f"{v:.5f}" for v in o.quat)
        size = " ".join(f"{v:.4f}" for v in o.size)
        pos = " ".join(f"{v:.4f}" for v in o.pos)
        rgba = " ".join(f"{v:.3f}" for v in o.rgba)
        out.append(f'    <geom name="{o.name}" type="{o.kind}" size="{size}" pos="{pos}" quat="{q}" rgba="{rgba}"/>')
    colours = {"start": "0.2 0.8 0.2", "gate": "0.2 0.5 0.95", "checkpoint": "0.95 0.6 0.1", "finish": "0.9 0.2 0.2"}
    for i, z in enumerate(course.zones):
        pos = " ".join(f"{v:.3f}" for v in z.center)
        half = " ".join(f"{v:.3f}" for v in z.half)
        out.append(f'    <geom name="zone_{i:02d}_{z.kind}" type="box" size="{half}" pos="{pos}" rgba="{colours[z.kind]} 0.25" contype="0" conaffinity="0"/>')
        out.append(f'    <site name="zone_{i:02d}" pos="{pos}" size="0.1" rgba="{colours[z.kind]} 1"/>')
    for i, g in enumerate(course.start_grid):
        out.append(f'    <geom name="pad{i}" type="box" size="0.4 0.4 0.01" pos="{g[0]:.3f} {g[1]:.3f} 0.01" rgba="0.9 0.9 0.9 1" contype="0" conaffinity="0"/>')
    if with_quad:
        out.append("    " + refquad.mjcf_body("quad", start).replace("\n", "\n    "))
    out += ['  </worldbody>', '</mujoco>']
    return "\n".join(out)


def build_all(out_dir: Path = SCENES, **kw) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for slug, fn in COURSES.items():
        c = fn()
        p = out_dir / f"{slug}.xml"
        p.write_text(to_mjcf(c, **kw))
        paths[slug] = p
    return paths
