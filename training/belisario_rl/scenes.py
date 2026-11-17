"""The scenes a training session is made of: each a terrain, Chiron props (geometry the robot can touch, with
passive joints where something swings or hangs), where the robot starts, and a task - plain locomotion, or a goal
to reach through or over the scene. A session (``session.py``) lists scenes with weights and parameters; the
vectorised environments are shared out among them.

Scenes (``SCENES``), with their parameters and defaults:

* **flat** / **rough** / **steps** - locomotion on the terrains of ``terrain.py`` (``rms``, ``correlation_length``,
  ``step_height``, ``step_spacing``).
* **door** - the doorway of ``kerchak_tasks.door_prop`` (a hinged door with a sprung lever, ``width``, ``mass_kg``,
  ``damping``); the goal is 2.5 m beyond it.
* **window** - a wall with a window opening (``sill``, ``width``, ``height``, ``wall_thickness``); the goal beyond.
* **tree** - a trunk (``radius``, ``height``) with branches every ``branch_spacing`` from ``first_branch`` up,
  alternating sides (``branch_length``, ``branch_radius``); the goal is the pelvis at ``goal_height`` up the trunk.
* **hull** - a ship's side: a plate sloping at ``slope_deg`` up to a rail at ``height``, the deck beyond; the goal
  on the deck.
* **obstacles** - bars across the path (``count``, ``height``, ``spacing``, ``bar_radius``) to jump or clamber over;
  the goal past the last.
* **rope** - a rope hanging from a beam (``length``, ``links``, ``radius``, ``top_height``), its links on
  two-axis hinges; the goal is the pelvis at ``goal_height``.
* **wall** - the climbing wall of ``kerchak_scenarios`` (rungs; ``x``, ``rung_spacing``, ``first_rung``, ``cycles``).

``wall``, ``tree``, ``hull`` and ``rope`` take their props from ``kerchak_scenarios`` (the scenery the pre-programmed
climbs were solved on), so a session can train on exactly the scenes the references cover.

A scene's props are built once per environment; the terrain and the props go into the MJCF through Chiron's
``Robot.to_mjcf(terrain, props=...)``. Nothing here invents physical parameters of the robot: the scenes are scenery.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field

import numpy as np
from vegeta.chiron import Flat, Geom, Joint, Link, Prop

from .robot_model import DESIGNS
from .terrain import make_terrain

__all__ = ["SceneSpec", "SCENES", "build_scene", "scene_names"]

GREY, WOOD, BARK, STEEL, ROPE = (0.45, 0.45, 0.47, 1.0), (0.55, 0.42, 0.28, 1.0), (0.4, 0.3, 0.2, 1.0), (0.5, 0.52, 0.56, 1.0), (0.75, 0.65, 0.45, 1.0)


@dataclass
class SceneSpec:
    """A built scene: what goes into the model and what the episode is about."""
    name: str
    terrain: object
    props: tuple = ()
    start: tuple = (0.0, 0.0, 0.0)       #: the pelvis's x, y and yaw at reset (the height comes from the pose)
    goal: tuple | None = None            #: (x, y, z) the pelvis should reach, or None for plain locomotion
    goal_radius: float = 0.4
    task: str = "locomotion"             #: locomotion | reach | climb
    extent: tuple = (-2.0, 8.0, -2.0, 2.0)
    params: dict = field(default_factory=dict)
    notes: str = ""


def _box(name, half, pos, rgba=GREY, mass=50.0, quat=(1.0, 0.0, 0.0, 0.0)):
    return Geom(name, "box", tuple(half), pos=tuple(pos), quat=tuple(quat), mass=mass, role="link", rgba=rgba)


def _cyl(name, r, a, b, rgba=BARK, mass=20.0):
    return Geom(name, "capsule", (r,), fromto=tuple(a) + tuple(b), mass=mass, role="link", rgba=rgba)


# ---- the scenes
def scene_flat(cfg, p):
    return SceneSpec("flat", Flat(), (), task="locomotion", params=p)


def scene_rough(cfg, p):
    t = cfg.terrain
    t = type(t)(**{**t.__dict__, "kind": "rough", **{k: v for k, v in p.items() if k in t.__dict__}})
    return SceneSpec("rough", make_terrain(t, seed=p.get("seed", t.seed)), (), task="locomotion", extent=tuple(t.extent), params=p)


def scene_steps(cfg, p):
    t = cfg.terrain
    t = type(t)(**{**t.__dict__, "kind": "steps", **{k: v for k, v in p.items() if k in t.__dict__}})
    return SceneSpec("steps", make_terrain(t, seed=p.get("seed", t.seed)), (), task="locomotion", extent=tuple(t.extent), params=p)


def scene_door(cfg, p):
    if str(DESIGNS) not in sys.path:
        sys.path.insert(0, str(DESIGNS))
    import kerchak_tasks as gt
    for k in ("width", "mass_kg", "damping", "open_rad"):
        if k in p:
            gt.DOOR[k] = p[k]
    x = gt.DOOR["x"]
    return SceneSpec("door", Flat(), (gt.door_prop(),), start=(0.0, 0.0, 0.0), goal=(x + 2.5, 0.0, 0.62), task="reach",
                     extent=(-2.0, x + 5.0, -2.5, 2.5), params=p, notes="the lever turns and the door swings under contacts")


def scene_window(cfg, p):
    x, sill, w, h, t = p.get("x", 3.0), p.get("sill", 0.9), p.get("width", 1.1), p.get("height", 1.2), p.get("wall_thickness", 0.25)
    wall_h, wall_w = 3.0, 3.0
    frame = Link("window_wall", pos=(x, 0.0, 0.0),
                 geoms=[_box("wall_below", (t / 2, wall_w, sill / 2), (0.0, 0.0, sill / 2), mass=200.0),
                        _box("wall_above", (t / 2, wall_w, (wall_h - sill - h) / 2), (0.0, 0.0, sill + h + (wall_h - sill - h) / 2), mass=200.0),
                        _box("wall_left", (t / 2, (wall_w - w / 2) / 2, h / 2), (0.0, w / 2 + (wall_w - w / 2) / 2, sill + h / 2), mass=200.0),
                        _box("wall_right", (t / 2, (wall_w - w / 2) / 2, h / 2), (0.0, -w / 2 - (wall_w - w / 2) / 2, sill + h / 2), mass=200.0),
                        _box("sill_plate", (t / 2 + 0.04, w / 2, 0.03), (0.0, 0.0, sill + 0.03), rgba=WOOD, mass=20.0)])
    return SceneSpec("window", Flat(), (Prop(frame, free=False),), goal=(x + 2.5, 0.0, 0.62), task="reach",
                     extent=(-2.0, x + 5.0, -2.5, 2.5), params=p, notes="pass through the opening: sill at %.2f m" % sill)


def scene_tree_branches(cfg, p):
    x, r, h = p.get("x", 2.5), p.get("radius", 0.3), p.get("height", 6.0)
    first, spacing, bl, br = p.get("first_branch", 1.4), p.get("branch_spacing", 0.9), p.get("branch_length", 1.2), p.get("branch_radius", 0.07)
    goal_h = p.get("goal_height", 3.0)
    geoms = [_cyl("trunk", r, (0.0, 0.0, 0.0), (0.0, 0.0, h), rgba=BARK, mass=800.0)]
    z, k = first, 0
    while z < h - 0.5:
        side = 1.0 if k % 2 == 0 else -1.0
        geoms.append(_cyl(f"branch_{k}", br, (0.0, side * r * 0.8, z), (0.0, side * (r + bl), z + 0.15), rgba=BARK, mass=15.0))
        geoms.append(_cyl(f"branch_{k}_fore", br * 0.8, (r * 0.8, 0.0, z + 0.45), (r + bl * 0.8, 0.0, z + 0.55), rgba=BARK, mass=10.0))
        z += spacing
        k += 1
    tree = Link("tree", pos=(x, 0.0, 0.0), geoms=geoms)
    return SceneSpec("tree", Flat(), (Prop(tree, free=False),), goal=(x - r - 0.5, 0.0, goal_h), goal_radius=0.5, task="climb",
                     extent=(-2.0, x + 3.0, -2.5, 2.5), params=p, notes="climb the trunk by its branches to %.1f m" % goal_h)


def scene_hull_plate(cfg, p):
    x, slope, height, rail = p.get("x", 2.5), math.radians(p.get("slope_deg", 60.0)), p.get("height", 2.2), p.get("rail_radius", 0.05)
    length = height / math.sin(slope)
    cx, cz = x + 0.5 * length * math.cos(slope), 0.5 * height
    a = -(math.pi / 2 - slope)                                             # the plate tilted up from the ground about y
    quat = (math.cos(a / 2), 0.0, math.sin(a / 2), 0.0)
    hull = Link("hull", pos=(0.0, 0.0, 0.0),
                geoms=[_box("hull_plate", (length / 2, 2.5, 0.06), (cx, 0.0, cz), rgba=STEEL, mass=2000.0, quat=quat),
                       _cyl("hull_rail", rail, (x + length * math.cos(slope), -2.5, height + 0.08), (x + length * math.cos(slope), 2.5, height + 0.08), rgba=STEEL, mass=50.0),
                       _box("deck", (1.5, 2.5, 0.06), (x + length * math.cos(slope) + 1.5, 0.0, height - 0.06), rgba=STEEL, mass=2000.0)])
    top_x = x + length * math.cos(slope)
    return SceneSpec("hull", Flat(), (Prop(hull, free=False),), goal=(top_x + 1.5, 0.0, height + 0.62), goal_radius=0.5, task="climb",
                     extent=(-2.0, top_x + 4.0, -3.0, 3.0), params=p, notes="over the side: %.0f° plate to a rail at %.1f m" % (p.get("slope_deg", 60.0), height))


def scene_obstacles(cfg, p):
    n, h, spacing, br, x0 = int(p.get("count", 3)), p.get("height", 0.35), p.get("spacing", 2.0), p.get("bar_radius", 0.05), p.get("x", 2.5)
    geoms = []
    for k in range(n):
        x = x0 + k * spacing
        geoms.append(_cyl(f"bar_{k}", br, (x, -1.5, h), (x, 1.5, h), rgba=STEEL, mass=20.0))
        for sy in (-1.5, 1.5):
            geoms.append(_box(f"post_{k}_{'l' if sy > 0 else 'r'}", (0.04, 0.04, h / 2), (x, sy, h / 2), rgba=STEEL, mass=10.0))
    bars = Link("obstacles", pos=(0.0, 0.0, 0.0), geoms=geoms)
    last = x0 + (n - 1) * spacing
    return SceneSpec("obstacles", Flat(), (Prop(bars, free=False),), goal=(last + 2.0, 0.0, 0.62), task="reach",
                     extent=(-2.0, last + 4.0, -2.5, 2.5), params=p, notes="%d bars at %.2f m to jump or clamber over" % (n, h))


def scene_rope_links(cfg, p):
    x, length, links, r, top = p.get("x", 1.8), p.get("length", 4.0), int(p.get("links", 12)), p.get("radius", 0.025), p.get("top_height", 5.0)
    seg = length / links
    beam = Link("rope_beam", pos=(x, 0.0, top), geoms=[_cyl("beam", 0.06, (0.0, -1.5, 0.0), (0.0, 1.5, 0.0), rgba=STEEL, mass=100.0)])
    parent = beam
    for k in range(links):
        link = Link(f"rope_{k}", pos=(0.0, 0.0, 0.0 if k == 0 else -seg),
                    joints=[Joint(f"rope_{k}_x", axis=(1, 0, 0), range=(-1.2, 1.2), damping=0.3, stiffness=0.5),
                            Joint(f"rope_{k}_y", axis=(0, 1, 0), range=(-1.2, 1.2), damping=0.3, stiffness=0.5)],
                    geoms=[_cyl(f"rope_seg_{k}", r, (0.0, 0.0, 0.0), (0.0, 0.0, -seg), rgba=ROPE, mass=0.8)])
        parent.children.append(link)
        parent = link
    goal_h = p.get("goal_height", 2.5)
    return SceneSpec("rope", Flat(), (Prop(beam, free=False, solref=(-2000.0, -40.0)),), goal=(x, 0.0, goal_h), goal_radius=0.5, task="climb",
                     extent=(-2.0, x + 3.0, -2.5, 2.5), params=p, notes="a %d-link rope from %.1f m; climb to %.1f m" % (links, top, goal_h))


def _scenario_scene(name, task, goal_fn, extent_fn, note):
    """A scene built from ``kerchak_scenarios``' props (the same scenery the pre-programmed sequences climb)."""
    def build(cfg, p):
        if str(DESIGNS) not in sys.path:
            sys.path.insert(0, str(DESIGNS))
        import kerchak_scenarios as gs
        params = {**gs.DEFAULTS[name], **{k: v for k, v in p.items() if k in gs.DEFAULTS[name]}}
        props = gs.props(name, params)
        return SceneSpec(name, Flat(), tuple(props), goal=goal_fn(params), goal_radius=0.5, task=task, extent=extent_fn(params), params=params,
                         notes=note(params))
    return build


scene_wall = _scenario_scene("wall", "climb", lambda p: (p["x"] - 0.5, 0.0, p["first_rung"] + 3 * p["rung_spacing"] + p["cycles"] * p["rung_spacing"]),
                             lambda p: (-2.0, p["x"] + 2.0, -2.5, 2.5), lambda p: "the climbing wall: %d rungs %.2f m apart" % (int((p["height"] - 0.2 - p["first_rung"]) / p["rung_spacing"]) + 1, p["rung_spacing"]))
scene_tree = _scenario_scene("tree", "climb", lambda p: (p["x"] - p["radius"] - 0.45, 0.0, p["first_branch"] + p["cycles"] * p["branch_spacing"] + 0.55),
                             lambda p: (-2.0, p["x"] + 3.0, -2.5, 2.5), lambda p: "the trunk with branches every %.2f m" % p["branch_spacing"])
scene_hull = _scenario_scene("hull", "climb", lambda p: (p["x"] + p["height"] / math.tan(math.radians(p["slope_deg"])) + 1.5, 0.0, p["height"] + 0.62),
                             lambda p: (-2.0, p["x"] + 6.0, -3.0, 3.0), lambda p: "the ship's side: %.0f° to a rail at %.1f m, ribs every %.1f m" % (p["slope_deg"], p["height"], p["rib_spacing"]))
scene_rope = _scenario_scene("rope", "climb", lambda p: (p["x"] - 0.45, 0.0, 0.35 + p["cycles"] * 0.3 + 0.55),
                             lambda p: (-2.0, p["x"] + 3.0, -2.5, 2.5), lambda p: "a %d-link rope from %.1f m" % (int(p["links"]), p["top_height"]))

SCENES = {"flat": scene_flat, "rough": scene_rough, "steps": scene_steps, "door": scene_door, "window": scene_window, "wall": scene_wall,
          "tree": scene_tree, "hull": scene_hull, "obstacles": scene_obstacles, "rope": scene_rope}
# the scenes the pre-programmed sequences climb (wall, tree, hull, rope) are kerchak_scenarios' props, so a policy
# trains on exactly the scenery the references were made for; the local builders above stay for custom variants


def scene_names() -> list:
    return list(SCENES)


def build_scene(name: str, cfg, params: dict | None = None) -> SceneSpec:
    if name not in SCENES:
        raise KeyError(f"scene {name!r}: one of {', '.join(SCENES)}")
    return SCENES[name](cfg, dict(params or {}))
