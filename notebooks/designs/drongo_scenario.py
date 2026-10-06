"""Drongo delivers a potato and a cup of cream to hungry people (notebook 08b, ``scenarios/drongo_delivery.py``).

The scene (``Scene``): a kitchen with Drongo's pad by the door, a supply zone beside it with a basket (an open cube)
holding the two items, a garden
hedge, and 36 m away a picnic with hungry people and a drop zone (a blanket). Two variants:

* **drop** — four of the people hold a 2 m × 2 m net at waist height over the zone. Drongo hovers over the net and
  lets the item fall; the net is **not simulated**: an item that reaches the net's plane inside its square is caught
  there (it then rests on it), *if* it fell at most ``NET['max_drop_m']`` (5 m) — from higher up it tears the net and
  falls on (``Watch``). The net's arrest is the assumed stretch (``drongo_robot.net_catch``);
* **place** — the people stand back; Drongo lands on the zone with the item so that the item stands on it, opens the
  jaws and leaves.

``Watch`` (a scene hook) judges every item from the physics alone: the pads' squeeze while it is held (MuJoCo's
contact forces) against its squeeze limit, a slip in the jaws, the moment the grip is lost (resting on the ground: put
down, inside the zone or not; in the air: falling, from what height), the catch in the net and the speed at which it
meets the ground. Those are the **deliveries**: the potato when it is caught / put down on the zone, the cream likewise;
``wait_times`` is how long the hungry people wait for each, counted from the order (t = 0, Drongo on its pad).

    import drongo_scenario as ds
    scene = ds.Scene("drop")
    lab = ds.make_lab(scene, log_geoms=True)
    ep = ds.run(lab, scene)
    ds.deliveries(ep), ds.wait_times(ep), ds.render_movie(ep, scene, "drop.mp4")
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vegeta import chiron as ch
from vegeta.chiron.robot import TERRAIN_NAME

import drongo_controller as dc
import drongo_robot as dr
from onager_controller import smooth

__all__ = ["Scene", "Watch", "item_props", "scenery", "make_lab", "run", "timeseries", "phase_table", "deliveries",
           "wait_times", "outcome", "time_budget", "activity", "ACTIVITIES", "sweep", "render_movie", "stills", "VARIANTS"]

VARIANTS = ("drop", "place")


@dataclass
class Scene:
    """Where things are (world, m; the ground is flat at z = 0). ``variant`` 'drop' or 'place'; ``home`` Drongo's pad
    by the kitchen door; the supply ``basket`` (an open cube of ``basket_size`` with walls and floor ``basket_wall``
    thick, centred at ``basket_at``) holding the items at ``potato_at`` / ``cream_at`` on its floor — Drongo lands in it
    over each (its rotors span 0.30 m: the cube leaves room for one item beside it); ``zone`` the drop zone's centre
    (a ``zone_size`` square blanket) with the net held ``NET['height_m']`` above it; ``cruise_alt`` above the ground
    (over the 2.5 m hedge at ``hedge_x`` and the people)."""

    variant: str = "drop"
    home: tuple = (0.0, 0.0)
    basket_at: tuple = (1.5, 0.0)
    basket_size: float = 0.7
    basket_wall: float = 0.015
    potato_at: tuple = (1.5, 0.13)
    cream_at: tuple = (1.5, -0.13)
    zone: tuple = (36.0, 6.0)
    zone_size: float = 2.4
    net_height: float = dr.NET["height_m"]
    net_size: float = dr.NET["size_m"]
    net_max_drop: float = dr.NET["max_drop_m"]
    cruise_alt: float = 8.0
    hedge_x: float = 18.0
    hedge_height: float = 2.5
    spot_offset: float = 0.45
    take_after_s: float = 3.0
    robot_geometry: dict = field(default_factory=dr.geometry)

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")

    def ground(self, x, y) -> float:
        """What Drongo or an item stands on at (x, y): the basket's floor inside it, the flat garden elsewhere."""
        h = self.basket_size / 2 - self.basket_wall
        bx, by = self.basket_at
        return self.basket_wall if abs(x - bx) < h and abs(y - by) < h else 0.0

    def item_at(self, name: str) -> tuple:
        return {"potato": self.potato_at, "cream": self.cream_at}[name]

    def spot(self, name: str) -> tuple:
        """Where on the zone Drongo puts ``name`` down (place) — side by side, ``spot_offset`` either side of the
        centre — or drops it (drop: the net's centre)."""
        zx, zy = self.zone
        if self.variant == "drop":
            return zx, zy
        return zx, zy + {"potato": 1, "cream": -1}[name] * self.spot_offset

    @property
    def table(self) -> tuple:
        """The picnic table's top centre (x, y, z): the hungry people put what they took there."""
        return self.zone[0] + 3.5, self.zone[1] + 3.0, 0.76

    def plate(self, name: str) -> tuple:
        x, y, z = self.table
        return {"potato": (x - 0.4, y + 0.15, z + 0.012), "cream": (x + 0.4, y + 0.15, z + 0.012)}[name]

    def in_zone(self, x, y, margin: float = 0.0) -> bool:
        h = self.zone_size / 2 - margin
        return abs(x - self.zone[0]) <= h and abs(y - self.zone[1]) <= h

    def in_net(self, x, y) -> bool:
        h = self.net_size / 2
        return abs(x - self.zone[0]) <= h and abs(y - self.zone[1]) <= h

    @property
    def distance_m(self) -> float:
        """From the kitchen pad to the drop zone, as the crow flies."""
        return math.hypot(self.zone[0] - self.home[0], self.zone[1] - self.home[1])


# ----------------------------------------------------------------------------------------------- props
def item_props(scene: Scene, items=("potato", "cream")) -> list:
    """The potato (a sphere) and the cream (an upright cylinder) as free props on the ground at their places; their
    friction governs their contacts with the pads and the ground (``priority`` 2)."""
    out = []
    for name in items:
        it = dr.ITEMS[name]
        x, y = scene.item_at(name)
        fr = (it["mu"], it["torsional_m"], max(it["rolling_m"], 1e-5))
        if it["shape"] == "sphere":
            r = it["diameter_m"] / 2
            geom = ch.Geom(f"{name}_g", "sphere", (r,), mass=it["mass_kg"], friction=fr, rgba=it["rgba"])
            z = scene.ground(x, y) + r
        else:
            geom = ch.Geom(f"{name}_g", "cylinder", (it["diameter_m"] / 2, it["height_m"] / 2), mass=it["mass_kg"],
                           friction=fr, rgba=it["rgba"])
            z = scene.ground(x, y) + it["height_m"] / 2
        geoms = [geom]
        if name == "cream":                                    # the foil lid and the label, drawn
            geoms.append(ch.Geom("cream_lid", "cylinder", (it["diameter_m"] / 2 - 0.001, 0.0008),
                                 pos=(0, 0, it["height_m"] / 2), role="visual", rgba=(0.85, 0.1, 0.12, 1.0)))
            geoms.append(ch.Geom("cream_label", "cylinder", (it["diameter_m"] / 2 + 0.0005, it["height_m"] / 5),
                                 pos=(0, 0, -it["height_m"] / 8), role="visual", rgba=(0.15, 0.35, 0.75, 1.0)))
        out.append(ch.Prop(ch.Link(name, pos=(x, y, z + 0.0005), geoms=geoms), free=True,
                           condim=6 if it["shape"] == "sphere" else 4))
    return out


def _box(name, centre, half, rgba):
    return ch.Geom(name, "box", tuple(half), pos=tuple(centre), role="visual", rgba=rgba)


def _basket(scene: Scene) -> list:
    """The supply basket as an open cube: a floor and four walls (they collide; drawn translucent)."""
    a, t = scene.basket_size / 2, scene.basket_wall
    rgba = (0.72, 0.55, 0.32, 0.55)
    g = [ch.Geom("basket_floor", "box", (a, a, t / 2), pos=(0, 0, t / 2), rgba=rgba)]
    for k, (cx, cy, hx, hy) in enumerate(((a - t / 2, 0, t / 2, a), (-a + t / 2, 0, t / 2, a),
                                          (0, a - t / 2, a, t / 2), (0, -a + t / 2, a, t / 2))):
        g.append(ch.Geom(f"basket_wall{k}", "box", (hx, hy, a), pos=(cx, cy, a), rgba=rgba))
    return g


def _person(name, x, y, facing, shirt, holding=None):
    """A standing person (1.75 m) as capsules and a sphere; ``holding`` (x, y, z) a point the arms reach to."""
    c, s = math.cos(facing), math.sin(facing)
    skin, legs = (0.93, 0.76, 0.62, 1.0), (0.20, 0.25, 0.45, 1.0)
    g = [ch.Geom(f"{name}_legs", "capsule", (0.07,), fromto=(0, 0, 0.07, 0, 0, 0.85), role="visual", rgba=legs),
         ch.Geom(f"{name}_body", "capsule", (0.15,), fromto=(0, 0, 0.95, 0, 0, 1.38), role="visual", rgba=shirt),
         ch.Geom(f"{name}_head", "sphere", (0.11,), pos=(0, 0, 1.66), role="visual", rgba=skin)]
    for side in (1, -1):
        sh = (-s * side * 0.19, c * side * 0.19, 1.42)
        if holding is not None:
            hx, hy, hz = holding[0] - x, holding[1] - y, holding[2]
            hand = (hx - s * side * 0.12, hy + c * side * 0.12, hz)
        else:
            hand = (sh[0] + 0.02 * c, sh[1] + 0.02 * s, 0.82)
        g.append(ch.Geom(f"{name}_arm{side}", "capsule", (0.04,), fromto=(*sh, *hand), role="visual", rgba=shirt))
    return ch.Prop(ch.Link(name, pos=(x, y, 0.0), geoms=g), log=False)


def scenery(scene: Scene) -> list:
    """What is drawn and does not collide: the kitchen, the pad, the hedge, a tree, the picnic, the zone and the people
    — holding the net (drop) or standing back (place)."""
    hx, hy = scene.home
    zx, zy = scene.zone
    wall, roof = (0.93, 0.89, 0.80, 1.0), (0.62, 0.25, 0.18, 1.0)
    out = [ch.Prop(ch.Link("kitchen", pos=(hx - 4.5, hy, 0.0), geoms=[
        _box("kitchen_walls", (0, 0, 1.5), (2.5, 4.0, 1.5), wall),
        _box("kitchen_roof", (0, 0, 3.25), (2.8, 4.3, 0.25), roof),
        _box("kitchen_door", (2.51, 0, 1.0), (0.01, 0.5, 1.0), (0.35, 0.22, 0.12, 1.0)),
        _box("kitchen_window", (2.51, 2.2, 1.6), (0.01, 0.6, 0.45), (0.55, 0.75, 0.9, 1.0))]), log=False),
        ch.Prop(ch.Link("pad", pos=(hx, hy, 0.0), geoms=[
            _box("pad_mat", (0, 0, 0.002), (0.35, 0.35, 0.002), (0.25, 0.25, 0.28, 1.0)),
            _box("pad_h1", (0, 0.12, 0.0045), (0.18, 0.025, 0.0006), (1, 1, 1, 1)),
            _box("pad_h2", (0, -0.12, 0.0045), (0.18, 0.025, 0.0006), (1, 1, 1, 1)),
            _box("pad_h3", (0, 0, 0.0045), (0.025, 0.12, 0.0006), (1, 1, 1, 1))]), log=False),
        ch.Prop(ch.Link("basket", pos=(scene.basket_at[0], scene.basket_at[1], 0.0), geoms=_basket(scene)), log=False),
        ch.Prop(ch.Link("hedge", pos=(scene.hedge_x, (hy + zy) / 2, 0.0), geoms=[
            _box("hedge_g", (0, 0, scene.hedge_height / 2), (0.5, 12.0, scene.hedge_height / 2), (0.18, 0.42, 0.16, 1.0))]),
            log=False),
        ch.Prop(ch.Link("tree", pos=(26.0, -3.5, 0.0), geoms=[
            ch.Geom("tree_trunk", "cylinder", (0.18, 1.6), pos=(0, 0, 1.6), role="visual", rgba=(0.4, 0.27, 0.15, 1.0)),
            ch.Geom("tree_crown", "sphere", (2.0,), pos=(0, 0, 4.2), role="visual", rgba=(0.16, 0.48, 0.18, 1.0))]),
            log=False),
        ch.Prop(ch.Link("zone", pos=(zx, zy, 0.0), geoms=[
            _box("zone_blanket", (0, 0, 0.002), (scene.zone_size / 2, scene.zone_size / 2, 0.002), (0.80, 0.15, 0.15, 1.0)),
            _box("zone_x1", (0, 0, 0.0045), (scene.zone_size / 2, 0.06, 0.0006), (1, 1, 1, 1)),
            _box("zone_x2", (0, 0, 0.0045), (0.06, scene.zone_size / 2, 0.0006), (1, 1, 1, 1))]), log=False),
        ch.Prop(ch.Link("picnic", pos=(scene.table[0], scene.table[1], 0.0), geoms=[
            _box("table_top", (0, 0, 0.74), (0.9, 0.45, 0.02), (0.55, 0.38, 0.22, 1.0)),
            _box("table_leg1", (0.8, 0, 0.36), (0.04, 0.4, 0.36), (0.45, 0.30, 0.18, 1.0)),
            _box("table_leg2", (-0.8, 0, 0.36), (0.04, 0.4, 0.36), (0.45, 0.30, 0.18, 1.0)),
            ch.Geom("plate1", "cylinder", (0.12, 0.006), pos=(-0.4, 0.15, 0.766), role="visual", rgba=(1, 1, 1, 1)),
            ch.Geom("plate2", "cylinder", (0.12, 0.006), pos=(0.4, 0.15, 0.766), role="visual", rgba=(1, 1, 1, 1)),
            ch.Geom("plate3", "cylinder", (0.12, 0.006), pos=(0.0, -0.18, 0.766), role="visual", rgba=(1, 1, 1, 1))]),
            log=False)]
    shirts = [(0.85, 0.2, 0.2, 1.0), (0.2, 0.55, 0.85, 1.0), (0.95, 0.75, 0.1, 1.0), (0.3, 0.7, 0.35, 1.0)]
    h = scene.net_size / 2
    if scene.variant == "drop":
        net_z = scene.ground(zx, zy) + scene.net_height
        rope = (0.95, 0.95, 0.95, 1.0)
        geoms = [_box("net_mesh", (0, 0, 0), (h, h, 0.004), (0.92, 0.92, 0.95, 0.45))]
        for k, (a, b) in enumerate((((-h, -h), (h, -h)), ((h, -h), (h, h)), ((h, h), (-h, h)), ((-h, h), (-h, -h)))):
            geoms.append(ch.Geom(f"net_rope{k}", "capsule", (0.012,), fromto=(a[0], a[1], 0, b[0], b[1], 0), role="visual",
                                 rgba=rope))
        for k in range(1, 6):                                   # the mesh, drawn
            u = -h + k * 2 * h / 6
            geoms.append(ch.Geom(f"net_u{k}", "capsule", (0.004,), fromto=(u, -h, 0, u, h, 0), role="visual", rgba=rope))
            geoms.append(ch.Geom(f"net_v{k}", "capsule", (0.004,), fromto=(-h, u, 0, h, u, 0), role="visual", rgba=rope))
        out.append(ch.Prop(ch.Link("net", pos=(zx, zy, net_z), geoms=geoms), log=False))
        for k, (sx, sy) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
            px, py = zx + sx * (h + 0.35), zy + sy * (h + 0.35)
            out.append(_person(f"person{k}", px, py, math.atan2(zy - py, zx - px), shirts[k],
                               holding=(zx + sx * h, zy + sy * h, net_z)))
    else:
        for k, ang in enumerate((0.35, 1.4, 2.45, 3.5)):          # clear of the camera's side (south)
            px, py = zx + 2.6 * math.cos(ang), zy + 2.6 * math.sin(ang)
            out.append(_person(f"person{k}", px, py, math.atan2(zy - py, zx - px), shirts[k]))
    return out


# ----------------------------------------------------------------------------------------------- the judge
class Watch:
    """Scene hook: what happens to each item (see the module). ``records[item]`` holds its story: ``picked_up`` (both
    pads squeeze it), ``peak_squeeze_N``, ``slip_mm`` (how far it moved in the jaws while squeezed), ``released`` (the
    grip is lost) and how (``put_down`` on the ground — ``in_zone`` — or ``falling`` from ``release_height_m``),
    ``caught`` (the net), ``impact_m_s`` (meeting the ground), ``delivered`` (caught in the net, or put down on the zone),
    ``taken`` (the hungry people carry it to the table ``take_after_s`` after it arrived, once Drongo is 2.5 m
    away; it is then held on its plate) and ``damage`` (what spoilt it). ``history`` keeps [t, item index, squeeze
    front, squeeze rear, ground force] every ``log_every`` calls while an item is held."""

    def __init__(self, scene: Scene, items=("potato", "cream"), log_every: int = 10):
        self.scene, self.items, self.log_every = scene, tuple(items), log_every

    def reset(self, lab):
        import mujoco

        m = lab.model
        self._mj = mujoco
        self.buf = np.zeros(6)
        gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
        self.pads = [gid(p) for p in dr.PADS]
        self.terrain = gid(TERRAIN_NAME)
        root = lab._root_body
        self.robot_geoms = {g for g in range(m.ngeom) if int(m.body_rootid[m.geom_bodyid[g]]) == root}
        self.item_geom = {gid(f"{n}_g"): n for n in self.items}
        self.body = {n: lab._body_id(n) for n in self.items}
        self.qadr = {n: int(m.jnt_qposadr[m.body_jntadr[self.body[n]]]) for n in self.items}
        self.dadr = {n: int(m.jnt_dofadr[m.body_jntadr[self.body[n]]]) for n in self.items}
        self.half = {n: dr.item_geometry(dr.ITEMS[n])["half_h"] for n in self.items}
        self.root = root
        self.state = {n: "waiting" for n in self.items}
        self.records = {n: {"item": n, "picked_up": None, "peak_squeeze_N": 0.0, "slip_mm": 0.0, "released": None,
                            "release": None, "release_height_m": None, "in_zone": None, "caught": None,
                            "catch_speed_m_s": None, "catch_load_N": None, "impact_m_s": None, "touch_speed_m_s": None,
                            "delivered": None, "how": None, "taken": None, "damage": [], "final_pos": None}
                        for n in self.items}
        self.v_prev = {n: np.zeros(3) for n in self.items}
        self.lost_for = {n: 0.0 for n in self.items}
        self.held_at = {}
        self.net_pose, self.table_pose, self.settled_at = {}, {}, {}
        self.history, self.n = [], 0

    def _damage(self, lab, name, what):
        rec = self.records[name]
        if what not in rec["damage"]:
            rec["damage"].append(what)
            lab.log_event(name, f"SPOILT: {what}")

    def _local(self, lab, p):
        d = lab.data
        R = d.xmat[self.root].reshape(3, 3)
        return R.T @ (p - d.xpos[self.root])

    def __call__(self, lab):
        d, m = lab.data, lab.model
        sq = {n: [0.0, 0.0] for n in self.items}
        ground = {n: 0.0 for n in self.items}
        for c in range(d.ncon):
            con = d.contact[c]
            if con.efc_address < 0:
                continue
            g1, g2 = int(con.geom1), int(con.geom2)
            name = self.item_geom.get(g1) or self.item_geom.get(g2)
            if name is None:
                continue
            other = g2 if self.item_geom.get(g1) == name else g1
            self._mj.mj_contactForce(m, d, c, self.buf)
            fn = float(self.buf[0])
            if other == self.pads[0]:
                sq[name][0] += fn
            elif other == self.pads[1]:
                sq[name][1] += fn
            elif other == self.terrain:
                ground[name] += fn
        t, dt = float(lab.time), lab.control_dt
        sc = self.scene
        for k, name in enumerate(self.items):
            rec, st = self.records[name], self.state[name]
            it = dr.ITEMS[name]
            qa, da = self.qadr[name], self.dadr[name]
            pos = np.array(d.qpos[qa:qa + 3])
            vel = np.array(d.qvel[da:da + 3])
            bottom = pos[2] - self.half[name]
            held = min(sq[name]) > 0.5
            if st == "waiting" and held:
                self.state[name], rec["picked_up"] = "held", t
                self.held_at[name] = self._local(lab, pos)
                lab.log_event(name, f"in the jaws (squeeze {sq[name][0]:.1f} / {sq[name][1]:.1f} N)")
            elif st == "held":
                rec["peak_squeeze_N"] = max(rec["peak_squeeze_N"], *sq[name])
                if max(sq[name]) > it["squeeze_limit_N"]:
                    self._damage(lab, name, f"crushed in the jaws ({max(sq[name]):.0f} N > {it['squeeze_limit_N']:.0f} N)")
                if held and t - rec["picked_up"] >= 0.3:       # squeezed (the jaws have centred it): has it moved?
                    if name not in self.settled_at:
                        self.settled_at[name] = self._local(lab, pos)
                    slip = float(np.linalg.norm(self._local(lab, pos) - self.settled_at[name])) * 1000
                    rec["slip_mm"] = max(rec["slip_mm"], slip)
                airborne = bottom > sc.ground(pos[0], pos[1]) + 0.02
                rec["airborne"] = rec.get("airborne", False) or airborne
                if rec.get("airborne") and ground[name] > 0.05 and rec["touch_speed_m_s"] is None:
                    rec["touch_speed_m_s"] = float(np.linalg.norm(self.v_prev[name]))
                    lab.log_event(name, f"touches the ground at {rec['touch_speed_m_s']:.2f} m/s, "
                                        f"{'inside' if sc.in_zone(pos[0], pos[1]) else 'OUTSIDE'} the zone")
                if self.n % self.log_every == 0:
                    self.history.append([t, k, sq[name][0], sq[name][1], ground[name]])
                self.lost_for[name] = self.lost_for[name] + dt if max(sq[name]) < 0.2 else 0.0
                if self.lost_for[name] >= 0.06:
                    rec["released"] = t
                    on_ground = ground[name] > 0.05 or bottom < sc.ground(pos[0], pos[1]) + 0.005
                    if on_ground and rec.get("airborne"):
                        self.state[name] = "on the ground"
                        rec["release"], rec["in_zone"] = "put down", sc.in_zone(pos[0], pos[1])
                        lab.log_event(name, f"let go on the ground {'inside' if rec['in_zone'] else 'OUTSIDE'} the zone")
                        if rec["in_zone"] and sc.variant == "place":
                            rec["delivered"], rec["how"] = t, "put down on the zone"
                    else:
                        self.state[name] = "falling"
                        rec["release"], rec["release_height_m"] = "in the air", float(bottom)
                        lab.log_event(name, f"falls from {bottom:.2f} m (its bottom above the ground)")
            elif st in ("falling", "through the net"):
                net_z = sc.ground(*sc.zone) + sc.net_height
                if st == "falling" and sc.variant == "drop" and bottom <= net_z and vel[2] < 0:
                    if sc.in_net(pos[0], pos[1]):
                        fall = rec["release_height_m"] - net_z
                        v = float(np.linalg.norm(vel))
                        load = it["mass_kg"] * (dr.G + v * v / (2 * dr.NET["stretch_m"]))
                        rec["catch_speed_m_s"], rec["catch_load_N"] = v, load
                        if fall > sc.net_max_drop:
                            self.state[name] = "through the net"
                            lab.log_event("net", f"TORN by the {name}: it fell {fall:.2f} m (> {sc.net_max_drop:g} m)")
                            self._damage(lab, name, f"tore the net ({fall:.1f} m fall)")
                        else:
                            self.state[name], rec["caught"] = "in the net", t
                            rec["delivered"], rec["how"] = t, "caught in the net"
                            self.net_pose[name] = np.array([pos[0], pos[1], net_z + self.half[name]])
                            lab.log_event("net", f"caught the {name} after a {fall:.2f} m fall at {v:.1f} m/s "
                                                 f"(arrest load {load:.0f} N)")
                            if load > it["catch_limit_N"]:
                                self._damage(lab, name, f"bruised by the net's arrest ({load:.0f} N > {it['catch_limit_N']:.0f} N)")
                    else:
                        self.state[name] = "through the net"
                        lab.log_event("net", f"MISSED the {name} ({pos[0]:.2f}, {pos[1]:.2f} m)")
                elif ground[name] > 0.05:
                    v = float(np.linalg.norm(self.v_prev[name]))
                    rec["impact_m_s"] = v
                    self.state[name] = "on the ground"
                    lab.log_event(name, f"hits the ground at {v:.1f} m/s")
                    if v > it["impact_limit_m_s"]:
                        self._damage(lab, name, f"smashed on the ground ({v:.1f} m/s > {it['impact_limit_m_s']:.1f} m/s)")
            elif st == "in the net":                           # the net is not simulated: it holds what it caught
                d.qpos[qa:qa + 3] = self.net_pose[name]
                d.qvel[da:da + 6] = 0.0
            elif st == "on the table":                         # held on its plate (the table is only drawn)
                d.qpos[qa:qa + 3] = self.table_pose[name]
                d.qpos[qa + 3:qa + 7] = (1.0, 0.0, 0.0, 0.0)
                d.qvel[da:da + 6] = 0.0
            if rec["delivered"] is not None and rec["taken"] is None and self.state[name] in ("in the net", "on the ground") \
                    and t - rec["delivered"] >= sc.take_after_s \
                    and np.linalg.norm(d.xpos[self.root][:2] - pos[:2]) > 2.5:
                rec["taken"] = t
                self.state[name] = "on the table"
                self.table_pose[name] = np.array(sc.plate(name)) + np.array([0.0, 0.0, self.half[name]])
                lab.log_event(name, "taken to the table by the hungry people")
            if st == "on the ground" and rec["touch_speed_m_s"] is not None and rec["release"] == "put down":
                if rec["touch_speed_m_s"] > it["impact_limit_m_s"]:
                    self._damage(lab, name, f"put down too hard ({rec['touch_speed_m_s']:.1f} m/s)")
            rec["final_pos"] = pos.tolist()
            self.v_prev[name] = vel
        self.n += 1


# ----------------------------------------------------------------------------------------------- lab and run
def make_lab(scene: Scene, robot=None, prop: dr.Propulsion | None = None, **kwargs):
    """Drongo on the flat garden with the items and the scenery, the ``Rotors`` hook (``lab.rotors``) and ``Watch``
    (``lab.watch``)."""
    robot = robot or dr.drongo()
    prop = prop or dr.propulsion()
    opts = dict(dr.LAB_OPTIONS)
    opts.update(kwargs)
    lab = ch.ChironLab(robot, ch.Flat(), props=item_props(scene) + scenery(scene), **opts)
    lab.prop_model = prop
    lab.rotors = dr.Rotors(robot, prop)
    lab.add_hook(lab.rotors)
    lab.watch = Watch(scene)
    lab.add_hook(lab.watch)
    return lab


def run(lab, scene: Scene, plan: dc.Plan | None = None, duration: float = 130.0, seed: int = 0, flight_kw=None):
    """The whole delivery. ``ep.log``: ``mission`` (the phase log), ``mission_finished``, ``records`` (``Watch``),
    ``rotor_history`` [t, T1..T4, P], ``squeeze_history``, ``energy_Wh``; ``ep.outcome``: success when both items were
    delivered unspoilt, ``t_end`` when the last of them was."""
    plan = plan or dc.Plan()
    mission = dc.Mission(dc.delivery(scene, plan), dc.Flight(lab.prop_model, **(flight_kw or {})),
                         name=f"Drongo: potato and cream, {scene.variant}")
    ep = lab.run(mission, duration=duration, rules=None, settle=0.3, seed=seed, base_pos=tuple(scene.home),
                 info={"controller": mission.name, "treatment": scene.variant})
    ep.log["mission"] = [list(x) for x in mission.log]
    ep.log["mission_finished"] = mission.finished
    ep.log["records"] = {k: dict(v) for k, v in lab.watch.records.items()}
    ep.log["rotor_history"] = np.asarray(lab.rotors.history, dtype=float).reshape(-1, 6)
    ep.log["squeeze_history"] = np.asarray(lab.watch.history, dtype=float).reshape(-1, 5)
    ep.log["energy_Wh"] = lab.rotors.energy_Wh
    ep.log["variant"] = scene.variant
    ep.log["plan"] = dict(plan.__dict__)
    ends = [t for t, name, note in mission.log if note == "done"]
    ep.log["mission_end"] = float(ends[-1]) if (mission.finished and ends) else float(ep.log["t"][-1])
    ep.outcome = outcome(ep)
    return ep


def outcome(ep) -> dict:
    """Success when every item was delivered and nothing spoilt it; ``t_end`` the last delivery."""
    recs = ep.log["records"]
    spoilt = {k: r["damage"] for k, r in recs.items() if r["damage"]}
    missing = [k for k, r in recs.items() if r["delivered"] is None]
    t_last = max((r["delivered"] for r in recs.values() if r["delivered"] is not None), default=float("nan"))
    if not spoilt and not missing:
        return {"success": True, "reason": "delivered", "t_end": t_last,
                "detail": "; ".join(f"{k} {r['how']} at {r['delivered']:.1f} s" for k, r in recs.items())}
    parts = [f"{k}: {', '.join(v)}" for k, v in spoilt.items()] + [f"{k} not delivered" for k in missing]
    return {"success": False, "reason": "spoilt" if spoilt else "not delivered",
            "t_end": t_last if not missing else float(ep.log["mission_end"]), "detail": "; ".join(parts)}


# ----------------------------------------------------------------------------------------------- tables
def _yaw_tilt(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
    yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return yaw, tilt


def timeseries(ep) -> pd.DataFrame:
    """Per log sample: Drongo's position, speed, tilt, the four rotor thrusts and the electrical power (from the rotor
    history), the pad gap, the items' positions."""
    log = ep.log
    t = np.asarray(log["t"])
    bp = np.asarray(log["body_pos"])[:, 0]
    bv = np.asarray(log["body_linvel"])[:, 0]
    _, tilt = _yaw_tilt(np.asarray(log["body_quat"])[:, 0])
    df = pd.DataFrame({"t": t, "x": bp[:, 0], "y": bp[:, 1], "z": bp[:, 2], "speed": np.linalg.norm(bv, axis=1),
                       "vz": bv[:, 2], "tilt_deg": tilt})
    rh = np.asarray(log["rotor_history"])
    for k in range(4):
        df[f"T{k + 1}"] = np.interp(t, rh[:, 0], rh[:, 1 + k]) if len(rh) else 0.0
    df["P_el"] = np.interp(t, rh[:, 0], rh[:, 5]) if len(rh) else 0.0
    names = list(log["joints"])
    q = np.asarray(log["q"])
    df["gap_mm"] = (dr.geometry()["closed_gap"] + q[:, names.index(dr.JAW_JOINTS[0])] + q[:, names.index(dr.JAW_JOINTS[1])]) * 1000
    pp = np.asarray(log["prop_pos"])
    for i, n in enumerate(log["props"]):
        if n in dr.ITEMS:
            df[f"{n}_x"], df[f"{n}_y"], df[f"{n}_z"] = pp[:, i, 0], pp[:, i, 1], pp[:, i, 2]
    return df


def phase_table(ep, prop: dr.Propulsion | None = None) -> pd.DataFrame:
    """Per mission phase: start, duration, where it ended, peak tilt, speed and rotor thrust (share of the maximum),
    and the electrical energy it took."""
    prop = prop or dr.propulsion()
    log = ep.log
    ts = timeseries(ep)
    rh = np.asarray(log["rotor_history"])
    starts = [(t, name) for t, name, note in log["mission"] if note == "start"]
    rows = {}
    for k, (t0, name) in enumerate(starts):
        t1 = starts[k + 1][0] if k + 1 < len(starts) else float(log["mission_end"])
        s = ts[(ts.t >= t0) & (ts.t <= t1)]
        sel = (rh[:, 0] >= t0) & (rh[:, 0] <= t1) if len(rh) else np.zeros(0, bool)
        e = float(np.trapezoid(rh[sel, 5], rh[sel, 0])) / 3600 if sel.sum() > 1 else 0.0
        rows[f"{k + 1:02d} {name}"] = {
            "start [s]": t0, "duration [s]": t1 - t0, "x [m]": float(s.x.iloc[-1]) if len(s) else np.nan,
            "y [m]": float(s.y.iloc[-1]) if len(s) else np.nan, "z [m]": float(s.z.iloc[-1]) if len(s) else np.nan,
            "peak speed [m/s]": float(s.speed.max()) if len(s) else np.nan,
            "peak tilt [deg]": float(s.tilt_deg.max()) if len(s) else np.nan,
            "peak thrust / max": float(s[[f"T{i}" for i in range(1, 5)]].max().max() / prop.max_thrust_N) if len(s) else np.nan,
            "energy [Wh]": e}
    return pd.DataFrame(rows).T


def deliveries(ep) -> pd.DataFrame:
    """Per item: when it was picked up and delivered, how, the loads it saw against its limits, and whether it
    arrived unspoilt."""
    rows = {}
    for name, r in ep.log["records"].items():
        it = dr.ITEMS[name]
        rows[name] = {"picked up [s]": r["picked_up"], "delivered [s]": r["delivered"], "how": r["how"] or "NOT DELIVERED",
                      "peak squeeze [N]": r["peak_squeeze_N"], "squeeze limit [N]": it["squeeze_limit_N"],
                      "slip in the jaws [mm]": r["slip_mm"],
                      "fall onto the net [m]": (r["release_height_m"] - (ep.log.get("net_height") or dr.NET["height_m"]))
                      if r["release_height_m"] is not None else None,
                      "speed at the net [m/s]": r["catch_speed_m_s"], "net arrest load [N]": r["catch_load_N"],
                      "catch limit [N]": it["catch_limit_N"] if r["catch_load_N"] is not None else None,
                      "touchdown speed [m/s]": r["touch_speed_m_s"], "impact speed [m/s]": r["impact_m_s"],
                      "impact limit [m/s]": it["impact_limit_m_s"],
                      "unspoilt": not r["damage"], "damage": "; ".join(r["damage"]) or "none"}
    return pd.DataFrame(rows).T


def wait_times(ep) -> dict:
    """How long the hungry people wait [s] (from the order at t = 0): for the potato, for the cream, for both."""
    recs = ep.log["records"]
    out = {f"{k} [s]": r["delivered"] for k, r in recs.items()}
    out["both [s]"] = None if any(r["delivered"] is None for r in recs.values()) else max(r["delivered"] for r in recs.values())
    out["Drongo home [s]"] = ep.log["mission_end"]
    return out


#: What each mission phase is, for the time budget (the first word of the phase's name decides).
ACTIVITIES = {"flying": ("hop", "fly"), "climbing": ("take", "climb"),
              "descending and landing": ("descend", "touch", "spool", "down", "lower"),
              "handling the item": ("open", "grip", "drop", "let", "hold", "steady", "close", "release")}


def activity(phase: str) -> str:
    first = phase.split()[0]
    return next((a for a, words in ACTIVITIES.items() if first in words), "other")


def time_budget(ep, until: float | None = None) -> dict:
    """Seconds of each activity (``ACTIVITIES``) from the order until ``until`` (default: the last delivery)."""
    until = float(ep.outcome["t_end"] if until is None else until)
    starts = [(t, name) for t, name, note in ep.log["mission"] if note == "start"]
    out = {a: 0.0 for a in ACTIVITIES}
    for k, (t0, name) in enumerate(starts):
        t1 = starts[k + 1][0] if k + 1 < len(starts) else float(ep.log["mission_end"])
        dt = max(0.0, min(t1, until) - t0)
        out[activity(name)] = out.get(activity(name), 0.0) + dt
    return out


def _sweep_one(args):
    variant, kw, duration = args
    fields = set(Scene.__dataclass_fields__)
    scene = Scene(variant, **{k: v for k, v in kw.items() if k in fields})
    lab = make_lab(scene)
    ep = run(lab, scene, dc.Plan(**{k: v for k, v in kw.items() if k not in fields}), duration=duration)
    ts = timeseries(ep)
    w = wait_times(ep)
    row = {"variant": variant, **kw, "potato [s]": w["potato [s]"], "both [s]": w["both [s]"],
           "Drongo home [s]": w["Drongo home [s]"], "unspoilt": bool(ep.outcome["success"]),
           "peak tilt [deg]": float(ts.tilt_deg.max()), "peak speed [m/s]": float(ts.speed.max()),
           "energy [Wh]": float(ep.log["energy_Wh"]), "detail": ep.outcome["detail"]}
    row.update({f"{k} [s]": v for k, v in time_budget(ep).items()})
    return row


def sweep(plans: list, variants=VARIANTS, processes: int = 4, duration: float = 110.0) -> pd.DataFrame:
    """Every plan (a dict of ``drongo_controller.Plan`` fields, and ``Scene`` fields such as ``cruise_alt``) in every
    variant, ``processes`` runs at a time (forked processes): wait times, whether everything arrived unspoilt, peak tilt
    and speed, energy, the time budget."""
    import multiprocessing as mp

    jobs = [(v, dict(kw), duration) for kw in plans for v in variants]
    if processes > 1:
        with mp.get_context("fork").Pool(min(processes, len(jobs))) as pool:
            rows = pool.map(_sweep_one, jobs)
    else:
        rows = [_sweep_one(j) for j in jobs]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- the movie
def _cameras(ep, idx, scene: Scene, smooth_frames: float = 10.0) -> list:
    """A camera per frame (position, focal point, up): a chase shot behind Drongo along its direction of travel and to
    its right, closer near the ground; within 7 m of the drop zone it blends into a wider shot of the zone that holds
    Drongo and the net (or the zone) in view, so a falling item is seen all the way. The camera's offsets from Drongo
    are low-pass filtered (it glides) while Drongo itself never leaves the frame."""
    log = ep.log
    com = np.asarray(log["body_pos"])[:, 0]
    vel = np.asarray(log["body_linvel"])[:, 0]
    zone = np.asarray(scene.zone, dtype=float)
    heading = zone - np.asarray(scene.home, dtype=float)
    heading /= np.linalg.norm(heading)
    z_low = scene.ground(*zone) + (scene.net_height if scene.variant == "drop" else 0.0)
    side = np.array([-0.25, -0.97, 0.38])                # from the south: between two of the net's holders
    side /= np.linalg.norm(side)
    cams, f_s, p_s = [], None, None
    a = 1.0 / smooth_frames
    half_view = math.radians(12.5)                      # inside pyvista's 30° vertical view angle, with a margin
    for i in idx:
        c, vh = com[i], vel[i, :2]
        if np.linalg.norm(vh) > 0.8:
            heading = heading + 0.15 * (vh / np.linalg.norm(vh) - heading)
            heading /= np.linalg.norm(heading)
        right = np.array([heading[1], -heading[0]])
        h = max(0.0, c[2] - scene.ground(c[0], c[1]))
        dist = 1.3 + 0.13 * h
        chase_f = np.array([*(0.3 * heading), -0.05])
        chase_p = np.array([*(-0.75 * heading + 0.6 * right) * dist, 0.35 * dist + 0.15])
        d_zone = float(np.linalg.norm(c[:2] - zone))
        w = smooth(float(np.clip((7.0 - d_zone) / 4.0, 0.0, 1.0)))
        mid = np.array([0.5 * (c[0] + zone[0]), 0.5 * (c[1] + zone[1]), 0.5 * (c[2] + z_low)])
        span = max(c[2] - z_low + 1.2, 3.0)
        zone_f = mid - c
        zone_p = zone_f + side * max(7.0, 0.5 * span / math.tan(half_view))
        f_off, p_off = (1 - w) * chase_f + w * zone_f, (1 - w) * chase_p + w * zone_p
        f_s = f_off if f_s is None else f_s + a * (f_off - f_s)
        p_s = p_off if p_s is None else p_s + a * (p_off - p_s)
        pos = c + p_s
        pos[2] = max(pos[2], 0.35)
        cams.append((tuple(pos), tuple(c + f_s), (0, 0, 1)))
    return cams


def _hud(img, lines, title):
    """The title bar and the panel's lines on a translucent dark box (one stroke per glyph: this OpenCV draws a
    thicker stroke as a bolder, wider font, so an outline would not line up)."""
    import cv2

    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(img, (0, 0), (w, 30), (25, 30, 40), -1)
    cv2.putText(img, title, (12, 21), font, 0.58, (240, 240, 240), 1, cv2.LINE_AA)
    width = max(cv2.getTextSize(text, font, 0.5, 1)[0][0] for text, _ in lines) + 24
    box = img[36:36 + 22 * len(lines) + 10, 0:width]
    box[:] = (0.45 * box + 0.55 * np.array([20, 24, 32])).astype(np.uint8)
    y = 54
    for text, colour in lines:
        cv2.putText(img, text, (12, y), font, 0.5, colour, 1, cv2.LINE_AA)
        y += 22
    return img


def render_movie(ep, scene: Scene, path, *, speed: float = 2.0, fps: int = 25, size=(960, 540), progress=False):
    """The delivery as a movie: Chiron's renderer (pyvista, from the log; needs ``log_geoms=True``) with a chase
    camera, a panel with the clock — how long the hungry people have waited —, the phase, height, speed and the
    items' fate, the outcome on the last frame (``pekari_controller.end_card``) and the Vegeta watermark; MP4 through
    OpenCV. ``speed`` × real time, up to the end of the mission."""
    import cv2
    from vegeta.aeromant._watermark import watermark
    from vegeta.chiron import viz

    from pekari_controller import end_card

    log = ep.log
    t = np.asarray(log["t"])
    dt = float(t[1] - t[0])
    every = max(1, int(round(speed / (fps * dt))))
    stop = int(np.searchsorted(t, float(log["mission_end"]) + 1.0)) + 1
    idx = list(range(0, min(stop, len(t)), every))
    cams = iter(_cameras(ep, idx, scene))
    zx, zy = scene.zone
    imgs = viz.frames(ep, camera=lambda com: next(cams), every=every, stop=idx[-1] + 1, size=size, show_time=False,
                      ground=(scene.home[0] - 250, zx + 250, scene.home[1] - 250, zy + 250),
                      scenery_range=500.0, ground_color="#7fa860", background="#cfe3f5")
    ts = timeseries(ep)
    starts = [(tt, name) for tt, name, note in log["mission"] if note == "start"]
    events = sorted(log.get("events", []), key=lambda e: e[0])
    recs = log["records"]
    title = (f"Drongo delivers a potato and a cream - variant: {'drop into the net (max 5 m)' if scene.variant == 'drop' else 'place on the zone'}"
             f"   ({speed:g}x speed)")
    out = []
    for k, i in enumerate(idx[:len(imgs)]):
        ti = t[i]
        phase = next((name for tt, name in reversed(starts) if tt <= ti + 1e-9), "on the pad")
        row = ts.iloc[i]
        lines = [(f"hungry people waiting: {ti:5.1f} s", (255, 225, 120)),
                 (f"phase: {phase}", (235, 235, 235)),
                 (f"height {row.z - scene.robot_geometry['skid_h']:5.1f} m   speed {row.speed:4.1f} m/s   "
                  f"tilt {row.tilt_deg:4.1f} deg   power {row.P_el:4.0f} W", (210, 230, 255))]
        for name in ("potato", "cream"):
            r = recs[name]
            if r["delivered"] is not None and r["delivered"] <= ti:
                txt, col = f"{name}: {r['how']} at {r['delivered']:.1f} s", (140, 240, 140)
            elif r["picked_up"] is not None and r["picked_up"] <= ti:
                txt, col = f"{name}: in the pincer", (235, 235, 235)
            else:
                txt, col = f"{name}: waiting in the basket", (200, 200, 200)
            bad = [e for e in events if e[1] == name and "SPOILT" in e[2] and e[0] <= ti]
            if bad:
                txt, col = f"{name}: {bad[0][2]}", (255, 110, 110)
            lines.append((txt, col))
        recent = [e for e in events if 0 <= ti - e[0] < 2.5 and e[1] in ("net", "potato", "cream")]
        if recent:
            lines.append((f"> {recent[-1][2]}", (255, 200, 90)))
        out.append(_hud(np.ascontiguousarray(imgs[k]), lines, title))
    out = end_card(out, ep, fps=fps, hold_s=2.5)
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = out[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    try:
        for img in out:
            vw.write(watermark(cv2.cvtColor(np.ascontiguousarray(img, dtype=np.uint8), cv2.COLOR_RGB2BGR)))
    finally:
        vw.release()
    return path


def stills(path, times, *, speed: float = 2.0, fps: int = 25) -> list:
    """RGB frames of a movie made by ``render_movie`` at mission times ``times`` [s] (read in order: seeking in an
    MPEG-4 stream decodes from the wrong key frame)."""
    import cv2

    want = {int(round(t * fps / speed)): k for k, t in enumerate(times)}
    out = [None] * len(times)
    cap = cv2.VideoCapture(str(path))
    i = 0
    try:
        while want:
            ok, img = cap.read()
            if not ok:
                break
            if i in want:
                out[want.pop(i)] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            i += 1
    finally:
        cap.release()
    return [o for o in out if o is not None]
