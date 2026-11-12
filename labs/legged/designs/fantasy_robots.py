"""The fantasy objects as Chiron bodies, for export (``machines.py`` → ``tools/export.py``): loose rigid objects
(a free root, no actuators) whose collision shapes are a few boxes, cylinders, capsules and spheres over the CAD,
carrying the CAD's masses — each material's mass (``CAD`` volume × ``fantasy_specs.MATERIALS`` density) shared
among that material's shapes by their volumes, so the total is exact and the inertia close. The chest's lid is
its own link on a passive hinge (0 to 110°), opened by its motions. Units SI; x forward, y left, z up.

    import fantasy_robots as fr
    chest = fr.chest()              # also crate, sword, shield, coin, amulet, elixir_flask
"""
from __future__ import annotations

import math

from vegeta.chiron import Geom, Joint, Link, Robot

import fantasy_specs as fs

__all__ = ["CAD", "mass_of", "composition", "chest", "crate", "sword", "shield", "coin", "amulet", "elixir_flask",
           "materials_block", "parts", "BUILDERS"]

mm = 1e-3

#: CAD volumes [mm³] and centres of mass [mm] of each material part of the default designs (``fantasy_objects``;
#: the chest's lid alone in its hinge frame); ``tests/test_fantasy.py`` rebuilds them.
CAD = {
    "chest/oak": (35457519.8, (0.0, 0.0, 267.22)),
    "chest/iron": (803951.9, (-0.0, -15.0, 340.79)),
    "chest/lid_oak": (12070526.8, (210.0, 0.0, 117.3)),
    "chest/lid_iron": (387858.5, (248.2, -0.0, 106.46)),
    "crate/crate": (21585562.8, (0.0, 0.0, 250.0)),
    "sword/steel": (225845.8, (64.24, 0.0, 21.0)),
    "sword/leather": (97899.9, (-118.0, 0.0, 21.0)),
    "shield/oak": (5867099.2, (0.0, 0.0, 11.65)),
    "shield/iron": (269917.0, (-0.0, 0.0, 27.04)),
    "coin/coin": (1110.9, (-0.0, -0.0, 1.22)),
    "amulet/bronze": (8839.0, (82.56, -0.01, 2.33)),
    "amulet/gem": (3714.9, (0.0, 0.0, 3.5)),
    "elixir_flask/glass": (152129.3, (-0.0, 0.0, 58.68)),
    "elixir_flask/cork": (11474.7, (0.0, 0.0, 180.36)),
    "elixir_flask/elixir": (377594.2, (-0.0, -0.0, 47.21)),
}
LID = {"oak": "chest/lid_oak", "wrought iron": "chest/lid_iron"}
WOOD, IRON, GOLD = (0.55, 0.38, 0.22, 1.0), (0.3, 0.3, 0.32, 1.0), (0.85, 0.68, 0.2, 1.0)


def _material(key):
    obj, part = key.split("/")
    return fs.OBJECTS[obj][1]["_parts"].get(part) or {"lid_oak": "oak", "lid_iron": "wrought iron"}[part]


def mass_of(key) -> float:
    """A CAD part's mass [kg]."""
    return CAD[key][0] * 1e-9 * fs.MATERIALS[_material(key)]["density_kg_m3"]


def composition(obj) -> dict:
    """{part: {material, volume_m3, mass_kg, com_m}} of an object. The chest's ``oak`` and ``iron`` are the whole
    chest's; its ``lid_oak`` and ``lid_iron`` are the part of them in the lid (centres in the hinge frame)."""
    out = {}
    for key, (v, c) in CAD.items():
        if key.split("/")[0] == obj:
            out[key.split("/")[1]] = {"material": _material(key), "volume_m3": round(v * 1e-9, 9),
                                      "mass_kg": round(mass_of(key), 5), "com_m": [round(x * mm, 5) for x in c]}
    return out


def _volume(g):
    s = g["size"]
    if g["type"] == "box":
        return 8 * s[0] * s[1] * s[2]
    if g["type"] == "sphere":
        return 4 / 3 * math.pi * s[0] ** 3
    if g["type"] == "cylinder":
        return math.pi * s[0] ** 2 * 2 * s[1]
    if g["type"] == "capsule":
        a, b = g["fromto"][:3], g["fromto"][3:]
        return math.pi * s[0] ** 2 * math.dist(a, b) + 4 / 3 * math.pi * s[0] ** 3
    if g["type"] == "ellipsoid":
        return 4 / 3 * math.pi * s[0] * s[1] * s[2]
    raise ValueError(g["type"])


def _geoms(shapes, masses):
    """Chiron geoms from ``shapes`` ([{name, type, size, pos?, fromto?, quat?, material, rgba}]) with each
    material's mass ``masses[material]`` shared among its shapes by volume."""
    vol = {}
    for g in shapes:
        vol[g["material"]] = vol.get(g["material"], 0.0) + _volume(g)
    out = []
    for g in shapes:
        kw = {k: g[k] for k in ("pos", "fromto", "quat") if k in g}
        out.append(Geom(g["name"], g["type"], tuple(g["size"]), mass=masses[g["material"]] * _volume(g) / vol[g["material"]],
                        role="body", friction=g.get("friction", (0.6, 0.005, 0.0001)), rgba=g.get("rgba"), **kw))
    return out


def _masses(obj, *, skip_lid=False):
    m = {}
    for key in CAD:
        o, part = key.split("/")
        if o != obj or (part.startswith("lid_") and skip_lid):
            continue
        mat = _material(key)
        m[mat] = m.get(mat, 0.0) + mass_of(key)
    return m


def _object(name, shapes, obj, notes, children=(), masses=None):
    root = Link(name, log=True, group="body", geoms=_geoms(shapes, masses or _masses(obj)), children=list(children))
    return Robot(name, root, notes=notes,
                 sources={"geometry": f"designs/fantasy_objects.py ({fs.OBJECTS[obj][0]}) at fantasy_specs defaults",
                          "masses": "CAD volume × fantasy_specs.MATERIALS density per material (fantasy_robots.CAD)"})


Y_AXIS = (math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0)          # a cylinder's axis z -> y
X_AXIS = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)           # z -> x


def chest():
    p = fs.CHEST
    L, W, H, t, r = p["length"] * mm, p["width"] * mm, p["height"] * mm, p["wall"] * mm, p["width"] * mm / 2
    whole = _masses("chest", skip_lid=True)                              # the CAD's chest/oak and /iron hold the lid
    lid_m = {mat: mass_of(LID[mat]) for mat in LID}
    box = {mat: whole[mat] - lid_m[mat] for mat in whole}
    base = [{"name": "chest_floor", "type": "box", "size": (L / 2, W / 2, t / 2), "pos": (0, 0, t / 2), "material": "oak"}]
    for sx in (-1, 1):
        base.append({"name": f"chest_end{'LR'[sx > 0]}", "type": "box", "size": (t / 2, W / 2, H / 2),
                     "pos": (sx * (L / 2 - t / 2), 0, H / 2), "material": "oak", "rgba": WOOD})
        base.append({"name": f"chest_side{'FB'[sx > 0]}", "type": "box", "size": (L / 2, t / 2, H / 2),
                     "pos": (0, sx * (W / 2 - t / 2), H / 2), "material": "oak", "rgba": WOOD})
        base.append({"name": f"chest_band{'LR'[sx > 0]}", "type": "box", "size": (0.02, W / 2 + 0.003, H / 2),
                     "pos": (sx * (L / 2 - 0.02), 0, H / 2), "material": "wrought iron", "rgba": IRON})
    lid_shapes = [{"name": "lid_shell", "type": "box", "size": (L / 2, r, r / 2), "pos": (0, -r, r / 2),
                   "material": "oak", "rgba": WOOD},
                  {"name": "lid_bands", "type": "box", "size": (L / 2, r + 0.003, 0.004), "pos": (0, -r, r + 0.002),
                   "material": "wrought iron", "rgba": IRON}]
    lid = Link("lid", pos=(0, W / 2, H), joints=[Joint("lid_hinge", axis=(-1, 0, 0), range=(0.0, math.radians(110)),
                                                      damping=0.5, tag="hinge")],
               geoms=_geoms(lid_shapes, lid_m))
    return _object("chest", base, "chest", "Treasure chest: oak and iron, 32.8 kg, a hinged lid (passive, 0–110°).",
                   children=[lid], masses=box)


def crate():
    S = fs.CRATE["size"] * mm
    return _object("crate", [{"name": "crate", "type": "box", "size": (S / 2, S / 2, S / 2), "pos": (0, 0, S / 2),
                              "material": "pine", "rgba": WOOD}], "crate", "Wooden crate: pine, 0.5 m, 11.0 kg.")


def sword():
    p = fs.SWORD
    z = p["pommel_diameter"] / 2 * mm
    Lb, g0, gl = p["blade_length"] * mm, p["guard_thickness"] * mm, p["grip_length"] * mm
    shapes = [{"name": "blade", "type": "box", "size": (Lb / 2, p["blade_width"] * mm / 2 * 0.8, p["blade_thickness"] * mm / 2),
               "pos": (Lb / 2, 0, z), "material": "spring steel", "rgba": IRON},
              {"name": "guard", "type": "box", "size": (g0 / 2, p["guard_width"] * mm / 2, 0.007), "pos": (-g0 / 2, 0, z),
               "material": "spring steel", "rgba": IRON},
              {"name": "grip", "type": "capsule", "size": (p["grip_diameter"] * mm / 2,),
               "fromto": (-g0, 0, z, -g0 - gl, 0, z), "material": "leather", "rgba": WOOD},
              {"name": "pommel", "type": "sphere", "size": (z,), "pos": (-g0 - gl - 0.4 * 2 * z, 0, z),
               "material": "spring steel", "rgba": IRON}]
    return _object("sword", shapes, "sword", "Longsword: spring-steel blade, leather grip, 1.86 kg, 1.16 m.")


def shield():
    p = fs.SHIELD
    D, t = p["diameter"] * mm, p["thickness"] * mm
    shapes = [{"name": "board", "type": "cylinder", "size": (D / 2, 2.5 * t), "pos": (0, 0, 2.5 * t),
               "material": "oak", "rgba": WOOD},
              {"name": "boss", "type": "sphere", "size": (p["boss_diameter"] * mm / 2 * 0.6,), "pos": (0, 0, 4 * t),
               "material": "wrought iron", "rgba": IRON},
              {"name": "rim", "type": "cylinder", "size": (D / 2 + 0.002, 0.002), "pos": (0, 0, 0.002),
               "material": "wrought iron", "rgba": IRON}]
    return _object("shield", shapes, "shield", "Round shield: oak planks, iron boss and rim, 6.5 kg, 0.8 m.")


def coin():
    p = fs.COIN
    return _object("coin", [{"name": "coin", "type": "cylinder", "size": (p["diameter"] * mm / 2, p["thickness"] * mm / 2),
                             "pos": (0, 0, p["thickness"] * mm / 2), "material": "gold 22k", "rgba": GOLD}],
                   "coin", "Gold coin: 22 carat, 28 mm, 20 g.")


def amulet():
    p = fs.AMULET
    R, t = p["setting_diameter"] * mm / 2, p["setting_thickness"] * mm
    Rc, n = p["chain_radius"] * mm, 12
    x0 = R + 0.0035 + Rc
    shapes = [{"name": "setting", "type": "cylinder", "size": (R, t / 2), "pos": (0, 0, t / 2), "material": "bronze",
               "rgba": GOLD},
              {"name": "gem", "type": "ellipsoid", "size": (p["gem_length"] * mm / 2, p["gem_width"] * mm / 2,
                                                           p["gem_height"] * mm / 2),
               "pos": (0, 0, t / 2 + 0.001), "material": "garnet", "rgba": (0.6, 0.05, 0.1, 1.0)}]
    for k in range(n):                                                    # the chain as a ring of capsules
        a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
        pa = (x0 - Rc * math.cos(a0), Rc * math.sin(a0), t / 2)
        pb = (x0 - Rc * math.cos(a1), Rc * math.sin(a1), t / 2)
        shapes.append({"name": f"chain{k}", "type": "capsule", "size": (p["wire_diameter"] * mm,), "fromto": pa + pb,
                       "material": "bronze", "rgba": GOLD})
    return _object("amulet", shapes, "amulet", "Amulet: a garnet in a bronze setting on a bronze chain, 93 g.")


def elixir_flask():
    p = fs.FLASK
    R, nr = p["bulb_diameter"] * mm / 2, p["neck_diameter"] * mm / 2
    zc = R + p["base_height"] * mm
    top = zc + R * 0.9
    shapes = [{"name": "bulb", "type": "sphere", "size": (R,), "pos": (0, 0, zc), "material": "soda-lime glass",
               "rgba": (0.7, 0.85, 0.9, 0.4)},
              {"name": "neck", "type": "cylinder", "size": (nr, p["neck_length"] * mm / 2),
               "pos": (0, 0, top + p["neck_length"] * mm / 2), "material": "soda-lime glass", "rgba": (0.7, 0.85, 0.9, 0.4)},
              {"name": "cork", "type": "cylinder", "size": (nr, p["cork_length"] * mm / 2),
               "pos": (0, 0, top + p["neck_length"] * mm), "material": "cork", "rgba": WOOD},
              {"name": "elixir", "type": "sphere", "size": (R * 0.8,), "pos": (0, 0, CAD["elixir_flask/elixir"][1][2] * mm),
               "material": "elixir", "rgba": (0.5, 0.1, 0.7, 0.8)}]
    return _object("elixir_flask", shapes, "elixir_flask",
                   "Elixir flask: a glass bulb 60 % full of a purple elixir, corked, 0.80 kg.")


BUILDERS = {"chest": chest, "crate": crate, "sword": sword, "shield": shield, "coin": coin, "amulet": amulet,
            "elixir_flask": elixir_flask}


def materials_block(obj) -> dict:
    """Yax ``materials``: the catalogue entries the object is made of."""
    mats = {c["material"] for c in composition(obj).values()}
    return {k: dict(v) for k, v in fs.MATERIALS.items() if k in mats}


def parts(obj):
    """A parts function for ``machines.register``: materials (00mat), each link's main material, the composition
    (part volumes, masses, CAD centres of mass) and, for the chest, the lid's motions."""
    def fn(m) -> dict:
        comp = composition(obj)
        out = {"materials": materials_block(obj), "composition": comp,
               "tags": {"data:composition*": "cad"}}
        by_mass = max((c for k, c in comp.items() if not k.startswith("lid_")), key=lambda c: c["mass_kg"])["material"]
        m.links[0].data["material"] = by_mass
        if obj == "chest":
            import fantasy_motions as fm
            lid = [lk for lk in m.links if lk.name == "lid"]
            if lid:
                lid[0].data["material"] = "oak"
            block = fm.parts_chest(m)
            out["motions"] = block["motions"]
            out["tags"].update(block["tags"])
        return out
    return fn


# one parts function per object, by name, for ``machines.py``
parts_chest, parts_crate, parts_sword, parts_shield, parts_coin, parts_amulet, parts_elixir_flask = (
    parts(o) for o in ("chest", "crate", "sword", "shield", "coin", "amulet", "elixir_flask"))
