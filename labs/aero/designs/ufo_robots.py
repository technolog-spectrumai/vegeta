"""The two UFOs (``ufo.py``) as Chiron robots, for the Yax export (``machines.py``): one rigid link each, its geoms the
card's collision primitives (``ufo.collision``), its mass (the class mass of the aircraft it stands in for) spread
over them by volume. No flight model here: how each moves is the card's ``kinematics`` (``ufo.step``), which rides in
the same Yax file as part data. Root frames as the cards': the saucer's pads' feet at z = 0, the cigar's axis z = 0."""
from __future__ import annotations

import math

from vegeta.chiron import Geom, Link, Robot

#: A local z axis turned onto x (a capsule or cylinder along the cigar).
Z_TO_X = (0.7071068, 0.0, 0.7071068, 0.0)
RGBA = {"saucer": (0.56, 0.6, 0.65, 1.0), "cigar": (0.48, 0.49, 0.49, 1.0)}


def _volume(p: dict) -> float:
    s = p["size"]
    if p["type"] == "ellipsoid":
        return 4 / 3 * math.pi * s[0] * s[1] * s[2]
    if p["type"] == "capsule":
        return math.pi * s[0] ** 2 * 2 * s[1] + 4 / 3 * math.pi * s[0] ** 3
    if p["type"] == "cylinder":
        return math.pi * s[0] ** 2 * 2 * s[1]
    if p["type"] == "box":
        return 8 * s[0] * s[1] * s[2]
    return 4 / 3 * math.pi * s[0] ** 3


def geoms(name: str) -> list:
    import ufo
    prims = ufo.collision(name)
    vol = sum(_volume(p) for p in prims)
    out = []
    for p in prims:
        quat = Z_TO_X if p.get("axis") == [1, 0, 0] else tuple(p.get("quat", (1.0, 0.0, 0.0, 0.0)))
        out.append(Geom(p["name"], p["type"], tuple(p["size"]), pos=tuple(p["pos"]), quat=quat,
                        mass=ufo.CRAFT[name]["mass_kg"] * _volume(p) / vol, role="body", rgba=RGBA[name]))
    return out


def ufo_robot(name: str) -> Robot:
    import ufo
    c = ufo.CRAFT[name]
    r = Robot(name.title(), Link("body", log=True, geoms=geoms(name)),
              notes=f"{c['what']} - fictional game craft: a rigid body of its collision primitives and class mass "
                    f"(moves as a {c['model']} by the card's kinematics; no weapons)",
              sources={"mass": "ufo.CRAFT mass_kg (the class of aircraft it stands in for; ufo.BASIS)",
                       "geoms": "ufo.collision", "looks": "designs/ufo.py"})
    r.validate()
    return r


def saucer() -> Robot:
    return ufo_robot("saucer")


def cigar() -> Robot:
    return ufo_robot("cigar")


def visual(name: str):
    """The craft's CAD for ``machines.register(visual=...)`` (the flag is a textured panel on the Ketos card)."""
    def build():
        import ufo
        return ufo.build(name)
    return build
