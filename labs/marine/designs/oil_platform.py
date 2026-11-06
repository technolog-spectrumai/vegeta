"""The oil field Ketos Oil works on: a fixed steel jacket in ~110 m of North Sea water, the DP support vessel the
work-class ROV is launched from (its LARS: an A-frame over the side), the tether management system (TMS) the ROV
rides down in, and a subsea manifold with an ROV valve panel. None is a real installation; the numbers are class
values for a central-North-Sea 4-legged drilling/production jacket (assumed).

* ``Jacket`` - 4 battered legs (1:``batter`` on both faces), horizontal frames at ``levels``, an X-brace in every
  bay of every face, node cans where the braces land on the legs, stand-off anodes on every brace and leg bay, a
  2 x 3 conductor bundle in the middle, two risers and a caisson clamped outside the legs, the topside deck on deck
  legs. ``members(p)`` is the geometry as data (every tube from end to end with its diameter, the nodes, the
  anodes) - what Ketos builds its inspection panels from, so the card and the mesh are the same structure.
* ``SupportVessel`` - an ~90 m IMR/ROV support vessel on DP: the merchant hull's lines at 90 x 20 m, the
  accommodation forward, the hangar and the A-frame LARS on the starboard side, the work deck aft.
* ``Tms`` - a top-hat TMS: the tether drum under a steel hat, the ROV latched underneath; 250 m excursion tether.
* ``Manifold`` - a 4-slot production manifold (12 x 8 x 4 m) under a protection roof, its headers, and the ROV
  panel on its south face: six class-4 rotary valve buckets (ISO 13628-8) and two hot-stab receptacles.

Layout (all): x east, y north, z up; z = 0 the sea surface, the jacket's centre at x = y = 0 (the vessel and the
TMS: their own centre and waterline / middle height). Metres in the parameters, millimetres in the CAD.
"""
from __future__ import annotations

import math

try:
    import cadquery as cq
except ImportError:
    cq = None

from vegeta.dedalus import Design, Parameter

MM = 1000.0
LEGS = {"A1": (-1, -1), "A2": (1, -1), "B1": (-1, 1), "B2": (1, 1)}
FACES = {"S": ("A1", "A2"), "E": ("A2", "B2"), "N": ("B2", "B1"), "W": ("B1", "A1")}


def _tube(a, b, d):
    """A cylinder from a to b (metres) of diameter d (metres), in mm."""
    va, vb = cq.Vector(*[c * MM for c in a]), cq.Vector(*[c * MM for c in b])
    ax = vb - va
    return cq.Solid.makeCylinder(d * MM / 2, ax.Length, va, ax.normalized())


def _compound(solids):
    return cq.Workplane("XY").newObject([cq.Compound.makeCompound(solids)])


class Jacket(Design):
    parameters = [
        Parameter("part", "jacket", choices=("jacket", "structure", "anodes", "conductors", "risers", "topside")),
        Parameter("water_depth", 110.0, "m"),
        Parameter("top", 12.0, "m", description="the jacket's top (the deck legs above it)"),
        Parameter("half_top", 10.0, "m", description="half the leg spacing at the top (a 20 x 20 m square)"),
        Parameter("batter", 10.0, "", description="leg batter 1:n on both faces"),
        Parameter("levels", "12,-10,-35,-65,-110", description="horizontal frames (m, the last the mudline)"),
        Parameter("leg_d", 2.0, "m"),
        Parameter("brace_d", 0.9, "m"),
        Parameter("horizontal_d", 0.8, "m"),
        Parameter("can_d", 2.4, "m", description="the node cans on the legs"),
        Parameter("conductors", 6, min=0, max=12),
        Parameter("conductor_d", 0.762, "m", description="30 in"),
        Parameter("riser_d", 0.406, "m", description="16 in"),
        Parameter("caisson_d", 1.0, "m"),
        Parameter("anode_kg", 350.0, "kg", description="each aluminium stand-off anode, new"),
        Parameter("deck", 20.0, "m", description="the cellar deck's level"),
    ]

    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in Jacket.parameters} | (over or {})

    @staticmethod
    def level_list(p):
        lv = [float(v) for v in str(p["levels"]).split(",")]
        lv[-1] = -float(p["water_depth"])
        return lv

    @staticmethod
    def leg_at(p, leg, z):
        sx, sy = LEGS[leg]
        h = p["half_top"] + (p["top"] - z) / p["batter"]
        return (sx * h, sy * h, z)

    @classmethod
    def members(cls, p=None):
        """The structure as data: {"members": [{id, kind, a, b, d}], "nodes": [{id, p, kind}], "anodes": [{id,
        member, p, kg}], "levels", "water_depth"} (metres)."""
        p = cls.p(p)
        lv = cls.level_list(p)
        r3 = lambda q: [round(c, 3) for c in q]  # noqa: E731
        mem, nodes, anodes = [], [], []
        for leg in LEGS:
            mem.append({"id": f"DL-{leg}", "kind": "deck_leg", "a": r3(cls.leg_at(p, leg, p["top"])),
                        "b": r3(cls.leg_at(p, leg, p["deck"])), "d": p["leg_d"]})
            for k in range(len(lv) - 1):
                mem.append({"id": f"L-{leg}-{k + 1}", "kind": "leg", "a": r3(cls.leg_at(p, leg, lv[k])),
                            "b": r3(cls.leg_at(p, leg, lv[k + 1])), "d": p["leg_d"]})
            for k, z in enumerate(lv):
                nodes.append({"id": f"N-{leg}-{k}", "p": r3(cls.leg_at(p, leg, z)), "kind": "leg", "level": k})
        for k, z in enumerate(lv):
            for f, (a, b) in FACES.items():
                mem.append({"id": f"H-{k}-{f}", "kind": "horizontal", "a": r3(cls.leg_at(p, a, z)),
                            "b": r3(cls.leg_at(p, b, z)), "d": p["horizontal_d"]})
            if 0 < k < len(lv) - 1:
                mem.append({"id": f"H-{k}-D", "kind": "horizontal", "a": r3(cls.leg_at(p, "A1", z)),
                            "b": r3(cls.leg_at(p, "B2", z)), "d": p["horizontal_d"]})
        for f, (a, b) in FACES.items():
            for k in range(len(lv) - 1):
                if lv[k] > 0 and lv[k + 1] > 0:
                    continue
                p00, p01 = cls.leg_at(p, a, lv[k]), cls.leg_at(p, b, lv[k])
                p10, p11 = cls.leg_at(p, a, lv[k + 1]), cls.leg_at(p, b, lv[k + 1])
                mid = tuple((p00[i] + p01[i] + p10[i] + p11[i]) / 4 for i in range(3))
                nodes.append({"id": f"X-{f}-{k + 1}", "p": r3(mid), "kind": "x", "level": k + 0.5})
                for s, (u, w) in (("a", (p00, p11)), ("b", (p01, p10))):
                    mem.append({"id": f"X-{f}-{k + 1}{s}", "kind": "brace", "a": r3(u), "b": r3(w), "d": p["brace_d"]})
        # the anodes: two on every brace and diagonal horizontal under water, one per leg bay, 0.6 m off the wall
        for m in mem:
            if m["kind"] not in ("brace", "leg") or max(m["a"][2], m["b"][2]) > -5:
                continue
            fs = (0.5,) if m["kind"] == "leg" else (0.25, 0.75)
            for i, f in enumerate(fs):
                q = [m["a"][j] + (m["b"][j] - m["a"][j]) * f for j in range(3)]
                c = math.hypot(q[0], q[1]) or 1.0
                off = m["d"] / 2 + 0.35
                q = [q[0] + q[0] / c * off, q[1] + q[1] / c * off, q[2]]
                anodes.append({"id": f"AN-{m['id']}-{i}", "member": m["id"], "p": r3(q), "kg": p["anode_kg"]})
        # conductors, risers, a caisson
        n = int(p["conductors"])
        for i in range(n):
            x, y = -2.5 + 2.5 * (i % 3), -1.25 + 2.5 * (i // 3)
            mem.append({"id": f"C-{i + 1}", "kind": "conductor", "a": [x, y, p["deck"]], "b": [x, y, -p["water_depth"]],
                        "d": p["conductor_d"]})
        for i, leg, out in ((1, "A2", (2.2, -1.2)), (2, "A2", (2.2, 1.2))):
            top, bot = cls.leg_at(p, leg, p["deck"]), cls.leg_at(p, leg, -p["water_depth"] + 2)
            a = [top[0] + out[0], top[1] + out[1], p["deck"]]
            b = [bot[0] + out[0], bot[1] + out[1], -p["water_depth"] + 2]
            mem.append({"id": f"R-{i}", "kind": "riser", "a": r3(a), "b": r3(b), "d": p["riser_d"]})
            mem.append({"id": f"R-{i}-spool", "kind": "riser", "a": r3(b), "b": r3([b[0] + 30, b[1], b[2]]), "d": p["riser_d"]})
        top, bot = cls.leg_at(p, "B1", p["deck"]), cls.leg_at(p, "B1", -60)
        mem.append({"id": "CS-1", "kind": "caisson", "a": r3([top[0] - 2.5, top[1], p["deck"]]),
                    "b": r3([bot[0] - 2.5, bot[1], -60]), "d": p["caisson_d"]})
        return {"members": mem, "nodes": nodes, "anodes": anodes, "levels": lv, "water_depth": p["water_depth"]}

    @classmethod
    def sizing(cls, over=None):
        g = cls.members(over)
        steel = 0.0
        for m in g["members"]:
            L = math.dist(m["a"], m["b"])
            t = m["d"] / 40                                      # D/t 40, class value
            steel += math.pi * m["d"] * t * L * 7850 / 1000
        wet = sum(math.pi * m["d"] * math.dist(m["a"], m["b"]) for m in g["members"] if min(m["a"][2], m["b"][2]) < 0)
        return {"members": len(g["members"]), "nodes": len(g["nodes"]), "anodes": len(g["anodes"]),
                "steel_t": round(steel), "wetted_m2": round(wet)}

    def build(self, p):
        if cq is None:
            raise ImportError("the jacket's CAD needs CadQuery")
        g = self.members(p)
        groups = {"structure": [], "anodes": [], "conductors": [], "risers": [], "topside": []}
        for m in g["members"]:
            k = {"conductor": "conductors", "riser": "risers", "caisson": "risers"}.get(m["kind"], "structure")
            groups[k].append(_tube(m["a"], m["b"], m["d"]))
        for nd in g["nodes"]:
            if nd["kind"] == "leg":
                q = nd["p"]
                a = (q[0], q[1], q[2] + 2.0)
                b = (q[0], q[1], max(q[2] - 2.0, -p["water_depth"]))
                groups["structure"].append(_tube(a, b, p["can_d"]))
        for an in g["anodes"]:
            x, y, z = (c * MM for c in an["p"])
            groups["anodes"].append(cq.Solid.makeBox(260, 260, 2400, cq.Vector(x - 130, y - 130, z - 1200)))
        h = p["half_top"] + (p["top"] - p["deck"]) / p["batter"] + 6
        groups["topside"].append(cq.Solid.makeBox(2 * h * MM, 2 * h * MM, 8 * MM, cq.Vector(-h * MM, -h * MM, p["deck"] * MM)))
        groups["topside"].append(cq.Solid.makeBox(24 * MM, 18 * MM, 14 * MM, cq.Vector(-14 * MM, -9 * MM, (p["deck"] + 8) * MM)))
        groups["topside"].append(cq.Solid.makeBox(6 * MM, 6 * MM, 40 * MM, cq.Vector(4 * MM, 4 * MM, (p["deck"] + 8) * MM)))
        if p["part"] in groups:
            return _compound(groups[p["part"]])
        return _compound([s for v in groups.values() for s in v])


class SupportVessel(Design):
    parameters = [
        Parameter("part", "vessel", choices=("vessel", "hull", "house", "lars")),
        Parameter("length", 90.0, "m"),
        Parameter("beam", 20.0, "m"),
        Parameter("draft", 6.5, "m"),
        Parameter("depth", 9.0, "m"),
        Parameter("lars_x", -8.0, "m", description="the A-frame along the ship (from midships)"),
        Parameter("dp_wind_m_s", 20.0, "m/s", description="DP holds station up to this mean wind (DP2 class, assumed)"),
        Parameter("dp_current_m_s", 1.5, "m/s"),
        Parameter("lars_hs_m", 3.0, "m", description="launch and recovery of the ROV up to this Hs"),
        Parameter("day_rate_eur", 65000.0, "EUR/day", description="vessel + ROV spread, class value (assumed)"),
    ]

    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in SupportVessel.parameters} | (over or {})

    def overboard(self, p):
        """Where the LARS lowers the TMS (m, the vessel's frame: x forward, y left, z up)."""
        return (p["lars_x"], -(p["beam"] / 2 + 3.5), 0.0)

    def build(self, p):
        if cq is None:
            raise ImportError("the vessel's CAD needs CadQuery")
        from merchant_hull import MerchantHull
        L, B, D, T = p["length"], p["beam"], p["depth"], p["draft"]
        hp = MerchantHull.p({"length": L, "beam": B, "draft": T, "depth": D})
        hull = MerchantHull().build(hp)
        dz = (D - T) * MM
        house = _compound([cq.Solid.makeBox(22 * MM, (B - 2) * MM, 12 * MM, cq.Vector(14 * MM, -(B / 2 - 1) * MM, dz)),
                           cq.Solid.makeBox(12 * MM, (B - 4) * MM, 4 * MM, cq.Vector(20 * MM, -(B / 2 - 2) * MM, dz + 12 * MM)),
                           cq.Solid.makeBox(12 * MM, 7 * MM, 7 * MM, cq.Vector((p["lars_x"] - 6) * MM, -(B / 2 - 0.5) * MM, dz))])
        x = p["lars_x"] * MM
        y0, y1 = -(B / 2 - 1.0) * MM, -(B / 2 + 3.5) * MM
        top = dz + 11 * MM
        a = []
        for dx in (-2.5 * MM, 2.5 * MM):
            va, vb = cq.Vector(x + dx, y0, dz), cq.Vector(x + dx, y1, top)
            a.append(cq.Solid.makeCylinder(350, (vb - va).Length, va, (vb - va).normalized()))
        a.append(cq.Solid.makeCylinder(400, 5.6 * MM, cq.Vector(x - 2.8 * MM, y1, top), cq.Vector(1, 0, 0)))
        a.append(cq.Solid.makeBox(4 * MM, 3 * MM, 3 * MM, cq.Vector(x - 2 * MM, y0 + 2 * MM, dz)))     # the umbilical winch
        lars = _compound(a)
        parts = {"hull": hull, "house": house, "lars": lars}
        if p["part"] in parts:
            return parts[p["part"]]
        return _compound([hull.val(), house.val(), lars.val()])


class Tms(Design):
    parameters = [
        Parameter("diameter", 2.6, "m"),
        Parameter("height", 1.8, "m"),
        Parameter("tether_m", 250.0, "m"),
        Parameter("tether_d", 0.032, "m"),
        Parameter("umbilical_d", 0.042, "m", description="the vessel's lift umbilical"),
        Parameter("mass_kg", 3200.0, "kg"),
    ]

    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in Tms.parameters} | (over or {})

    def build(self, p):
        if cq is None:
            raise ImportError("the TMS's CAD needs CadQuery")
        D, H = p["diameter"] * MM, p["height"] * MM
        s = [cq.Solid.makeCylinder(D / 2, H * 0.7, cq.Vector(0, 0, -H / 2 + H * 0.3)),
             cq.Solid.makeCylinder(D / 2 + 400, 120, cq.Vector(0, 0, -H / 2 + H * 0.3)),          # the hat's brim
             cq.Solid.makeCylinder(250, H * 0.3, cq.Vector(0, 0, -H / 2)),                         # the latch
             cq.Solid.makeCylinder(300, 600, cq.Vector(0, 0, H / 2 - 100))]                        # the bullet
        return _compound(s)


class Manifold(Design):
    parameters = [
        Parameter("part", "manifold", choices=("manifold", "frame", "panel")),
        Parameter("length", 12.0, "m"),
        Parameter("width", 8.0, "m"),
        Parameter("height", 4.0, "m"),
        Parameter("valves", 6, min=1, max=12, description="class-4 rotary buckets on the ROV panel"),
        Parameter("hot_stabs", 2, min=0, max=4),
    ]

    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in Manifold.parameters} | (over or {})

    @staticmethod
    def panel(p):
        """The ROV panel on the south face (y = -W/2): valves and hot stabs (m, the manifold's frame, z up from the
        mudmat), the ROV's docking bars 0.5 m in front of them."""
        n, L, W = int(p["valves"]), p["length"], p["width"]
        valves = [{"id": f"V{i + 1}", "p": [round(-L / 4 + i * (L / 2) / max(n - 1, 1), 3), -W / 2 - 0.25, 1.6],
                   "kind": "class4", "torque_Nm": 2700, "turns": 18 + 2 * (i % 3), "open": i % 2 == 0} for i in range(n)]
        stabs = [{"id": f"HS{i + 1}", "p": [round(-L / 4 + i * 1.5, 3), -W / 2 - 0.25, 2.6], "kind": "hot_stab"}
                 for i in range(int(p["hot_stabs"]))]
        return {"valves": valves, "hot_stabs": stabs}

    def build(self, p):
        if cq is None:
            raise ImportError("the manifold's CAD needs CadQuery")
        L, W, H = p["length"] * MM, p["width"] * MM, p["height"] * MM
        f = [cq.Solid.makeBox(L, W, 300, cq.Vector(-L / 2, -W / 2, 0))]                           # the mudmat
        for sx in (-1, 1):
            for sy in (-1, 1):
                f.append(cq.Solid.makeBox(300, 300, H, cq.Vector(sx * (L / 2 - 150) - 150, sy * (W / 2 - 150) - 150, 0)))
        f.append(cq.Solid.makeBox(L, W, 200, cq.Vector(-L / 2, -W / 2, H)))                        # the roof
        for y in (-W / 4, 0, W / 4):
            f.append(cq.Solid.makeCylinder(170, L - 600, cq.Vector(-L / 2 + 300, y, 1200), cq.Vector(1, 0, 0)))
        pn = []
        pv = self.panel(p)
        pn.append(cq.Solid.makeBox(L / 2 + 1500, 200, 2200, cq.Vector(-L / 4 - 750, -W / 2 - 100, 900)))
        for v in pv["valves"] + pv["hot_stabs"]:
            x, y, z = (c * MM for c in v["p"])
            pn.append(cq.Solid.makeCylinder(160, 350, cq.Vector(x, y + 250, z), cq.Vector(0, -1, 0)))
        frame, panel = _compound(f), _compound(pn)
        if p["part"] == "frame":
            return frame
        if p["part"] == "panel":
            return panel
        return _compound([frame.val(), panel.val()])


class ResidentDock(Design):
    """The resident drone's subsea dock on the jacket (Ketos Oil stage 3): a garage frame clamped to a leg at -80 m,
    an open-ended tube the AUV backs into, an inductive charging coupler and a fibre/wet-mate connection to the
    topside. z = 0 the garage's middle height, x out of the garage's mouth."""
    parameters = [
        Parameter("length", 3.0, "m"),
        Parameter("width", 1.6, "m"),
        Parameter("height", 1.4, "m"),
        Parameter("depth", 80.0, "m"),
        Parameter("charge_kw", 2.0, "kW", description="inductive charger (class value)"),
        Parameter("mass_kg", 1800.0, "kg"),
        Parameter("price_eur", 900000.0, "EUR", description="garage, charger, topside fibre and power (assumed)"),
    ]

    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in ResidentDock.parameters} | (over or {})

    def build(self, p):
        if cq is None:
            raise ImportError("the dock's CAD needs CadQuery")
        L, W, H = p["length"] * MM, p["width"] * MM, p["height"] * MM
        s = []
        for sy in (-1, 1):
            for sz in (-1, 1):
                s.append(cq.Solid.makeCylinder(60, L, cq.Vector(-L / 2, sy * W / 2, sz * H / 2), cq.Vector(1, 0, 0)))
        for x in (-L / 2, 0.0, L / 2 - 120):
            s.append(cq.Solid.makeBox(120, W + 120, 120, cq.Vector(x, -W / 2 - 60, -H / 2 - 60)))
            s.append(cq.Solid.makeBox(120, W + 120, 120, cq.Vector(x, -W / 2 - 60, H / 2 - 60)))
        s.append(cq.Solid.makeBox(200, W * 0.6, H * 0.6, cq.Vector(-L / 2 - 200, -W * 0.3, -H * 0.3)))   # the charger
        s.append(cq.Solid.makeBox(500, 300, H * 1.6, cq.Vector(-L / 2 - 700, -150, -H * 0.8)))           # the leg clamp
        return _compound(s)


class RiserClamp(Design):
    """A two-half riser clamp (the riser's guide on the jacket): bolted halves round the riser, an arm to the leg."""
    parameters = [Parameter("riser_d", 0.406, "m"), Parameter("arm", 1.5, "m"), Parameter("height", 0.6, "m")]

    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in RiserClamp.parameters} | (over or {})

    def build(self, p):
        if cq is None:
            raise ImportError("the clamp's CAD needs CadQuery")
        R, H = p["riser_d"] * MM / 2, p["height"] * MM
        ring = cq.Solid.makeCylinder(R + 80, H, cq.Vector(0, 0, -H / 2)).cut(cq.Solid.makeCylinder(R, H, cq.Vector(0, 0, -H / 2)))
        s = [ring, cq.Solid.makeBox(p["arm"] * MM, 160, H * 0.6, cq.Vector(-p["arm"] * MM - R, -80, -H * 0.3))]
        for sy in (-1, 1):
            s.append(cq.Solid.makeBox(140, 60, H, cq.Vector(-70, sy * (R + 80) - 30, -H / 2)))
        return _compound(s)


class Debris(Design):
    """Dropped objects on the seabed (class shapes): a scaffold tube, an oil drum, a coil of wire rope, a grating."""
    parameters = [Parameter("kind", "drum", choices=("drum", "tube", "coil", "grating"))]

    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in Debris.parameters} | (over or {})

    def build(self, p):
        if cq is None:
            raise ImportError("the debris CAD needs CadQuery")
        k = p["kind"]
        if k == "drum":
            return cq.Workplane("XY").add(cq.Solid.makeCylinder(290, 880, cq.Vector(-440, 0, 290), cq.Vector(1, 0, 0)))
        if k == "tube":
            return cq.Workplane("XY").add(cq.Solid.makeCylinder(24, 6000, cq.Vector(-3000, 0, 24), cq.Vector(1, 0, 0)))
        if k == "coil":
            return cq.Workplane("XY").add(cq.Solid.makeTorus(450, 80, cq.Vector(0, 0, 80)))
        return cq.Workplane("XY").box(1500, 1000, 40).translate((0, 0, 20))


if __name__ == "__main__":
    print(Jacket.sizing())
