"""Two fictional extraterrestrial vessels for the sea game: non-functional exterior models (simple solids, lofts,
extrusions, chamfers and fillets; nothing inside), for a real-time 3D naval simulation. Their flag and hull emblem is
a white alien head with large almond-shaped black eyes above two crossed bones, on black (``emblem``).

* **AlienShip** - a 140 m surface vessel: an elongated, angular hull with a sharp raised bow, sloping sides
  (tumblehome above the chine) and a broad flat transom; a stepped, faceted superstructure (56 x 13 m) with a bridge
  and its narrow panoramic window band; a prominent geometric sensor tower; the flag on the tower; the emblem on both
  sides of the bow. Dark graphite hull, medium-grey superstructure, black windows.
* **AlienSubmarine** - a 95 m submarine: a body of revolution (rounded bow, cylindrical middle, tapered stern), an
  angular slightly swept sail (12 x 4 x 7 m) with sail planes, cruciform stern fins, decorative sensor masts; the
  emblem on both sides of the sail; a small flag on a mast for the surfaced configuration. Matte graphite-black hull,
  dark-grey secondary surfaces.

Every dimension is a parameter (metres in the parameters, mm in the CAD). x forward (the bow at +x), y left, z up;
the ship's waterline at z = 0 and its keel below; the submarine's axis at z = 0. The only physics is the hull's
volume: the ship's displacement below its waterline, the submarine's submerged displacement (``sizing``) - the mass
a rigid body of it has. ``part`` selects one component; ``vessel`` is all of them.
"""
from __future__ import annotations

import math

try:
    import cadquery as cq
except ImportError:
    cq = None
from vegeta.dedalus import Design, Parameter

RHO_SW = 1025.0
COLOURS = {"hull": "#2b2d30", "superstructure": "#7a7d80", "bridge": "#7a7d80", "windows": "#050505", "tower": "#6b6e72",
           "emblem": "#f2f2f2", "emblem_eyes": "#050505", "flag": "#0a0a0a", "flag_emblem": "#f2f2f2",
           "sub_hull": "#1c1d1f", "sail": "#1c1d1f", "fins": "#3a3c3f", "masts": "#3a3c3f"}


# ------------------------------------------------------------------------------------------------------------ emblem
def _almond(a, b):
    """An almond (pointed at both ends) of half-length ``a`` and half-height ``b``, centred, along x."""
    return cq.Workplane("XY").moveTo(-a, 0).threePointArc((0, b), (a, 0)).threePointArc((0, -b), (-a, 0)).close()


def emblem_2d(height: float, thick: float):
    """The emblem in the XY plane (x right, y up), ``height`` its full height, extruded ``thick`` along +z, centred:
    (white: the head with its eye holes and the crossed bones; black: the two eyes). Proportions measured on the
    owner's flag image (the head's width the unit ``s``): the head 1.27 s tall, its eyes almonds 0.4 s long slanted
    38 deg, centred 0.53 s above the chin and 0.23 s off the middle; the bones 1.4 s long, 0.09 s thick, at 30 deg,
    crossing 0.21 s under the chin, each end two knobs."""
    s = height / 1.98                                          # crown 1.48 s above the crossing, bones 0.50 s below it
    half = [(0.0, 0.0), (0.10, 0.04), (0.27, 0.24), (0.42, 0.52), (0.50, 0.80), (0.47, 1.00), (0.36, 1.15), (0.19, 1.24)]
    pts = [(x, y) for x, y in half] + [(0.0, 1.27)] + [(-x, y) for x, y in reversed(half[1:])]
    y0 = 0.21                                                   # the chin above the bones' crossing
    head = cq.Workplane("XY").spline([(x * s, (y + y0) * s) for x, y in pts], periodic=True).close().extrude(thick)
    eyes = None
    for sx in (1, -1):                                          # slanted almonds, outer corners up
        e = (_almond(0.20 * s, 0.115 * s).extrude(thick * 1.2).rotate((0, 0, 0), (0, 0, 1), sx * 38)
             .translate((sx * 0.23 * s, (0.53 + y0) * s, -thick * 0.1)))
        eyes = e if eyes is None else eyes.union(e)
    head = head.cut(eyes)
    bones = None
    for sgn in (1, -1):                                         # two bones crossing under the chin, knobbed ends
        L, w, r = 1.40 * s, 0.09 * s, 0.085 * s
        b = cq.Workplane("XY").rect(L, w).extrude(thick)
        for ex in (-L / 2, L / 2):
            for ey in (-0.06 * s, 0.06 * s):
                b = b.union(cq.Workplane("XY").center(ex, ey).circle(r).extrude(thick))
        b = b.rotate((0, 0, 0), (0, 0, 1), sgn * 30)
        bones = b if bones is None else bones.union(b)
    white = head.union(bones).translate((0, -0.49 * s, 0))            # centred on its height
    black = eyes.translate((0, -0.49 * s, 0)).intersect(
        cq.Workplane("XY").box(10 * s, 10 * s, thick).translate((0, 0, thick / 2)))
    return white, black


def place(shape, origin, normal, tilt_deg: float = 0.0):
    """Place an XY-plane plate (extruded along +z) on a surface: its +z along ``normal`` ("+y" or "-y" on a side),
    tilted ``tilt_deg`` about the vessel's x axis (a sloping side), its centre at ``origin`` (mm)."""
    if normal == "+y":
        s = shape.rotate((0, 0, 0), (1, 0, 0), -90).rotate((0, 0, 0), (0, 0, 1), 180)    # face out to port, x forward
        s = s.rotate((0, 0, 0), (1, 0, 0), tilt_deg)
    else:
        s = shape.rotate((0, 0, 0), (1, 0, 0), 90)                                       # face out to starboard
        s = s.rotate((0, 0, 0), (1, 0, 0), -tilt_deg)
    return s.translate(origin)


def flag(width: float, height: float, thick: float, mast_top, pole_h: float, pole_r: float):
    """A black flag on a pole (``mast_top`` the pole's foot), the emblem on both faces: (black: cloth and pole,
    white: the emblems, black: their eyes)."""
    x0, y0, z0 = mast_top
    pole = cq.Workplane("XY").add(cq.Solid.makeCylinder(pole_r, pole_h, cq.Vector(x0, y0, z0)))
    cloth = cq.Workplane("XY").box(width, thick, height).translate((x0 - pole_r - width / 2, y0, z0 + pole_h - height / 2))
    whites, blacks = None, None
    for side in ("+y", "-y"):
        w, b = emblem_2d(height * 0.82, thick * 0.6)
        sy = 1 if side == "+y" else -1
        o = (x0 - pole_r - width / 2, y0 + sy * thick / 2, z0 + pole_h - height / 2)
        w, b = place(w, o, side), place(b, o, side)
        whites = w if whites is None else whites.union(w)
        blacks = b if blacks is None else blacks.union(b)
    return pole.union(cloth).union(blacks), whites


# ------------------------------------------------------------------------------------------------------------- ship
class AlienShip(Design):
    parameters = [
        Parameter("part", "vessel", choices=("vessel", "hull", "superstructure", "bridge", "windows", "tower", "emblem", "emblem_eyes",
                                             "flag", "flag_emblem")),
        Parameter("length", 140.0, "m"), Parameter("beam", 17.0, "m", description="maximum width"),
        Parameter("freeboard", 8.0, "m", description="the visible hull height at midship"),
        Parameter("draught", 4.5, "m", description="below the waterline (not visible)"),
        Parameter("height", 31.0, "m", description="the total height above the waterline (the flag's top)"),
        Parameter("bow_rise", 3.5, "m", description="the sheer's rise to the bow"),
        Parameter("rake_deg", 32.0, "deg", description="the stem's rake from the vertical: a sharp raked bow"),
        Parameter("tumblehome", 0.84, "", description="the deck edge's half-beam over the chine's: sloping sides"),
        Parameter("super_length", 56.0, "m"), Parameter("super_width", 13.0, "m"), Parameter("super_x", -6.0, "m", description="its centre from midship"),
        Parameter("emblem_h", 6.0, "m", description="the hull emblem's height"),
        Parameter("flag_w", 6.0, "m"),
    ]

    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in AlienShip.parameters} | (over or {})

    @staticmethod
    def section(p, x):
        """The hull's section at x (m from midship): keel, chine (max beam, just above the waterline), deck edge; a
        sharp raised bow (beam to a point, the deck and the keel rising), a broad flat transom."""
        L, B, D, T = p["length"], p["beam"], p["freeboard"], p["draught"]
        s = (x + L / 2) / L                                       # 0 the transom .. 1 the bow
        half = B / 2 * (0.88 + 0.12 * math.sin(math.pi * min(s / 0.55, 1.0))) if s < 0.55 else \
            B / 2 * max(0.02, math.cos(math.pi / 2 * (s - 0.55) / 0.45)) ** 0.9
        deck = D + p["bow_rise"] * max(0.0, (s - 0.6) / 0.4) ** 2
        keel = -T * (1 - max(0.0, (s - 0.8) / 0.2) ** 1.5 * 0.85) * (0.75 if s < 0.04 else 1.0)
        chine = 1.0 + 0.6 * max(0.0, (s - 0.7) / 0.3)
        return half, deck, keel, chine

    def _hull(self, p):
        L = p["length"]
        wires = []
        for k in range(15):
            s = k / 14
            x = -L / 2 + L * min(s, 0.9995)
            half, deck, keel, chine = self.section(p, x)
            th = p["tumblehome"]
            pts = [(0.0, keel), (half * 0.35, keel), (half, chine), (half * th, deck), (-half * th, deck), (-half, chine), (-half * 0.35, keel)]
            wires.append(cq.Wire.makePolygon([cq.Vector(x * 1000, y * 1000, z * 1000) for y, z in pts], close=True))
        hull = cq.Workplane("XY").add(cq.Solid.makeLoft(wires, ruled=True))
        # the raked stem: the forefoot cut away under a line from the bow's deck back to the keel (rake_deg from vertical)
        top = (L / 2 * 1000, (p["freeboard"] + p["bow_rise"] + 1) * 1000)
        drop = (p["freeboard"] + p["bow_rise"] + p["draught"] + 2) * 1000
        foot = (top[0] - drop * math.tan(math.radians(p["rake_deg"])), top[1] - drop)
        wedge = (cq.Workplane("XZ").polyline([top, (top[0] + 5000, top[1]), (top[0] + 5000, foot[1]), foot]).close()
                 .extrude(p["beam"] * 1000, both=True))
        return hull.cut(wedge)

    def _tiers(self, p):
        """The stepped, faceted superstructure: three tiers, each narrower, shorter and chamfered."""
        D = p["freeboard"] * 1000
        x0, Ls, Ws = p["super_x"] * 1000, p["super_length"] * 1000, p["super_width"] * 1000
        tiers = [(Ls, Ws, 5000.0, 0.0), (Ls * 0.68, Ws * 0.82, 4000.0, Ls * 0.10)]
        out, z = None, D
        for Lt, Wt, Ht, dx in tiers:
            t = cq.Workplane("XY").box(Lt, Wt, Ht).edges("|X").chamfer(min(Wt, Ht) * 0.22).edges("|Y and >Z").chamfer(Ht * 0.35)
            t = t.translate((x0 + dx, 0, z + Ht / 2))
            out = t if out is None else out.union(t)
            z += Ht
        return out, z

    def _bridge(self, p):
        """The bridge on the top tier, forward: a faceted block, its front raked."""
        sup, z = self._tiers(p)
        Lb, Wb, Hb = 16000.0, p["super_width"] * 1000 * 0.68, 3600.0
        xb = p["super_x"] * 1000 + p["super_length"] * 1000 * 0.10 + p["super_length"] * 1000 * 0.68 / 2 - Lb / 2
        b = (cq.Workplane("XY").box(Lb, Wb, Hb).edges("|Y and >X and >Z").chamfer(1400).edges("|X and >Z").chamfer(700)
             .translate((xb, 0, z + Hb / 2)))
        return b, (xb, Lb, Wb, z, Hb)

    def _windows(self, p):
        """The narrow panoramic window band round the bridge's front and sides (black, proud of the faces)."""
        _, (xb, Lb, Wb, z, Hb) = self._bridge(p)
        zw = z + Hb * 0.62
        front = cq.Workplane("XY").box(300, Wb * 0.86, 650).translate((xb + Lb / 2 - 900, 0, zw))
        sides = None
        for sy in (1, -1):
            sd = cq.Workplane("XY").box(Lb * 0.7, 200, 650).translate((xb + Lb * 0.05, sy * (Wb / 2 - 900 + 100), zw))
            sides = sd if sides is None else sides.union(sd)
        return front.union(sides)

    def _tower(self, p):
        """The geometric sensor tower behind the bridge: an octagonal frustum, a faceted sensor head, a mast."""
        _, (xb, Lb, Wb, z, Hb) = self._bridge(p)
        xt = xb - Lb / 2 - 2500
        top = (p["height"] - 3.2) * 1000                           # the flag's pole above it
        h1 = (top - z) * 0.62
        base = cq.Workplane("XY").polygon(8, 7000).workplane(offset=h1).polygon(8, 3600).loft().translate((xt, 0, z))
        head = (cq.Workplane("XY").polygon(6, 5200).workplane(offset=1800).polygon(6, 6000).workplane(offset=1800).polygon(6, 2400)
                .loft(ruled=True).translate((xt, 0, z + h1)))
        mast = cq.Workplane("XY").add(cq.Solid.makeCylinder(260, top - z - h1 - 3600, cq.Vector(xt, 0, z + h1 + 3600)))
        return base.union(head).union(mast), (xt, top)

    def _emblems(self, p):
        """The emblem on both sides of the bow, on the sloping side's plane between the chine and the deck edge."""
        x = p["length"] * 0.24
        half, deck, keel, chine = self.section(p, x)
        th = p["tumblehome"]
        tilt = math.degrees(math.atan2(half * (1 - th), deck - chine))
        zc = (chine + deck) / 2 + 0.2
        yc = half - (zc - chine) / (deck - chine) * half * (1 - th) + 0.06
        whites, blacks = None, None
        for side in ("+y", "-y"):
            w, b = emblem_2d(p["emblem_h"] * 1000, 60.0)
            sy = 1 if side == "+y" else -1
            o = (x * 1000, sy * yc * 1000, zc * 1000)
            w, b = place(w, o, side, tilt), place(b, o, side, tilt)
            whites = w if whites is None else whites.union(w)
            blacks = b if blacks is None else blacks.union(b)
        return whites, blacks

    def _flag(self, p):
        _, (xt, top) = self._tower(p)
        H = p["flag_w"] * 1000 * 2 / 3
        return flag(p["flag_w"] * 1000, H, 40.0, (xt, 0, top), p["height"] * 1000 - top, 90)

    @classmethod
    def sizing(cls, over=None):
        """The displacement: the hull's volume below the waterline."""
        p = cls.p(over)
        if cq is None:
            raise ImportError("the volume needs CadQuery")
        hull = cls()._hull(p)
        below = hull.intersect(cq.Workplane("XY").box(p["length"] * 1100, p["beam"] * 1100, p["draught"] * 1000 * 1.2)
                               .translate((0, 0, -p["draught"] * 600))).val().Volume() / 1e9
        return {"length_m": p["length"], "beam_m": p["beam"], "freeboard_m": p["freeboard"], "height_m": p["height"],
                "draught_m": p["draught"], "displaced_m3": round(below, 1), "displacement_t": round(below * RHO_SW / 1000, 1),
                "superstructure_m": [p["super_length"], p["super_width"]], "fictional": True, "functional": False}

    def build(self, p):
        if cq is None:
            raise ImportError("the alien ship's CAD needs CadQuery")
        parts = {"hull": lambda: self._hull(p), "superstructure": lambda: self._tiers(p)[0], "bridge": lambda: self._bridge(p)[0],
                 "windows": lambda: self._windows(p), "tower": lambda: self._tower(p)[0], "emblem": lambda: self._emblems(p)[0],
                 "emblem_eyes": lambda: self._emblems(p)[1], "flag": lambda: self._flag(p)[0], "flag_emblem": lambda: self._flag(p)[1]}
        if p["part"] in parts:
            return parts[p["part"]]()
        return cq.Workplane("XY").newObject([cq.Compound.makeCompound([f().val() for f in parts.values()])])


# -------------------------------------------------------------------------------------------------------- submarine
class AlienSubmarine(Design):
    parameters = [
        Parameter("part", "vessel", choices=("vessel", "sub_hull", "sail", "fins", "masts", "emblem", "emblem_eyes", "flag", "flag_emblem")),
        Parameter("length", 95.0, "m"), Parameter("diameter", 10.0, "m", description="the hull's width and height"),
        Parameter("bow_length", 14.0, "m", description="the rounded bow (elliptic)"),
        Parameter("stern_length", 30.0, "m", description="the tapered stern"),
        Parameter("sail_length", 12.0, "m"), Parameter("sail_width", 4.0, "m"), Parameter("sail_height", 7.0, "m", description="above the hull"),
        Parameter("sail_x", 18.0, "m", description="the sail's centre forward of midship"),
        Parameter("sail_sweep_deg", 12.0, "deg", description="the sail's leading edge swept back"),
        Parameter("emblem_h", 4.0, "m"), Parameter("flag_w", 2.4, "m"),
    ]

    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in AlienSubmarine.parameters} | (over or {})

    def _hull(self, p):
        """The body of revolution about x: an elliptic bow, a cylinder, a tapered stern to a small cone."""
        L, R = p["length"] * 1000, p["diameter"] * 500
        lb, ls = p["bow_length"] * 1000, p["stern_length"] * 1000
        x_bow, x_st = L / 2, -L / 2
        bow = [(x_bow - lb + lb * math.cos(t), R * math.sin(t)) for t in (k / 12 * math.pi / 2 for k in range(1, 13))]   # the tip to the shoulder
        stern = [(x_st + ls * (1 - u), R * ((1 - u) ** 1.6 * 0.92 + 0.08)) for u in (k / 12 for k in range(1, 13))]      # to r = 0.08 R
        w = (cq.Workplane("XY").moveTo(x_bow, 0).spline(bow, includeCurrent=True).lineTo(x_st + ls, R)
             .spline(stern, includeCurrent=True).lineTo(x_st, 0).close())
        return w.revolve(360, (0, 0, 0), (1, 0, 0))

    def _sail(self, p):
        """The angular sail: a loft from its root on the hull to a smaller, swept-back, chamfered crown."""
        R = p["diameter"] * 500
        Ls, Ws, Hs = p["sail_length"] * 1000, p["sail_width"] * 1000, p["sail_height"] * 1000
        x0 = p["sail_x"] * 1000
        sweep = Hs * math.tan(math.radians(p["sail_sweep_deg"]))

        def sec(L, W, nose):
            return [(L / 2, 0), (L / 2 - nose, W / 2), (-L / 2 + W * 0.3, W / 2), (-L / 2, 0), (-L / 2 + W * 0.3, -W / 2), (L / 2 - nose, -W / 2)]
        root = cq.Wire.makePolygon([cq.Vector(x0 + x, y, R - 800) for x, y in sec(Ls * 1.05, Ws, Ws * 0.9)], close=True)
        crown = cq.Wire.makePolygon([cq.Vector(x0 - sweep + x, y, R + Hs) for x, y in sec(Ls * 0.82, Ws * 0.78, Ws * 0.6)], close=True)
        return cq.Workplane("XY").add(cq.Solid.makeLoft([root, crown], ruled=True)), (x0, sweep, R, Hs, Ls, Ws)

    def _fins(self, p):
        """The sail planes and the cruciform stern fins: thin trapezoids."""
        _, (x0, sweep, R, Hs, Ls, Ws) = self._sail(p)
        out = None
        for sy in (1, -1):                                            # the sail planes, two thirds up the sail
            z = R + Hs * 0.6
            pl = cq.Workplane("XY").polyline([(1800, 0), (-1600, 0), (-2400, sy * 3500), (200, sy * 3500)]).close().extrude(220)
            pl = pl.translate((x0 - sweep * 0.6, sy * Ws * 0.36, z))
            out = pl if out is None else out.union(pl)
        xs = -p["length"] * 500 + p["stern_length"] * 1000 * 0.22
        rs = R * 0.35
        for ang in (0, 90, 180, 270):                                 # the cruciform stern fins
            f = cq.Workplane("XZ").polyline([(2600, 0), (-2200, 0), (-3200, 4200), (-200, 4200)]).close().extrude(-130, both=True)
            f = f.translate((xs, 0, rs * 0.6)).rotate((0, 0, 0), (1, 0, 0), ang)
            out = out.union(f)
        return out

    def _masts(self, p):
        """Decorative sensor masts on the sail's crown: a periscope-like column, a faceted sensor mast, a whip; the
        flag's mast aft."""
        _, (x0, sweep, R, Hs, Ls, Ws) = self._sail(p)
        z = R + Hs
        xc = x0 - sweep
        a = cq.Workplane("XY").add(cq.Solid.makeCylinder(220, 2600, cq.Vector(xc + 2600, 0, z)))
        b = cq.Workplane("XY").polygon(6, 700).workplane(offset=2000).polygon(6, 380).loft().translate((xc + 700, 0, z))
        c = cq.Workplane("XY").add(cq.Solid.makeCylinder(60, 3200, cq.Vector(xc - 900, 500, z)))
        return a.union(b).union(c)

    def _emblems(self, p):
        _, (x0, sweep, R, Hs, Ls, Ws) = self._sail(p)
        zc = R + Hs * 0.42
        whites, blacks = None, None
        tilt = math.degrees(math.atan2(Ws * 0.11, Hs + 800))           # the sail's sides lean in
        for side in ("+y", "-y"):
            w, b = emblem_2d(p["emblem_h"] * 1000, 40.0)
            sy = 1 if side == "+y" else -1
            yc = Ws / 2 * (1 - 0.22 * (zc - R + 800) / (Hs + 800)) + 30
            o = (x0 - sweep * 0.45 - 600, sy * yc, zc)
            w, b = place(w, o, side, tilt), place(b, o, side, tilt)
            whites = w if whites is None else whites.union(w)
            blacks = b if blacks is None else blacks.union(b)
        return whites, blacks

    def _flag(self, p):
        _, (x0, sweep, R, Hs, Ls, Ws) = self._sail(p)
        top = (x0 - sweep - Ls * 0.82 / 2 + 1200, 0, R + Hs)
        return flag(p["flag_w"] * 1000, p["flag_w"] * 1000 * 2 / 3, 30.0, top, 2600.0, 50)

    @classmethod
    def sizing(cls, over=None):
        """The submerged displacement: the hull's and the sail's volume."""
        p = cls.p(over)
        if cq is None:
            raise ImportError("the volume needs CadQuery")
        d = cls()
        v = (d._hull(p).val().Volume() + d._sail(p)[0].val().Volume()) / 1e9
        return {"length_m": p["length"], "diameter_m": p["diameter"], "height_m": p["diameter"] + p["sail_height"],
                "sail_m": [p["sail_length"], p["sail_width"], p["sail_height"]], "displaced_m3": round(v, 1),
                "displacement_t": round(v * RHO_SW / 1000, 1), "fictional": True, "functional": False}

    def build(self, p):
        if cq is None:
            raise ImportError("the alien submarine's CAD needs CadQuery")
        parts = {"sub_hull": lambda: self._hull(p), "sail": lambda: self._sail(p)[0], "fins": lambda: self._fins(p),
                 "masts": lambda: self._masts(p), "emblem": lambda: self._emblems(p)[0], "emblem_eyes": lambda: self._emblems(p)[1],
                 "flag": lambda: self._flag(p)[0], "flag_emblem": lambda: self._flag(p)[1]}
        if p["part"] in parts:
            return parts[p["part"]]()
        return cq.Workplane("XY").newObject([cq.Compound.makeCompound([f().val() for f in parts.values()])])


SHIP_PARTS = ("hull", "superstructure", "bridge", "windows", "tower", "emblem", "emblem_eyes", "flag", "flag_emblem")
SUB_PARTS = ("sub_hull", "sail", "fins", "masts", "emblem", "emblem_eyes", "flag", "flag_emblem")


if __name__ == "__main__":
    print(AlienShip.sizing())
    print(AlienSubmarine.sizing())
