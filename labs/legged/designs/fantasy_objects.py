"""Fantasy objects (CAD): a treasure chest with a hinged lid, a wooden crate, a sword, a shield, a coin, an amulet
and an elixir flask. Each ``part`` choice is one material, so the masses come from the CAD volumes × the
densities in ``fantasy_specs.MATERIALS``. Numbers in ``fantasy_specs``; mm, x forward, y left, z up, resting on
the ground at z = 0 (the sword and the shield lying flat, the coin flat, the amulet laid out)."""
import math

import cadquery as cq
from vegeta.dedalus import Design

import fantasy_specs as fs
import shape_kit as sk
from zoo_quadruped import parameters_of


def _choices(spec):
    return tuple(spec["_parts"])


def _params(spec):
    return parameters_of({k: v for k, v in spec.items() if not k.startswith("_")}, _choices(spec))


class Chest(Design):
    """A treasure chest: an oak box of planks (grooves between them) with iron corner bands, a barrel lid (half a
    cylinder) on two hinges at the back, opened ``open_deg``, an iron lock plate with a keyhole and two side
    handles. ``part``: the chest, its oak (box + lid wood) or its iron (bands, hinges, lock, handles)."""

    spec = fs.CHEST
    parameters = _params(fs.CHEST)

    def _lid(self, p):
        """The lid in its hinge frame: hinge axis along y at the origin, the lid closed over +x."""
        L, W, r, t = p["length"], p["width"], p["width"] / 2, p["wall"]
        outer = cq.Workplane("XZ").center(W / 2, 0).circle(r).extrude(L / 2, both=True)
        outer = outer.intersect(cq.Workplane("XY").box(W, L, r).translate((W / 2, 0, r / 2)))
        inner = cq.Workplane("XZ").center(W / 2, 0).circle(r - t).extrude(L / 2 - t, both=True)
        return outer.cut(inner)

    def _lid_iron(self, p):
        L, W, r, b = p["length"], p["width"], p["width"] / 2, p["band_width"]
        bands = None
        for y in (-L / 2 + b * 1.5, L / 2 - b * 1.5, 0.0):
            ring = (cq.Workplane("XZ").center(W / 2, 0).circle(r + 3).circle(r - 1).extrude(b / 2, both=True)
                    .translate((0, y, 0)).intersect(cq.Workplane("XY").box(W + 10, L, r + 5).translate((W / 2, 0, r / 2 + 2))))
            bands = ring if bands is None else bands.union(ring)
        hasp = cq.Workplane("XY").box(16, b * 1.6, 70).translate((W + 4, 0, -20))
        return bands.union(hasp)

    def _place_lid(self, solid, p):
        H = p["height"]
        return (solid.rotate((0, 0, 0), (0, 1, 0), -p["open_deg"])
                .translate((-p["width"] / 2, 0, H)).rotate((0, 0, 0), (0, 0, 1), -90))   # hinge at the back (+y)

    def oak(self, p):
        L, W, H, t = p["length"], p["width"], p["height"], p["wall"]
        box = cq.Workplane("XY").box(W, L, H).translate((0, 0, H / 2))
        box = box.cut(cq.Workplane("XY").box(W - 2 * t, L - 2 * t, H).translate((0, 0, H / 2 + t)))
        n = int(p["planks"])
        for i in range(1, n):                                            # grooves between the planks
            z = H * i / n
            box = box.cut(cq.Workplane("XY").box(W + 2, L + 2, 3).translate((0, 0, z)).cut(
                cq.Workplane("XY").box(W - 3, L - 3, 4).translate((0, 0, z))))
        lid = self._place_lid(self._lid(p), p)
        return box.rotate((0, 0, 0), (0, 0, 1), 90).union(lid)

    def iron(self, p):
        L, W, H, b = p["length"], p["width"], p["height"], p["band_width"]
        parts = []
        for sx in (-1, 1):
            for sy in (-1, 1):                                           # corner bands: two strips per corner
                parts.append(cq.Workplane("XY").box(b, 3, H).translate((sx * (L / 2 - b / 2), sy * (W / 2 + 1.5), H / 2)))
                parts.append(cq.Workplane("XY").box(3, b, H).translate((sx * (L / 2 + 1.5), sy * (W / 2 - b / 2), H / 2)))
        lock = cq.Workplane("XZ").rect(70, 80).extrude(4).translate((0, -W / 2 - 4 + 4, H - 50))
        lock = lock.cut(sk.capsule((0, -W / 2 - 10, H - 40), (0, -W / 2 - 10, H - 60), 5))          # the keyhole
        for sx in (-1, 1):                                               # side handles: half rings
            parts.append(cq.Workplane("YZ").center(0, H * 0.6).circle(30).circle(24).extrude(8, both=True)
                         .intersect(cq.Workplane("XY").box(20, 80, 40).translate((0, 0, H * 0.6 - 20)))
                         .translate((sx * (L / 2 + 6), 0, 0)))
        for sy in (-1, 1):                                               # hinges at the back
            parts.append(cq.Workplane("YZ").circle(8).extrude(30, both=True).translate((sy * L * 0.3, W / 2, H)))
        lid = self._place_lid(self._lid_iron(p), p)
        return sk.union_all(parts + [lock, lid])

    def build(self, p):
        if p["part"] == "oak":
            return self.oak(p)
        if p["part"] == "iron":
            return self.iron(p)
        return self.oak(p).union(self.iron(p))


class Crate(Design):
    """A wooden crate: six slatted pine panels (``slats`` per side, gaps between them) on a square frame, with a
    diagonal brace on each side. ``part``: the crate (all pine)."""

    spec = fs.CRATE
    parameters = _params(fs.CRATE)

    def build(self, p):
        S, t, n, g = p["size"], p["board"], int(p["slats"]), p["gap"]
        w = (S - (n - 1) * g) / n
        parts = []
        for k in range(n):                                               # slats on the four sides, top and bottom
            o = -S / 2 + w / 2 + k * (w + g)
            for sx in (-1, 1):
                parts.append(cq.Workplane("XY").box(t, S, w).translate((sx * (S / 2 - t / 2), 0, S / 2 + o)))
                parts.append(cq.Workplane("XY").box(S, t, w).translate((0, sx * (S / 2 - t / 2), S / 2 + o)))
                parts.append(cq.Workplane("XY").box(S, w, t).translate((0, o, S / 2 + sx * (S / 2 - t / 2))))
        f = p["frame"]
        for sx in (-1, 1):                                               # frame posts
            for sy in (-1, 1):
                parts.append(cq.Workplane("XY").box(f, f, S).translate((sx * (S / 2 - f / 2), sy * (S / 2 - f / 2), S / 2)))
        for sx in (-1, 1):                                               # diagonal braces on the outside
            parts.append(sk.rod((sx * (S / 2 + t / 2), -S / 2 + f, f), (sx * (S / 2 + t / 2), S / 2 - f, S - f), f / 2))
            parts.append(sk.rod((-S / 2 + f, sx * (S / 2 + t / 2), f), (S / 2 - f, sx * (S / 2 + t / 2), S - f), f / 2))
        return sk.union_all(parts)


class Sword(Design):
    """A longsword lying flat on its side: a diamond-section blade tapering to the point, with a fuller (a groove)
    down both faces, a crossguard with flared ends, a grip wrapped in ``grip_rings`` leather rings over the tang
    and a round pommel. x along the blade (the point at +x), z up. ``part``: the sword, the steel (blade, guard,
    pommel) or the leather grip."""

    spec = fs.SWORD
    parameters = _params(fs.SWORD)

    def steel(self, p):
        L, W, T = p["blade_length"], p["blade_width"], p["blade_thickness"]
        z = p["pommel_diameter"] / 2                                      # lying flat, resting on the pommel
        sec = lambda w, t: [(0, -w / 2), (t / 2, 0), (0, w / 2), (-t / 2, 0)]   # noqa: E731  (y, z) diamond
        blade = (cq.Workplane("YZ").polyline([(y, zz) for zz, y in sec(W, T)]).close()
                 .workplane(offset=L * 0.85).polyline([(y, zz) for zz, y in sec(W * 0.75, T * 0.8)]).close()
                 .workplane(offset=L * 0.15).polyline([(y, zz) for zz, y in sec(2, 1)]).close().loft(ruled=True))
        blade = blade.translate((0, 0, z))
        fl = p["fuller_width"]
        for sz in (-1, 1):
            blade = blade.cut(sk.capsule((40, 0, z + sz * T / 2), (L * 0.6, 0, z + sz * T / 2), fl / 2))
        gw = p["guard_width"]
        guard = (cq.Workplane("XY").box(p["guard_thickness"], gw, 14).translate((-p["guard_thickness"] / 2, 0, z))
                 .union(sk.ball((-p["guard_thickness"] / 2, gw / 2, z), 11)).union(sk.ball((-p["guard_thickness"] / 2, -gw / 2, z), 11)))
        gl = p["grip_length"]
        tang = sk.rod((-p["guard_thickness"], 0, z), (-p["guard_thickness"] - gl, 0, z), p["grip_diameter"] * 0.3)
        pommel = sk.ball((-p["guard_thickness"] - gl - p["pommel_diameter"] * 0.4, 0, z), p["pommel_diameter"] / 2)
        return sk.union_all([blade, guard, tang, pommel])

    def leather(self, p):
        z = p["pommel_diameter"] / 2
        g0, gl, d, n = p["guard_thickness"], p["grip_length"], p["grip_diameter"], int(p["grip_rings"])
        grip = sk.rod((-g0, 0, z), (-g0 - gl, 0, z), d / 2).cut(sk.rod((-g0 + 1, 0, z), (-g0 - gl - 1, 0, z), d * 0.3))
        for k in range(n):                                               # wrap ridges
            x = -g0 - gl * (k + 0.5) / n
            grip = grip.union(cq.Workplane("YZ").circle(d / 2 + 1.5).circle(d / 2 - 1).extrude(2.5, both=True)
                              .translate((x, 0, z)))
        return grip

    def build(self, p):
        if p["part"] == "steel":
            return self.steel(p)
        if p["part"] == "leather":
            return self.leather(p)
        return self.steel(p).union(self.leather(p))


class Shield(Design):
    """A shield lying face up: ``shape`` 'round' (a dished disc of oak planks with an iron boss and rim) or
    'heater' (the classic knightly outline, curved across its width, with an iron rim and boss). The boss covers
    the hand grip behind it. ``part``: the shield, its oak or its iron."""

    spec = fs.SHIELD
    parameters = _params(fs.SHIELD)

    def _outline(self, p):
        D, t = p["diameter"], p["thickness"]
        if p["shape_heater"] < 0.5:
            return cq.Workplane("XY").circle(D / 2).extrude(t)
        w, h = D * 0.8, D                                                # heater: a flat top, curved sides to a point
        pts = [(-w / 2, h * 0.45), (w / 2, h * 0.45), (w / 2, 0.0), (0.0, -h * 0.55), (-w / 2, 0.0)]
        return (cq.Workplane("XY").moveTo(*pts[0]).lineTo(*pts[1]).lineTo(*pts[2])
                .threePointArc((w * 0.28, -h * 0.35), pts[3]).threePointArc((-w * 0.28, -h * 0.35), pts[4]).close()
                .extrude(t))

    def _dish(self, p, grow=0.0):
        """Curve the board: keep what lies between two spheres (round) or two cylinders (heater) about one centre,
        radius R and R − thickness; ``grow`` thickens the shell both ways (the rim over the board's edge)."""
        R, t = p["curvature_radius"], p["thickness"]
        z0 = -R + t * 4                                                  # the board's top at the centre: z = 4 t
        if p["shape_heater"] < 0.5:
            outer = cq.Workplane("XY").sphere(R + grow).translate((0, 0, z0))
            inner = cq.Workplane("XY").sphere(R - t - grow).translate((0, 0, z0))
        else:
            outer = cq.Workplane("YZ").circle(R + grow).extrude(p["diameter"], both=True).translate((0, 0, z0))
            inner = cq.Workplane("YZ").circle(R - t - grow).extrude(p["diameter"], both=True).translate((0, 0, z0))
        tall = self._outline(dict(p, thickness=R)).translate((0, 0, -R / 2))   # the outline, deep enough
        return tall.intersect(outer.cut(inner))

    def oak(self, p):
        board = self._dish(p)
        n = int(p["planks"])
        D = p["diameter"]
        for k in range(1, n):                                            # plank seams
            y = -D / 2 + D * k / n
            board = board.cut(cq.Workplane("XY").box(D * 1.2, 2.0, p["thickness"] * 10).translate((0, y, 0)))
        return board.cut(cq.Workplane("XY").circle(p["boss_diameter"] / 2 - 5).extrude(p["thickness"] * 10)
                         .translate((0, 0, -p["thickness"] * 5)))

    def iron(self, p):
        t, D = p["thickness"], p["diameter"]
        rim_in = self._outline(dict(p, diameter=D - 2 * p["rim_width"], thickness=p["curvature_radius"])) \
            .translate((0, 0, -p["curvature_radius"] / 2))
        rim = self._dish(p, grow=1.5).cut(self._dish(p)).cut(rim_in)      # a 1.5 mm skin over the edge, both faces
        b = p["boss_diameter"] / 2
        top = t * 4                                                      # the board's top at the centre
        dome = cq.Workplane("XY").sphere(b).cut(cq.Workplane("XY").sphere(b - 3.0))   # a 3 mm iron dome
        boss = (dome.intersect(cq.Workplane("XY").box(3 * b, 3 * b, b).translate((0, 0, b / 2)))
                .translate((0, 0, top - t - 2)))
        boss = boss.union(cq.Workplane("XY").circle(b + 8).circle(b - 5).extrude(3).translate((0, 0, top - 3)))
        return rim.union(boss)

    def build(self, p):
        if p["part"] == "oak":
            return self.oak(p)
        if p["part"] == "iron":
            return self.iron(p)
        return self.oak(p).union(self.iron(p))


class Coin(Design):
    """A gold coin lying flat: a disc with a raised rim on both faces, a five-pointed star in relief on the obverse,
    a ring of dots on the reverse and a reeded edge (``reeds`` notches). ``part``: the coin (all gold)."""

    spec = fs.COIN
    parameters = _params(fs.COIN)

    def build(self, p):
        D, t, rim = p["diameter"], p["thickness"], p["rim_width"]
        coin = cq.Workplane("XY").circle(D / 2).extrude(t)
        relief = p["relief"]
        for z0, sgn in ((t, 1), (0, -1)):                                 # sunken fields inside the rims
            coin = coin.cut(cq.Workplane("XY").circle(D / 2 - rim).extrude(relief * sgn).translate((0, 0, z0 - relief * sgn)))
        r1, r2 = D * 0.32, D * 0.13
        star = [(r1 * math.cos(math.pi / 2 + k * math.pi / 5) if k % 2 == 0 else r2 * math.cos(math.pi / 2 + k * math.pi / 5),
                 r1 * math.sin(math.pi / 2 + k * math.pi / 5) if k % 2 == 0 else r2 * math.sin(math.pi / 2 + k * math.pi / 5))
                for k in range(10)]
        coin = coin.union(cq.Workplane("XY").polyline(star).close().extrude(relief).translate((0, 0, t - relief)))
        dots = cq.Workplane("XY").pushPoints([(D * 0.3 * math.cos(2 * math.pi * k / 12), D * 0.3 * math.sin(2 * math.pi * k / 12))
                                              for k in range(12)]).circle(D * 0.025).extrude(relief)
        coin = coin.union(dots)
        n = int(p["reeds"])
        for k in range(n):
            a = 2 * math.pi * k / n
            coin = coin.cut(cq.Workplane("XY").box(0.8, 0.8, t * 3).rotate((0, 0, 0), (0, 0, 1), math.degrees(a))
                            .translate((D / 2 * math.cos(a), D / 2 * math.sin(a), 0)))
        return coin


class Amulet(Design):
    """An amulet laid out flat: a bronze pendant (a ring setting with four claws round an oval gem, a bail on top)
    on a chain of ``chain_links`` interlocking round links, every other one on edge (separate solids, as a chain is). ``part``: the amulet, its bronze (setting, bail, chain)
    or its gem."""

    spec = fs.AMULET
    parameters = _params(fs.AMULET)

    def gem(self, p):
        a, b = p["gem_length"] / 2, p["gem_width"] / 2
        z = p["setting_thickness"] / 2 + 1
        return sk.egg(2 * a, 2 * b, p["gem_height"], (0, 0, z), nose=0.5, tail=0.5)

    def bronze(self, p):
        R, w, t = p["setting_diameter"] / 2, p["setting_width"], p["setting_thickness"]
        ring = cq.Workplane("XY").circle(R).circle(R - w).extrude(t)
        back = cq.Workplane("XY").circle(R - w + 0.5).extrude(1.0)
        claws = [sk.capsule(((R - w / 2) * math.cos(a), (R - w / 2) * math.sin(a), t),
                            ((R - w * 1.6) * math.cos(a), (R - w * 1.6) * math.sin(a), t + 2.5), 1.0)
                 for a in [math.pi / 4 + k * math.pi / 2 for k in range(4)]]
        bail = (cq.Workplane("XZ").circle(4.5).circle(3.0).extrude(1.5, both=True).translate((R + 3.5, 0, t / 2)))
        links = []
        n, r, Rc = int(p["chain_links"]), p["wire_diameter"] / 2, p["chain_radius"]
        Rl = 0.62 * 2 * math.pi * Rc / n                                 # link radius: neighbours interlock
        for k in range(n):                                               # links round a circle from the bail
            a = 2 * math.pi * (k + 0.5) / n
            c = (R + 3.5 + Rc - Rc * math.cos(a), Rc * math.sin(a), t / 2)
            link = cq.Workplane("XY").add(cq.Solid.makeTorus(Rl, r))
            if k % 2:                                                    # every other link stands on edge
                link = link.rotate((0, 0, 0), (1, 0, 0), 90)
            links.append(link.rotate((0, 0, 0), (0, 0, 1), 90 - math.degrees(a)).translate(c))
        return sk.union_all([ring, back, bail] + claws + links)

    def build(self, p):
        if p["part"] == "gem":
            return self.gem(p)
        if p["part"] == "bronze":
            return self.bronze(p)
        return self.bronze(p).union(self.gem(p))


class ElixirFlask(Design):
    """An elixir flask standing upright: a round-bottomed glass bulb on a flat base ring, a neck with a rolled lip,
    a cork stopper, and the elixir filling the bulb to ``fill`` of its height. ``part``: the flask, its glass, its
    cork or the elixir."""

    spec = fs.FLASK
    parameters = _params(fs.FLASK)

    def _bulb_centre(self, p):
        return (0, 0, p["bulb_diameter"] / 2 + p["base_height"])

    def glass(self, p):
        R, t = p["bulb_diameter"] / 2, p["glass_thickness"]
        c = self._bulb_centre(p)
        nr, nl = p["neck_diameter"] / 2, p["neck_length"]
        top = c[2] + R * 0.9
        outer = (sk.ball(c, R).union(cq.Workplane("XY").circle(nr).extrude(nl + R * 0.2).translate((0, 0, top - R * 0.2)))
                 .union(cq.Workplane("XY").circle(R * 0.55).extrude(p["base_height"] + R * 0.3)))
        lip = cq.Workplane("XY").add(cq.Solid.makeTorus(nr + 1.0, 2.0)).translate((0, 0, top + nl))
        inner = (sk.ball(c, R - t).union(cq.Workplane("XY").circle(nr - t).extrude(nl + R * 0.5)
                                         .translate((0, 0, top - R * 0.3))))
        return outer.union(lip).cut(inner)

    def cork(self, p):
        c = self._bulb_centre(p)
        top = c[2] + p["bulb_diameter"] / 2 * 0.9 + p["neck_length"]
        nr = p["neck_diameter"] / 2 - p["glass_thickness"]
        plug = (cq.Workplane("XY").circle(nr * 0.95).workplane(offset=p["cork_length"]).circle(nr * 1.25)
                .loft().translate((0, 0, top - p["cork_length"] * 0.55)))
        return plug

    def elixir(self, p):
        R, t = p["bulb_diameter"] / 2, p["glass_thickness"]
        c = self._bulb_centre(p)
        level = c[2] - R + t + (2 * (R - t)) * p["fill"]
        return sk.ball(c, R - t - 0.2).intersect(cq.Workplane("XY").box(4 * R, 4 * R, level).translate((0, 0, level / 2)))

    def build(self, p):
        parts = {"glass": self.glass, "cork": self.cork, "elixir": self.elixir}
        if p["part"] in parts:
            return parts[p["part"]](p)
        return sk.union_all([self.glass(p), self.cork(p), self.elixir(p)])
