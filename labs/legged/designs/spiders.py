"""The second-generation spiders' CAD: ARACHNE-2, TARANTULA and SALTY (numbers in ``spider_specs``; docs/spiders.md).
mm; x forward, y left, z up, the ground (or the wall) at z = 0.

Every spider: a **prosoma** (the front body: the motor bank, the board) and an **opisthosoma** (the rear body: the
battery) joined at a waist; eight legs, each **coxa → femur (rising) → knee (high) → tibia → metatarsus → tarsus**,
the joints driven from the body: the coxa turns on a hip-yaw knuckle in the prosoma's side, a pushrod lifts the femur,
a tendon in a sleeve along the femur bends the knee against a **torsion spring** that extends it, and a second tendon
couples the tarsus to the knee through a spring-loaded stop. The foot is a **claw + pad** tarsus: a pad under it, a
curved claw at its toe (shallow tarsus: the pad loads; steep: the claw hooks).

* ``Arachne2`` — palm-sized wall climber: an oval carapace with the hips round its rim and 24 Ø6 mm gearmotors
  inside it (2 rows × 2 layers × 6, along the body; tendon sleeves to the hips), the 1S battery low in the
  opisthosoma, a tether eye, time-of-flight sensors at the front.
* ``Tarantula`` — a 1.5 kg walker and climber: 24 micro metal gearmotors, an armoured dorsal shell of overlapping
  plates (Apheloria's tergites), a sensor turret (camera + lidar), big scopula pads with claws, a payload deck.
* ``Salty`` — a jumping spider: a boxy carapace with a flat face of camera eyes (two big principal eyes, two anterior
  laterals looking out at 40°, two small posterior eyes on top), legs III longer and latched in a crouch with a
  hydraulic extensor cylinder each, a gas accumulator, pump and battery in the opisthosoma, a dragline reel at the
  spinnerets.

``part``: 'robot' (all of it), 'cutaway' (the robot without its body shells: the insides), 'body', 'internals',
'leg' (one leg in its own frame, r → +x), 'foot'.
"""
import math

import cadquery as cq
import numpy as np
import shape_kit as sk
import spider_specs as ss
from vegeta.dedalus import Design, Parameter

__all__ = ["LEGS", "Arachne2", "Salty", "Tarantula", "comb_row", "foot", "foot_parts", "leg_parts"]

LEGS = [f"{s}{i}" for s in "LR" for i in (1, 2, 3, 4)]


# --------------------------------------------------------------------------------------------- small parts
def _v(p):
    return cq.Vector(*map(float, p))


def tube(a, b, r, wall=None):
    """A carbon tube from a to b (hollow when ``wall`` is given)."""
    s = sk.rod(a, b, r)
    if wall:
        s = s.cut(sk.rod(a, b, r - wall))
    return s


def knuckle(c, axis, r, w):
    """A hinge knuckle: a cylinder of radius ``r``, width ``w`` along ``axis`` centred on ``c``."""
    ax = _v(axis).normalized()
    a = _v(c) - ax * (w / 2)
    return cq.Workplane("XY").add(cq.Solid.makeCylinder(r, w, a, ax))


def coil(c, axis, r, wire, turns, pitch):
    """A torsion spring: ``turns`` rings of ``wire`` radius on radius ``r`` along ``axis``."""
    ax = _v(axis).normalized()
    out = None
    for k in range(turns):
        o = _v(c) + ax * ((k - (turns - 1) / 2) * pitch)
        ring = cq.Workplane("XY").add(cq.Solid.makeTorus(r, wire, o, ax))
        out = ring if out is None else out.union(ring)
    return out


def line(points, r):
    """A tendon or wire: a thin polyline."""
    return sk.polyline_limb(points, [r])


def motor_cyl(c, direction, d, L):
    """A cylindrical micro gearmotor (motor + gearbox + output) centred at c along ``direction``."""
    ax = _v(direction).normalized()
    a = _v(c) - ax * (L / 2)
    body = cq.Workplane("XY").add(cq.Solid.makeCylinder(d / 2, L, a, ax))
    shaft = cq.Workplane("XY").add(cq.Solid.makeCylinder(d * 0.15, L * 0.18, a + ax * L, ax))
    return body.union(shaft)


def motor_box(c, sx, w, h, L):
    """A micro metal gearmotor (w × h section, L long) centred at c along x, its output shaft towards ``sx`` (±1)."""
    b = cq.Workplane("XY").box(L, w, h).edges("|X").fillet(min(w, h) * 0.15).translate(tuple(c))
    shaft = sk.rod((c[0] + sx * L / 2, c[1], c[2]), (c[0] + sx * (L / 2 + 0.35 * L * 0.4), c[1], c[2]), 1.5)
    return b.union(shaft)


def ellipsoid(a, b, c, center):
    """An ellipsoid with semi-axes a, b, c (a scaled sphere)."""
    s = cq.Solid.makeSphere(1.0, angleDegrees1=-90, angleDegrees2=90)
    return cq.Workplane("XY").add(s.transformGeometry(cq.Matrix([[a, 0, 0, center[0]], [0, b, 0, center[1]],
                                                                  [0, 0, c, center[2]]])))


def carapace(spec, L, W, H, zc, lift=0.0):
    """The prosoma's shell: an oval puck (elliptical plan, domed top) or, for ``plan`` 'box', a rounded box (the
    salticid's square carapace with its flat face); mid-height ``zc``."""
    if spec.get("plan") == "box":
        w = (cq.Workplane("XY").box(L, W, H).edges("|Z").fillet(0.2 * W).edges(">Z").fillet(0.35 * H)
             .edges("<Z").fillet(0.2 * H))
        return w.translate((0, 0, zc + lift))
    return (cq.Workplane("XY").ellipse(L / 2, W / 2).extrude(H).edges(">Z").fillet(0.35 * H).edges("<Z")
            .fillet(0.25 * H).translate((0, 0, zc - H / 2 + lift)))


# --------------------------------------------------------------------------------------------- the leg
def _claws(spec):
    a, b, t = spec["pad"]
    c = spec["claw"]
    claws = None
    for y in (-b * 0.18, b * 0.18):
        pts = [(a * 0.42, y, t * 0.9), (a * 0.5 + c * 0.45, y, t * 0.95), (a * 0.5 + c * 0.75, y, t * 0.35),
               (a * 0.5 + c * 0.7, y, -t * 0.25)]
        k = sk.polyline_limb(pts, [c * 0.16, c * 0.12, c * 0.08])
        claws = k if claws is None else claws.union(k)
    return claws


def comb_row(spec, x, detail=True):
    """One row of the spine comb (``spec['spine_comb']``), its top edge at x under the backing plate: a 0.6 mm plate
    split into one finger flexure per spine, each ending in a spine tip ``reach`` below the sole, tilted towards the
    heel. ``detail`` False: the row's envelope (a plain plate down to the tips — for the whole-robot assembly)."""
    sc = spec["spine_comb"]
    t = spec["pad"][2]
    n, pitch = sc["per_row"], sc["pitch"]
    top, root = t * 0.58, t * 0.42                                        # under the plate; where the fingers split
    tip = -sc["reach"] / math.cos(math.radians(sc["tilt_deg"])) - (top * (1 / math.cos(math.radians(sc["tilt_deg"])) - 1))
    w = n * pitch
    half = pitch / 3
    if not detail:
        pts = [(-w / 2, top), (w / 2, top), (w / 2, tip), (-w / 2, tip)]
        row = cq.Workplane("YZ").polyline(pts).close().extrude(0.6).translate((x - 0.3, 0, 0))
        return row.rotate((x, 0, top), (x, 1, top), sc["tilt_deg"])
    pts = [(-w / 2, top), (w / 2, top), (w / 2, root)]
    for i in reversed(range(n)):
        y = -w / 2 + pitch * (i + 0.5)
        pts += [(y + half, root), (y + half, t * 0.1), (y, tip), (y - half, t * 0.1), (y - half, root)]
    pts.append((-w / 2, root))
    row = cq.Workplane("YZ").polyline(pts).close().extrude(0.6).translate((x - 0.3, 0, 0))
    return row.rotate((x, 0, top), (x, 1, top), sc["tilt_deg"])


def foot_parts(spec, detail=True):
    """The foot in the tarsus frame (the sole on z = 0, centred at the origin, toe to +x): {"foot": the rigid foot,
    "spines": [comb rows], "named": its pieces (comb feet)} — the spines stay separate solids (a union with the fine
    comb is not reliably valid).

    Without ``spine_comb``: the claw + pad foot (ARACHNE-2, SALTY): a pad, its foam layer and backing, and two curved
    claws at the toe hooking down past the sole. With it (TARANTULA, §10): a backing plate, a rubber sole under the
    heel two thirds (the swappable sole: rubber; the fibrillar pad or the magnet are kits), the comb rows at the toe
    between two side guards, and the two claws."""
    a, b, t = spec["pad"]
    if "spine_comb" not in spec:
        pad = cq.Workplane("XY").box(a, b, t * 0.45).edges("|Z").fillet(min(a, b) * 0.2).translate((0, 0, t * 0.225))
        sole = cq.Workplane("XY").box(a * 0.96, b * 0.96, t * 0.25).edges("|Z").fillet(min(a, b) * 0.18).translate(
            (0, 0, t * 0.45 + t * 0.125))                                     # the foam / setae layer
        back = cq.Workplane("XY").box(a, b, t * 0.3).edges("|Z").fillet(min(a, b) * 0.2).translate(
            (0, 0, t * 0.7 + t * 0.15))
        return {"foot": sk.union_all([pad, sole, back, _claws(spec)]), "spines": []}
    sc = spec["spine_comb"]
    top = t * 0.58
    plate = cq.Workplane("XY").box(a, b, t - top).edges("|Z").fillet(min(a, b) * 0.2).translate((0, 0, (t + top) / 2))
    x_heel_end = sc["x0"] - 2.8
    sole_l = x_heel_end + a / 2
    sole = (cq.Workplane("XY").box(sole_l, b * 0.94, top).edges("|Z").fillet(min(a, b) * 0.15)
            .translate((-a / 2 + sole_l / 2, 0, top / 2)))
    x_last = sc["x0"] + sc["dx"] * (sc["rows"] - 1)
    w = sc["per_row"] * sc["pitch"]
    guards = [cq.Workplane("XY").box(x_last - x_heel_end + 1.0, 1.0, top * 0.8)
              .translate(((x_heel_end + x_last + 1.0) / 2, sy * (w / 2 + 0.8), top * 0.6)) for sy in (-1, 1)]
    rows = [comb_row(spec, sc["x0"] + sc["dx"] * k, detail) for k in range(sc["rows"])]
    claws = _claws(spec)
    return {"foot": sk.union_all([plate, sole, *guards, claws]), "spines": rows,
            "named": {"plate": plate, "sole": sole, "guards": guards, "claws": claws}}


def foot(spec):
    """The whole foot (``foot_parts``) as one compound."""
    fp = foot_parts(spec)
    return _compound([fp["foot"]] + fp["spines"])


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def tendon_routes(spec, scale=1.0, angles=None):
    """The cords of a leg (leg frame), as the joints carry them: {name: {"pts": [(link, point)], "radius", "wrap"}},
    link 'hip' (the coxa), 'femur', 'tibia' (the tibia, metatarsus and tarsus) or 'wrap' (the point where the cord
    rides over a joint's knuckle in this pose). ``wrap`` = (centre, radius, side): the knee knuckle the cord wraps,
    above (+1) or below (−1) it.

    * ``flexor`` — the knee tendon: from the hip boss over the top of the yaw knuckle, over the lift knuckle, along
      the top of the femur, over the top of the knee knuckle, to a lug on the tibia's top: pulled, it folds the knee;
    * ``coupling`` — the tarsus tendon: from the femur's underside near the knee, under the knee knuckle, along the
      tibia's underside, under the metatarsal knuckle, to the tarsus joint's underside: it turns the foot with the
      knee;
    * ``pushrod`` — the lift pushrod from under the coxa to under the femur's root."""
    c = ss.leg_chain(spec, scale, angles)
    rf, rt, rm, _ = (x * (1.0 if scale == 1.0 else 1.25) for x in spec["tube"])
    kn = rf * 1.9
    P = {k: np.array([c[k][0], 0.0, c[k][1]]) for k in ("lift", "knee", "meta", "tarsus")}
    f, tt, m = _unit(P["knee"] - P["lift"]), _unit(P["meta"] - P["knee"]), _unit(P["tarsus"] - P["meta"])
    up = lambda d: np.array([-d[2], 0.0, d[0]])
    nf, nt, nm = up(f), up(tt), up(m)
    nk, nmeta = _unit(nf + nt), _unit(nt + nm)
    Lf, Lt = (np.linalg.norm(P["knee"] - P["lift"]), np.linalg.norm(P["meta"] - P["knee"]))
    ya, yb = np.array([0, rf * 0.9, 0]), np.array([0, -rf * 0.9, 0])
    RA = RB = kn * 1.05 + rf * 0.15
    zl = c["z_lift"]
    flexor = [("hip", np.array([kn * 0.6, 0, zl + kn * 1.2]) + ya), ("hip", np.array([c["lift"][0], 0, zl + kn + rf * 0.2]) + ya),
              ("femur", P["lift"] + f * 0.18 * Lf + nf * rf * 1.5 + ya), ("femur", P["knee"] - f * 0.14 * Lf + nf * rf * 1.5 + ya),
              ("wrap", P["knee"] + nk * RA + ya), ("tibia", P["knee"] + tt * 0.22 * Lt + nt * rt * 1.5 + ya)]
    coupling = [("femur", P["knee"] - f * 0.16 * Lf - nf * rf * 1.4 + yb), ("wrap", P["knee"] - nk * RB + yb),
                ("tibia", P["meta"] - tt * 0.12 * Lt - nt * rt * 1.4 + yb), ("tibia", P["meta"] - nmeta * rt * 1.8 + yb),
                ("tibia", P["tarsus"] - nm * rm * 1.8 + yb)]
    pushrod = [("hip", np.array([c["lift"][0] * 0.5, 0, zl - rf * 1.6])), ("femur", P["lift"] + f * 0.25 * Lf - nf * rf * 1.6)]
    return {"flexor": {"pts": flexor, "radius": max(0.25, rf * 0.12), "wrap": (P["knee"] + ya, RA, 1)},
            "coupling": {"pts": coupling, "radius": max(0.2, rf * 0.1), "wrap": (P["knee"] + yb, RB, -1)},
            "pushrod": {"pts": pushrod, "radius": rf * 0.4, "wrap": None}}


def leg_parts(spec, scale=1.0, angles=None, detail=True):
    """One leg in its own frame (r → +x, across the leg → +y, z up, ground z = 0): a dict of parts in place."""
    c = ss.leg_chain(spec, scale, angles)
    rf, rt, rm, rs = (x * (1.0 if scale == 1.0 else 1.25) for x in spec["tube"])
    p = lambda q: (q[0], 0.0, q[1])
    lift, knee, meta, tars, toe = (p(c[k]) for k in ("lift", "knee", "meta", "tarsus", "toe"))
    yaw = (0.0, 0.0, c["z_lift"])
    kn = rf * 1.9                                                             # knuckle radius at the joints
    parts = {}
    parts["coxa"] = sk.union_all([knuckle(yaw, (0, 0, 1), kn * 1.15, kn * 2.4),
                                  tube(yaw, lift, rf * 1.35),
                                  knuckle(lift, (0, 1, 0), kn, rf * 4.2)])
    parts["femur"] = tube(lift, knee, rf, rf * 0.35)
    routed = spec.get("cords") == "routed"                                # TARANTULA: the cords of tendon_routes
    if routed:
        parts["knee"] = knuckle(knee, (0, 1, 0), kn * 1.05, rf * 3.6)      # also the cords' pulley
    else:
        parts["knee"] = sk.union_all([knuckle(knee, (0, 1, 0), kn * 1.05, rf * 3.6),
                                      knuckle((knee[0], rf * 2.6, knee[2]), (0, 1, 0), kn * 1.35, rf * 0.9)])  # pulley
    parts["knee_spring"] = coil((knee[0], -rf * 2.6, knee[2]), (0, 1, 0), kn * 0.9, rf * 0.18, 4, rf * 0.45)
    parts["tibia"] = tube(knee, meta, rt, rt * 0.35)
    parts["meta_joint"] = knuckle(meta, (0, 1, 0), rt * 1.7, rt * 3.2)
    parts["metatarsus"] = tube(meta, tars, rm, rm * 0.4)
    parts["tarsus_joint"] = sk.union_all([knuckle(tars, (0, 1, 0), rm * 1.7, rm * 3.0),
                                          coil((tars[0], -rm * 2.0, tars[2]), (0, 1, 0), rm * 1.3, rm * 0.15, 3, rm * 0.4)])
    if routed:
        # the cords (tendon_routes): the knee flexor over the knuckles' tops, the tarsus coupling under them, the pushrod
        routes = tendon_routes(spec, scale, angles)
        parts["tendons"] = sk.union_all([line([tuple(p) for _, p in r["pts"]], r["radius"]) if name != "pushrod" else
                                         tube(tuple(r["pts"][0][1]), tuple(r["pts"][1][1]), r["radius"])
                                         for name, r in routes.items()])
    else:
        # the tendons: lift pushrod at the femur's root, the knee tendon along the femur's top in a sleeve, the tarsus
        # coupling along the tibia and the metatarsus
        up = lambda q, d: (q[0], 0.0, q[2] + d)
        parts["tendons"] = sk.union_all([
            line([(0.0, rf * 2.2, c["z_lift"] + rf * 1.5), (lift[0] * 0.5 + knee[0] * 0.5, rf * 2.2,
                                                             (lift[2] + knee[2]) / 2 + rf * 1.5),
                  (knee[0], rf * 2.6, knee[2] + kn * 1.35)], max(0.25, rf * 0.12)),
            line([(knee[0], -rf * 1.6, knee[2] + rf), up(meta, rt * 1.6), (tars[0], -rf * 1.6, tars[2] + rm * 1.6)],
                 max(0.2, rf * 0.09)),
            tube(up(lift, -rf * 1.8), (lift[0] + (knee[0] - lift[0]) * 0.25, 0.0,
                                       lift[2] + (knee[2] - lift[2]) * 0.25 - rf * 1.8), rf * 0.4)])  # the lift pushrod
    # the foot: the tarsus tube, then the foot along it, its sole on the ground
    fp = foot_parts(spec, detail)
    mid = ((tars[0] + toe[0]) / 2, 0.0, 0.0)
    parts["tarsus"] = tube(tars, (toe[0], 0.0, max(toe[2], spec["pad"][2])), rs)
    parts["foot"] = fp["foot"].translate(mid)
    if fp["spines"]:
        parts["spines"] = _compound([r.translate(mid) for r in fp["spines"]])
    return parts


def leg_solid(spec, scale=1.0, angles=None, skip=(), detail=True):
    """The leg as one solid with its cords and the spine comb's rows beside it (a compound; the cords ride over the
    knuckles, so they stay separate solids)."""
    parts = leg_parts(spec, scale, angles, detail)
    apart = ("spines", "tendons") if spec.get("cords") == "routed" else ("spines",)
    solid = sk.union_all([v for k, v in parts.items() if k not in skip and k not in apart])
    extra = [parts[k] for k in apart if k in parts and k not in skip]
    return _compound([solid, *extra]) if extra else solid


# --------------------------------------------------------------------------------------------- the bodies
def _place(solid, leg, spec):
    """Leg frame → body: turned to the leg's splay, its yaw axis on the hip point."""
    (hx, hy), splay, sy = ss.hip_points(spec, leg)
    a = math.radians(splay)
    ang = math.degrees(math.atan2(sy * math.cos(a), math.sin(a)))
    return solid.rotate((0, 0, 0), (0, 0, 1), ang).translate((hx, hy, 0))


def _compound(parts):
    solids = [v for q in parts if q is not None for v in q.vals()]
    return cq.Workplane("XY").add(cq.Compound.makeCompound(solids))


class _Spider(Design):
    spec = None
    parameters = [Parameter("part", "robot", choices=("robot", "cutaway", "body", "internals", "leg", "foot"),
                            description="what to build")]

    # the layout
    def zc(self, s):
        return ss.body_z(s)

    def x_opi(self, s):
        return -(s["prosoma"][0] / 2 + s["waist"] + s["opisthosoma"][0] / 2) * 0.94

    def z_opi(self, s):
        return s["belly"] + s["opisthosoma"][2] / 2

    # the shells
    def prosoma(self, s):
        L, W, H = s["prosoma"]
        return carapace(s, L, W, H, self.zc(s))

    def opisthosoma(self, s):
        L, W, H = s["opisthosoma"]
        return ellipsoid(L / 2, W / 2, H / 2, (self.x_opi(s), 0, self.z_opi(s)))

    def waist(self, s):
        """The pedicel: a short narrow neck between the two bodies."""
        H = min(s["prosoma"][2], s["opisthosoma"][2])
        z = (self.zc(s) + self.z_opi(s)) / 2
        x0 = -s["prosoma"][0] / 2 * 0.85
        x1 = self.x_opi(s) + s["opisthosoma"][0] / 2 * 0.85
        return sk.capsule((x0, 0, z), (x1, 0, z), H * 0.2)

    def shells(self, s):
        return [self.prosoma(s), self.waist(s), self.opisthosoma(s)]

    def hip_sockets(self, s):
        """A boss from the rim out to each hip, carrying the coxa's yaw knuckle."""
        out = []
        z = ss.leg_chain(s)["z_lift"]
        r = s["tube"][0] * 1.9 * 1.3
        for leg in LEGS:
            (hx, hy), _, sy = ss.hip_points(s, leg)
            out.append(sk.rod((hx, sy * (ss.rim_y(s, hx) - 1.0), z), (hx, hy, z), r))
        return out

    # the insides
    def bank_layout(self, s):
        """The joint motors: ``bank`` = (across, layers, rows) lying along x, the rows back to back, each output shaft
        outward (to the front or the back of the body); centred at ``bank_at`` (dx, dz) from the prosoma's middle."""
        nx, nz, nr = s["bank"]
        dx, dz = s.get("bank_at", (0.0, 0.0))
        m = s["motor"]
        if len(m) == 2:
            w = h = m[0]
            L = m[1]
        else:
            w, h, L = m
        py, pz = w + 0.5, h + 1.0
        out = []
        for row in range(nr):
            sx = 1 if row % 2 == 0 else -1
            x = dx + sx * (L / 2 + 1.0)
            for j in range(nz):
                z = self.zc(s) + dz + (j - (nz - 1) / 2) * pz
                for i in range(nx):
                    y = (i - (nx - 1) / 2) * py
                    out.append(((x, y, z), sx))
        return out[: s["motors"]], (w, h, L), (nx * py / 2, nz * pz / 2, L + 1.0)

    def motor_bank(self, s):
        lay, (w, h, L), _ = self.bank_layout(s)
        if len(s["motor"]) == 2:
            return [motor_cyl(c, (sx, 0, 0), w, L) for c, sx in lay]
        return [motor_box(c, sx, w, h, L) for c, sx in lay]

    def sleeves(self, s):
        """The tendon sleeves: from the bank's end faces to each hip."""
        _, _, (hy_b, _, hx_b) = self.bank_layout(s)
        dx, dz = s.get("bank_at", (0.0, 0.0))
        z = ss.leg_chain(s)["z_lift"]
        r = max(0.5, s["tube"][0] * 0.3)
        out = []
        for leg in LEGS:
            (hx, hy), _, sy = ss.hip_points(s, leg)
            ex = dx + (hx_b if hx >= dx else -hx_b)
            out.append(line([(ex, sy * hy_b * 0.5, self.zc(s) + dz), (ex + (hx - ex) * 0.3, sy * hy_b * 1.05, z),
                             (hx, hy - sy * s["tube"][0] * 2.5, z)], r))
        return out

    def board(self, s):
        _, _, (_, hz_b, _) = self.bank_layout(s)
        dx, dz = s.get("bank_at", (0.0, 0.0))
        bw, bl = s["board"]
        return cq.Workplane("XY").box(bw, bl, 1.2).translate((dx, 0, self.zc(s) + dz + hz_b + 1.2))

    def battery(self, s, dx=0.0, dz=0.0):
        bx, by, bz = s["battery"]
        return sk.rounded_box(bx, by, bz, (self.x_opi(s) + dx, 0, self.z_opi(s) + dz), fillet=min(bz * 0.3, 4.0))

    def internals(self, s):
        """[motors…, battery, board, sleeves…] — the order the renders colour them by."""
        return self.motor_bank(s) + [self.battery(s), self.board(s)] + self.sleeves(s)

    def extras(self, s, cut):
        return []

    _detail = True                                                        # the spine comb in full (or its envelope)

    def legs(self, s):
        return [_place(leg_solid(s, detail=self._detail), leg, s) for leg in LEGS]

    def build(self, p):
        s = self.spec
        part = p["part"]
        if part == "leg":
            return leg_solid(s)
        if part == "foot":
            return foot(s)
        if part == "body":
            return sk.union_all(self.shells(s) + self.hip_sockets(s))
        if part == "internals":
            return _compound(self.internals(s))
        cut = part == "cutaway"
        parts = ([] if cut else self.shells(s)) + self.hip_sockets(s) + self.legs(s) + self.extras(s, cut)
        if cut:
            parts += self.internals(s)
        return _compound(parts)


class Arachne2(_Spider):
    """ARACHNE-2: the palm-sized wall climber with spider-like legs (docs/spiders.md §3)."""

    spec = ss.ARACHNE2

    def battery(self, s, dx=0.0, dz=0.0):
        return super().battery(s, dx, -1.0)                               # low: the centre of mass near the wall

    def extras(self, s, cut):
        L, W, H = s["prosoma"]
        zc = self.zc(s)
        eye = cq.Workplane("XZ").circle(3.0).circle(1.8).extrude(0.8, both=True).translate((-L * 0.3, 0, zc + H / 2 + 2.6))
        post = sk.rod((-L * 0.3, 0, zc + H / 2 - 1.0), (-L * 0.3, 0, zc + H / 2 + 0.2), 1.2)
        x = L / 2 * math.sqrt(1 - (8.5 / (W / 2)) ** 2)
        tof = [cq.Workplane("XY").box(2.5, 5.0, 4.0).translate((x - 1.0, sy * 6.0, zc)) for sy in (-1, 1)]
        return [eye, post] + tof


class Tarantula(_Spider):
    """TARANTULA: the 1.5 kg walker and climber (docs/spiders.md §5): armoured, a sensor turret, a payload deck."""

    spec = ss.TARANTULA

    def plates(self, s):
        """Overlapping dorsal plates over the carapace (Apheloria's tergites): a 3 mm skin over the top, cut into
        ``shell_plates`` bands across the body, each a little higher than the one in front of it."""
        L, W, H = s["prosoma"]
        zc = self.zc(s)
        n = s["shell_plates"]
        inner = carapace(s, L, W, H, zc)
        out = []
        for k in range(n):
            lift = 1.0 + 0.8 * (n - 1 - k)
            skin = carapace(s, L + 6, W + 6, H + 4, zc, lift).cut(inner)
            x0 = -L / 2 - 4 + (L + 8) * k / n - (1.5 if k else 0)
            x1 = -L / 2 - 4 + (L + 8) * (k + 1) / n + 1.5
            band = cq.Workplane("XY").box(x1 - x0, W + 10, H * 0.6).translate(((x0 + x1) / 2, 0, zc + H * 0.5))
            out.append(skin.intersect(band))
        return out

    def tergites(self, s, n=4):
        """The opisthosoma's armour: a 3 mm skin over its top in ``n`` overlapping bands, each rear band a little lower
        (they slide under each other as the abdomen flexes — Apheloria's tergites)."""
        L, W, H = s["opisthosoma"]
        x, z = self.x_opi(s), self.z_opi(s)
        inner = ellipsoid(L / 2, W / 2, H / 2, (x, 0, z))
        out = []
        S_, _, _ = self.pod_frame(s)
        notch = cq.Workplane("XY").box(60.0, 36.0, 70.0).translate((S_[0] - 4.0, 0.0, S_[2]))   # the dragline pod's seat
        for k in range(n):
            lift = 0.8 * (n - 1 - k)
            skin = ellipsoid(L / 2 + 3, W / 2 + 3, H / 2 + 3, (x, 0, z + lift)).cut(inner).cut(notch)
            x0 = x + L / 2 + 4 - (L + 8) * (k + 1) / n - 1.5
            x1 = x + L / 2 + 4 - (L + 8) * k / n + (1.5 if k else 0)
            band = cq.Workplane("XY").box(x1 - x0, W + 10, H * 0.75).translate(((x0 + x1) / 2, 0, z + H * 0.4))
            out.append(skin.intersect(band))
        return out

    def turret(self, s):
        L, _, H = s["prosoma"]
        d, h = s["turret"]
        x = L / 2 - d * 0.8
        z = self.zc(s) + H / 2 + 3.0
        base = cq.Workplane("XY").circle(d / 2).extrude(h * 0.5).translate((x, 0, z))
        head = sk.rounded_box(d * 0.8, d * 1.1, h * 0.55, (x + 2, 0, z + h * 0.5 + h * 0.27), fillet=4)
        lens = [cq.Workplane("YZ").circle(5.5).extrude(4.0).translate((x + d * 0.4 + 2, yy, z + h * 0.77))
                for yy in (-9.0, 9.0)]
        lidar = cq.Workplane("XY").circle(d * 0.32).extrude(h * 0.35).translate((x, 0, z + h * 1.05))
        return [base, head, lidar] + lens

    def deck(self, s):
        """The payload deck: a plate on four posts over the opisthosoma."""
        H = s["opisthosoma"][2]
        x, z = self.x_opi(s), self.z_opi(s) + H / 2
        plate = sk.rounded_box(80.0, 60.0, 3.0, (x, 0, z + 6.0), fillet=1.0)
        posts = [sk.rod((x + ax, ay, z - 6.0), (x + ax, ay, z + 5.0), 3.0) for ax in (-25.0, 25.0) for ay in (-18.0, 18.0)]
        return [plate] + posts

    # the chelicerae: TARANTULA's cutting jaws ----------------------------------------------------------------------
    JAW_OUTER = ((-6.0, 2.0), (0.0, 6.5), (8.0, 7.0), (20.0, 6.5), (32.0, 3.5), (42.0, -3.0), (49.0, -11.0),
                 (53.0, -20.0), (54.0, -27.0))
    JAW_HOOK = ((50.0, -23.0), (44.0, -17.5), (37.0, -13.5))           # the inner edge from the fang tip to the blade

    def jaw(self, s, side, open_deg):
        """One jaw in place ({"jaw", "blade"}): a curved, fang-tipped plate with its pivot boss and an outward tail,
        a hardened blade insert along the straight cutting edge (``edge_y`` inside the pivot, ``cut_zone``). ``side``
        +1 = the left jaw (above the pivot plane), −1 = the right (below, mirrored); turned ``open_deg`` outward."""
        c = s["cutter"]
        e, (a0, a1) = c["edge_y"], c["cut_zone"]
        tx, ty = c["tail"]
        outline = list(self.JAW_OUTER) + list(self.JAW_HOOK) + [(a1, -e), (0.0, -e), (-6.0, -6.0)]
        z0 = 0.0 if side > 0 else -c["jaw_t"]
        plate = cq.Workplane("XY").polyline(outline).close().extrude(c["jaw_t"])
        boss = cq.Workplane("XY").circle(6.0).extrude(c["jaw_t"])
        L = math.hypot(tx, ty)
        arm = (cq.Workplane("XY").center(tx / 2, ty / 2).rect(L, 6.0).extrude(c["jaw_t"])
               .rotate((tx / 2, ty / 2, 0), (tx / 2, ty / 2, 1), math.degrees(math.atan2(ty, tx))))
        tip = cq.Workplane("XY").center(tx, ty).circle(2.5).extrude(c["jaw_t"])
        jaw = sk.union_all([plate, boss, arm, tip])
        blade = (cq.Workplane("XY").polyline([(a0, -e), (a1, -e), (a1 + 2.0, -e + 3.0), (a0, -e + 3.0)]).close()
                 .extrude(c["jaw_t"]))
        px, py, pz = c["pivot"]
        out = {}
        for k, w in (("jaw", jaw), ("blade", blade)):
            w = w.translate((0, 0, z0))
            if side < 0:
                w = w.mirror("XZ")
            out[k] = w.rotate((0, 0, 0), (0, 0, 1), side * open_deg).translate((px, side * py, pz))
        return out

    def chelicerae(self, s, open_deg):
        """{name: [solids]}: the housing (two lobes over one hollow, open-fronted box), the pivot pins and their
        tongues, the two jaws and blades, and the drive (the crossbar yoke, two links, the leadscrew, its motor)."""
        c = s["cutter"]
        x0, x1, z0, z1 = c["housing_at"]
        px, py, pz = c["pivot"]
        zm = (z0 + z1) / 2
        lobes = [sk.rounded_box(x1 - x0, 23.0, z1 - z0, ((x0 + x1) / 2, sy * 15.5, zm), fillet=8.0) for sy in (-1, 1)]
        bridge = cq.Workplane("XY").box(x1 - x0, 12.0, z1 - z0 - 6.0).translate(((x0 + x1) / 2, 0, zm - 3.0))
        cavity = cq.Workplane("XY").box(x1 - x0 + 4.0, 50.0, z1 - z0 - 4.0).translate(((x0 + x1) / 2 + 3.0, 0, zm))
        hole = sk.rod((x0 - 1, 0, pz), (x0 + 4, 0, pz), 2.6)
        housing = sk.union_all(lobes + [bridge]).cut(cavity).cut(hole)
        t = c["jaw_t"]
        pins, tongues = [], []
        for sy in (-1, 1):
            pins.append(sk.rod((px, sy * py, z0), (px, sy * py, pz + t + 3.0), 2.5))
            yy = (sy * (py - 6.0), sy * 25.0)
            for zz in (z0 + 1.0, pz + t + 2.0):                           # the lower plate (the floor's) and the tongue
                xa = x0 + 20.0 if zz > pz else x1
                tongues.append(cq.Workplane("XY").box(px + 6.0 - xa, abs(yy[1] - yy[0]), 2.0)
                               .translate(((px + 6.0 + xa) / 2, (yy[0] + yy[1]) / 2, zz)))
        jl, jr = self.jaw(s, 1, open_deg), self.jaw(s, -1, open_deg)
        lk = ss.cutter_linkage(s, open_deg)
        yx, (tx, ty) = lk["yoke_x"], lk["tail"]
        yoke = cq.Workplane("XY").box(6.0, 2 * (c["yoke_y"] + 2.0), 2 * t + 1.0).translate((yx, 0, pz))
        links = [sk.rod((yx, sy * c["yoke_y"], pz + sy * t / 2), (tx, sy * ty, pz + sy * t / 2), 2.0) for sy in (-1, 1)]
        mx, mz = c["motor_at"]
        motor = motor_box((mx, 0, mz), 1, *s["motor"])
        screw = sk.rod((mx + s["motor"][2] / 2, 0, pz), (yx + 4.0, 0, pz), 2.0)
        return {"housing": [housing], "pins": pins + tongues, "jaws": [jl["jaw"], jr["jaw"]],
                "blades": [jl["blade"], jr["blade"]], "drive": [yoke] + links, "motor": [motor, screw]}

    def internals(self, s):
        return super().internals(s) + self.chelicerae(s, self._jaw)["motor"]

    # the dragline pod: TARANTULA's spool and harpoon at the spinnerets (§12) ----------------------------------------
    def pod_frame(self, s):
        """(spool centre, launcher axis d, the tube's rear end T0) in the robot frame [mm]."""
        w = s["winch"]
        S_ = np.array([self.x_opi(s) + w["at"][0], 0.0, self.z_opi(s) + w["at"][1]])
        e = math.radians(w["elevation_deg"])
        d = np.array([-math.cos(e), 0.0, math.sin(e)])
        T0 = S_ + np.array([-4.0, 0.0, w["flange"] + 3.0])
        return S_, d, T0

    def harpoon(self, s):
        """The harpoon in its own frame: the line's eye at the origin, the shaft along +x, a hardened tip and two
        toggle barbs folded along it (they open under load and lock behind the fibres)."""
        r, L, tip = (v / 2 if i == 0 else v for i, v in enumerate(s["winch"]["harpoon"]))
        shaft = sk.rod((0, 0, 0), (L, 0, 0), r)
        cone = cq.Workplane("XY").add(cq.Solid.makeCone(r, 0.0, tip, _v((L, 0, 0)), _v((1, 0, 0))))
        barbs = [sk.polyline_limb([(L - 2, 0, sz * r * 0.6), (L - 14, 0, sz * (r + 2.2))], [0.7, 0.45]) for sz in (-1, 1)]
        eye = cq.Workplane("XY").add(cq.Solid.makeTorus(3.0, 0.9, _v((-2.2, 0, 0)), _v((0, 1, 0))))
        collar = sk.rod((L * 0.45, 0, 0), (L * 0.55, 0, 0), r + 0.8)
        return {"shaft": sk.union_all([shaft, cone, collar]), "barbs": barbs, "eye": eye}

    def harpoon_pose(self, s):
        """(position of its eye, direction) of the harpoon seated in the launcher, tip out of the muzzle."""
        _, d, T0 = self.pod_frame(s)
        return T0 + 10.0 * d, d

    def place_harpoon(self, s, solid, at=None, d=None):
        p, dd = self.harpoon_pose(s) if at is None else (np.asarray(at), np.asarray(d))
        ang = math.degrees(math.atan2(-dd[2], dd[0]))                         # about y: x → d (in the x–z plane)
        return solid.rotate((0, 0, 0), (0, 1, 0), ang).translate(tuple(p))

    def dragline_pod(self, s):
        """{name: [solids]}: the spool (drum, flanges, the line wound on it), the gearmotor under it, the band brake
        and its servo, the bracket, the launcher tube, the fairlead and the line from the spool into the tube."""
        w = s["winch"]
        S_, d, T0 = self.pod_frame(s)
        y0 = w["width"] / 2
        Sx, Sz = S_[0], S_[2]
        drum = knuckle(tuple(S_), (0, 1, 0), w["drum"], w["width"])
        flanges = [knuckle((Sx, sy * (y0 + 0.75), Sz), (0, 1, 0), w["flange"], 1.5) for sy in (-1, 1)]
        wound = knuckle(tuple(S_), (0, 1, 0), w["flange"] - 3.0, w["width"] - 0.5)
        motor = motor_box((Sx + 11.0, 0.0, Sz - 17.0), 1, 12.0, 10.0, 24.0).rotate((Sx + 11.0, 0, Sz - 17.0),
                                                                                  (Sx + 11.0, 0, Sz - 16.0), 90)
        band = (knuckle((Sx, y0 + 0.75, Sz), (0, 1, 0), w["flange"] + 1.2, 3.0)
                .cut(knuckle((Sx, y0 + 0.75, Sz), (0, 1, 0), w["flange"] + 0.2, 3.0)))
        servo = sk.rounded_box(12.0, 6.0, 10.0, (Sx - 8.0, y0 + 7.0, Sz - 10.0), fillet=1.0)
        bracket = [cq.Workplane("XY").box(44.0, 2.0, 34.0).translate((Sx + 2.0, sy * (y0 + 3.5), Sz - 8.0)) for sy in (-1, 1)]
        ro, ri, L = w["tube"]
        tube_ = tube(tuple(T0), tuple(T0 + L * d), ro / 2, (ro - ri) / 2)
        clamp = sk.rod(tuple(T0 + 6 * d), tuple(T0 + 14 * d), ro / 2 + 1.5)
        post = sk.rod(tuple(T0 + 10 * d), (T0[0] + 10 * d[0], 0.0, Sz + 4.0), 2.5)
        fairlead = cq.Workplane("XY").add(cq.Solid.makeTorus(3.0, 0.8, _v(tuple(T0)), _v(tuple(d))))
        line_in = line([(Sx, 0.0, Sz + w["flange"] - 3.0), tuple(T0)], 0.5)
        return {"spool": [drum, *flanges], "line": [wound, line_in], "motor": [motor], "brake": [band, servo],
                "bracket": bracket + [post, clamp], "launcher": [tube_, fairlead]}

    def extras(self, s, cut):
        pod = [v for vs in self.dragline_pod(s).values() for v in vs]
        hp = self.harpoon(s)
        harpoon = [self.place_harpoon(s, v) for v in [hp["shaft"], *hp["barbs"], hp["eye"]]]
        ch = self.chelicerae(s, self._jaw)
        jaws = ch["housing"] + ch["pins"] + ch["jaws"] + ch["blades"] + ch["drive"]
        return ([] if cut else self.plates(s) + self.tergites(s)) + self.turret(s) + self.deck(s) + jaws + pod + harpoon

    _jaw = 20.0
    parameters = [Parameter("part", "robot", choices=("robot", "cutaway", "body", "internals", "leg", "foot", "jaws",
                                                     "winch"),
                            description="what to build"),
                  Parameter("jaw", 20.0, units="deg", min=0.0, max=30.0, description="the jaws' opening, each side"),
                  Parameter("comb", "full", choices=("full", "envelope"),
                            description="the feet's spine combs: every spine, or each row's envelope (a light assembly)")]

    def build(self, p):
        self._jaw = float(p.get("jaw", 20.0))
        self._detail = p.get("comb", "full") == "full"
        if p["part"] == "winch":
            hp = self.harpoon(self.spec)
            return _compound([v for vs in self.dragline_pod(self.spec).values() for v in vs] +
                             [self.place_harpoon(self.spec, v) for v in [hp["shaft"], *hp["barbs"], hp["eye"]]])
        if p["part"] == "jaws":
            ch = self.chelicerae(self.spec, self._jaw)
            return _compound([v for k in ("housing", "pins", "jaws", "blades", "drive", "motor") for v in ch[k]])
        return super().build(p)


class Salty(_Spider):
    """SALTY: the hydraulic jumping spider (docs/spiders.md §6)."""

    spec = ss.SALTY
    JUMP = ("L3", "R3")

    def jump_chain(self, s):
        return ss.leg_chain(s, s["jump_leg"], s["jump_angles"])

    def legs(self, s):
        out = []
        for leg in LEGS:
            if leg in self.JUMP:
                out.append(_place(leg_solid(s, s["jump_leg"], s["jump_angles"]), leg, s))
                out.append(_place(self.jump_cylinder(s), leg, s))           # kept apart: the union is not valid
            else:
                out.append(_place(leg_solid(s), leg, s))
        return out

    def jump_cylinder(self, s):
        """The extensor cylinder on a jumping leg (leg frame): its barrel under the femur, in the leg's plane (so it is
        the same on both sides), its rod to the knee; the hose into the body; the latch hook over the knee."""
        c = self.jump_chain(s)
        d, L = s["cylinder"]
        rf = s["tube"][0] * 1.25
        lift = (c["lift"][0], 0.0, c["lift"][1])
        knee = (c["knee"][0], 0.0, c["knee"][1])
        ux, uz = knee[0] - lift[0], knee[2] - lift[2]
        n = math.hypot(ux, uz)
        ux, uz = ux / n, uz / n
        nx, nz = uz, -ux                                                  # the femur's underside
        off = d / 2 + rf + 0.6
        b0 = (lift[0] + ux * 4 + nx * off, 0.0, lift[2] + uz * 4 + nz * off)
        b1 = (b0[0] + ux * L * 0.75, 0.0, b0[2] + uz * L * 0.75)
        barrel = sk.rod(b0, b1, d / 2).union(sk.ball(b0, d / 2))
        rod = sk.rod(b1, (knee[0] + nx * off * 0.6, 0.0, knee[2] + nz * off * 0.6), d * 0.18)
        hose = line([b0, (b0[0] - 3, 0.0, b0[2] - 3), (-3.0, 0.0, lift[2] - 2.0)], 0.9)
        kn = rf * 1.9
        latch = sk.polyline_limb([(knee[0] - ux * 6, 0.0, knee[2] - uz * 6 - nz * kn), (knee[0] - 2, 0.0, knee[2] + kn + 2.0),
                                  (knee[0] + 3, 0.0, knee[2] + kn + 0.5)], [0.8])
        return sk.union_all([barrel, rod, hose, latch])

    def eye(self, c, axis, d):
        """A camera eye: a lens (a ball standing proud of the face by a fifth of its diameter) in a rim."""
        ax = _v(axis).normalized()
        o = _v(c)
        lens = sk.ball(tuple(o - ax * (d * 0.3)), d / 2 * 0.98)
        rim = (cq.Workplane("XY").add(cq.Solid.makeCylinder(d / 2 + 1.2, 2.0, o - ax * 1.5, ax))
               .cut(cq.Workplane("XY").add(cq.Solid.makeCylinder(d / 2, 2.0, o - ax * 1.5, ax))))
        return rim, lens

    def face(self, s):
        """[rim, lens, …]: the principal eyes in the flat face, the anterior laterals on the front corners looking out,
        the small posterior eyes on top."""
        L, W, H = s["prosoma"]
        zc = self.zc(s)
        (dp, yp), (dl, look) = s["eyes"]
        out = []
        for sy in (-1, 1):
            out += self.eye((L / 2, sy * yp, zc - 1.0), (1, 0, 0), dp)
        rc = 0.2 * W
        a = math.radians(look)
        for sy in (-1, 1):
            c = (L / 2 - rc + rc * math.cos(a), sy * (W / 2 - rc + rc * math.sin(a)), zc + 1.0)
            out += self.eye(c, (math.cos(a), sy * math.sin(a), 0), dl)
        rt = 0.35 * H
        for sy in (-1, 1):
            y = W / 2 - 7.0
            z = zc + H / 2 - rt + math.sqrt(rt * rt - (y - (W / 2 - rt)) ** 2)
            out += [cq.Workplane("XY").add(cq.Solid.makeCylinder(3.4, 0.8, _v((L * 0.05, sy * y, z - 0.6)), _v((0, 0, 1)))),
                    sk.ball((L * 0.05, sy * y, z), 2.4)]
        return out

    def hydraulics(self, s):
        """The accumulator high along the opisthosoma, the pump low in front of the battery."""
        da, la = s["accumulator"]
        x, z = self.x_opi(s), self.z_opi(s)
        acc = sk.capsule((x + la / 2 - da / 2, 0, z + 7.0), (x - la / 2 + da / 2, 0, z + 7.0), da / 2)
        pw, ph, pl = s["pump"]
        pump = sk.rounded_box(pl, pw, ph, (x + 12.0, 0, z - 5.5), fillet=2)
        return [acc, pump]

    def battery(self, s, dx=0.0, dz=0.0):
        return super().battery(s, -8.0, -5.0)

    def spinnerets(self, s):
        """The dragline reel at the opisthosoma's tip: a spool and the line leaving it."""
        dr, wr = s["reel"]
        L = s["opisthosoma"][0]
        x = self.x_opi(s) - L / 2 * 0.95
        z = self.z_opi(s) - s["opisthosoma"][2] * 0.1
        c = (x - dr * 0.2, 0, z)
        spool = knuckle(c, (0, 1, 0), dr / 2, wr).cut(knuckle(c, (0, 1, 0), dr / 2 - 1.5, wr - 2.0))
        hub = knuckle(c, (0, 1, 0), dr * 0.18, wr)
        dragline = line([(c[0], 0, z - dr / 2 + 1), (x - dr * 1.5, 0, z - dr * 0.6), (x - dr * 3.5, 0, 0.6)], 0.25)
        return [spool, hub, dragline]

    def internals(self, s):
        return super().internals(s) + self.hydraulics(s)

    def extras(self, s, cut):
        return self.face(s) + self.spinnerets(s)
