"""ARACHNE — the CAD: a palm-sized eight-legged wall climber standing on a wall (the wall is z = 0, z out of it; x along
the heading, y left). Every number from ``arachne_specs`` (the budget, the geometry, the BOM), so the CAD, the robot
model (``arachne_robot``), the statics and the drawings agree.

The body is a two-plate sandwich: the belly plate (6 mm off the wall) and the top plate 19 mm above it, joined by
corner and centre posts; the eight hip-yaw servos stand between the plates in two rows, shafts up through the top
plate; the 1S battery lies on the belly plate between the rows; the main board on the top plate; two time-of-flight
sensors at the nose (one looking ahead for a wall, one down past the nose for an edge); a tether eye on top.

A leg (its own frame: r along the leg, t across it, z out of the wall; built at the yaw axis then turned and placed):
the yaw horn on the top plate carries the **coxa C-bracket**, which reaches over the body's edge and down to a cradle
holding the **lift servo** (shaft across the leg, 12 mm off the wall) with the **knee servo** stacked on it. The
**femur** hangs on the lift horn; the knee servo's horn drives the **tibia** through a carbon **pushrod** parallel to
the femur. The tibia ends in a socket that takes a **foot module** — a snap-in dovetail clip, a crossed-flexure
silicone ankle (virtual pivot 0.6 mm above the adhesive), a magnet read by the Hall sensor in the tibia tip, and one of:

* ``pad`` — an 18 × 14 mm FR4 backing plate, 1 mm of soft foam, the fibrillar dry-adhesive film (glass, acrylic,
  varnished wood);
* ``spine`` — a TPU toe with sixteen 0.2 mm steel spines in two rows, each on its own flexure, 50° to the surface,
  tips facing down the body (the clip indexes in 15° steps so every leg's spines face down) (bark, brick);
* ``hybrid`` — a 14 × 14 mm pad with six spines along its toe edge.

``part``: 'arachne' (the whole robot on the wall), 'body', 'leg' (one leg in its own frame), 'servo',
'foot_pad', 'foot_spine', 'foot_hybrid' (in the foot's frame: the pad's centre at the origin, toe to +x), 'tail'.
``feet``: which foot module the whole robot wears; ``tail``: the bark kit's tail fitted.
"""
import math

import cadquery as cq
from vegeta.dedalus import Design, Parameter

import arachne_specs as sp
import shape_kit as sk

__all__ = ["Arachne", "servo_body", "leg_local", "foot_module"]


def _box(x0, x1, y0, y1, z0, z1):
    return cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0).translate(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))


def servo_body(axis="z"):
    """The micro servo, its shaft at the origin pointing +z (``axis`` 'z'), the case below the shaft and extending to
    +x from 5 mm behind it: case, mounting tabs, shaft and a round horn."""
    L, W, H = sp.SERVO["case"]
    tl, tt = sp.SERVO["tabs"]
    e = sp.SERVO["shaft_from_end"]
    case = _box(-e, L - e, -W / 2, W / 2, -H, 0)
    tabs = _box(-e - (tl - L) / 2, L - e + (tl - L) / 2, -W / 2, W / 2, -H * 0.3 - tt, -H * 0.3)
    boss = cq.Workplane("XY").circle(2.6).extrude(1.0)
    shaft = cq.Workplane("XY").circle(1.0).extrude(2.6)
    return sk.union_all([case, tabs, boss, shaft])


def foot_module(kind="pad", spine_dir_deg=180.0):
    """A foot module in its own frame: the adhesive (or spine tips) on z = 0, the pad's centre at the origin, toe to +x.
    ``spine_dir_deg``: the direction the spine tips face in this frame (the clip's index)."""
    t_film = sp.FOOT_PAD["adhesive_film"]
    t_foam = sp.FOOT_PAD["backing_foam"]
    t_plate = sp.FOOT_PAD["backing_plate"]
    z_plate = t_film + t_foam
    z_top = z_plate + t_plate                                             # the backing plate's top: 1.85 mm
    parts = []
    if kind in ("pad", "hybrid"):
        a, b = sp.FOOT_PAD["pad"] if kind == "pad" else sp.FOOT_HYBRID["pad"]
        x_off = 0.0 if kind == "pad" else -2.0
        parts.append(cq.Workplane("XY").box(a, b, t_film).edges("|Z").fillet(1.5).translate((x_off, 0, t_film / 2)))
        parts.append(cq.Workplane("XY").box(a, b, t_foam).edges("|Z").fillet(1.5).translate((x_off, 0, t_film + t_foam / 2)))
        parts.append(cq.Workplane("XY").box(a, b, t_plate).edges("|Z").fillet(1.5).translate((x_off, 0, z_plate + t_plate / 2)))
    if kind in ("spine", "hybrid"):
        n = sp.FOOT_SPINE["spines"] if kind == "spine" else sp.FOOT_HYBRID["spines"]
        rows = sp.FOOT_SPINE["rows"] if kind == "spine" else 1
        per = n // rows
        if kind == "spine":
            toe = cq.Workplane("XY").box(18.0, 14.0, 2.4).edges("|Z").fillet(2.0).translate((0, 0, 1.6 + 1.2))
            parts.append(toe)
            for k in range(per):                                          # flexure slots between the spines
                y = -6.0 + 12.0 * (k + 0.5) / per
                parts[-1] = parts[-1].cut(_box(-8.0, 8.0, y - 0.15, y + 0.15, 1.6, 3.0))
            x_rows = (-3.0, 3.0)
            z_base = 1.8
        else:
            x_rows = (6.0,)
            z_base = z_top
        a = math.radians(sp.FOOT_SPINE["spine_angle_deg"])
        d = math.radians(spine_dir_deg)
        reach = z_base / math.tan(a)
        for xr in x_rows:
            for k in range(per):
                y = -6.0 + 12.0 * (k + 0.5) / per + (0.4 if xr > 0 else -0.4)
                base = (xr, y, z_base)
                tip = (xr + reach * math.cos(d), y + reach * math.sin(d), 0.0)
                parts.append(sk.rod(base, tip, sp.FOOT_SPINE["pin_diameter"] / 2))
    # the crossed-flexure ankle: two silicone strips whose lines cross 0.6 mm above the interface
    pv = sp.FOOT_PAD["pivot_height"]
    z0, z1 = (z_top if kind != "spine" else 4.0), 6.0
    slope = (z0 - pv) / 1.2
    for sx in (-1, 1):
        a0 = (sx * 1.2, 0.0, z0)
        a1 = (sx * (1.2 + (z1 - z0) / slope), 0.0, z1)
        parts.append(sk.rod(a0, a1, 0.45).union(sk.rod((a0[0], 1.8, a0[2]), (a1[0], 1.8, a1[2]), 0.45))
                     .union(sk.rod((a0[0], -1.8, a0[2]), (a1[0], -1.8, a1[2]), 0.45)))
    magnet = cq.Workplane("XY").circle(sp.FOOT_PAD["magnet"][0] / 2).extrude(sp.FOOT_PAD["magnet"][1]).translate((0, 0, z0))
    clip = _box(-5.0, 5.0, -4.0, 4.0, z1, z1 + 2.6)
    clip = clip.cut(_box(-6.0, 6.0, -1.6, 1.6, z1 + 1.0, z1 + 2.7))      # the dovetail slot the tibia tip slides into
    stop = _box(4.2, 5.0, -4.0, 4.0, z1 - 0.8, z1)                       # the peel stop (meets the tibia at 25°)
    return sk.union_all(parts + [magnet, clip, stop])


def leg_local(foot="pad", spine_dir_deg=180.0, with_foot=True):
    """One leg in its own frame (r → +x along the leg, t → +y across it, z out of the wall), the yaw axis at x = y = 0,
    standing: horn, C-bracket, lift and knee servos, femur, pushrod, tibia with the Hall sensor, the foot module
    (``with_foot`` False: without it; "parts": a dict of the parts, each in place, for drawings and colour)."""
    B, Lg = sp.BODY, sp.LEG
    z_top = B["belly_clearance"] + B["height"]
    rl, zl = Lg["coxa"], Lg["lift_axis_height"]
    zk_s = zl + Lg["knee_servo_above"]
    a = math.radians(Lg["femur_rise_deg"])
    knee = (rl + Lg["femur"] * math.cos(a), 0.0, zl + Lg["femur"] * math.sin(a))
    dz = knee[2] - Lg["ankle_height"]
    ankle = (knee[0] + math.sqrt(Lg["tibia"] ** 2 - dz ** 2), 0.0, Lg["ankle_height"])
    L, W, H = sp.SERVO["case"]
    e = sp.SERVO["shaft_from_end"]
    horn = cq.Workplane("XY").circle(6.5).extrude(1.5).translate((0, 0, z_top))
    # the C-bracket: a top arm from the horn over the body edge, a side plate down the far side of the servo stack,
    # a cradle under the lift servo
    zt = z_top + 1.5
    top_arm = _box(-4.0, rl + L - e + 1.0, -H / 2 - 2.2, 4.0, zt, zt + 2.0)
    side = _box(rl - e - 1.0, rl + L - e + 1.0, -H / 2 - 2.2, -H / 2 - 0.2, zl - W / 2 - 1.0, zt + 2.0)
    cradle = _box(rl - e - 1.0, rl + L - e + 1.0, -H / 2 - 2.2, H / 2, zl - W / 2 - 1.0, zl - W / 2)
    side = side.cut(_box(rl - e + 1.0, rl + L - e - 1.0, -H / 2 - 3.0, -H / 2 + 1.0, zl - W / 2 + 1.5, zt - 1.5))
    top_arm = top_arm.cut(_box(4.0, rl + L - e - 2.0, -H / 2 + 0.5, 2.5, zt - 1.0, zt + 3.0))  # open frames, not slabs
    bracket = top_arm.union(side).union(cradle)
    # the servos lie on their side: shaft across the leg (+t), the case from 5 mm behind the shaft outwards
    def lying(z_axis):
        return servo_body().rotate((0, 0, 0), (1, 0, 0), -90).translate((rl, H / 2, z_axis))
    lift_servo = lying(zl)
    knee_servo = lying(zk_s)
    t_link = H / 2 + 3.6                                                  # the links run outside the horns
    femur = sk.rod((rl, t_link, zl), (knee[0], t_link, knee[2]), 1.8).union(sk.ball((rl, t_link, zl), 3.5))
    # the parallelogram: the knee servo's horn arm sits 8.3 mm above the lift axis, the tibia's lever 8.3 mm above
    # the knee, and the pushrod joins them parallel to the femur
    k_horn = (rl, t_link + 2.6, zk_s)
    k_lever = (knee[0], t_link + 2.6, knee[2] + (zk_s - zl))
    pushrod = sk.rod(k_horn, k_lever, sp.LEG["pushrod"][0] / 2).union(sk.ball(k_horn, 1.0)).union(sk.ball(k_lever, 1.0))
    tibia = sk.rod((knee[0], t_link, knee[2]), (ankle[0], 0.0, ankle[2] + 2.0), 1.6) \
        .union(sk.rod((knee[0], t_link, knee[2]), (knee[0], t_link + 2.6, knee[2] + (zk_s - zl)), 1.2)) \
        .union(sk.ball((knee[0], t_link, knee[2]), 2.6))
    hall = _box(ankle[0] - 1.0, ankle[0] + 1.0, -1.0, 1.0, ankle[2] + 2.0, ankle[2] + 2.6)
    parts = {"horn": horn, "bracket": bracket, "lift_servo": lift_servo, "knee_servo": knee_servo, "femur": femur,
             "pushrod": pushrod, "tibia": tibia, "hall": hall}
    if with_foot:
        parts["foot"] = foot_module(foot, spine_dir_deg).translate((ankle[0], 0.0, 0.0))
    if with_foot == "parts":
        return parts
    return sk.union_all(list(parts.values()))


def _rounded_plate(L, W, t, z0, fillet=10.0):
    return cq.Workplane("XY").box(L, W, t).edges("|Z").fillet(fillet).translate((0, 0, z0 + t / 2))


class Arachne(Design):
    """ARACHNE, the palm-sized eight-legged wall climber, standing on a wall (see the module docstring)."""

    parameters = [
        Parameter("part", "arachne", choices=("arachne", "body", "leg", "servo", "foot_pad", "foot_spine",
                                              "foot_hybrid", "tail"), description="what to build"),
        Parameter("feet", "pad", choices=("pad", "spine", "hybrid"), description="the foot module fitted"),
        Parameter("tail", False, description="the bark kit's passive tail fitted"),
    ]

    def body(self, p):
        B = sp.BODY
        z0, t, gap = B["belly_clearance"], B["plate"], B["gap"]
        L, W = B["length"], B["width"]
        belly = _rounded_plate(L, W, t, z0)
        top = _rounded_plate(L, W, t, z0 + t + gap)
        for x in (-30.0, 0.0, 30.0):                                      # lightening holes in both plates
            belly = belly.cut(cq.Workplane("XY").box(14, 10, 10).edges("|Z").fillet(3).translate((x, 0, z0)))
        posts = [cq.Workplane("XY").circle(2.2).extrude(gap).translate((x, y, z0 + t))
                 for x in (-46.0, 0.0, 46.0) for y in (-28.0, 28.0)]
        servos = []
        for leg in sp.LEGS:                                                # the yaw servos: shaft up at the hip
            (hx, hy, _), _ = sp.hip(leg)
            sy = 1 if hy > 0 else -1
            s = servo_body().rotate((0, 0, 0), (0, 0, 1), -90 * sy)         # the case runs inwards from the shaft
            servos.append(s.translate((hx, hy, z0 + t + sp.SERVO["case"][2])))   # the case on the belly plate
        bx, by, bz = B["battery"]
        battery = cq.Workplane("XY").box(bx, by, bz).edges("|X").fillet(1.5).translate((0, 0, z0 + t + bz / 2))
        board = _box(-30.0, 30.0, -13.0, 13.0, z0 + 2 * t + gap, z0 + 2 * t + gap + 0.8)
        esp = _box(-8.0, 8.0, -6.5, 6.5, z0 + 2 * t + gap + 0.8, z0 + 2 * t + gap + 3.0)
        tof_ahead = _box(L / 2 - 1.0, L / 2 + 1.5, -2.5, 2.5, z0 + t + 8.0, z0 + t + 13.0)
        tof_down = _box(L / 2 - 4.0, L / 2 + 1.0, -2.5, 2.5, z0 - 1.0, z0)
        eye = cq.Workplane("XZ").circle(4.0).circle(2.6).extrude(1.0, both=True).translate((-10.0, 0, z0 + 2 * t + gap + 4.0))
        return sk.union_all([belly, top] + posts + servos + [battery, board, esp, tof_ahead, tof_down, eye])

    def leg(self, leg, p):
        u = sp.radial(leg)
        (hx, hy, _), _ = sp.hip(leg)
        ang = math.degrees(math.atan2(u[1], u[0]))
        spine_dir = 180.0 - ang                                           # in the leg frame: tips face body −x
        part = leg_local(p["feet"], spine_dir)
        return part.rotate((0, 0, 0), (0, 0, 1), ang).translate((hx, hy, 0.0))

    def tail(self, p):
        T = sp.TAIL
        B = sp.BODY
        x0 = -B["length"] / 2
        z0 = B["belly_clearance"] + 4.0
        strip = sk.polyline_limb([(x0 - 2.0, 0, z0), (x0 - 25.0, 0, z0 - 1.0), (T["x"], 0, 2.2)], [1.2, 1.0])
        return strip.union(sk.ball((T["x"], 0, 2.2), 2.2))

    def build(self, p):
        part = p["part"]
        if part == "servo":
            return servo_body()
        if part.startswith("foot_"):
            return foot_module(part[5:])
        if part == "leg":
            return leg_local(p["feet"])
        if part == "tail":
            return self.tail(p)
        body = self.body(p)
        if part == "body":
            return body
        parts = [body] + [self.leg(leg, p) for leg in sp.LEGS]
        if p["tail"]:
            parts.append(self.tail(p))
        solids = [v for q in parts for v in q.vals()]                     # an assembly of parts, not one fused solid
        return cq.Workplane("XY").add(cq.Compound.makeCompound(solids))
