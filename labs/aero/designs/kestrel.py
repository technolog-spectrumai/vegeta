"""Kestrel-4Q — a zero-stagger, truss-braced biplane quadrotor tailsitter (notebook 40).

Two straight, equal-span wings one above the other with their leading edges at the **same** station (zero stagger:
there is no stagger parameter, both wings are built from one ``wing_le_x``), a slim central pod spanning the gap with
the camera in its nose, carbon posts at the motor stations and an X of diagonals in each bay, four **fixed** nacelles
— one at each wing at ``±motor_y`` — whose thrust axes are all parallel to the pod's axis (no tilt, no pivot), a
vertical tail plate in the centre plane through both wings (the dorsal and ventral fin), and the feet the tailsitter
stands on: one at each fin tip and a rod behind each nacelle, all ending at ``x_feet``.

Frame: x from the nose tip **aft** (the free stream comes along +x in level flight), y to the right (spanwise), z up in
level flight; the pod's axis is the x axis, the gap is centred on it (wings at z = ±gap/2). The propellers thrust
toward −x (toward the nose). Motors: M1 upper left, M2 upper right, M3 lower left, M4 lower right (seen from behind,
y < 0 is left). Standing on its feet the aircraft is nose-up with the ground at x = ``x_feet``.

``part``: ``aircraft`` (everything, the propellers as thin discs when ``show_props``), ``airframe`` (everything but
the propeller discs: the CFD body, the rotor disks replace the propellers), ``wing`` (one wing, the upper; the lower is
the same shape at −gap/2), ``pod``, ``truss`` (posts and diagonals), ``nacelle`` (one, at M2), ``tail`` (the fin plate
with its feet and the nacelle foot rods). ``angle_of_attack_deg`` turns the whole build nose-up about the y axis
through (``rotation_x``, 0, 0) — the CFD's body at an angle of attack (rotate the disk centres with ``rotate``).

Units mm. Concept geometry: every number is an input of this concept study, none is a validated design value.
"""
from __future__ import annotations

import math

import numpy as np

try:
    import cadquery as cq
except ImportError:                     # the layout (areas, stations, clearances) needs no CAD kernel
    cq = None
from vegeta.dedalus import Design, Parameter

PARTS = ("aircraft", "airframe", "wing", "pod", "truss", "nacelle", "tail", "elevon", "capsule")
MOTORS = ("M1", "M2", "M3", "M4")
MOTOR_SIDES = {"M1": (-1, 1), "M2": (1, 1), "M3": (-1, -1), "M4": (1, -1)}      # (sign of y, sign of z)
PER_CALL = ("part", "angle_of_attack_deg", "rotation_x", "show_props", "elevon_deg")


def naca4(camber, camber_pos, thickness, n=40, te_cut=0.98):
    """NACA 4-digit section, chord 1: (upper, lower) point lists from the trailing edge to the leading edge, cut at
    ``te_cut`` so the trailing edge has a finite thickness (as ``fixed_wing.naca4``, without importing CadQuery)."""
    m, p, t = camber, camber_pos, thickness
    xs = [te_cut * 0.5 * (1 - math.cos(math.pi * i / n)) for i in range(n + 1)]
    upper, lower = [], []
    for x in xs:
        yt = 5 * t * (0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)
        if m > 0 and x < p:
            yc, dyc = m / p ** 2 * (2 * p * x - x ** 2), 2 * m / p ** 2 * (p - x)
        elif m > 0:
            yc, dyc = m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x ** 2), 2 * m / (1 - p) ** 2 * (p - x)
        else:
            yc, dyc = 0.0, 0.0
        th = math.atan(dyc)
        upper.append((x - yt * math.sin(th), yc + yt * math.cos(th)))
        lower.append((x + yt * math.sin(th), yc - yt * math.cos(th)))
    return upper[::-1], lower[::-1]


class Kestrel4Q(Design):
    parameters = [
        Parameter("part", "aircraft", choices=PARTS, description="what to build"),
        # wings: equal span, equal chord, zero stagger
        Parameter("span", 1000.0, "mm", min=300, description="span of each wing (both equal)"),
        Parameter("chord", 140.0, "mm", min=40, description="chord of each wing (constant, straight wings)"),
        Parameter("gap", 280.0, "mm", min=80, description="vertical distance between the wings' chord planes"),
        Parameter("wing_le_x", 200.0, "mm", min=50, description="both leading edges, from the nose tip (zero stagger)"),
        Parameter("thickness", 0.12, "", min=0.06, max=0.2, description="NACA thickness ratio"),
        Parameter("camber", 0.0, "", min=0.0, max=0.06, description="NACA camber (0: symmetric, no pitching moment to trim without a tail)"),
        Parameter("camber_pos", 0.4, "", min=0.1, max=0.9),
        Parameter("spar_x", 0.30, "", min=0.1, max=0.6, description="main spar (and post) station, fraction of chord"),
        # pod
        Parameter("pod_length", 470.0, "mm", min=150, description="nose tip to pod tail (a 220 mm constant mid-body between the tapers: the bays)"),
        Parameter("pod_width", 110.0, "mm", min=40),
        Parameter("camera_diameter", 36.0, "mm", min=10, description="the forward-looking camera's dome in the nose"),
        Parameter("nose_length", 150.0, "mm", min=30, description="the pod's ellipsoidal nose, tip to full section (an ogive of ~1.3 widths)"),
        Parameter("tail_length", 100.0, "mm", min=20, description="the pod's tail taper, full section to the tail end"),
        Parameter("pod_wall", 6.0, "mm", min=2, description="the pod's shell and frames: the bays keep this clear of the skin"),
        # the payload bay (Part II): a removable insulated capsule for medical supplies, on the pod's axis
        Parameter("bay_length", 160.0, "mm", min=40, description="the capsule bay along x"),
        Parameter("bay_width", 72.0, "mm", min=20, description="the capsule bay across (y)"),
        Parameter("bay_height", 72.0, "mm", min=20, description="the capsule bay in z"),
        Parameter("bay_x", 152.0, "mm", min=0, description="the bay's front face from the nose tip (inside the constant mid-body, close behind the CG)"),
        Parameter("bay_z", 10.0, "mm", description="the bay's centre height above the pod's axis (the battery below, the electronics above)"),
        # propulsion: four fixed nacelles, thrust axes parallel to x
        Parameter("motor_y", 390.0, "mm", min=50, description="motor stations either side of the centre plane"),
        Parameter("prop_diameter", 228.6, "mm", min=50, description="9 inch"),
        Parameter("prop_x", 125.0, "mm", min=0, description="propeller plane from the nose tip (ahead of the leading edge)"),
        Parameter("nacelle_diameter", 34.0, "mm", min=10),
        Parameter("prop_clearance", 25.0, "mm", min=5, description="least clearance of a propeller disc to the other disc and to the pod"),
        # truss
        Parameter("post_diameter", 8.0, "mm", min=3, description="carbon interplane posts at the motor stations"),
        Parameter("diagonal_diameter", 6.0, "mm", min=2, description="carbon diagonals: an X in each bay"),
        Parameter("bracing", "x", choices=("x", "posts", "none"), description="the interplane truss: posts and X diagonals, posts only, none (Part II)"),
        Parameter("strut_fairing", False, "", description="posts and diagonals in streamlined sleeves (Kestrel-4Q-2's option)"),
        Parameter("fairing_tc", 0.33, "", min=0.15, max=0.6, description="the sleeve's thickness / chord (a symmetric NACA 00xx)"),
        Parameter("fairing_wall", 0.8, "mm", min=0.2, description="sleeve material outside the tube at its thickest"),
        # elevons (Part II): trailing-edge surfaces on the outer panels of both wings, behind the propellers
        Parameter("elevons", False, "", description="elevons on the outer panels of both wings"),
        Parameter("elevon_chord", 0.28, "", min=0.1, max=0.5, description="elevon chord, fraction of the wing chord"),
        Parameter("elevon_inner_y", 275.0, "mm", min=30, description="the elevons' inner end either side of the centre plane (the slipstream's inner edge)"),
        Parameter("elevon_gap", 1.0, "mm", min=0.2, description="gap between the wing and the elevon at the hinge and the inner end"),
        Parameter("elevon_deg", 0.0, "deg", min=-45, max=45, description="all four elevons turned (trailing edge down positive; display)"),
        # tail and feet
        Parameter("x_feet", 540.0, "mm", min=200, description="the feet's ends from the nose tip: the ground when standing"),
        Parameter("fin_height", 190.0, "mm", min=20, description="dorsal fin above the upper wing"),
        Parameter("ventral_height", 150.0, "mm", min=20, description="ventral fin below the lower wing"),
        Parameter("fin_tip_chord", 90.0, "mm", min=10),
        Parameter("fin_thickness", 4.0, "mm", min=1),
        Parameter("foot_rod_diameter", 6.0, "mm", min=2, description="carbon foot rods behind the nacelles"),
        # display and CFD
        Parameter("show_props", True, "", description="draw the propellers as thin discs (aircraft only)"),
        Parameter("angle_of_attack_deg", 0.0, "deg", min=-30, max=120, description="the whole build turned nose-up about y"),
        Parameter("rotation_x", 220.0, "mm", min=0, description="x of the axis the angle of attack turns about (the CG)"),
    ]

    # ------------------------------------------------------------------------------------------- the layout
    @staticmethod
    def layout(p) -> dict:
        """The numbers the other modules use, in metres (areas m², angles rad): no CAD kernel needed."""
        mm = 1e-3
        b, c, gap = p["span"] * mm, p["chord"] * mm, p["gap"] * mm
        h = gap / 2
        x_le = p["wing_le_x"] * mm
        R = p["prop_diameter"] * mm / 2
        y_m = p["motor_y"] * mm
        x_spar = x_le + p["spar_x"] * c
        motors = {k: {"x": p["prop_x"] * mm, "y": sy * y_m, "z": sz * h} for k, (sy, sz) in MOTOR_SIDES.items()}
        post_len = gap
        y_in = p["pod_width"] * mm / 2
        diag_len = math.hypot(y_m - y_in, gap)
        x_te = x_le + c
        return {
            "span": b, "chord": c, "gap": gap, "h": h, "x_le": x_le, "x_te": x_te, "x_ac": x_le + 0.25 * c,
            "x_spar": x_spar, "S_wing": b * c, "S_ref": 2 * b * c, "AR_wing": b / c, "gap_chord": gap / c,
            "gap_span": gap / b, "R_prop": R, "A_disc": math.pi * R ** 2, "motor_y": y_m, "motors": motors,
            "prop_x": p["prop_x"] * mm, "prop_ahead_of_le": x_le - p["prop_x"] * mm,
            "prop_gap": gap - 2 * R, "prop_to_pod": y_m - R - y_in, "prop_beyond_tip": y_m + R - b / 2,
            "pod_length": p["pod_length"] * mm, "pod_width": p["pod_width"] * mm, "pod_height": gap,
            "post_length": post_len, "diagonal_length": diag_len,
            "x_feet": p["x_feet"] * mm, "fin_height": p["fin_height"] * mm, "ventral_height": p["ventral_height"] * mm,
            "fin_tip_chord": p["fin_tip_chord"] * mm, "fin_root_x": x_te - 0.06 * 1.0,
            "foot_rod_length": p["x_feet"] * mm - (x_le + 0.6 * c),
            # the ground footprint standing nose-up (ground plane x = x_feet): the feet's (y, z)
            "feet": [(0.0, h + p["fin_height"] * mm), (0.0, -h - p["ventral_height"] * mm)]
                    + [(sy * y_m, sz * h) for sy, sz in MOTOR_SIDES.values()],
            # the elevons (four: both sides of both wings) and the bracing
            "elevons": bool(p["elevons"]), "elevon_cf": p["elevon_chord"], "elevon_y_in": p["elevon_inner_y"] * mm,
            "elevon_span_side": max(b / 2 - p["elevon_inner_y"] * mm, 0.0), "x_hinge": x_le + (1 - p["elevon_chord"]) * c,
            "S_elevon": 4 * max(b / 2 - p["elevon_inner_y"] * mm, 0.0) * p["elevon_chord"] * c, "bracing": p["bracing"],
            "diagonals": 4 if p["bracing"] == "x" else 0, "posts": 0 if p["bracing"] == "none" else 2,
        }

    @staticmethod
    def bays(p, battery_x: float | None = None) -> dict:
        """The pod's internal layout [mm]: boxes (x0, x1, y0, y1, z0, z1) for the camera, the flight stack, the mission
        computer, the battery (its centre at ``battery_x``, the mass table's choice), the capsule bay. The capsule sits
        on the axis so that its contents move the CG along x only; the battery lies below it, the electronics above."""
        bl, bw, bh, bx, bz = p["bay_length"], p["bay_width"], p["bay_height"], p["bay_x"], p["bay_z"]
        batt_l, batt_w, batt_h = 140.0, 44.0, 32.0
        bxc = battery_x if battery_x is not None else bx + bl / 2
        z_batt = bz - bh / 2 - 8.0 - batt_h / 2
        return {"camera": (0.0, 40.0, -18.0, 18.0, -18.0, 18.0),
                "flight stack": (bx, bx + 90.0, -30.0, 30.0, bz + bh / 2 + 8.0, bz + bh / 2 + 8.0 + 30.0),
                "mission computer": (bx + 95.0, bx + 95.0 + 60.0, -30.0, 30.0, bz + bh / 2 + 8.0, bz + bh / 2 + 8.0 + 30.0),
                "capsule bay": (bx, bx + bl, -bw / 2, bw / 2, bz - bh / 2, bz + bh / 2),
                "battery": (bxc - batt_l / 2, bxc + batt_l / 2, -batt_w / 2, batt_w / 2, z_batt - batt_h / 2, z_batt + batt_h / 2)}

    @staticmethod
    def bay_checks(p) -> list[str]:
        """The bays inside the pod's constant mid-body, clear of the skin by ``pod_wall``, and clear of each other."""
        out = []
        b = Kestrel4Q.bays(p)
        x_mid0, x_mid1 = p["nose_length"], p["pod_length"] - p["tail_length"]
        w2, h2 = p["pod_width"] / 2 - p["pod_wall"], 0.49 * p["gap"] - p["pod_wall"]
        for name in ("capsule bay", "flight stack", "mission computer"):
            x0, x1, y0, y1, z0, z1 = b[name]
            if x0 < x_mid0 - 1e-9 or x1 > x_mid1 + 1e-9:
                out.append(f"the {name} ({x0:.0f}-{x1:.0f} mm) reaches out of the constant mid-body ({x_mid0:.0f}-{x_mid1:.0f} mm)")
            if -y0 > w2 or y1 > w2 or -z0 > h2 or z1 > h2:
                out.append(f"the {name} does not clear the pod's skin by {p['pod_wall']:.0f} mm")
        names = list(b)
        for i, a in enumerate(names):
            for c in names[i + 1:]:
                A, C = b[a], b[c]
                if all(A[2 * k] < C[2 * k + 1] and C[2 * k] < A[2 * k + 1] for k in range(3)):
                    out.append(f"the {a} and the {c} overlap")
        return out

    @staticmethod
    def checks(p) -> list[str]:
        """The geometric rules the concept must keep; an empty list when all hold."""
        L = Kestrel4Q.layout(p)
        out = []
        if L["prop_gap"] * 1e3 < p["prop_clearance"]:
            out.append(f"the upper and lower propeller discs clear each other by {L['prop_gap'] * 1e3:.0f} mm "
                       f"(< {p['prop_clearance']:.0f} mm): open the gap or take a smaller propeller")
        if L["prop_to_pod"] * 1e3 < p["prop_clearance"]:
            out.append(f"the propeller discs clear the pod by {L['prop_to_pod'] * 1e3:.0f} mm (< {p['prop_clearance']:.0f} mm): "
                       "move the motors outboard")
        if L["prop_ahead_of_le"] * 1e3 < 30.0:
            out.append(f"the propeller plane is {L['prop_ahead_of_le'] * 1e3:.0f} mm ahead of the leading edge (< 30 mm)")
        if p["motor_y"] > p["span"] / 2 - p["nacelle_diameter"] / 2:
            out.append("the nacelles overhang the wing tips")
        if p["pod_length"] < p["wing_le_x"] + p["chord"]:
            out.append("the pod ends ahead of the trailing edge: it must carry both wings over their whole chord")
        if p["nose_length"] + p["tail_length"] > p["pod_length"] - 40.0:
            out.append("the nose and tail tapers leave no constant mid-body (shorten nose_length or tail_length)")
        if p["nose_length"] > p["wing_le_x"]:
            out.append("the nose taper must end at or ahead of the leading edge")
        if p["x_feet"] <= p["pod_length"]:
            out.append("the feet must reach aft of the pod's tail")
        if p["prop_x"] < p["camera_diameter"] / 2:
            out.append("the propeller plane is ahead of the camera dome")
        if p["elevons"] and not p["pod_width"] / 2 + 10.0 <= p["elevon_inner_y"] <= p["span"] / 2 - 30.0:
            out.append("the elevons' inner end must lie between the pod and 30 mm short of the tip")
        out += Kestrel4Q.bay_checks(p)
        return out

    # ------------------------------------------------------------------------------------------- parts
    def _section(self, p):
        """(upper, lower) point lists of the wing section, chord 1, from the trailing edge to the leading edge."""
        return naca4(p["camber"], p["camber_pos"], p["thickness"])

    def _wing(self, p, z0):
        c, x_le = p["chord"], p["wing_le_x"]
        up, lo = self._section(p)
        pts = [(x_le + c * u, z0 + c * v) for u, v in up] + [(x_le + c * u, z0 + c * v) for u, v in lo[::-1][1:]]
        wing = cq.Workplane("XZ").polyline(pts).close().extrude(p["span"] / 2, both=True)
        if p["elevons"]:                                   # the elevons' bays cut out of the trailing edge
            xh, g = x_le + (1 - p["elevon_chord"]) * c, p["elevon_gap"]
            y0, y1 = p["elevon_inner_y"] - g, p["span"] / 2 + 2.0
            for s in (1, -1):
                bay = cq.Workplane("XY").box(c, y1 - y0, c).translate((xh - g + c / 2, s * (y0 + y1) / 2, z0))
                wing = wing.cut(bay)
        return wing

    def _elevon(self, p, z0, s, deflect=True):
        """One elevon (side ``s``: +1 right, −1 left) on the wing at ``z0``: the section's aft ``elevon_chord`` from the
        hinge to the trailing edge, over the outer panel, turned by ``elevon_deg`` about the hinge line (on the camber
        line at the hinge station; trailing edge down positive)."""
        c, x_le, cf = p["chord"], p["wing_le_x"], p["elevon_chord"]
        up, lo = self._section(p)
        xh = 1 - cf
        up_aft = [(u, v) for u, v in up if u >= xh]          # TE → hinge
        lo_aft = [(u, v) for u, v in lo if u >= xh]
        pts = [(x_le + c * u, z0 + c * v) for u, v in up_aft[::-1]] + [(x_le + c * u, z0 + c * v) for u, v in lo_aft[1:]]
        y0, y1 = p["elevon_inner_y"], p["span"] / 2
        el = cq.Workplane("XZ").polyline(pts).close().extrude((y1 - y0) / 2, both=True).translate((0, s * (y0 + y1) / 2, 0))
        if deflect and p["elevon_deg"]:
            zh = z0 + c * float(np.interp(xh, [u for u, _ in up[::-1]], [0.5 * (a + b) for (_, a), (_, b) in zip(up[::-1], lo[::-1])]))
            el = el.rotate((x_le + c * xh, 0, zh), (x_le + c * xh, 1, zh), p["elevon_deg"])
        return el

    @staticmethod
    def _superellipse(wp, w, hgt, n=24, e=3.0):
        pts = []
        for k in range(n):
            t = 2 * math.pi * k / n
            ct, st = math.cos(t), math.sin(t)
            pts.append((w / 2 * math.copysign(abs(ct) ** (2 / e), ct), hgt / 2 * math.copysign(abs(st) ** (2 / e), st)))
        return wp.spline(pts, periodic=True).close()

    @staticmethod
    def pod_profile(p, n_nose: int = 7, n_tail: int = 4):
        """The pod's loft stations (x, width, height) [mm]: an ellipsoidal nose over ``nose_length`` (the section's
        width and height follow √(1 − ((L_n − x)/L_n)²): the tangent-ogive family's smooth member, no shoulder), a
        constant mid-body, a tail taper over ``tail_length`` to 45 % width / 55 % height. The nose tip carries the
        camera dome flush (a small flat of the dome's diameter)."""
        w, Hh, Lp, Ln, Lt = p["pod_width"], 0.98 * p["gap"], p["pod_length"], p["nose_length"], p["tail_length"]
        d_cam = p["camera_diameter"]
        out = []
        for k in range(n_nose + 1):
            x = Ln * k / n_nose
            f = math.sqrt(max(1.0 - ((Ln - x) / Ln) ** 2, 0.0))
            f = max(f, d_cam / w * 0.95 if k == 0 else 0.0)               # the tip: the camera's flat, not a point
            out.append((x, max(w * f, 2.0), max(Hh * f, 2.0)))
        out.append((Lp - Lt, w, Hh))
        for k in range(1, n_tail + 1):
            x = Lp - Lt + Lt * k / n_tail
            g = 1.0 - 0.55 * (k / n_tail) ** 1.5
            out.append((x, w * g, Hh * (1.0 - 0.45 * (k / n_tail) ** 1.5)))
        return out

    def _loft(self, stations, ruled):
        wp = cq.Workplane("YZ")
        for i, (x, ww, hh) in enumerate(stations):
            wp = self._superellipse(wp.workplane(offset=x - (stations[i - 1][0] if i else 0.0)), ww, hh)
        return wp.loft(ruled=ruled, combine=True)

    def _pod(self, p):
        """The nose and the mid-body as one smooth loft (the ellipsoid's stations are monotone), the tail taper as a
        ruled loft (a smooth loft through shrinking stations bulges between them), joined at the mid-body's end."""
        stations = self.pod_profile(p)
        i_mid = next(i for i, s in enumerate(stations) if s[0] >= p["pod_length"] - p["tail_length"] - 1e-6)
        body = self._loft(stations[:i_mid + 1], ruled=False).union(self._loft(stations[i_mid:], ruled=True))
        r = p["camera_diameter"] / 2
        cam = cq.Workplane("XY").sphere(r).translate((r * 0.75, 0, 0))     # the dome: a quarter of its radius proud of the tip
        return body.union(cam)

    def _faired(self, a, b, d, p):
        """A tube from a to b inside a streamlined sleeve whose chord lies along x (the free stream in cruise and the
        slipstream in the hover). The struts lie in the y-z plane, so the section perpendicular to the strut is the
        streamwise section: no sweep, no stretching."""
        a, b = cq.Vector(*a), cq.Vector(*b)
        axis = (b - a).normalized()
        c, pts = fairing_section(d, p["fairing_tc"], p["fairing_wall"])
        plane = cq.Plane(origin=a, xDir=cq.Vector(1, 0, 0), normal=axis)
        return cq.Workplane(plane).polyline(pts).close().extrude((b - a).Length)

    def _rod(self, a, b, d):
        a, b = cq.Vector(*a), cq.Vector(*b)
        return cq.Workplane("XY").add(cq.Solid.makeCylinder(d / 2, (b - a).Length, a, b - a))

    def _truss(self, p):
        h, x = p["gap"] / 2, p["wing_le_x"] + p["spar_x"] * p["chord"]
        y_m, y_in = p["motor_y"], p["pod_width"] / 2
        out = None
        if p["bracing"] == "none":
            return None
        for s in (1, -1):
            strut = (lambda a, b, d: self._faired(a, b, d, p)) if p["strut_fairing"] else self._rod
            parts = [strut((x, s * y_m, -h), (x, s * y_m, h), p["post_diameter"])]
            if p["bracing"] == "x":
                parts += [strut((x, s * y_in, h), (x, s * y_m, -h), p["diagonal_diameter"]),
                          strut((x, s * y_in, -h), (x, s * y_m, h), p["diagonal_diameter"])]
            for q in parts:
                out = q if out is None else out.union(q)
        return out

    def _nacelle(self, p, sy, sz, prop=False):
        y, z = sy * p["motor_y"], sz * p["gap"] / 2
        r = p["nacelle_diameter"] / 2
        x0, x1 = p["prop_x"] + 12.0, p["wing_le_x"] + 0.6 * p["chord"]
        can = cq.Workplane("YZ").workplane(offset=x0).center(y, z).circle(r).extrude(x1 - x0)
        spinner = cq.Workplane("XY").add(cq.Solid.makeCone(0.0, r * 0.8, 30.0, cq.Vector(p["prop_x"] - 18.0, y, z), cq.Vector(1, 0, 0)))
        hub = cq.Workplane("YZ").workplane(offset=p["prop_x"] + 10.0).center(y, z).circle(r * 0.8).extrude(4.0)
        out = can.union(spinner).union(hub)
        if prop:
            disc = cq.Workplane("YZ").workplane(offset=p["prop_x"] - 1.0).center(y, z).circle(p["prop_diameter"] / 2).extrude(2.0)
            out = out.union(disc)
        return out

    def _tail(self, p):
        h, xt, tc = p["gap"] / 2, p["x_feet"], p["fin_tip_chord"]
        x0 = p["wing_le_x"] + p["chord"] - 60.0
        zt, zb = h + p["fin_height"], -h - p["ventral_height"]
        pts = [(x0, h), (xt - tc, zt), (xt, zt), (xt, zb), (xt - tc, zb), (x0, -h)]
        fin = cq.Workplane("XZ").polyline(pts).close().extrude(p["fin_thickness"] / 2, both=True)
        for z in (zt, zb):
            fin = fin.union(cq.Workplane("XY").sphere(6.0).translate((xt - 6.0, 0, z)))
        x_rod = p["wing_le_x"] + 0.6 * p["chord"]
        for sy, sz in MOTOR_SIDES.values():
            fin = fin.union(self._rod((x_rod - 5.0, sy * p["motor_y"], sz * h), (xt, sy * p["motor_y"], sz * h), p["foot_rod_diameter"]))
        return fin

    def _capsule(self, p):
        """The payload capsule: a rounded box filling the bay less 2 mm all round."""
        x0, x1, y0, y1, z0, z1 = self.bays(p)["capsule bay"]
        box = cq.Workplane("XY").box(x1 - x0 - 4.0, y1 - y0 - 4.0, z1 - z0 - 4.0).edges().fillet(8.0)
        return box.translate(((x0 + x1) / 2, 0.0, (z0 + z1) / 2))

    def build(self, p):
        problems = self.checks(p)
        if problems:
            raise ValueError("; ".join(problems))
        part = p["part"]
        h = p["gap"] / 2
        if part == "wing":
            shape = self._wing(p, h)
        elif part == "pod":
            shape = self._pod(p)
        elif part == "truss":
            shape = self._truss(p)
            if shape is None:
                raise ValueError("bracing 'none': there is no truss to build")
        elif part == "elevon":
            if not p["elevons"]:
                raise ValueError("elevons are off: set elevons=True")
            shape = self._elevon(p, h, 1)
        elif part == "capsule":
            shape = self._capsule(p)
        elif part == "nacelle":
            shape = self._nacelle(p, 1, 1)
        elif part == "tail":
            shape = self._tail(p)
        else:
            shape = self._wing(p, h).union(self._wing(p, -h)).union(self._pod(p)).union(self._tail(p))
            truss = self._truss(p)
            if truss is not None:
                shape = shape.union(truss)
            if p["elevons"]:
                for z0 in (h, -h):
                    for s in (1, -1):
                        shape = shape.union(self._elevon(p, z0, s))
            for sy, sz in MOTOR_SIDES.values():
                shape = shape.union(self._nacelle(p, sy, sz, prop=(part == "aircraft" and p["show_props"])))
        a = p["angle_of_attack_deg"]
        if a:
            x0 = p["rotation_x"]
            shape = shape.rotate((x0, 0, 0), (x0, 1, 0), a)
        return shape


def fairing_section(d, tc=0.33, wall=0.8, n=30, tube_at=0.30):
    """A strut sleeve around a tube of diameter ``d`` [mm]: a symmetric NACA 00xx of thickness ratio ``tc`` whose
    thickest part (30 % chord) clears the tube by ``wall`` each side. Returns (chord, closed point list (x, t)) with the
    tube's centre at (0, 0): the leading edge at −``tube_at`` × chord."""
    c = (d + 2 * wall) / tc
    up, lo = naca4(0.0, 0.4, tc, n=n, te_cut=1.0)
    pts = [(c * (u - tube_at), c * v) for u, v in up] + [(c * (u - tube_at), c * v) for u, v in lo[::-1][1:-1]]
    return c, pts


def resolve(p=None, **kw) -> dict:
    d = dict(p or {})
    d.update(kw)
    return Kestrel4Q().resolve(**d)


def overrides(p=None) -> dict:
    """A parameter set without the per-call keys, to pass to ``Kestrel4Q().generate(**overrides(p), part=...)``."""
    return {k: v for k, v in dict(p or {}).items() if k not in PER_CALL}


def layout(p=None) -> dict:
    return Kestrel4Q.layout(resolve(p))


def rotate(point, alpha_deg, x0):
    """A point (x, y, z) of the level aircraft turned nose-up by ``alpha_deg`` about the y axis through (x0, 0, 0),
    as ``build`` turns the shape (CadQuery's right-hand rotation about +y: the nose, at smaller x, goes up)."""
    a = math.radians(alpha_deg)
    x, y, z = point
    dx = x - x0
    return (x0 + dx * math.cos(a) + z * math.sin(a), y, -dx * math.sin(a) + z * math.cos(a))
