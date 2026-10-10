"""FALCO — a 2.4 m twin-boom pusher for mountain survey: NISUS's style, bigger, with the energy and the power to climb
fast to 4500 m and the crow flaps to come down fast (notebook 33).

The aircraft: NISUS's layout scaled up and reworked for the mountains — a straight-tapered wing of 2400 mm span (300/200
mm chords, 0.60 m², AR 9.6) in **three pieces** for a backpack (a 1.1 m centre section with the flaps, two 0.65 m outer
panels with the ailerons on a carbon spar joiner; a 20 mm carbon spar), a rounded pod (760 x 130 x 140 mm) that carries a 6S Li-ion pack of
21700 cells under the wing, the Jetson Orin, a lidar and a pitot in the nose; one electric pusher (a fixed 15 x 8 propeller on a
41xx-class motor) between two 16 mm carbon booms 500 mm apart; an H-tail on the booms; a belly skid.

What is new against NISUS (whose shapes this class inherits — nothing is designed twice):

- **flaps and ailerons as real cut surfaces**: inboard flaps (from the boom fittings to the panel joint, 30 % chord) and
  outboard ailerons (25 % chord); ``flap_deg`` and ``aileron_deg`` deflect them in the CAD (positive: trailing edge
  down). **Crow** is flaps down and ailerons up (``flap_deg=55, aileron_deg=-25``): the drag brake of the fast descent;
- **the panel joint** at ``wing_joint_y`` with a carbon **spar joiner** inside the main spar (``spar_joiner``);
- the motor mount's bolt pattern, the trays, the tail fittings and the skid are parameters (NISUS had them as literals);
- ``transport_check``: the longest piece against a backpack.

Frame and conventions as ``nisus`` (wing root leading edge at the origin, x aft, y right, z up; mm). The module helpers
(``resolve``, ``planform``, ``wetted_areas``, ``drag_buildup``, ``outline``, ``exploded_parts``, ``boom_check``) are
NISUS's with ``design=Falco()``; ``planform_split`` cuts the wing's quads at the flap and aileron stations for the
lattice (``falco_flight``).

``part``: as NISUS's (``aircraft``, ``wing``, ``spar``, ``rear_spar``, ``pod``, ``nose``, ``boom``, ``boom_fitting``,
``tail``, ``tail_fitting``, ``motor_mount``, ``tray``, ``skid``, ``battery_tray``) plus ``flap`` and ``aileron`` (the right
surfaces, undeflected) and ``spar_joiner``; FEA parts NISUS's plus ``spar_joiner_fea`` (the joiner in five pieces).
"""
from __future__ import annotations

import math

import cadquery as cq
import numpy as np
from vegeta.dedalus import Parameter

import nisus
from fixed_wing import FixedWing
from nisus import Nisus, naca4

NAME = "Falco-Zero (autonomous mountain survey, Jetson Orin onboard)"
PARTS = nisus.PARTS + ("flap", "aileron", "spar_joiner")
FEA_PARTS = nisus.FEA_PARTS + ("spar_joiner_fea",)
PER_CALL = ("part", "show_prop", "angle_of_attack_deg", "flap_deg", "aileron_deg")


def _nf(name, default=None, **kw):
    """NISUS's parameter ``name`` with a new default where FALCO differs."""
    p = next(q for q in Nisus.parameters if q.name == name)
    f = dict(name=p.name, default=p.default, units=p.units, min=p.min, max=p.max, description=p.description, choices=p.choices)
    if default is not None:
        f["default"] = default
    f.update(kw)
    return Parameter(**f)


class Falco(Nisus):
    """FALCO: NISUS's twin-boom pusher, 2.4 m, with flaps, ailerons, a three-piece wing and a 6S Li-ion pack."""

    parameters = [
        Parameter("part", "aircraft", choices=PARTS + FEA_PARTS, description="what to build"),
        _nf("show_prop"),
        _nf("span", 2400.0),
        _nf("root_chord", 300.0),
        _nf("tip_chord", 200.0),
        _nf("dihedral_deg", 3.0),
        _nf("wing_incidence_deg", 1.5),
        _nf("camber", 0.03), _nf("camber_pos", 0.4), _nf("thickness", 0.12),
        _nf("spar_x_frac", 0.30),
        _nf("spar_od", 20.0, description="20/17 carbon: the mountain gust (Pratt, 10 m/s at 22 m/s EAS) sized it; 16/14 failed the hand check"), _nf("spar_id", 17.0),
        _nf("spar_half_length", 1000.0, description="the carbon tube (centre section and outer panels with the joiner) reaches this far from the centre"),
        _nf("rear_spar_x_frac", 0.55),
        _nf("rear_spar_od", 18.0), _nf("rear_spar_id", 14.0, description="rear carry-through tube ID: a 2 mm wall (18/15 reached margin -0.04 in the tail case; a 20 mm tube thins the boom fitting's web under its rear ring)"),
        _nf("rear_spar_half_length", 300.0),
        _nf("aileron_chord_frac", 0.25),
        _nf("aileron_span_frac", 0.56, description="outer share of the half span with an aileron (the outer panel from just outboard of the joint)"),
        Parameter("flap_y0", 275.0, "mm", min=50, description="flap inboard end (just outboard of the boom fitting)"),
        Parameter("flap_y1", 535.0, "mm", min=100, description="flap outboard end (just inboard of the panel joint)"),
        Parameter("flap_chord_frac", 0.30, "", min=0.1, max=0.45),
        Parameter("wing_joint_y", 550.0, "mm", min=200, description="centre section / outer panel joint: the centre piece is 2 x this long"),
        Parameter("joiner_od", 16.8, "mm", min=4, description="spar joiner tube, a slide fit inside the main spar"),
        Parameter("joiner_id", 13.5, "mm", min=0),
        Parameter("joiner_length", 240.0, "mm", min=40, description="half in the centre section's spar, half in the panel's"),
        Parameter("flap_deg", 0.0, "deg", min=-10, max=70, description="flap deflection in the CAD (trailing edge down +)"),
        Parameter("aileron_deg", 0.0, "deg", min=-40, max=30, description="both ailerons, symmetric (crow: up, negative)"),
        _nf("pod_length", 760.0), _nf("pod_width", 130.0), _nf("pod_height", 140.0),
        _nf("nose_length", 400.0, description="pod nose tip ahead of the wing leading edge (the pack and the Orin sit in it)"),
        _nf("nose_cone_length", 90.0), _nf("pod_wall", 1.5),
        _nf("boom_y", 250.0, description="boom centre from the symmetry plane (500 mm spacing)"),
        _nf("boom_od", 20.0, description="boom tube OD (16/14 bent ~50 mm at the tail under the tail + fin ultimate load in the FEA, margin 0.41 by hand: 20/18 doubles the stiffness for ~35 g)"), _nf("boom_id", 18.0),
        _nf("boom_length", 880.0), _nf("boom_x0", 120.0, description="tube front end inside the root fitting: a 170 mm socket (an 88 mm one let the boom pry the fitting apart: the first FEA)"), _nf("boom_z", -28.0),
        _nf("fitting_width", 30.0), _nf("fitting_ring_wall", 4.0), _nf("fitting_web", 18.0), _nf("socket_od", 30.0),
        _nf("tail_span", 560.0), _nf("tail_chord", 190.0), _nf("tail_thickness", 0.06), _nf("tail_incidence_deg", 1.0),
        _nf("elevator_frac", 0.35),
        _nf("fin_height", 200.0), _nf("fin_ventral", 110.0), _nf("fin_chord", 190.0), _nf("rudder_frac", 0.35),
        _nf("motor_diameter", 49.0, description="motor bell (T-Motor AT4125 class)"),
        _nf("motor_length", 42.0),
        _nf("motor_z", 40.0, description="thrust line height at the pod's tail, above the pod's top line: the 15-inch disc clears the ground at rest"),
        _nf("motor_downthrust_deg", 6.0),
        _nf("mount_thickness", 6.0),
        Parameter("motor_bolt_a", 25.0, "mm", min=8, description="motor cross bolt pattern, horizontal pair spacing (assumed for the 41xx class: check the drawing)"),
        Parameter("motor_bolt_b", 30.0, "mm", min=8, description="motor cross bolt pattern, vertical pair spacing"),
        Parameter("motor_bolt_d", 4.2, "mm", min=2, description="bolt clearance hole (M4)"),
        _nf("prop_diameter", 381.0, description="15 inch (APC 15x8E, fixed: it brakes and regenerates)"),
        _nf("prop_gap", 10.0),
        _nf("skid_depth", 70.0), _nf("skid_x0", -80.0), _nf("skid_x1", 190.0),
        Parameter("skid_width", 10.0, "mm", min=3, description="belly keel width (TPU)"),
        Parameter("battery_tray_width", 80.0, "mm", min=30, description="battery cradle inner width (the 6S pack's 74 mm + foam)"),
        _nf("angle_of_attack_deg"),
    ]

    # ------------------------------------------------------------------------------------------- numbers
    @staticmethod
    def layout(p) -> dict:
        """NISUS's layout plus the flaps, the panel joint, the transport pieces."""
        L = Nisus.layout(p)
        c0, c1, b2, yc = p["root_chord"], p["tip_chord"], L["b2"], L["yc"]

        def chord(y):
            return c0 + (c1 - c0) * max(0.0, (abs(y) - yc) / (b2 - yc))

        y0, y1 = p["flap_y0"], p["flap_y1"]
        S_flap = p["flap_chord_frac"] * 0.5 * (chord(y0) + chord(y1)) * (y1 - y0) * 1e-6
        pieces = {"centre section": 2 * p["wing_joint_y"], "outer panel": b2 - p["wing_joint_y"], "boom": p["boom_length"],
                  "pod": p["pod_length"], "tail (stabiliser)": p["tail_span"]}
        L.update({"S_flap_each": S_flap, "y_flap0": y0, "y_flap1": y1, "flap_span_share": 2 * (y1 - y0) / p["span"],
                  "flapped_area_share": 2 * 0.5 * (chord(y0) + chord(y1)) * (y1 - y0) * 1e-6 / L["S_ref"],
                  "y_joint": p["wing_joint_y"], "chord_joint": chord(p["wing_joint_y"]), "pieces_mm": pieces,
                  "longest_piece_mm": max(pieces.values())})
        return L

    @staticmethod
    def bays(p) -> dict:
        """The pod's bays: camera, lidar and pitot in the nose cone; the Jetson Orin behind it (the computer bay); the 6S
        pack on the floor of the battery bay under the wing — near the centre of gravity, where it slides to set it; the
        flight controller tray above the pack; the ESC, the buck regulators and the wiring at the tail with the vents."""
        L = Falco.layout(p)
        w = p["pod_width"] - 2 * p["pod_wall"]
        h = p["pod_height"] - 2 * p["pod_wall"]
        x0 = L["x_nose"]
        return {"camera, lidar, pitot (nose cone)": (x0 + 10, x0 + p["nose_cone_length"] - 10.0, w * 0.6, h * 0.6),
                "computer bay (the Orin)": (x0 + p["nose_cone_length"], x0 + p["nose_cone_length"] + 130.0, w, h),
                "battery bay": (-180.0, 130.0, w, h),
                "flight controller tray (above the pack)": (-30.0, 100.0, w, h),
                "ESC, regulators, wiring, vents": (140.0, L["x_pod_end"] - 20.0, w * 0.8, h * 0.8)}

    # ------------------------------------------------------------------------------------------- shapes
    def _hinge(self, p, y, frac):
        """(x, z) [mm] of a hinge line at ``frac`` of the local chord at station y, on the camber line, before the incidence."""
        L = self.layout(p)
        f = max(0.0, (abs(y) - L["yc"]) / (L["b2"] - L["yc"]))
        c = p["root_chord"] + (p["tip_chord"] - p["root_chord"]) * f
        x = L["le_sweep_tip"] * f + frac * c
        m_, pc = p["camber"], p["camber_pos"]
        ycam = (m_ / pc ** 2 * (2 * pc * frac - frac ** 2)) if frac < pc else (m_ / (1 - pc) ** 2 * (1 - 2 * pc + 2 * pc * frac - frac ** 2))
        z = max(0.0, abs(y) - L["yc"]) * math.tan(math.radians(p["dihedral_deg"])) + ycam * c
        return x, z

    def surfaces(self, p) -> list:
        """The movable surfaces: (name, side, y0, y1 (|y|), chord fraction of the hinge, deflection [deg])."""
        L = self.layout(p)
        out = []
        for s in (1, -1):
            out.append(("flap", s, p["flap_y0"], p["flap_y1"], 1 - p["flap_chord_frac"], p["flap_deg"]))
            out.append(("aileron", s, L["y_aileron0"], L["b2"] - 2.0, 1 - p["aileron_chord_frac"], p["aileron_deg"]))
        return out

    def _cutter(self, p, s, y0, y1, xfrac):
        """A prism behind the hinge line between |y| = y0 and y1 on side s (it holds the surface)."""
        xa, _ = self._hinge(p, y0, xfrac)
        xb, _ = self._hinge(p, y1, xfrac)
        pts = [(xa, s * y0), (xb, s * y1), (xb + 400.0, s * y1), (xa + 400.0, s * y0)]
        return cq.Workplane("XY").workplane(offset=-150.0).polyline(pts).close().extrude(400.0)

    def _surface(self, p, wing, s, y0, y1, xfrac, deg):
        part = wing.intersect(self._cutter(p, s, y0, y1, xfrac))
        if deg:
            ya, yb = (y0, y1) if s > 0 else (y1, y0)
            xa, za = self._hinge(p, ya, xfrac)
            xb, zb = self._hinge(p, yb, xfrac)
            part = part.rotate((xa, s * ya, za), (xb, s * yb, zb), deg)       # the axis points +y on both sides: + is trailing edge down
        return part

    def _wing_solid(self, p):
        """NISUS's wing (FixedWing's centre section and tapered panels) with the flaps and the ailerons cut out and turned
        by their deflections, then turned by the incidence. Undeflected, the wing stays one solid."""
        w = FixedWing._wing(self, self._fwp(p))
        if p["flap_deg"] or p["aileron_deg"]:
            moved = []
            for name, s, y0, y1, xfrac, deg in self.surfaces(p):
                if deg:
                    moved.append(self._surface(p, w, s, y0, y1, xfrac, deg))
                    w = w.cut(self._cutter(p, s, y0, y1, xfrac))
            for m in moved:
                w = w.union(m)
        if p["wing_incidence_deg"]:
            x = 0.25 * p["root_chord"]
            w = w.rotate((x, 0, 0), (x, 1, 0), p["wing_incidence_deg"])
        return w

    def _flap_part(self, p, name):
        w = FixedWing._wing(self, self._fwp(p))
        sfc = next(t for t in self.surfaces(p) if t[0] == name and t[1] == 1)
        return self._surface(p, w, 1, sfc[2], sfc[3], sfc[4], 0.0)

    def _joiner(self, p, pieces=False):
        """The spar joiner: a carbon tube inside the main spar, centred on the panel joint (right side), straight along the
        spar's outboard line. ``pieces``: in five pieces fused without cleaning for the FEA (in the centre spar, across
        the joint gap, then the part in the panel's spar as a 30 mm piece at the joint, the middle, a 30 mm piece at the
        end: the panel's bearing loads go on the two short pieces)."""
        yj, half = p["wing_joint_y"], p["joiner_length"] / 2
        xa, za = self.tube_point(p, yj - half, p["spar_x_frac"])
        xb, zb = self.tube_point(p, yj + half, p["spar_x_frac"])
        a, b = np.array([xa, yj - half, za]), np.array([xb, yj + half, zb])
        u = (b - a) / np.linalg.norm(b - a)
        cuts = [yj - half, yj - 2.0, yj + 2.0, yj + 32.0, yj + half - 30.0, yj + half] if pieces else [yj - half, yj + half]
        parts = []
        for y0, y1 in zip(cuts[:-1], cuts[1:]):
            p0 = a + u * (y0 - (yj - half)) / u[1]
            p1 = a + u * (y1 - (yj - half)) / u[1]
            d = cq.Vector(*(p1 - p0))
            seg = cq.Solid.makeCylinder(p["joiner_od"] / 2, d.Length, cq.Vector(*p0), d)
            if p["joiner_id"] > 0:
                seg = seg.cut(cq.Solid.makeCylinder(p["joiner_id"] / 2, d.Length + 0.2, cq.Vector(*(p0 - u * 0.1)), d))
            parts.append(seg)
        return cq.Workplane("XY").add(parts[0].fuse(*parts[1:]) if len(parts) > 1 else parts[0])

    def _tail_fitting(self, p, side=1):
        """The printed sleeve at the boom's end that carries the stabiliser and the fin (NISUS's, sized from the boom and
        the tail): a 2 mm wall sleeve over the tail chord, a saddle for the stabiliser's spar at 30 % of its chord, a
        post for the fin."""
        L = self.layout(p)
        y = side * p["boom_y"]
        x0, x1 = L["tail_le"], L["boom_x1"]
        r = p["boom_od"] / 2 + 2.0
        sleeve = cq.Workplane("YZ").workplane(offset=x0).center(y, p["boom_z"]).circle(r).extrude(x1 - x0)
        k = p["tail_chord"] / 140.0
        saddle = cq.Workplane("XY").box(30.0 * k, 2 * r + 4.0, 12.0, centered=(True, True, False)).translate((x0 + 0.3 * p["tail_chord"], y, p["boom_z"]))
        post = cq.Workplane("XY").box(40.0 * k, 5.0, 30.0, centered=(True, True, False)).translate((x0 + 0.3 * p["fin_chord"], y, p["boom_z"] - 15.0))
        fit = sleeve.union(saddle).union(post)
        bore = cq.Workplane("YZ").workplane(offset=x0 - 1).center(y, p["boom_z"]).circle(p["boom_od"] / 2 + 0.1).extrude(x1 - x0 + 2)
        return fit.cut(bore)

    def _motor_mount(self, p, tilt=True):
        """The printed cup on the pod's tail: a plate ``mount_thickness`` thick with the motor's ``motor_bolt_a`` /
        ``motor_bolt_b`` cross pattern of M4 holes and a 12 mm centre hole, a 15 mm skirt bonded inside the pod's end ring."""
        L = self.layout(p)
        r_out = p["motor_diameter"] / 2 + 2.0
        xe = L["x_pod_end"]
        plate = cq.Workplane("YZ").workplane(offset=xe).center(0, p["motor_z"]).circle(r_out - 0.5).extrude(p["mount_thickness"])
        skirt = cq.Workplane("YZ").workplane(offset=xe - 15.0).center(0, p["motor_z"]).circle(r_out).circle(r_out - 2.5).extrude(15.0)
        mount = plate.union(skirt)
        a, b = p["motor_bolt_a"] / 2, p["motor_bolt_b"] / 2
        holes = (cq.Workplane("YZ").workplane(offset=xe - 1).center(0, p["motor_z"]).pushPoints([(a, 0), (-a, 0), (0, b), (0, -b)])
                 .circle(p["motor_bolt_d"] / 2).extrude(p["mount_thickness"] + 2))
        centre = cq.Workplane("YZ").workplane(offset=xe - 16).center(0, p["motor_z"]).circle(6.0).extrude(25.0)
        mount = mount.cut(holes).cut(centre)
        return self._tilt(mount, p) if tilt else mount

    def motor_holes(self, p) -> dict:
        a, b = p["motor_bolt_a"] / 2, p["motor_bolt_b"] / 2
        return {"h1": (a, 0.0), "h2": (-a, 0.0), "h3": (0.0, b), "h4": (0.0, -b)}

    def _skid(self, p):
        """The belly keel: a ``skid_width`` TPU plate under the pod from ``skid_x0`` to ``skid_x1`` (NISUS's, wider)."""
        if p["skid_depth"] <= 0:
            return None
        prof = self.pod_profile(p)
        xs = np.linspace(p["skid_x0"], p["skid_x1"], 12)
        z_low = np.interp(xs, prof[:, 0], prof[:, 3] - prof[:, 2])
        bottom = -p["pod_height"] - p["skid_depth"]
        top = [(float(x), float(z) + 3.0) for x, z in zip(xs, z_low)]
        ramp = 0.3 * (p["skid_x1"] - p["skid_x0"])
        poly = top + [(p["skid_x1"], bottom), (p["skid_x0"] + ramp, bottom)]
        return cq.Workplane("XZ").polyline(poly).close().extrude(p["skid_width"] / 2, both=True)

    def _tray(self, p, bay="flight controller tray (above the pack)"):
        """The removable electronics tray: a 2.5 mm PETG plate with 30.5 x 30.5 mm M3 holes for the flight controller."""
        x0, x1, w, h = self.bays(p)[bay]
        tray = cq.Workplane("XY").box(x1 - x0 - 6.0, w - 10.0, 2.5, centered=(False, True, False)).translate((x0 + 3.0, 0, -p["pod_height"] / 2 - 12.0))
        holes = (cq.Workplane("XY").workplane(offset=-p["pod_height"] / 2 - 13.0).center(0.5 * (x0 + x1), 0)
                 .pushPoints([(15.25, 15.25), (-15.25, 15.25), (15.25, -15.25), (-15.25, -15.25)]).circle(1.6).extrude(5.0))
        slots = cq.Workplane("XY").workplane(offset=-p["pod_height"] / 2 - 13.0).center(0.5 * (x0 + x1), 0).pushPoints([(0, 40), (0, -40)]).slot2D(40.0, 5.0, 0).extrude(5.0)
        return tray.cut(holes).cut(slots)

    def _battery_tray(self, p):
        """The pack's cradle: a 2.5 mm floor with two 12 mm lips ``battery_tray_width`` apart and a strap slot."""
        x0, x1, w, h = self.bays(p)["battery bay"]
        z = -p["pod_height"] + p["pod_wall"] + 12.0
        bw = p["battery_tray_width"]
        floor = cq.Workplane("XY").box(x1 - x0 - 10.0, bw + 4.0, 2.5, centered=(False, True, False)).translate((x0 + 5.0, 0, z))
        lips = [cq.Workplane("XY").box(x1 - x0 - 10.0, 2.5, 14.0, centered=(False, True, False)).translate((x0 + 5.0, s * (bw / 2 + 1.25), z)) for s in (1, -1)]
        tray = floor.union(lips[0]).union(lips[1])
        slots = [cq.Workplane("XY").box(22.0, bw + 20.0, 5.0, centered=(True, True, False)).translate((x0 + f * (x1 - x0), 0, z - 1.0)) for f in (0.33, 0.67)]
        return tray.cut(slots[0]).cut(slots[1])

    def build(self, p):
        if p["flap_y1"] >= p["wing_joint_y"] or p["flap_y0"] <= p["boom_y"] + p["fitting_width"] / 2:
            raise ValueError("the flap must lie between the boom fitting and the panel joint")
        L = self.layout(p)
        if L["y_aileron0"] <= p["wing_joint_y"]:
            raise ValueError("the aileron must start outboard of the panel joint (raise wing_joint_y or lower aileron_span_frac)")
        if p["joiner_od"] >= p["spar_id"]:
            raise ValueError("the joiner must slide inside the main spar")
        if p["wing_joint_y"] + p["joiner_length"] / 2 > p["spar_half_length"]:
            raise ValueError("the joiner must end inside the outer panel's spar")
        part = p["part"]
        if part == "flap":
            return self._flap_part(p, "flap")
        if part == "aileron":
            return self._flap_part(p, "aileron")
        if part == "spar_joiner":
            return self._joiner(p)
        if part == "spar_joiner_fea":
            return self._joiner(p, pieces=True)
        return super().build(p)


# --------------------------------------------------------------------------------------------------- helpers
def overrides(p=None) -> dict:
    """A parameter set without the per-call keys (``PER_CALL``), to pass to ``Falco().generate(**overrides(p), part=...)``."""
    return {k: v for k, v in dict(p or {}).items() if k not in PER_CALL}


def resolve(p=None, **kw) -> dict:
    return nisus.resolve(p, Falco(), **kw)


def exploded_parts(p=None, spread=1.0) -> dict:
    """NISUS's exploded view of FALCO plus the flaps, ailerons and joiners; the wing in three pieces."""
    d = Falco()
    p = resolve(p)
    parts = nisus.exploded_parts(p, 1.4 * spread, design=d)
    s = 1.4 * spread
    parts["flap R"] = d._flap_part(p, "flap").translate((90 * s, 40 * s, 120 * s))
    parts["aileron R"] = d._flap_part(p, "aileron").translate((90 * s, 160 * s, 120 * s))
    parts["spar joiner R"] = d._joiner(p).translate((0, 60 * s, 220 * s))
    return parts


def planform(p=None) -> dict:
    return nisus.planform(p, Falco())


def wetted_areas(p=None) -> dict:
    return nisus.wetted_areas(p, Falco())


def drag_buildup(p=None, speed: float = 20.0, nu: float = 1.5e-5) -> dict:
    """NISUS's build-up (Raymer) on FALCO's geometry at ``speed`` and the kinematic viscosity ``nu`` of the altitude
    (``falco_systems.atmosphere``: ν grows from 1.46e-5 at sea level to 2.1e-5 at 4500 m, the Reynolds numbers drop)."""
    return nisus.drag_buildup(p, speed, nu, design=Falco())


def boom_check(p=None, **kw) -> dict:
    kw.setdefault("tail_load_N", 40.0)
    kw.setdefault("fin_side_load_N", 15.0)
    kw.setdefault("tail_mass_kg", 0.12)
    return nisus.boom_check(p, design=Falco(), **kw)


def planform_split(p=None) -> dict:
    """``planform`` with each wing half cut into spanwise quads at the pod side, the boom, the flap's ends, the joint and
    the aileron's start, and each quad tagged (``surface``: 'centre', 'plain', 'flap', 'aileron'), so the lattice can
    deflect the flaps and the ailerons (an incidence increment on their quads) — ``falco_flight``."""
    p = resolve(p)
    L = Falco.layout(p)
    pl = planform(p)
    c0, c1, yc, b2 = p["root_chord"], p["tip_chord"], L["yc"], L["b2"]
    dxt, dzt = L["le_sweep_tip"], (b2 - yc) * math.tan(math.radians(p["dihedral_deg"]))

    def le(y):
        f = max(0.0, (y - yc) / (b2 - yc))
        return np.array([dxt * f, y, dzt * f])

    def ch(y):
        f = max(0.0, (y - yc) / (b2 - yc))
        return c0 + (c1 - c0) * f

    cuts = sorted({yc, p["flap_y0"], p["flap_y1"], p["wing_joint_y"], L["y_aileron0"], b2})
    quads, tags = [], []
    centre = np.array([[0, 0, 0], [0, -yc, 0], [c0, -yc, 0], [c0, 0, 0]], float)
    for side in (-1, 1):
        quads.append(centre * (np.array([1, -1, 1]) if side == 1 else 1)); tags.append("centre")
        for ya, yb in zip(cuts[:-1], cuts[1:]):
            A, B = le(ya), le(yb)
            q = np.array([A, B, B + [ch(yb), 0, 0], A + [ch(ya), 0, 0]])
            q[:, 1] *= -1                                                # the -y half as nisus builds it, mirrored for +y
            if side == 1:
                q[:, 1] *= -1
            ym = 0.5 * (ya + yb)
            tag = "flap" if p["flap_y0"] <= ym <= p["flap_y1"] else ("aileron" if ym >= L["y_aileron0"] else "plain")
            quads.append(q); tags.append(tag)
    out = dict(pl)
    out["wing"] = [q / 1000 for q in quads]
    out["wing_tags"] = tags
    out["wing_side"] = [(-1 if np.mean(q[:, 1]) < 0 else 1) for q in quads]
    return out


def outline(p=None) -> dict:
    """NISUS's outline of FALCO plus the flap hinge lines, the panel joints (top view) and the pieces."""
    d = Falco()
    p = resolve(p)
    o = nisus.outline(p, design=d)
    L = Falco.layout(p)
    hinges = list(o["top"]["hinges"])
    joints = []
    for s in (1, -1):
        xa, _ = d._hinge(p, p["flap_y0"], 1 - p["flap_chord_frac"])
        xb, _ = d._hinge(p, p["flap_y1"], 1 - p["flap_chord_frac"])
        hinges.append(np.array([[xa, s * p["flap_y0"]], [xb, s * p["flap_y1"]]]) / 1000)
        xj = L["le_sweep_tip"] * (p["wing_joint_y"] - L["yc"]) / (L["b2"] - L["yc"])
        joints.append(np.array([[xj, s * p["wing_joint_y"]], [xj + L["chord_joint"], s * p["wing_joint_y"]]]) / 1000)
    o["top"]["hinges"] = hinges
    o["top"]["joints"] = joints
    o["layout"] = L
    return o


def transport_check(p=None, backpack_mm: float = 1150.0) -> dict:
    """The pieces the aircraft comes apart into and the longest against a backpack's carrying length (a 70-80 l
    mountain pack with the pieces strapped alongside: ~1.1-1.2 m; an assumption)."""
    p = resolve(p)
    L = Falco.layout(p)
    return {"pieces_mm": L["pieces_mm"], "longest_mm": L["longest_piece_mm"], "backpack_mm": backpack_mm,
            "fits": L["longest_piece_mm"] <= backpack_mm,
            "note": "centre section with the booms' root fittings, two outer panels on the joiner, two booms with the tail (the stabiliser "
                    "unbolts from one boom), the pod; the propeller off"}


__all__ = ["Falco", "NAME", "PARTS", "FEA_PARTS", "PER_CALL", "overrides", "resolve", "exploded_parts", "planform", "planform_split", "wetted_areas",
           "drag_buildup", "boom_check", "outline", "transport_check"]
