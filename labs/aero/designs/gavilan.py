"""GAVILAN — a tailsitter that hunts quadcopters the way a goshawk hunts birds (notebook 38).

Gavilán is the sparrowhawk and goshawk in Spanish: the short-winged hawk that wins by manoeuvre, not by speed. The
machine is a **tailsitter with nothing that tilts but its camera**: a slender body, a short broad wing, four fixed
motors — two above and two below the wing, an X seen from the front — and a long tail that ends in four fins it
stands on. It lifts off and hovers on its tail as a quadcopter (pitch, roll and yaw by the four thrusts alone: no
control surface, no servo in the airframe), pitches over and flies on the wing. The one servo aboard tilts the nose
camera, so the camera stays on a bandit while the body points where the thrust has to go.

What this file builds (CadQuery through Dedalus):

- the **body**: an ogive nose that ends in the camera ball, a cylindrical bay the 6S packs slide into in line, and a
  cone that closes onto the tail boom;
- the **wing**: a short, broad, tapered wing on a symmetric section (it turns either way, and it flies inverted as
  well as upright), mid-mounted, the span held to what passes a 4 m gap with room to spare;
- two **pylons**: a vertical strut through each wing half, a motor pod at each end — the upper and the lower motor of
  that side; the propeller planes ahead of the wing's leading edge, so the wing flies in their wash;
- the **tail**: a long boom and four swept fins in an X, between the propeller wakes — the weathercock, the damper,
  and the four feet;
- the **camera**: a ball in the nose on one tilt axis (``camera_tilt_deg``: 0 looks along the nose, 90 looks toward
  the belly — the horizon when the aircraft hangs on its propellers).

Reused, not redrawn: the nose and pod ogives, the strut section and the pod meridian are ÉMERILLON's
(``emerillon.ogive``, ``emerillon.strut``, ``Emerillon.pod_profile``); the wing section is notebook 09's
(``fixed_wing.naca4``).

Frame: ``x`` runs aft from the nose tip along the body axis (the free stream along +X), ``y`` right, ``z`` up when it
flies level; units mm. Standing on its fins the nose is up and the ground is at ``x = length``. ``part``:
``aircraft`` (the whole machine, solid: pictures, wetted area, CFD; the propellers are thin discs when
``show_props``), ``wing`` (the wing alone), ``pylon`` (one pylon with its two pods, for the FEA and the printer),
``nose`` (the nose cone as a printed shell, without the camera), ``tail`` (the boom and its four fins).
"""
import math

import cadquery as cq
import numpy as np

from vegeta.dedalus import Design, Parameter

from emerillon import Emerillon, ogive, strut
from fixed_wing import FixedWing, naca4

FIN_ANGLES = (45.0, 135.0, 225.0, 315.0)
# the four motors seen from behind (y right, z up): M1 upper left, M2 upper right, M3 lower left, M4 lower right
MOTORS = ((-1, 1), (1, 1), (-1, -1), (1, -1))


class Gavilan(Design):
    """GAVILAN: body of revolution, a short broad wing, two pylons with four motor pods, a long X tail, a tilting camera."""

    parameters = [
        Parameter("part", "aircraft", choices=("aircraft", "wing", "pylon", "nose", "tail"), description="what to build"),
        Parameter("show_props", False, description="part='aircraft': the propellers as thin discs (pictures only)"),
        # ---- body
        Parameter("body_diameter", 61.9, "mm", min=30.0, description="outer diameter of the battery bay (the pack's diagonal and two skins)"),
        Parameter("nose_length", 150.0, "mm", min=30.0, description="ogive nose, the camera ball's centre to the bay"),
        Parameter("bay_length", 219.0, "mm", min=40.0, description="cylindrical bay: the stack and two packs in line"),
        Parameter("cone_length", 150.0, "mm", min=30.0, description="the cone from the bay down to the tail boom"),
        Parameter("boom_diameter", 20.0, "mm", min=6.0, description="tail boom (a printed tube over a bought carbon tube)"),
        Parameter("length", 860.0, "mm", min=300.0, description="nose tip to the fins' trailing edges: the feet"),
        Parameter("wall", 1.4, "mm", min=0.4, description="printed skin thickness"),
        # ---- camera
        Parameter("camera_diameter", 32.0, "mm", min=10.0, description="camera ball in the nose (a 19 mm camera, its cradle)"),
        Parameter("camera_tilt_deg", 0.0, "deg", min=-10.0, max=100.0, description="camera axis from the nose toward the belly"),
        # ---- wing
        Parameter("span", 700.0, "mm", min=200.0, description="wing span: the widest thing on the aircraft"),
        Parameter("root_chord", 230.0, "mm", min=40.0, description="wing chord on the body axis"),
        Parameter("taper", 0.65, "", min=0.2, max=1.0, description="tip chord / root chord"),
        Parameter("thickness", 0.10, "", min=0.05, max=0.20, description="wing section thickness / chord (symmetric NACA 00xx)"),
        Parameter("le_sweep_deg", 12.0, "deg", min=-10.0, max=45.0, description="leading-edge sweep"),
        Parameter("wing_x", 235.0, "mm", min=0.0, description="wing root leading edge from the nose tip"),
        # ---- pylons, pods, propellers
        Parameter("motor_y", 200.0, "mm", min=40.0, description="pylon (and motor axes) from the centre plane"),
        Parameter("motor_z", 120.0, "mm", min=20.0, description="motor axes above and below the wing"),
        Parameter("pylon_shape", "strut", choices=("strut", "panel"),
                  description="strut: a slim streamlined post through the wing (A); panel: a broad six-sided side plate at the wing tip (B)"),
        Parameter("pylon_chord", 62.0, "mm", min=10.0, description="pylon chord along the flow (panel: at the wing)"),
        Parameter("panel_taper", 0.55, "", min=0.2, max=1.0, description="panel: chord at the pods / chord at the wing"),
        Parameter("pylon_thickness", 9.0, "mm", min=3.0, description="pylon section thickness"),
        Parameter("pod_diameter", 37.5, "mm", min=10.0, description="motor pod diameter (the motor's can and a skin)"),
        Parameter("pod_length", 125.0, "mm", min=30.0, description="pod, spinner tip to the cone's end"),
        Parameter("motor_height", 20.0, "mm", min=5.0, description="motor can height: the pod's motor face behind the propeller plane"),
        Parameter("prop_diameter", 177.8, "mm", min=50.0, description="propeller diameter (clearances; a disc when show_props)"),
        Parameter("prop_gap", 22.0, "mm", min=2.0, description="propeller plane ahead of the wing's leading edge at the pylon"),
        # ---- tail
        Parameter("fin_span", 165.0, "mm", min=20.0, description="fin, out from the boom axis to its tip"),
        Parameter("fin_root_chord", 170.0, "mm", min=20.0, description="fin chord on the boom"),
        Parameter("fin_tip_chord", 70.0, "mm", min=10.0, description="fin chord at the tip (the foot)"),
        Parameter("fin_thickness", 4.0, "mm", min=1.0, description="fin plate thickness"),
        # ---- talons and guards (Gavilan-J)
        Parameter("talons", False, description="the talons on the belly: four sprung legs with a net between them"),
        Parameter("talon_deg", 90.0, "deg", min=0.0, max=110.0, description="talons: 0 folded back along the belly, 90 thrown out from it"),
        Parameter("talon_length", 420.0, "mm", min=100.0, description="talons: each leg, hub to hook"),
        Parameter("talon_splay_deg", 32.0, "deg", min=5.0, max=45.0, description="talons: the legs' half-angle about their common axis"),
        Parameter("guards", False, description="a guard ring round each propeller (it touches things)"),
        Parameter("guard_clearance", 7.0, "mm", min=2.0, description="guards: ring inside radius over the propeller's"),
        Parameter("angle_of_attack_deg", 0.0, "deg", min=-20.0, max=90.0, description="pitch the aircraft nose-up about y (CFD, pictures)"),
    ]

    # ------------------------------------------------------------------------------------------------ stations
    @staticmethod
    def layout(p) -> dict:
        """Stations [mm]: the body's sections, the wing's chords and areas, the pylon, the pods and the propeller
        planes, the fins and the feet; and the clearances between the propeller tips, the body, the wing and each other."""
        R = p["body_diameter"] / 2
        x_nose = p["camera_diameter"] / 2                               # the ball's centre; the ogive starts here
        x_bay = x_nose + p["nose_length"]
        x_cone = x_bay + p["bay_length"]
        x_boom = x_cone + p["cone_length"]
        b2 = p["span"] / 2
        c0, c1 = p["root_chord"], p["root_chord"] * p["taper"]
        tan_le = math.tan(math.radians(p["le_sweep_deg"]))

        def chord(y):
            return c0 + (c1 - c0) * abs(y) / b2

        def x_le(y):
            return p["wing_x"] + abs(y) * tan_le

        S = (c0 + c1) * b2                                               # trapezoid, both halves [mm2]
        mac = 2 / 3 * (c0 ** 2 + c0 * c1 + c1 ** 2) / (c0 + c1)
        y_mac = b2 / 3 * (c0 + 2 * c1) / (c0 + c1)
        x_prop = x_le(p["motor_y"]) - p["prop_gap"]
        spinner = 1.3 * p["pod_diameter"]
        x_pylon = x_le(p["motor_y"]) + 0.30 * chord(p["motor_y"]) - p["pylon_chord"] / 2
        if p["pylon_shape"] == "panel":                                  # the panel's pod ends start at the motor faces
            x_pylon = x_prop + p["motor_height"] - 0.35 * p["pylon_chord"] * (1 - p["panel_taper"])
        r_prop = p["prop_diameter"] / 2
        return {"R": R, "x_nose": x_nose, "x_bay": x_bay, "x_cone": x_cone, "x_boom": x_boom, "length": p["length"],
                "c_root": c0, "c_tip": c1, "S_mm2": S, "AR": p["span"] ** 2 / S, "mac": mac, "y_mac": y_mac,
                "x_mac_le": x_le(y_mac), "x_ac": x_le(y_mac) + 0.25 * mac, "x_le_pylon": x_le(p["motor_y"]),
                "c_pylon": chord(p["motor_y"]), "x_prop": x_prop, "x_pod": x_prop - spinner, "spinner": spinner,
                "x_mount": x_prop + p["motor_height"], "x_pylon": x_pylon,
                "x_fin": p["length"] - p["fin_root_chord"], "foot_radius": p["fin_span"],
                "foot_half_width": p["fin_span"] * math.sin(math.radians(45.0)),
                "tip_to_body": p["motor_y"] - r_prop - R, "tip_to_wing": p["motor_z"] - r_prop - 0.5 * p["thickness"] * c0,
                "tip_to_tip": 2 * p["motor_z"] - p["prop_diameter"], "tip_inside_span": b2 - p["motor_y"] - r_prop,
                "S_panels_mm2": (2 * p["motor_z"] * p["pylon_chord"] * (1 + p["panel_taper"]) / 2 * 2) if p["pylon_shape"] == "panel" else 2 * 2 * p["motor_z"] * p["pylon_chord"],
                "width": max(p["span"], 2 * (p["motor_y"] + r_prop)), "height": 2 * (p["motor_z"] + r_prop)}

    @staticmethod
    def body_profile(p, n=24) -> np.ndarray:
        """The body's meridian (x, r): the camera ball's socket, the ogive nose, the bay, the cone, the boom."""
        L = Gavilan.layout(p)
        r_cam, r_boom = p["camera_diameter"] / 2, p["boom_diameter"] / 2
        x0 = L["x_nose"] + 0.6 * r_cam                                    # a blunt start inside the ball: the ball sits in it
        nose = ogive(L["x_bay"] - x0, L["R"] - 0.55 * r_cam, n)
        nose = np.column_stack([nose[:, 0] + x0, nose[:, 1] + 0.55 * r_cam])
        cone = ogive(p["cone_length"], L["R"] - 0.8 * r_boom, n)[::-1]      # it closes a little inside the boom's radius
        cone = np.column_stack([L["x_boom"] - cone[:, 0], cone[:, 1] + 0.8 * r_boom])
        return np.vstack([[[x0, 0.0]], nose, cone])

    # ---------------------------------------------------------------------------------------------------- solids
    @staticmethod
    def _revolve(profile, x0=0.0):
        return Emerillon._revolve(profile, x0)

    def _body(self, p):
        return self._revolve(self.body_profile(p))

    def _camera(self, p):
        """The camera ball with its lens, tilted about y: 0 along the nose (−x), 90 toward the belly (−z)."""
        L = self.layout(p)
        r = p["camera_diameter"] / 2
        ball = cq.Workplane("XY").sphere(r)
        lens = cq.Workplane("YZ").circle(0.42 * r).extrude(-(r + 5.0))                 # toward −x: ahead of the nose
        cam = ball.union(lens).rotate((0, 0, 0), (0, 1, 0), p["camera_tilt_deg"])       # +: the lens swings to −z
        return cam.translate((L["x_nose"], 0, 0))

    @staticmethod
    def _section(wp, chord, x0, t):
        up, lo = naca4(0.0, 0.4, t, n=28)
        pts = [(x0 + x * chord, z * chord) for x, z in up] + [(x0 + x * chord, z * chord) for x, z in lo[::-1][1:]]
        return wp.polyline(pts).close()

    def _wing(self, p):
        """Two tapered halves lofted from the centre plane to the tips; the tips rounded off by the section itself."""
        L = self.layout(p)
        b2 = p["span"] / 2
        dx = b2 * math.tan(math.radians(p["le_sweep_deg"]))
        half = self._section(cq.Workplane("XZ"), L["c_root"], p["wing_x"], p["thickness"]).workplane(offset=b2)
        half = self._section(half, L["c_tip"], p["wing_x"] + dx, p["thickness"]).loft(combine=True, ruled=True)   # toward −y
        return half.union(half.mirror("XZ"))

    def _pod(self, p, y, z):
        L = self.layout(p)
        return self._revolve(Emerillon.pod_profile(p), L["x_pod"]).translate((0, y, z))

    def _pylon(self, p, side=1):
        """One pylon: a vertical strut through the wing at ``side * motor_y`` with a pod at each end."""
        L = self.layout(p)
        y, zm = side * p["motor_y"], p["motor_z"]
        if p["pylon_shape"] == "panel":                                   # a six-sided plate, widest at the wing
            x0, c, ce = L["x_pylon"], p["pylon_chord"], p["pylon_chord"] * p["panel_taper"]
            xe = x0 + 0.35 * (c - ce)
            pts = [(x0, 0.0), (xe, zm), (xe + ce, zm), (x0 + c, 0.0), (xe + ce, -zm), (xe, -zm)]
            blade = cq.Workplane("XZ").polyline(pts).close().extrude(p["pylon_thickness"]).translate((0, y + p["pylon_thickness"] / 2, 0))
        else:
            pts = [(float(x) + L["x_pylon"], float(t)) for x, t in strut(p["pylon_chord"], p["pylon_thickness"])]
            blade = cq.Workplane("XY").polyline(pts).close().extrude(2 * zm).translate((0, y, -zm))
        return blade.union(self._pod(p, y, zm)).union(self._pod(p, y, -zm))

    def _props(self, p):
        L = self.layout(p)
        out = None
        for sy, sz in MOTORS:
            d = cq.Workplane("YZ").circle(p["prop_diameter"] / 2).extrude(1.2).translate((L["x_prop"], sy * p["motor_y"], sz * p["motor_z"]))
            out = d if out is None else out.union(d)
        return out

    @classmethod
    def talon_layout(cls, p) -> dict:
        """The talons' hub on the belly and each leg's tip [mm], and how far the open claw reaches and spans."""
        L = cls.layout(p)
        hub = np.array([L["x_ac"], 0.0, -L["R"] + 4.0])
        a = math.radians(p["talon_deg"])
        s = math.radians(3.0 + (p["talon_splay_deg"] - 3.0) * min(max(math.sin(a), 0.0), 1.0))     # the legs close as they fold back
        d0 = np.array([math.cos(a), 0.0, -math.sin(a)])                    # folded: aft along the belly; thrown: straight out of it
        e1 = np.array([0.0, 1.0, 0.0])
        e2 = np.cross(d0, e1)
        dirs = [math.cos(s) * d0 + math.sin(s) * (math.cos(k) * e1 + math.sin(k) * e2) for k in np.radians((45.0, 135.0, 225.0, 315.0))]
        tips = [hub + p["talon_length"] * d for d in dirs]
        return {"hub": hub, "dirs": dirs, "tips": tips, "reach": p["talon_length"] * math.cos(s),
                "open_width": 2 * p["talon_length"] * math.sin(s) * math.sin(math.radians(45.0)) * math.sqrt(2.0),
                "beyond_lower_propellers": p["talon_length"] * math.cos(s) * math.sin(a) - 4.0 - (p["motor_z"] + p["prop_diameter"] / 2 - L["R"])}

    def _talons(self, p):
        T = self.talon_layout(p)
        hub = cq.Workplane("XY").sphere(13.0).translate(tuple(T["hub"]))
        for d, tip in zip(T["dirs"], T["tips"]):
            leg = cq.Workplane("XY").add(cq.Solid.makeCylinder(2.2, p["talon_length"], cq.Vector(*T["hub"]), cq.Vector(*d)))
            hub = hub.union(leg).union(cq.Workplane("XY").sphere(5.0).translate(tuple(tip)))
        return hub

    def _guards(self, p):
        """A ring round each propeller disc on two struts from its pod."""
        L = self.layout(p)
        R = p["prop_diameter"] / 2 + p["guard_clearance"]
        out = None
        for sy, sz in MOTORS:
            c = cq.Vector(L["x_prop"], sy * p["motor_y"], sz * p["motor_z"])
            ring = cq.Workplane("XY").add(cq.Solid.makeTorus(R, 1.6, c, cq.Vector(1, 0, 0)))
            for dy in (1, -1):
                ring = ring.union(cq.Workplane("XY").add(cq.Solid.makeCylinder(1.6, R, cq.Vector(L["x_prop"] + 14.0, c.y, c.z), cq.Vector(-14.0 / R, dy * 1.0, 0.0))))
            out = ring if out is None else out.union(ring)
        return out

    def _fin(self, p):
        """One fin in the xy plane, out along +y from the boom: swept, its trailing edge the foot line at ``length``."""
        L = self.layout(p)
        r0 = p["boom_diameter"] / 2 - 3.0
        x0, x1 = L["x_fin"], p["length"]
        pts = [(x0, r0), (x1 - p["fin_tip_chord"], p["fin_span"]), (x1, p["fin_span"]), (x1 - 0.12 * p["fin_root_chord"], r0)]
        return cq.Workplane("XY").polyline(pts).close().extrude(p["fin_thickness"]).translate((0, 0, -p["fin_thickness"] / 2))

    def _tail(self, p):
        L = self.layout(p)
        r = p["boom_diameter"] / 2
        x0 = L["x_boom"] - 0.25 * p["cone_length"]                        # the boom starts inside the cone
        tail = cq.Workplane("YZ").circle(r).extrude(p["length"] - 0.25 * p["fin_root_chord"] - x0).translate((x0, 0, 0))
        fin = self._fin(p)
        for a in FIN_ANGLES:
            tail = tail.union(fin.rotate((0, 0, 0), (1, 0, 0), a - 90.0))
        return tail

    def _nose(self, p):
        L = self.layout(p)
        prof = self.body_profile(p)
        prof = prof[prof[:, 0] <= L["x_bay"] + 1e-6]
        outer = self._revolve(prof)
        skin = np.column_stack([prof[1:, 0], prof[1:, 1] - p["wall"]])
        skin[0, 0] += p["wall"]
        skin[-1, 0] += 1.0                                                # open at the bay
        inner = self._revolve(np.vstack([[[skin[0, 0], 0.0]], skin]))
        return outer.cut(inner)

    def build(self, p):
        L = self.layout(p)
        if L["tip_to_body"] < 2.0:
            raise ValueError("the propeller tips reach the body: raise motor_y")
        if L["tip_to_wing"] < 2.0 or L["tip_to_tip"] < 2.0:
            raise ValueError("the propeller tips reach the wing or each other: raise motor_z")
        if L["x_fin"] < L["x_boom"]:
            raise ValueError("the fins run into the cone: lengthen the aircraft or shorten fin_root_chord")
        if p["part"] == "wing":
            return self._wing(p)
        if p["part"] == "pylon":
            return self._pylon(p)
        if p["part"] == "nose":
            return self._nose(p)
        if p["part"] == "tail":
            return self._tail(p)
        aircraft = self._body(p).union(self._camera(p)).union(self._wing(p)).union(self._tail(p))
        for side in (1, -1):
            aircraft = aircraft.union(self._pylon(p, side))
        if p["talons"]:
            aircraft = aircraft.union(self._talons(p))
        if p["guards"]:
            aircraft = aircraft.union(self._guards(p))
        if p["show_props"]:
            aircraft = aircraft.union(self._props(p))
        if p["angle_of_attack_deg"]:
            aircraft = aircraft.rotate((0, 0, 0), (0, 1, 0), p["angle_of_attack_deg"])
        return aircraft

    def parts(self, p) -> dict:
        """The aircraft's solids by name, for a coloured picture (the same solids ``build`` unites)."""
        out = {"body": self._body(p), "camera": self._camera(p), "wing": self._wing(p), "tail": self._tail(p),
               "pylons": self._pylon(p, 1).union(self._pylon(p, -1)), "props": self._props(p)}
        if p["talons"]:
            out["talons"] = self._talons(p)
        if p["guards"]:
            out["guards"] = self._guards(p)
        return out


# ------------------------------------------------------------------------------------------------- GAVILAN-X: the X-wing
def _gp(name, default):
    """One of ``Gavilan``'s parameters with another default."""
    import dataclasses
    return dataclasses.replace(next(q for q in Gavilan.parameters if q.name == name), default=default)


def _x_kept():
    skip = ("part", "bay_length", "length", "motor_y", "motor_z", "talons", "guards", "prop_gap")
    return [q for q in Gavilan.parameters if q.name not in skip]


class GavilanX(Gavilan):
    """GAVILAN-X: GAVILAN-J's hunt on an airframe that is easier to build. J's body (camera, one pack), motors,
    propellers, guards and talons — but the wing, the two pylons, the boom and the four fins become **four identical
    panels** at 45, 135, 225 and 315 degrees: each carries one motor pod, and each sweeps back to a tip that is one of
    the four feet. Seen from the front it is an X; the talons leave through the gap between the two lower panels.
    Printed part types: nose, bay with its cone, panel, pod (J: nose, bay with its cone, wing, pylon, pod, fin).
    ``part``: ``aircraft``, ``panel`` (one panel with its pod: the part printed four times), ``nose``."""

    parameters = _x_kept() + [
        Parameter("part", "aircraft", choices=("aircraft", "panel", "nose"), description="what to build"),
        _gp("bay_length", 134.5), _gp("length", 580.0), _gp("talons", True), _gp("guards", True), _gp("prop_gap", 24.0),
        _gp("motor_y", 141.4), _gp("motor_z", 141.4),                    # the motor axes: 200 mm from the body axis, on the 45 degree lines
        Parameter("panel_radius", 340.0, "mm", min=120.0, description="each panel's tip from the body axis (the foot)"),
        Parameter("panel_root_chord", 220.0, "mm", min=60.0, description="panel chord at the body axis"),
        Parameter("panel_tip_chord", 110.0, "mm", min=30.0, description="panel chord at the tip"),
        Parameter("panel_x", 215.0, "mm", min=0.0, description="panel root leading edge from the nose tip"),
    ]

    @staticmethod
    def layout(p) -> dict:
        """``Gavilan``'s body stations, and the panels: chords, the leading edge along the panel, the motor's station,
        the areas, the feet, the clearances [mm]."""
        L = dict(Gavilan.layout(p))
        r1, c0, c1 = p["panel_radius"], p["panel_root_chord"], p["panel_tip_chord"]
        r_m = math.hypot(p["motor_y"], p["motor_z"])
        x_le_tip = p["length"] - c1                                       # the tip's trailing edge is the foot line

        def x_le(r):
            return p["panel_x"] + (x_le_tip - p["panel_x"]) * r / r1

        def chord(r):
            return c0 + (c1 - c0) * r / r1

        r_prop, R = p["prop_diameter"] / 2, L["R"]
        S_panel = 0.5 * (chord(R) + c1) * (r1 - R)                        # outside the body
        mac = 2 / 3 * (c0 ** 2 + c0 * c1 + c1 ** 2) / (c0 + c1)
        r_mac = r1 / 3 * (c0 + 2 * c1) / (c0 + c1)
        x_prop = x_le(r_m) - p["prop_gap"]
        L.update({"r_motor": r_m, "x_le_tip": x_le_tip, "c_root": c0, "c_tip": c1, "S_panel_mm2": S_panel, "S_mm2": 4 * S_panel, "mac": mac,
                  "x_ac": x_le(r_mac) + 0.25 * mac, "le_sweep_deg": math.degrees(math.atan2(x_le_tip - p["panel_x"], r1)),
                  "x_prop": x_prop, "x_pod": x_prop - L["spinner"], "x_mount": x_prop + p["motor_height"], "x_le_motor": x_le(r_m),
                  "projected_span": 2 * r1 * math.cos(math.radians(45.0)), "AR": (2 * r1) ** 2 / (2 * S_panel + 2 * R * c0),
                  "foot_radius": r1, "foot_half_width": r1 * math.sin(math.radians(45.0)),
                  "tip_to_body": r_m - r_prop - R, "tip_to_tip": r_m * math.sqrt(2.0) - p["prop_diameter"],
                  "tip_to_wing": p["prop_gap"], "tip_inside_span": r1 - r_m - r_prop,
                  "width": 2 * max(r1 * math.cos(math.radians(45.0)), p["motor_y"] + r_prop), "height": 2 * max(r1 * math.sin(math.radians(45.0)), p["motor_z"] + r_prop),
                  "x_boom": L["x_boom"], "body_end": L["x_boom"] + 30.0})
        return L

    @staticmethod
    def body_profile(p, n=24) -> np.ndarray:
        """``Gavilan``'s nose, bay and cone, closed a little behind the cone (no boom: the panels are the tail)."""
        prof = Gavilan.body_profile(p, n)
        return np.vstack([prof, [[prof[-1, 0] + 30.0, 0.6 * prof[-1, 1]]]])

    def _body(self, p):
        return self._revolve(self.body_profile(p))

    def _panel(self, p):
        """One panel along -y from the body axis to its tip (the root inside the body), swept back to the foot."""
        L = self.layout(p)
        root = self._section(cq.Workplane("XZ"), L["c_root"], p["panel_x"], p["thickness"]).workplane(offset=p["panel_radius"])
        return self._section(root, L["c_tip"], L["x_le_tip"], p["thickness"] * L["c_root"] / L["c_tip"] * 0.75).loft(combine=True, ruled=True)

    def _panels(self, p):
        out = None
        for a in (45.0, 135.0, 225.0, 315.0):
            q = self._panel(p).rotate((0, 0, 0), (1, 0, 0), a)
            out = q if out is None else out.union(q)
        return out

    def _pods(self, p):
        out = None
        for sy, sz in MOTORS:
            q = self._pod(p, sy * p["motor_y"], sz * p["motor_z"])
            out = q if out is None else out.union(q)
        return out

    def parts(self, p) -> dict:
        out = {"body": self._body(p), "camera": self._camera(p), "wing": self._panels(p), "pylons": self._pods(p), "props": self._props(p)}
        if p["talons"]:
            out["talons"] = self._talons(p)
        if p["guards"]:
            out["guards"] = self._guards(p)
        return out

    def build(self, p):
        L = self.layout(p)
        if L["tip_to_body"] < 2.0 or L["tip_to_tip"] < 2.0:
            raise ValueError("the propeller tips reach the body or each other: move the motors out")
        if p["part"] == "nose":
            return self._nose(p)
        if p["part"] == "panel":
            a = math.radians(45.0)
            pod = self._pod(p, -L["r_motor"], 0.0)                         # on the panel's own line (-y), before the 45 degree turn
            return self._panel(p).union(pod)
        aircraft = self._body(p).union(self._camera(p)).union(self._panels(p)).union(self._pods(p))
        if p["talons"]:
            aircraft = aircraft.union(self._talons(p))
        if p["guards"]:
            aircraft = aircraft.union(self._guards(p))
        if p["show_props"]:
            aircraft = aircraft.union(self._props(p))
        if p["angle_of_attack_deg"]:
            aircraft = aircraft.rotate((0, 0, 0), (0, 1, 0), p["angle_of_attack_deg"])
        return aircraft


class GavilanXP(GavilanX):
    """GAVILAN-XP: GAVILAN-X with a hawk's feet in place of the four-legged net. Two thin legs hinge at the belly and
    swing forward out of it; each ends in a foot of four toes — ``front_toes`` forward (a hawk has three) and one back
    — that snap shut like a pincer. Each toe is two segments: the second curls in, as a talon does.
    ``leg_deg``: 0 folded back along the belly, 90 straight out of it, more reaches toward the nose.
    ``toe_open_deg``: about 80 spread to take the bandit, about 20 shut round it."""

    parameters = [q for q in GavilanX.parameters if q.name not in ("talons", "talon_deg", "talon_length", "talon_splay_deg")] + [
        Parameter("talons", False, description="X's four-legged net (off: XP carries feet instead)"),
        Parameter("talon_deg", 0.0, "deg", description="unused on XP"), Parameter("talon_length", 420.0, "mm", description="unused on XP"),
        Parameter("talon_splay_deg", 32.0, "deg", description="unused on XP"),
        Parameter("leg_deg", 100.0, "deg", min=0.0, max=180.0, description="legs: 90 straight out of the belly, 180 folded forward along it (the perch and the sprint), less than 90 trailing"),
        Parameter("leg_length", 240.0, "mm", min=80.0, description="each leg, hip to ankle"),
        Parameter("leg_y", 45.0, "mm", min=15.0, description="each leg from the centre plane"),
        Parameter("leg_diameter", 6.0, "mm", min=3.0, description="leg tube"),
        Parameter("front_toes", 3, "", min=2, max=3, description="toes pointing forward on each foot (one more points back)"),
        Parameter("toe_open_deg", 80.0, "deg", min=5.0, max=100.0, description="toes: 80 spread, 20 shut"),
        Parameter("toe_first", 60.0, "mm", min=20.0, description="each toe's first segment"),
        Parameter("toe_second", 50.0, "mm", min=15.0, description="each toe's second segment: the talon"),
        Parameter("toe_curl_deg", 50.0, "deg", min=0.0, max=90.0, description="how far the talon curls in from the first segment"),
        Parameter("toe_fan_deg", 35.0, "deg", min=10.0, max=60.0, description="angle between neighbouring front toes"),
    ]

    @classmethod
    def foot_layout(cls, p) -> dict:
        """Hips, ankles and every toe's joints and tip [mm]; the foot's open span front to back and the gap left when shut."""
        L = cls.layout(p)
        lam, phi, curl = math.radians(p["leg_deg"]), math.radians(p["toe_open_deg"]), math.radians(p["toe_curl_deg"])
        d = np.array([math.cos(lam), 0.0, -math.sin(lam)])                 # along the leg, hip to ankle
        f = np.array([-math.sin(lam), 0.0, -math.cos(lam)])                # the foot's front: toward the nose when the leg is straight out
        e = np.array([0.0, 1.0, 0.0])
        n = int(p["front_toes"])
        angles = [(k - (n - 1) / 2) * math.radians(p["toe_fan_deg"]) for k in range(n)] + [math.pi]
        feet = []
        for side in (1, -1):
            hip = np.array([L["x_ac"] - 20.0, side * p["leg_y"], -L["R"] + 4.0])
            ankle = hip + p["leg_length"] * d
            toes = []
            for a in angles:
                u = math.cos(a) * f + math.sin(a) * e
                j = ankle + p["toe_first"] * (math.cos(phi) * d + math.sin(phi) * u)
                tip = j + p["toe_second"] * (math.cos(phi - curl) * d + math.sin(phi - curl) * u)
                toes.append((j, tip))
            feet.append({"hip": hip, "ankle": ankle, "toes": toes})
        r_tip = p["toe_first"] * math.sin(phi) + p["toe_second"] * math.sin(phi - curl)      # a tip's distance from the leg's axis
        depth = p["toe_first"] * math.cos(phi) + p["toe_second"] * math.cos(phi - curl)      # and how far past the ankle
        return {"feet": feet, "leg_dir": d, "front": f, "aperture": 2 * r_tip, "tip_depth": depth, "between_feet": 2 * p["leg_y"],
                "reach_below_belly": p["leg_length"] * math.sin(lam) + depth * math.sin(lam),
                "past_lower_propellers": (p["leg_length"] + depth) * math.sin(lam) - (p["motor_z"] + p["prop_diameter"] / 2 - L["R"])}

    def _feet(self, p):
        F = self.foot_layout(p)
        r = p["leg_diameter"] / 2
        out = None
        for foot in F["feet"]:
            hip, ankle = foot["hip"], foot["ankle"]
            s = cq.Workplane("XY").sphere(9.0).translate(tuple(hip)).union(
                cq.Workplane("XY").add(cq.Solid.makeCylinder(r, p["leg_length"], cq.Vector(*hip), cq.Vector(*F["leg_dir"]))))
            s = s.union(cq.Workplane("XY").sphere(8.0).translate(tuple(ankle)))
            for j, tip in foot["toes"]:
                for a, b, rr in ((ankle, j, 2.6), (j, tip, 2.0)):
                    v = b - a
                    s = s.union(cq.Workplane("XY").add(cq.Solid.makeCylinder(rr, float(np.linalg.norm(v)), cq.Vector(*a), cq.Vector(*v))))
                s = s.union(cq.Workplane("XY").sphere(3.2).translate(tuple(j)))
            out = s if out is None else out.union(s)
        return out

    def parts(self, p) -> dict:
        out = GavilanX.parts(self, {**p, "talons": False})
        out["talons"] = self._feet(p)
        return out

    def build(self, p):
        if p["part"] != "aircraft":
            return GavilanX.build(self, p)
        aircraft = GavilanX.build(self, {**p, "talons": False, "angle_of_attack_deg": 0.0}).union(self._feet(p))
        if p["angle_of_attack_deg"]:
            aircraft = aircraft.rotate((0, 0, 0), (0, 1, 0), p["angle_of_attack_deg"])
        return aircraft


def xp_resolve(q=None) -> dict:
    return GavilanXP().resolve(**dict(q or {}))


def x_resolve(q=None) -> dict:
    return GavilanX().resolve(**dict(q or {}))


def x_planform(q=None) -> list:
    """GAVILAN-X's four panels as flat quadrilaterals [m] for the vortex lattice."""
    p = x_resolve(q)
    L = GavilanX.layout(p)
    quads = []
    for a in np.radians((45.0, 135.0, 225.0, 315.0)):
        c, s = math.cos(a), math.sin(a)
        r0, r1 = L["R"], p["panel_radius"]
        x0 = p["panel_x"] + (L["x_le_tip"] - p["panel_x"]) * r0 / r1
        c_r0 = L["c_root"] + (L["c_tip"] - L["c_root"]) * r0 / r1
        quads.append(np.array([[x0, r0 * c, r0 * s], [L["x_le_tip"], r1 * c, r1 * s], [L["x_le_tip"] + L["c_tip"], r1 * c, r1 * s],
                               [x0 + c_r0, r0 * c, r0 * s]]) / 1000.0)
    return quads


# ------------------------------------------------------------------------------------------------- GAVILAN-F: the fixed wing
def _fw(name, default):
    """One of ``FixedWing``'s parameters with another default."""
    import dataclasses
    return dataclasses.replace(next(q for q in FixedWing.parameters if q.name == name), default=default)


class GavilanF(FixedWing):
    """GAVILAN-F: notebook 09's twin-motor fixed wing (``fixed_wing.FixedWing``: its wing, fuselage, nacelles and tail,
    nothing redrawn) on a small wing, with the 8 mm recoilless emitter on the body axis — the two tractors on the wing
    leave the nose free for the muzzle and the tail free for the rear jet. It does not hover: it is thrown or
    catapulted and lands on its belly. Frame: ``FixedWing``'s (the wing root's leading edge at the origin, x aft,
    y span, z up; the body axis at z = −diameter/2). ``part``: ``FixedWing``'s, plus ``emitter`` (the tube alone)."""

    parameters = [
        Parameter("part", "aircraft", choices=("aircraft", "wing", "nacelle", "fuselage", "emitter"), description="what to build"),
        _fw("span", 800.0), _fw("root_chord", 230.0), _fw("taper", 0.65), _fw("dihedral_deg", 2.0),
        _fw("camber", 0.02), _fw("camber_pos", 0.4), _fw("thickness", 0.10),
        _fw("fuselage_length", 720.0), _fw("fuselage_diameter", 64.0), _fw("nose_length", 210.0),
        _fw("nacelle_y", 190.0), _fw("nacelle_diameter", 37.5), _fw("nacelle_length", 110.0), _fw("nacelle_forward", 60.0),
        _fw("tail_span", 300.0), _fw("tail_chord", 100.0), _fw("fin_height", 120.0), _fw("tail_thickness", 4.0),
        _fw("fuselage_wall", 2.0), _fw("angle_of_attack_deg", 0.0),
        Parameter("emitter_tube", 300.0, "mm", min=100.0, description="the emitter's tube, muzzle to the chamber"),
        Parameter("emitter_diameter", 12.0, "mm", min=9.0, description="tube outer diameter (8 mm bore)"),
        Parameter("muzzle_out", 25.0, "mm", min=0.0, description="how far the muzzle stands out of the nose"),
        Parameter("jet_diameter", 14.0, "mm", min=6.0, description="the rear jet's pipe through the tail (the venturi's exit and a liner)"),
    ]

    def _emitter(self, p):
        """The tube from the muzzle, the chamber behind it, and the jet pipe to the tail's end, on the body axis."""
        r, z = p["emitter_diameter"] / 2, -p["fuselage_diameter"] / 2
        x0 = -p["nose_length"] - p["muzzle_out"]
        x_end = -p["nose_length"] + p["fuselage_length"] + 6.0
        tube = cq.Workplane("YZ").circle(r).extrude(p["emitter_tube"]).translate((x0, 0, z))
        chamber = cq.Workplane("YZ").circle(17.0).extrude(64.0).translate((x0 + p["emitter_tube"], 0, z))
        x_j = x0 + p["emitter_tube"] + 64.0
        jet = cq.Workplane("YZ").circle(p["jet_diameter"] / 2).extrude(x_end - x_j).translate((x_j, 0, z))
        return tube.union(chamber).union(jet)

    def parts(self, p) -> dict:
        hstab, fin = self._tail(p, separate=True)
        wing = self._wing(p).union(self._nacelle(p, -p["nacelle_y"])).union(self._nacelle(p, p["nacelle_y"]))
        return {"fuselage": self._fuselage(p), "wing": wing, "tail": hstab.union(fin), "emitter": self._emitter(p)}

    def build(self, p):
        if p["part"] == "emitter":
            return self._emitter(p)
        if p["part"] != "aircraft":
            return FixedWing.build(self, p)
        aircraft = FixedWing.build(self, {**p, "angle_of_attack_deg": 0.0}).union(self._emitter(p))
        if p["angle_of_attack_deg"]:
            aircraft = aircraft.rotate((0, 0, 0), (0, 1, 0), p["angle_of_attack_deg"])
        return aircraft


def f_resolve(q=None) -> dict:
    return GavilanF().resolve(**dict(q or {}))


# ------------------------------------------------------------------------------------------------- GAVILAN-H: the hawk
class GavilanH(GavilanF):
    """GAVILAN-H: the fixed wing flown like a hawk. ``GavilanF``'s layout on a broad wing (a hawk's wing loading), with
    the one thing a hawk has that an aeroplane lacks: control far beyond the stall. The tailplane is two **all-moving
    halves** as wide as the two propellers stand apart, so each half sits in one propeller's wash: together they
    pitch the aircraft, against each other they roll it, at any airspeed down to none; the two motors' thrusts yaw it.
    It flares past its stall to brake and to turn, and can hang on its propellers; it has no feet: it is thrown and
    lands on its belly. ``weapon``: ``none`` (the hunter that watches) or ``emitter`` (F's tube on the body axis).
    ``taileron_deg`` deflects both halves (pictures); ``show_props`` adds the propeller discs."""

    parameters = [q for q in GavilanF.parameters if q.name not in ("part", "span", "root_chord", "tail_span", "tail_chord", "fin_height", "nacelle_y", "fuselage_length")] + [
        Parameter("part", "aircraft", choices=("aircraft", "wing", "nacelle", "fuselage", "emitter", "tail"), description="what to build"),
        _fw("span", 1000.0), _fw("root_chord", 280.0), _fw("nacelle_y", 200.0), _fw("fuselage_length", 860.0),
        _fw("tail_span", 580.0), _fw("tail_chord", 150.0), _fw("fin_height", 130.0),
        Parameter("weapon", "none", choices=("none", "emitter", "net"), description="what it carries on the body axis (net: the one-shot interceptor)"),
        Parameter("net_tube", 250.0, "mm", min=100.0, description="net: the launch tube, muzzle to the chamber (32 mm bore)"),
        Parameter("net_tube_od", 35.0, "mm", min=33.0, description="net: tube outer diameter"),
        Parameter("net_muzzle_out", 10.0, "mm", min=0.0, description="net: the muzzle ahead of the nose"),
        Parameter("net_pipe_od", 41.0, "mm", min=30.0, description="net: the rear jet's pipe on the axis to the tail's end"),
        Parameter("net_vessel_ml", 80.0, "mL", min=40.0, description="net: the one-shot charge vessel in the belly blister"),
        Parameter("taileron_deg", 0.0, "deg", min=-60.0, max=60.0, description="both tail halves, trailing edge up positive (pictures)"),
        Parameter("taileron_gap", 14.0, "mm", min=4.0, description="each half's root from the centre plane (the fuselage's end between them)"),
        Parameter("hinge_frac", 0.30, "", min=0.1, max=0.6, description="the halves' pivot along their chord"),
        Parameter("prop_diameter", 177.8, "mm", min=50.0, description="propeller diameter (the wash the tail sits in; a disc when show_props)"),
        Parameter("show_props", False, description="part='aircraft': the propellers as thin discs (pictures only)"),
    ]

    @staticmethod
    def layout(p) -> dict:
        """Where the tail sits against the propellers' wash [mm]: the wash's width at the tail (the disc contracted to
        0.71 of its diameter, momentum theory) and the share of each tail half inside it."""
        xt = -p["nose_length"] + p["fuselage_length"] - p["tail_chord"]
        wash = 0.707 * p["prop_diameter"]
        lo, hi = p["nacelle_y"] - wash / 2, p["nacelle_y"] + wash / 2
        half = (p["taileron_gap"], p["tail_span"] / 2)
        blown = max(0.0, min(hi, half[1]) - max(lo, half[0]))
        return {"x_tail": xt, "x_hinge": xt + p["hinge_frac"] * p["tail_chord"], "x_prop": -p["nacelle_forward"] - 12.0, "wash_width": wash,
                "blown_span_per_half": blown, "blown_share": blown / (half[1] - half[0]),
                "S_tail_mm2": 2 * (half[1] - half[0]) * p["tail_chord"], "S_blown_mm2": 2 * blown * p["tail_chord"],
                "tail_arm": xt + 0.25 * p["tail_chord"] - 0.25 * p["root_chord"]}

    def _tail(self, p, separate=False):
        """The two tail halves and the fin. With the net aboard the jet pipe runs between the halves: each then turns
        on its own stub shaft (and servo) instead of one shaft across, and the fin stands on the pipe's fairing."""
        L = self.layout(p)
        r, tt = p["fuselage_diameter"] / 2, p["tail_thickness"]
        net = p["weapon"] == "net"
        w = p["tail_span"] / 2 - p["taileron_gap"]
        half = cq.Workplane("XY").box(p["tail_chord"], w, tt, centered=(False, False, True)).translate((L["x_tail"], p["taileron_gap"], -r))
        half = half.rotate((L["x_hinge"], 0, -r), (L["x_hinge"], 1, -r), p["taileron_deg"])
        if net:
            stub = cq.Workplane("XZ").circle(3.0).extrude(-(p["taileron_gap"] - p["net_pipe_od"] / 2 + 6.0)).translate((L["x_hinge"], p["net_pipe_od"] / 2 - 2.0, -r))
            hstab = half.union(stub)
            hstab = hstab.union(hstab.mirror("XZ"))
        else:
            shaft = cq.Workplane("XZ").circle(4.0).extrude(2 * (p["taileron_gap"] + 12.0)).translate((L["x_hinge"], p["taileron_gap"] + 12.0, -r))
            hstab = half.union(half.mirror("XZ")).union(shaft)
        z_fin = -r + (p["net_pipe_od"] / 2 + 1.0 if net else 0.0)
        fin = cq.Workplane("XY").box(p["tail_chord"], tt, p["fin_height"], centered=(False, True, False)).translate((L["x_tail"] - 0.5 * p["tail_chord"], 0, z_fin))
        return (hstab, fin) if separate else hstab.union(fin)

    @staticmethod
    def net_layout(p) -> dict:
        """The one-shot net interceptor's stations on the body axis [mm]: muzzle, chamber, the jet pipe's end, the vessel."""
        x_muzzle = -p["nose_length"] - p["net_muzzle_out"]
        x_bore = x_muzzle + p["net_tube"]                                # the chamber's front face
        x_end = -p["nose_length"] + p["fuselage_length"] + 6.0
        vr = 22.0
        vl = p["net_vessel_ml"] * 1e3 / (math.pi * (vr - 2.0) ** 2) - 4 / 3 * (vr - 2.0)
        return {"x_muzzle": x_muzzle, "x_bore": x_bore, "x_chamber_end": x_bore + 55.0, "x_end": x_end, "pipe_length": x_end - (x_bore + 55.0),
                "vessel_r": vr, "vessel_l": vl, "x_vessel": x_bore + 10.0, "z_vessel": -p["fuselage_diameter"] / 2 - 35.0 - 20.0 - vr,
                "prop_plane_behind_muzzle": (-p["nacelle_forward"] - 12.0) - x_muzzle}

    def _net(self, p):
        """What the interceptor adds to the outside: the tube out of the nose, the pipe's fairing to the tail's end and
        the belly blister over the charge vessel (the chamber, the venturi and the burst diaphragm are inside)."""
        N, z = self.net_layout(p), -p["fuselage_diameter"] / 2
        tube = cq.Workplane("YZ").circle(p["net_tube_od"] / 2).extrude(p["net_tube"]).translate((N["x_muzzle"], 0, z))
        fairing = cq.Workplane("YZ").circle(p["net_pipe_od"] / 2 + 3.0).extrude(N["x_end"] - N["x_bore"]).translate((N["x_bore"], 0, z))
        blister = (cq.Workplane("YZ").ellipse(N["vessel_r"] + 9.0, (z - N["z_vessel"]) + N["vessel_r"] + 10.0).extrude(N["vessel_l"] + 2 * N["vessel_r"] + 70.0)
                   .translate((N["x_vessel"] - N["vessel_r"] - 35.0, 0, z)))
        blister = blister.intersect(cq.Workplane("XY").box(2000, 400, 400).translate((0, 0, z - 200)))
        return tube.union(fairing).union(blister)

    def _props(self, p):
        L = self.layout(p)
        d = cq.Workplane("YZ").circle(p["prop_diameter"] / 2).extrude(1.2)
        return d.translate((L["x_prop"], p["nacelle_y"], 0)).union(d.translate((L["x_prop"], -p["nacelle_y"], 0)))

    def parts(self, p) -> dict:
        hstab, fin = self._tail(p, separate=True)
        wing = self._wing(p).union(self._nacelle(p, -p["nacelle_y"])).union(self._nacelle(p, p["nacelle_y"]))
        out = {"fuselage": self._fuselage(p), "wing": wing, "tail": hstab.union(fin), "props": self._props(p)}
        if p["weapon"] == "emitter":
            out["emitter"] = self._emitter(p)
        if p["weapon"] == "net":
            out["emitter"] = self._net(p)
        return out

    def build(self, p):
        if p["part"] == "tail":
            return self._tail(p)
        if p["part"] != "aircraft":
            return GavilanF.build(self, p)
        aircraft = FixedWing.build(self, {**p, "angle_of_attack_deg": 0.0})
        if p["weapon"] == "emitter":
            aircraft = aircraft.union(self._emitter(p))
        if p["weapon"] == "net":
            if p["taileron_gap"] < p["net_pipe_od"] / 2 + 4.0:
                raise ValueError("the tail halves foul the jet pipe: raise taileron_gap (H_NET uses 26 mm)")
            aircraft = aircraft.union(self._net(p))
        if p["show_props"]:
            aircraft = aircraft.union(self._props(p))
        if p["angle_of_attack_deg"]:
            aircraft = aircraft.rotate((0, 0, 0), (0, 1, 0), p["angle_of_attack_deg"])
        return aircraft


H_NET = {"weapon": "net", "taileron_gap": 26.0}          # GAVILAN-H carrying the one-shot net interceptor


def h_resolve(q=None) -> dict:
    return GavilanH().resolve(**dict(q or {}))


# ------------------------------------------------------------------------------------------------------------ variants
# A — the goshawk: a short broad wing, the four motors on slim posts through it (the wing does the lifting).
# B — the "T-fighter" arrangement: the body in the middle, a stub wing to a broad six-sided panel on each side, a
#     motor at the top and the bottom of each panel. Rolled 90 degrees it is a biplane: the panels are its wings.
VARIANTS = {
    "A": {},
    "B": {"pylon_shape": "panel", "span": 420.0, "root_chord": 150.0, "taper": 1.0, "le_sweep_deg": 0.0, "wing_x": 255.0,
          "motor_y": 204.0, "motor_z": 150.0, "pylon_chord": 230.0, "pylon_thickness": 8.0, "prop_gap": 30.0},
}
VARIANTS["J"] = {"bay_length": 134.5, "length": 780.0, "talons": True, "guards": True}     # one pack: the bay 84.5 mm shorter
VARIANT_NAME = {"A": "Gavilan-A — broad wing, motors on posts", "B": "Gavilan-B — side panels (T-fighter)",
                "J": "Gavilan-J — the hawk's hunt: perch, sprint, talons"}


def variant(key: str = "A", **overrides) -> dict:
    """The parameters of a variant (``VARIANTS``) with ``overrides``."""
    return {**VARIANTS[key], **overrides}


# ------------------------------------------------------------------------------------------------------------ helpers
def resolve(p=None) -> dict:
    return Gavilan().resolve(**dict(p or {}))


def planform(p=None) -> dict:
    """The lifting surfaces as flat quadrilaterals [m] (x aft, y right, z up) for the vortex lattice (``ogma.aero.vlm``):
    the two wing halves, the two pylons (vertical: they lift sideways), the four fins."""
    p = resolve(p)
    L = Gavilan.layout(p)
    b2, tan_le = p["span"] / 2, math.tan(math.radians(p["le_sweep_deg"]))

    def wing(s):
        return np.array([[p["wing_x"], 0, 0], [p["wing_x"] + b2 * tan_le, s * b2, 0],
                         [p["wing_x"] + b2 * tan_le + L["c_tip"], s * b2, 0], [p["wing_x"] + L["c_root"], 0, 0]]) / 1000.0

    def fin(a):
        c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
        r0, r1 = p["boom_diameter"] / 2, p["fin_span"]
        x0, x1 = L["x_fin"], p["length"]
        return np.array([[x0, r0 * c, r0 * s], [x1 - p["fin_tip_chord"], r1 * c, r1 * s], [x1, r1 * c, r1 * s],
                         [x1 - 0.12 * p["fin_root_chord"], r0 * c, r0 * s]]) / 1000.0

    pyl = []
    for s in (1, -1):
        for o in pylon_outline(p):                                        # each pylon as an upper and a lower quad
            pyl.append(np.column_stack([o[:, 0], np.full(4, s * p["motor_y"]), o[:, 1]]) / 1000.0)
    return {"wing": [wing(1), wing(-1)], "pylons": pyl, "fins": [fin(a) for a in FIN_ANGLES]}


def pylon_outline(p) -> list:
    """One pylon's side outline [mm] as two quads (x, z): above and below the wing."""
    L = Gavilan.layout(p)
    x0, c, zm = L["x_pylon"], p["pylon_chord"], p["motor_z"]
    ce = c * p["panel_taper"] if p["pylon_shape"] == "panel" else c
    xe = x0 + (0.35 * (c - ce) if p["pylon_shape"] == "panel" else 0.0)
    return [np.array([[x0, 0.0], [xe, s * zm], [xe + ce, s * zm], [x0 + c, 0.0]]) for s in (1, -1)]


def outline(p=None) -> dict:
    """Top, side and front views [m] for a three-view drawing: polygons per view, the propeller discs and planes."""
    p = resolve(p)
    L = Gavilan.layout(p)
    k = 1000.0
    prof = Gavilan.body_profile(p) / k
    body = np.vstack([prof, prof[::-1] * [1, -1]])
    pf = planform(p)
    th = np.linspace(0, 2 * math.pi, 72)
    circle = np.column_stack([np.cos(th), np.sin(th)])
    pod = Emerillon.pod_profile(p) / k
    pod = np.vstack([pod, pod[::-1] * [1, -1]]) + [L["x_pod"] / k, 0.0]
    my, mz, rp = p["motor_y"] / k, p["motor_z"] / k, p["prop_diameter"] / 2 / k
    cam = circle * p["camera_diameter"] / 2 / k + [L["x_nose"] / k, 0.0]
    fin_side = [q[:, [0, 2]] for q in pf["fins"]]
    fin_top = [q[:, [0, 1]] for q in pf["fins"]]
    pyl_side = [o / k for o in pylon_outline(p)]
    up, lo = naca4(0.0, 0.4, p["thickness"], n=28)
    sec = (np.array(up + lo[::-1][1:]) * [L["c_root"], L["c_root"]] + [p["wing_x"], 0.0]) / k
    rb = p["boom_diameter"] / 2 / k
    boom = np.array([[L["x_boom"] / k - 0.02, -rb], [p["length"] / k - 0.25 * p["fin_root_chord"] / k, -rb],
                     [p["length"] / k - 0.25 * p["fin_root_chord"] / k, rb], [L["x_boom"] / k - 0.02, rb]])
    top = {"boom": [boom], "body": [body], "wing": [q[:, [0, 1]] for q in pf["wing"]], "fins": fin_top,
           "pods": [pod + [0.0, s * my] for s in (1, -1)], "camera": [cam],
           "pylons": [np.array([[L["x_pylon"], s * my - 0.0045], [L["x_pylon"] + p["pylon_chord"], s * my - 0.0045],
                                [L["x_pylon"] + p["pylon_chord"], s * my + 0.0045], [L["x_pylon"], s * my + 0.0045]] ) / [k, 1.0] for s in (1, -1)],
           "prop_lines": [((L["x_prop"] / k, s * my - rp), (L["x_prop"] / k, s * my + rp)) for s in (1, -1)]}
    side = {"boom": [boom], "body": [body], "wing": [sec], "fins": fin_side, "pylons": pyl_side, "pods": [pod + [0.0, s * mz] for s in (1, -1)],
            "camera": [cam], "prop_lines": [((L["x_prop"] / k, s * mz - rp), (L["x_prop"] / k, s * mz + rp)) for s in (1, -1)]}
    t0 = 0.5 * p["thickness"] * L["c_root"] / k
    b2 = p["span"] / 2 / k
    ft = p["fin_thickness"] / 2 / k
    front = {"body": [circle * L["R"] / k], "wing": [np.array([[-b2, -0.3 * t0], [-b2, 0.3 * t0], [0, t0], [b2, 0.3 * t0], [b2, -0.3 * t0], [0, -t0]])],
             "pylons": [np.array([[s * my - 0.0045, -mz], [s * my - 0.0045, mz], [s * my + 0.0045, mz], [s * my + 0.0045, -mz]]) for s in (1, -1)],
             "pods": [circle * p["pod_diameter"] / 2 / k + [sy * my, sz * mz] for sy, sz in MOTORS],
             "fins": [np.array([[0, -ft], [p["fin_span"] / k, -ft], [p["fin_span"] / k, ft], [0, ft]]) @
                      np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))], [-math.sin(math.radians(a)), math.cos(math.radians(a))]])
                      for a in FIN_ANGLES],
             "props": [circle * rp + [sy * my, sz * mz] for sy, sz in MOTORS]}
    return {"top": top, "side": side, "front": front, "length": p["length"] / k, "span": p["span"] / k,
            "width": L["width"] / k, "height": L["height"] / k}
