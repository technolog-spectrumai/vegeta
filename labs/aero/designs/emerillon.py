"""ÉMERILLON — an electric speed quadcopter: a bullet with four motors (notebook 37).

Émerillon is the merlin in French: the small falcon that hunts by outrunning its prey in level flight. The machine is
the layout the speed-record quadcopters converged on (the Bells' Peregreen V4 holds the record at 657.59 km/h): a
slender body of revolution that stands on its tail, lifts off as a quadcopter, pitches over almost 90° and flies
nose-first, the four propellers pulling along the body's axis. Everything is printed; everything else is bought.

What this file builds (CadQuery through Dedalus):

- the **body**: an ogive nose (camera, GPS, the stack), a cylindrical bay the 6S packs slide into in line, and a
  tail cone that closes on a small blunt end;
- four **arms**: symmetric strut sections running out from the bay at 45°, 135°, 225°, 315°, long enough for the
  propeller tips to clear the body and each other;
- four **pods**: a spinner-shaped nose, the motor's cylinder and a cone behind it, on the arm tips — the motors pull
  (tractor), their propeller planes ahead of the arms;
- four **fins** on the tail cone at 0°, 90°, 180°, 270°, between the propeller wakes: the feet it stands on and
  the weathercock that keeps the nose into the flow.

The numbers come from the parts: ``emerillon_flight.cad_parameters(config)`` gives this design's parameters for a
motor, a propeller and a battery layout; the defaults are the shelf build the trade settles on (notebook 37).

Frame: ``x`` runs aft from the nose tip along the body axis (the free stream along +X), ``y`` right, ``z`` up when it
flies level; units mm. ``part``: ``aircraft`` (the whole machine, solid: pictures, wetted area, CFD; the propellers
are not built), ``arm`` (one arm with its pod and its root pad, for the FEA and the printer), ``nose`` (the nose
cone as a printed shell).
"""
import math

import cadquery as cq
import numpy as np

from vegeta.dedalus import Design, Parameter

ARM_ANGLES = (45.0, 135.0, 225.0, 315.0)
FIN_ANGLES = (0.0, 90.0, 180.0, 270.0)


def ogive(length: float, radius: float, n: int = 24) -> np.ndarray:
    """A tangent ogive from the tip (0, 0) to (length, radius): ``n`` points (x, r). Tangent to a cylinder at its end."""
    rho = (radius ** 2 + length ** 2) / (2 * radius)
    x = length * (1 - np.cos(np.linspace(0.0, math.pi / 2, n))) if n > 2 else np.array([0.0, length])
    r = np.sqrt(np.maximum(rho ** 2 - (length - x) ** 2, 0.0)) + radius - rho
    r[0], r[-1] = 0.0, radius
    return np.column_stack([x, r])


def strut(chord: float, thickness: float, n: int = 16) -> np.ndarray:
    """A symmetric NACA 00xx outline (closed trailing edge) of ``chord`` and ``thickness``: points (x, z), the upper
    side from the nose to the tail then the lower side back."""
    t = thickness / chord
    xc = 0.5 * (1 - np.cos(np.linspace(0.0, math.pi, n)))
    yt = 5 * t * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc ** 2 + 0.2843 * xc ** 3 - 0.1036 * xc ** 4)
    up = np.column_stack([xc, yt])
    return np.vstack([up, up[-2:0:-1] * [1, -1]]) * chord


class Emerillon(Design):
    """ÉMERILLON: body of revolution, four arms with motor pods, four tail fins."""

    parameters = [
        Parameter("part", "aircraft", choices=("aircraft", "arm", "nose"), description="what to build"),
        Parameter("body_diameter", 61.9, "mm", min=30.0, description="outer diameter of the battery bay"),
        Parameter("nose_length", 160.9, "mm", min=30.0, description="ogive nose, tip to the bay"),
        Parameter("bay_length", 134.5, "mm", min=40.0, description="cylindrical bay: the stack and the packs in line"),
        Parameter("tail_length", 198.1, "mm", min=30.0, description="tail cone"),
        Parameter("tail_end_diameter", 12.0, "mm", min=2.0, description="the tail cone's blunt end (wires, the antenna)"),
        Parameter("wall", 1.4, "mm", min=0.4, description="printed skin thickness"),
        Parameter("arm_radius", 130.0, "mm", min=40.0, description="motor axis from the body axis"),
        Parameter("arm_chord", 51.1, "mm", min=10.0, description="arm section chord (along the flow)"),
        Parameter("arm_thickness", 11.3, "mm", min=3.0, description="arm section thickness"),
        Parameter("arm_station", 0.5, "", min=0.1, max=0.9, description="arm mid-chord along the bay (0 front, 1 back)"),
        Parameter("pod_diameter", 37.5, "mm", min=10.0, description="motor pod diameter (the motor's can and a skin)"),
        Parameter("pod_length", 125.0, "mm", min=30.0, description="pod, spinner tip to the cone's end"),
        Parameter("motor_diameter", 33.5, "mm", min=8.0, description="motor can diameter (bought; it sizes the pod)"),
        Parameter("motor_height", 20.0, "mm", min=5.0, description="motor can height: the pod's flat motor face sits this far behind the propeller plane"),
        Parameter("prop_diameter", 177.8, "mm", min=50.0, description="propeller diameter (clearances; not built)"),
        Parameter("prop_gap", 10.0, "mm", min=2.0, description="propeller plane ahead of the arm's leading edge"),
        Parameter("fin_span", 55.0, "mm", min=10.0, description="fin, out from the tail cone"),
        Parameter("fin_chord", 60.0, "mm", min=10.0, description="fin chord at the root"),
        Parameter("fin_thickness", 3.0, "mm", min=1.0, description="fin plate thickness"),
        Parameter("pad_thickness", 4.0, "mm", min=1.0, description="arm root pad on the body ring (part='arm')"),
        Parameter("angle_of_attack_deg", 0.0, "deg", min=-20.0, max=90.0, description="pitch the aircraft nose-up about y (CFD)"),
    ]

    # ------------------------------------------------------------------------------------------------ stations
    @staticmethod
    def layout(p) -> dict:
        """Axial stations [mm]: where the bay starts and ends, the arm's leading edge, the pod and the propeller plane,
        the fins; and the clearances between the propeller tips, the body and each other."""
        R = p["body_diameter"] / 2
        x_bay, x_tail = p["nose_length"], p["nose_length"] + p["bay_length"]
        length = x_tail + p["tail_length"]
        x_arm = x_bay + p["arm_station"] * p["bay_length"] - p["arm_chord"] / 2
        spinner = 1.3 * p["pod_diameter"]
        x_prop = x_arm - p["prop_gap"]                                 # the blades turn ahead of the arm's leading edge
        return {"R": R, "x_bay": x_bay, "x_tail": x_tail, "length": length, "x_arm": x_arm, "x_pod": x_prop - spinner,
                "spinner": spinner, "x_prop": x_prop, "x_mount": x_prop + p["motor_height"], "x_fin": length - p["fin_chord"] - 0.12 * p["tail_length"],
                "tip_to_body": p["arm_radius"] - p["prop_diameter"] / 2 - R,
                "tip_to_tip": p["arm_radius"] * math.sqrt(2) - p["prop_diameter"]}

    @staticmethod
    def body_profile(p, n=24) -> np.ndarray:
        """The body's meridian (x, r) from the nose tip to the tail's blunt end."""
        L = Emerillon.layout(p)
        nose = ogive(p["nose_length"], L["R"], n)
        r_end = p["tail_end_diameter"] / 2
        tail = ogive(p["tail_length"], L["R"] - r_end, n)[::-1]                     # an ogive closing aft
        tail = np.column_stack([L["length"] - tail[:, 0], tail[:, 1] + r_end])
        return np.vstack([nose, tail])

    @staticmethod
    def tail_radius(p, x: float) -> float:
        prof = Emerillon.body_profile(p)
        return float(np.interp(x, prof[:, 0], prof[:, 1]))

    @staticmethod
    def pod_profile(p, n=14) -> np.ndarray:
        """A pod's meridian (x, r) on its own axis, x from the spinner tip."""
        r = p["pod_diameter"] / 2
        s = 1.3 * p["pod_diameter"]
        cone = p["pod_length"] - s - p["motor_height"]
        nose = ogive(s, r, n)
        aft = ogive(cone, r - 1.0, n)[::-1]
        aft = np.column_stack([p["pod_length"] - aft[:, 0], aft[:, 1] + 1.0])
        return np.vstack([nose, aft])

    # ---------------------------------------------------------------------------------------------------- solids
    @staticmethod
    def _revolve(profile, x0=0.0):
        pts = [(float(x) + x0, float(r)) for x, r in profile]
        pts = [pts[0]] + [q for a, q in zip(pts, pts[1:]) if abs(q[0] - a[0]) + abs(q[1] - a[1]) > 1e-6]
        if pts[-1][1] > 1e-9:
            pts.append((pts[-1][0], 0.0))
        return cq.Workplane("XY").polyline(pts).close().revolve(360.0, (0, 0, 0), (1, 0, 0))

    def _body(self, p):
        return self._revolve(self.body_profile(p))

    def _arm_blade(self, p, inner, outer):
        """The arm's strut from radius ``inner`` to ``outer`` along +Y (chord along X)."""
        L = self.layout(p)
        pts = [(float(x) + L["x_arm"], float(z)) for x, z in strut(p["arm_chord"], p["arm_thickness"])]
        return cq.Workplane("XZ").polyline(pts).close().extrude(-(outer - inner)).translate((0, inner, 0))

    def _pod(self, p):
        L = self.layout(p)
        return self._revolve(self.pod_profile(p), L["x_pod"]).translate((0, p["arm_radius"], 0))

    def _arm(self, p, inner=None):
        L = self.layout(p)
        inner = 0.6 * L["R"] if inner is None else inner              # into the body: one solid after the union
        return self._arm_blade(p, inner, p["arm_radius"]).union(self._pod(p))

    def _fin(self, p):
        L = self.layout(p)
        r0 = max(self.tail_radius(p, L["x_fin"] + p["fin_chord"]) - 1.0, 1.0)
        tip = 0.55 * p["fin_chord"]
        x0, x1 = L["x_fin"], L["x_fin"] + p["fin_chord"]
        pts = [(x0, r0), (x1 - tip, r0 + p["fin_span"]), (x1 + 0.10 * p["fin_chord"], r0 + p["fin_span"]), (x1, r0)]
        return cq.Workplane("XY").polyline(pts).close().extrude(p["fin_thickness"]).translate((0, 0, -p["fin_thickness"] / 2))

    def _arm_part(self, p):
        """One arm as it is printed and loaded: the strut from the body's surface, the standing part of its pod (from
        the flat face the motor bolts to, back to the cone's end — the spinner turns with the propeller, the motor is
        bought) and a root pad that bolts to the body ring."""
        L = self.layout(p)
        arm = self._arm_blade(p, L["R"], p["arm_radius"]).union(self._pod(p))
        pad = (cq.Workplane("XZ").center(L["x_arm"] + p["arm_chord"] / 2, 0).rect(p["arm_chord"] * 1.25, p["arm_thickness"] * 2.4)
               .extrude(-p["pad_thickness"]).translate((0, L["R"] - 0.5, 0)))
        y0 = p["arm_radius"] - p["pod_diameter"] / 2 - 1.0
        ahead = cq.Workplane("XY").box(L["x_mount"] - L["x_pod"] + 2.0, p["pod_diameter"] + 4.0, p["pod_diameter"] + 4.0, centered=False) \
            .translate((L["x_pod"] - 2.0, y0, -p["pod_diameter"] / 2 - 2.0))
        return arm.union(pad).cut(ahead)

    def _nose(self, p):
        L = self.layout(p)
        outer = self._revolve(ogive(p["nose_length"], L["R"]))
        k = (L["R"] - p["wall"]) / L["R"]
        inner = self._revolve(ogive(p["nose_length"] * k, L["R"] * k), p["nose_length"] * (1 - k))
        return outer.cut(inner)

    def build(self, p):
        if p["arm_radius"] - p["prop_diameter"] / 2 < p["body_diameter"] / 2:
            raise ValueError("the propeller tips reach the body: raise arm_radius")
        if p["part"] == "arm":
            return self._arm_part(p)
        if p["part"] == "nose":
            return self._nose(p)
        aircraft = self._body(p)
        arm, fin = self._arm(p), self._fin(p)
        for a in ARM_ANGLES:
            aircraft = aircraft.union(arm.rotate((0, 0, 0), (1, 0, 0), a - 90.0))
        for a in FIN_ANGLES:
            aircraft = aircraft.union(fin.rotate((0, 0, 0), (1, 0, 0), a))
        if p["angle_of_attack_deg"]:
            aircraft = aircraft.rotate((0, 0, 0), (0, 1, 0), p["angle_of_attack_deg"])
        return aircraft


# ------------------------------------------------------------------------------------------------------------ helpers
def resolve(p=None) -> dict:
    return Emerillon().resolve(**dict(p or {}))


def outline(p=None) -> dict:
    """Side and front views [m] for a drawing: the body, the two visible arms with their pods, the fins, the propeller
    disks (front view) and planes (side view)."""
    p = resolve(p)
    L = Emerillon.layout(p)
    prof = Emerillon.body_profile(p) / 1000.0
    side = [np.vstack([prof, prof[::-1] * [1, -1]])]
    pod = Emerillon.pod_profile(p) / 1000.0
    off = p["arm_radius"] * math.sin(math.radians(45.0)) / 1000.0
    for s in (1, -1):
        side.append(np.vstack([pod, pod[::-1] * [1, -1]]) + [L["x_pod"] / 1000.0, s * off])
    th = np.linspace(0, 2 * math.pi, 60)
    circle = np.column_stack([np.cos(th), np.sin(th)])
    front = {"body": circle * L["R"] / 1000.0,
             "props": [circle * p["prop_diameter"] / 2000.0 + [p["arm_radius"] / 1000.0 * math.cos(math.radians(a)),
                                                                p["arm_radius"] / 1000.0 * math.sin(math.radians(a))] for a in ARM_ANGLES],
             "pods": [circle * p["pod_diameter"] / 2000.0 + [p["arm_radius"] / 1000.0 * math.cos(math.radians(a)),
                                                              p["arm_radius"] / 1000.0 * math.sin(math.radians(a))] for a in ARM_ANGLES]}
    return {"side": side, "front": front, "prop_x": L["x_prop"] / 1000.0, "prop_half": p["prop_diameter"] / 2000.0,
            "prop_offsets": (off, -off), "length": L["length"] / 1000.0}
