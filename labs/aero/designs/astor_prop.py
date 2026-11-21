"""ASTOR's folding propeller: a 9 x 11 two-blade tractor propeller whose blades hinge back along the nose when the
motor stops (CAD, Dedalus; and the numbers that say it works).

**How it is made.**

- A **yoke** on the motor shaft: a hub with two forks, one each side of the axis, ``hinge_radius`` out. Each fork
  holds a steel **hinge pin** across the blade's root (the pin lies in the propeller's plane, square to the blade).
- Two **blades**, each ending at its root in an eye that sits in the fork on the pin. A flat on the eye meets a flat in
  the fork when the blade is fully open: the **open stop**, set a few degrees forward of square (``cone_deg``) so
  thrust and centrifugal force share the root's bending.
- A short **spinner** ahead of the yoke; the blades fold behind it, along the nose.
- No spring and no actuator. **Running**, centrifugal force swings the blades out against the stops (hundreds of
  newtons per blade at speed; the air's push back is a few). **Stopped** (the ESC brakes the motor), the airflow
  folds them back in a fraction of a second; they trail along the nose, edge-on, until the motor turns again.

**What decides the sizes** (``checks``): the pin's shear and the eye's bearing under the centrifugal load at the top
speed of rotation; the speed of rotation at which the blades open against the air at a flight speed; the drag that
folds a stopped blade; the folded blades' clearance over the nose.

``state``: ``open``, ``folded``, or ``fold_deg`` for anything between. Frame: the shaft along x (the aircraft's: aft
positive, the propeller ahead of the nose), the blades along +-y when open; the hinge pins along z; mm.
"""
from __future__ import annotations

import math

import cadquery as cq

from fixed_wing import naca4
from vegeta.dedalus import Design, Parameter

RHO_BLADE = 1500.0                 # kg/m3: glass-filled nylon / carbon-nylon (HAND +-15 %)
PIN_SHEAR_MPA = 400.0              # hardened steel dowel, allowable in shear (HAND, with margin)
EYE_BEARING_MPA = 60.0             # carbon-nylon in bearing (HAND)


class FoldingProp(Design):
    """A two-blade folding propeller: yoke, two hinge pins, two blades, a spinner."""

    parameters = [
        Parameter("diameter", 228.6, "mm", min=100.0, description="9 inch"),
        Parameter("pitch", 279.4, "mm", min=50.0, description="11 inch: the advance per turn the blades are twisted for"),
        Parameter("hinge_radius", 18.0, "mm", min=8.0, description="the hinge pins from the shaft's axis"),
        Parameter("fold_deg", 0.0, "deg", min=0.0, max=88.0, description="0 open ... ~82 folded back along the nose"),
        Parameter("cone_deg", 3.0, "deg", min=0.0, max=10.0, description="open: the blades this far forward of square (the stop)"),
        Parameter("chord_root", 17.0, "mm", min=6.0), Parameter("chord_max", 25.0, "mm", min=8.0), Parameter("chord_tip", 9.0, "mm", min=3.0),
        Parameter("thickness", 0.10, "", min=0.05, max=0.2, description="blade section thickness / chord"),
        Parameter("pin_diameter", 3.0, "mm", min=1.5), Parameter("eye_width", 7.0, "mm", min=3.0),
        Parameter("spinner_diameter", 34.0, "mm", min=15.0), Parameter("spinner_length", 30.0, "mm", min=10.0),
        Parameter("hub_length", 16.0, "mm", min=8.0),
    ]

    # ------------------------------------------------------------------------------------------------ blade
    @staticmethod
    def stations(p, n=9):
        """(radius, chord, pitch angle deg) from the hinge to the tip."""
        R, rh = p["diameter"] / 2, p["hinge_radius"]
        out = []
        for k in range(n):
            f = k / (n - 1)
            r = rh + 6.0 + f * (R - rh - 6.0)
            c = (p["chord_root"] + (p["chord_max"] - p["chord_root"]) * math.sin(math.pi * min(f / 0.45, 1.0) / 2)) if f < 0.45 else \
                (p["chord_max"] + (p["chord_tip"] - p["chord_max"]) * ((f - 0.45) / 0.55) ** 1.6)
            out.append((r, c, math.degrees(math.atan(p["pitch"] / (2 * math.pi * r)))))
        return out

    def _blade(self, p):
        """One blade along +y, its root eye on the hinge at (0, hinge_radius, 0)."""
        up, lo = naca4(0.03, 0.4, p["thickness"], n=14)
        st = self.stations(p)
        wp = cq.Workplane("XZ")
        prev = st[0][0]
        for k, (r, c, beta) in enumerate(st):
            if k:
                wp = wp.workplane(offset=-(r - prev))
                prev = r
            b = math.radians(beta)
            pts = []
            for x, z in up + lo[::-1][1:]:
                u, w = (x - 0.35) * c, z * c                       # along the chord, across it
                pts.append((-u * math.sin(b) - w * math.cos(b), u * math.cos(b) - w * math.sin(b)))   # axial (x), tangential (z)
            wp = wp.polyline(pts).close()
        blade = wp.loft(combine=True, ruled=False).translate((0, st[0][0], 0))                 # along +y, the first section at its radius
        eye = (cq.Workplane("XY").circle(p["pin_diameter"] / 2 + 2.6).extrude(p["eye_width"]).translate((0, p["hinge_radius"], -p["eye_width"] / 2)))
        neck = cq.Workplane("XY").box(7.0, 8.0, p["eye_width"] * 0.8).translate((0, p["hinge_radius"] + 5.5, 0))
        return blade.union(eye).union(neck)

    def _blades(self, p):
        out = None
        ang = p["fold_deg"] - p["cone_deg"] if p["fold_deg"] > 0.5 else -p["cone_deg"]
        for s in (1, -1):
            b = self._blade(p)
            b = b.rotate((0, p["hinge_radius"], 0), (0, p["hinge_radius"], 1), -ang)       # about the pin (z): +y toward +x, the tip goes aft
            if s < 0:
                b = b.rotate((0, 0, 0), (1, 0, 0), 180.0)
            out = b if out is None else out.union(b)
        return out

    def _yoke(self, p):
        rh, ew = p["hinge_radius"], p["eye_width"]
        hub = cq.Workplane("YZ").circle(8.0).extrude(p["hub_length"]).translate((-p["hub_length"] / 2, 0, 0))
        bar = cq.Workplane("XY").box(10.0, 2 * rh + 10.0, ew + 8.0)
        slot = None
        for s in (1, -1):
            c = cq.Workplane("XY").box(14.0, 12.0, ew + 0.6).translate((0, s * (rh + 1.0), 0))
            slot = c if slot is None else slot.union(c)
        yoke = hub.union(bar.cut(slot))
        for s in (1, -1):
            pin = cq.Workplane("XY").circle(p["pin_diameter"] / 2).extrude(ew + 8.0).translate((0, s * rh, -(ew + 8.0) / 2))
            yoke = yoke.union(pin)
        return yoke

    def _spinner(self, p):
        r, L = p["spinner_diameter"] / 2, p["spinner_length"]
        pts = [(0.0, 0.0)] + [(-L * (1 - (k / 10) ** 0.5) * 0 - L * (1 - k / 10), r * math.sin(math.pi / 2 * (k / 10) ** 0.6)) for k in range(1, 11)] + [(0.0, 0.0)]
        pts = [(-L, 0.0)] + [(-L * (1 - k / 10), r * math.sin(math.pi / 2 * (k / 10) ** 0.6)) for k in range(1, 11)] + [(0.0, 0.0)]
        return cq.Workplane("XY").polyline(pts).close().revolve(360.0, (0, 0, 0), (1, 0, 0)).translate((-p["hub_length"] / 2 - 1.0, 0, 0))

    def parts(self, p) -> dict:
        return {"blades": self._blades(p), "yoke": self._yoke(p), "spinner": self._spinner(p)}

    def build(self, p):
        return self._yoke(p).union(self._spinner(p)).union(self._blades(p))


def checks(q=None, rpm_max=15000.0, V_flight=65.0) -> dict:
    """The hinge under the centrifugal load, the opening speed, the folding drag, the folded envelope."""
    P = FoldingProp()
    p = P.resolve(**(q or {}))
    st = P.stations(p, 40)
    rh = p["hinge_radius"]
    m, mr, area, a_r = 0.0, 0.0, 0.0, 0.0
    for (r0, c0, _), (r1, c1, _) in zip(st, st[1:]):
        c, r, dr = 0.5 * (c0 + c1), 0.5 * (r0 + r1), r1 - r0
        dm = RHO_BLADE * (0.68 * c * p["thickness"] * c) * dr * 1e-9          # section area ~0.68 t c
        m += dm; mr += dm * r; area += c * dr; a_r += c * dr * (r - rh)
    r_cg = mr / m
    w = rpm_max * 2 * math.pi / 60.0
    F = m * w * w * r_cg / 1000.0                                             # N per blade
    pin_area = math.pi * (p["pin_diameter"] / 2) ** 2
    tau = F / (2 * pin_area)                                                  # double shear
    bearing = F / (p["pin_diameter"] * p["eye_width"])
    # stopped, open, at V: the air's moment about the pin that folds the blade (flat plate, Cd 1.2)
    M_fold = 0.5 * 1.225 * V_flight ** 2 * 1.2 * (a_r * 1e-9)
    # running: the centrifugal moment that opens a blade folded by phi against the same drag at V: ~ m w^2 r_cg (r_cg - rh) sin(phi)
    arm = (r_cg - rh) / 1000.0
    M_drag_folded = 0.5 * 1.225 * V_flight ** 2 * 0.1 * (a_r * 1e-9)          # edge-on, Cd 0.1 (HAND)
    w_open = math.sqrt(M_drag_folded / max(m * (r_cg / 1000.0) * arm * math.sin(math.radians(60.0)), 1e-12))
    I_pin = sum(RHO_BLADE * 0.68 * (0.5 * (a[1] + b[1])) ** 2 * p["thickness"] * (b[0] - a[0]) * 1e-9 * ((0.5 * (a[0] + b[0]) - rh) / 1000.0) ** 2 for a, b in zip(st, st[1:]))
    t_fold = math.sqrt(2 * math.radians(80.0) * I_pin / max(0.5 * 1.225 * 20.0 ** 2 * 1.2 * (a_r * 1e-9), 1e-12))     # at 20 m/s, constant moment
    R = p["diameter"] / 2
    phi = math.radians(82.0)
    return {"blade_mass_g": round(m * 1000, 1), "blade_cg_radius_mm": round(r_cg, 0), "rpm_max": rpm_max,
            "tip_speed_m_s": round(w * R / 1000.0, 0),
            "centrifugal_load_per_blade_N": round(F, 0), "pin_shear_MPa": round(tau, 0), "pin_ok": bool(tau < PIN_SHEAR_MPA),
            "eye_bearing_MPa": round(bearing, 0), "eye_ok": bool(bearing < EYE_BEARING_MPA),
            "folding_moment_at_dash_Nmm": round(M_fold * 1000, 0), "opens_above_rpm_at_dash": round(w_open * 60 / (2 * math.pi), 0),
            "time_to_fold_at_20_m_s_s": round(t_fold, 2),
            "folded": {"fold_deg": 82.0, "reach_aft_of_hinge_mm": round((R - rh) * math.sin(phi), 0),
                       "tip_from_axis_mm": round(rh + (R - rh) * math.cos(phi), 0)},
            "note": "blade density, the pin's and the eye's allowables, and the drag coefficients are HAND"}
