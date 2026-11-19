"""ARIGO-2 — ARIGO-1's airframe with two tractor propellers on the wing (notebook 39/40).

Why two wing tractors: the nose is the camera's alone (no disc in its view, nothing ahead of it), the slipstream
blows the inner wing and the ailerons stay outboard of it, the two propellers turn opposite ways (no net torque
roll, no P-factor yaw at high alpha) and differential thrust gives yaw on top of the rudder. What it costs: two
motors and ESCs, two nacelles' drag, an engine out yaws the aircraft, and the discs hang below the belly — a belly
landing with the propellers turning strikes them, so the blades **fold** (motors stopped before touchdown: the
airflow lays them back along the nacelle).

ARIGO-1's design is imported, not changed: ``Arigo2`` is ``Arigo1`` with ``propulsion = "twin"`` fixed and the
nacelles redrawn (spinner, motor, the nacelle raised above the chord line, down-thrust and toe-out, two-blade
folding propellers, open or folded). The wing is one of ARIGO-1's variants (default ``hybrid``).

Frame: x aft from the nose tip, y right, z up; mm.
"""
from __future__ import annotations

import dataclasses
import math

import cadquery as cq
import numpy as np

import arigo1 as a1
from vegeta.dedalus import Parameter

_OVERRIDE = {"propulsion": "twin", "twin_prop_diameter": 203.2, "nacelle_y": 240.0, "nacelle_length": 135.0, "nacelle_diameter": 40.0}
_NEW = [
    Parameter("nacelle_z", 14.0, "mm", min=-30.0, max=60.0, description="the thrust line above the wing's chord line at the nacelle"),
    Parameter("down_thrust_deg", 2.0, "deg", min=-5.0, max=8.0),
    Parameter("toe_out_deg", 1.5, "deg", min=-5.0, max=8.0, description="each thrust line turned away from the fuselage"),
    Parameter("blades", 2, "", min=2, max=3),
    Parameter("folded", False, description="propellers stopped, the blades laid back along the nacelles (landing)"),
    Parameter("inboard_up", True, description="the blades rise on the inboard side (counter-rotating; tip vortex against the wing's)"),
]


class Arigo2(a1.Arigo1):
    """ARIGO-2: ARIGO-1's fuselage, wing and tail; two folding tractor propellers on wing nacelles."""

    parameters = [dataclasses.replace(q, default=_OVERRIDE[q.name]) if q.name in _OVERRIDE else q
                  for q in a1.Arigo1.parameters] + _NEW

    @staticmethod
    def layout(p) -> dict:
        L = a1.Arigo1.layout({**p, "propulsion": "twin"})
        st = np.array([s[:4] for s in L["stations"]])
        y_n = p["nacelle_y"]
        x_le = float(np.interp(y_n, st[:, 0], st[:, 1]))
        c_n = float(np.interp(y_n, st[:, 0], st[:, 2]))
        z_w = float(np.interp(y_n, st[:, 0], st[:, 3]))
        z_t = z_w + p["nacelle_z"]
        x_prop = x_le - 0.45 * p["nacelle_length"]
        R, D = L["R"], p["twin_prop_diameter"]
        L.update({"x_prop": x_prop, "props": [(y_n, z_t), (-y_n, z_t)], "prop_d": D, "chord_at_nacelle": c_n, "x_le_nacelle": x_le,
                  "tip_to_fuselage": y_n - D / 2 - R,
                  "disc_below_belly": (-R) - (z_t - D / 2),                # >0: the turning disc reaches below the belly
                  "blown_span_share": 2 * D / p["span"],                    # the span inside the two discs
                  "aileron_inner_y": p["span"] / 2 * (1 - p["aileron_span"]),
                  "aileron_clear_of_wash": p["span"] / 2 * (1 - p["aileron_span"]) - (y_n + D / 2),
                  "camera": (p["camera_diameter"] / 2, 0.0, 0.0)})
        return L

    def _motors(self, p):
        L = self.layout(p)
        out = None
        for y, z in L["props"]:
            s = 1 if y > 0 else -1
            body = (cq.Workplane("YZ").circle(p["nacelle_diameter"] / 2).extrude(p["nacelle_length"])
                    .edges(">X").chamfer(p["nacelle_diameter"] * 0.3))
            spinner = cq.Workplane("XY").sphere(p["nacelle_diameter"] * 0.42).translate((0, 0, 0))
            n = body.union(spinner).translate((L["x_prop"] + 4.0, 0, 0))
            pylon = cq.Workplane("XY").box(p["nacelle_length"] * 0.7, 8.0, abs(p["nacelle_z"]) + 8.0).translate(
                (L["x_prop"] + 4.0 + p["nacelle_length"] * 0.55, 0, -p["nacelle_z"] / 2))
            n = n.union(pylon)
            n = n.rotate((L["x_prop"], 0, 0), (L["x_prop"], 1, 0), -p["down_thrust_deg"])     # thrust line tipped down
            n = n.rotate((L["x_prop"], 0, 0), (L["x_prop"], 0, 1), s * p["toe_out_deg"])
            n = n.translate((0, y, z))
            out = n if out is None else out.union(n)
        return out

    def _props(self, p):
        """Two-(three-)blade propellers: open (a blade pair across the disc, at a fixed phase) or folded back."""
        L = self.layout(p)
        R = L["prop_d"] / 2
        out = None
        for y, z in L["props"]:
            hub = cq.Workplane("YZ").circle(9.0).extrude(10.0).translate((L["x_prop"] - 8.0, y, z))
            out = hub if out is None else out.union(hub)
            for k in range(p["blades"]):
                ph = math.radians(40.0 + 360.0 / p["blades"] * k)
                if p["folded"]:
                    blade = cq.Workplane("XY").box(R * 0.92, 12.0, 3.0).translate((L["x_prop"] - 4.0 + R * 0.46, 0, 0))
                    blade = blade.rotate((L["x_prop"], 0, 0), (L["x_prop"] + 1, 0, 0), math.degrees(ph))
                    blade = blade.translate((0, (p["nacelle_diameter"] / 2 + 2) * math.cos(ph), (p["nacelle_diameter"] / 2 + 2) * math.sin(ph)))
                else:
                    blade = (cq.Workplane("XY").box(4.0, R - 10.0, 18.0).translate((L["x_prop"] - 3.0, (R - 10.0) / 2 + 10.0, 0))
                             .rotate((L["x_prop"], 0, 0), (L["x_prop"] + 1, 0, 0), math.degrees(ph)))
                out = out.union(blade.translate((0, y, z)))
        return out

    def build(self, p):
        p = {**p, "propulsion": "twin"}
        if p["part"] != "aircraft":
            return super().build(p)
        ac = super().build({**p, "show_props": False})
        if p["show_props"] or p["folded"]:
            ac = ac.union(self._props(p))
        return ac

    def parts(self, p) -> dict:
        out = super().parts({**p, "propulsion": "twin"})
        out["blades" if p["folded"] else "props"] = self._props(p)       # folded blades are solid in pictures, not a disc
        return out


def variant(wing: str = "hybrid", **over) -> dict:
    """ARIGO-2 on one of ARIGO-1's wings."""
    q = a1.variant(wing, "twin")
    return {**q, **_OVERRIDE, **over}


def resolve(q=None) -> dict:
    return Arigo2().resolve(**dict(q or {}))


def engine_out(wing: str = "hybrid", speeds=(10, 12, 14, 16, 18, 20, 22, 25)) -> dict:
    """One motor stops at full power on the other: the yawing moment T y against the full rudder's (``arigo1_model``
    coefficients for the twin; the per-motor thrust is half of the model's hand-estimated curve). V_mc: the speed
    below which the rudder cannot hold it (the good motor must then be throttled back)."""
    import arigo1_model as am
    C = am.coefficients(wing, "twin")
    p = resolve(variant(wing))
    y = p["nacelle_y"] / 1e3
    rows, vmc = [], None
    for V in speeds:
        T1 = 0.5 * C["thrust"]["T0_N"] * max(0.0, 1 - V / C["thrust"]["V0_m_s"])
        N_rud = abs(C["Cndr"]) * am.DR_MAX * 0.5 * am.RHO * V * V * C["S"] * C["b"]
        rows.append({"V_m_s": V, "yaw_from_one_motor_Nm": round(T1 * y, 3), "full_rudder_Nm": round(N_rud, 3), "held": N_rud >= T1 * y})
    for r in rows:
        if r["held"]:
            vmc = r["V_m_s"]
            break
    return {"rows": rows, "V_mc_m_s_about": vmc, "stall_speed_m_s": round(math.sqrt(2 * C["mass"] * am.G / (am.RHO * C["S"] * C["CL_MAX"])), 1),
            "note": "thrust curve and rudder power are hand estimates (arigo1_model); one-motor thrust = half the twin's"}


def checks(wing: str = "hybrid") -> dict:
    p = resolve(variant(wing))
    L = Arigo2.layout(p)
    return {k: round(float(L[k]), 1) if isinstance(L[k], (int, float)) else L[k] for k in
            ("tip_to_fuselage", "disc_below_belly", "blown_span_share", "aileron_clear_of_wash", "chord_at_nacelle", "V_h", "V_v")}
