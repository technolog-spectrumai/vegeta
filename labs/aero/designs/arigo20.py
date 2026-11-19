"""ARIGO-20 — the fast one: two electric ducted fans high on the rear fuselage and a twin-fin tail, after the
A-10's layout (notebook 39/40).

The layout's reasons, kept here: the fans sit above and behind the wing — the wing shields the intakes from the
ground and from what the belly throws up on a belly landing; nothing turns ahead of the nose camera; the fans are
close to the centre line, so one stopping yaws the aircraft little; the jets pass over the stabiliser and between
the two fins at its tips (the fins also hide the jets' heat and noise from the sides). Ducted fans are efficient
only well above ARIGO-1's speeds, so ARIGO-20 has a smaller, more loaded, more swept wing for 30-40 m/s.

Built on ARIGO-1's design (imported, not changed): its fuselage, wing (with its own sizes) and section helpers;
the tail and the propulsion redrawn. Frame: x aft from the nose tip, y right, z up; mm.
"""
from __future__ import annotations

import dataclasses
import math

import cadquery as cq
import numpy as np

import arigo1 as a1
from fixed_wing import naca4
from vegeta.dedalus import Parameter

_OVERRIDE = {  # ARIGO-1 parameters with new defaults: the wing for 30-40 m/s, the tail as a twin-fin H
    "propulsion": "pusher",          # (ARIGO-1's switch is unused here: no propeller anywhere)
    "span": 820.0, "root_chord": 250.0, "taper": 0.45, "le_sweep_deg": 22.0, "dihedral_deg": 2.0, "washout_deg": 2.0,
    "incidence_deg": 1.0, "thickness": 0.11, "camber": 0.015, "wing_x": 220.0,
    "fuselage_length": 740.0, "fuselage_diameter": 78.0, "tail_cone": 300.0, "tail_end_diameter": 36.0,
    "htail_span": 440.0, "htail_root_chord": 165.0, "htail_taper": 0.80, "htail_sweep_deg": 8.0,
    "fin_height": 130.0, "fin_root_chord": 150.0, "fin_taper": 0.70, "fin_sweep_deg": 20.0,
    "skid": True,
}
_NEW = [
    Parameter("edf_diameter", 70.0, "mm", min=40.0, max=120.0, description="fan (rotor) diameter: the 70 mm class"),
    Parameter("edf_length", 185.0, "mm", min=80.0, description="nacelle: intake lip to nozzle"),
    Parameter("edf_wall", 3.0, "mm", min=1.0),
    Parameter("edf_y", 62.0, "mm", min=30.0, description="each nacelle's axis from the centre plane"),
    Parameter("edf_z", 66.0, "mm", min=30.0, description="the nacelles' axis above the fuselage axis"),
    Parameter("edf_x", 480.0, "mm", min=150.0, description="the intake lip from the nose tip"),
    Parameter("nozzle_ratio", 0.88, "", min=0.6, max=1.0, description="nozzle exit / fan diameter (the outlet area ~ 0.77 FSA at 0.88)"),
    Parameter("edf_toe_deg", 0.0, "deg", min=-5.0, max=5.0),
    Parameter("show_jets", False, description="the two jets as translucent cones (pictures)"),
]


class Arigo20(a1.Arigo1):
    """ARIGO-20: ARIGO-1's fuselage and wing family; two ducted fans on pylons over the rear fuselage; an H tail."""

    parameters = [dataclasses.replace(q, default=_OVERRIDE[q.name]) if q.name in _OVERRIDE else q
                  for q in a1.Arigo1.parameters] + _NEW

    @staticmethod
    def layout(p) -> dict:
        L = a1.Arigo1.layout(p)
        hb2 = p["htail_span"] / 2
        fct = p["fin_root_chord"] * p["fin_taper"]
        S_fin = 0.5 * (p["fin_root_chord"] + fct) * p["fin_height"]
        mac_v = 2 / 3 * (p["fin_root_chord"] ** 2 + p["fin_root_chord"] * fct + fct ** 2) / (p["fin_root_chord"] + fct)
        z_mac = p["fin_height"] / 3 * (p["fin_root_chord"] + 2 * fct) / (p["fin_root_chord"] + fct)
        x_tip_le = p["tail_le_x"] + hb2 * math.tan(math.radians(p["htail_sweep_deg"]))
        x_fin = x_tip_le + p["htail_root_chord"] * p["htail_taper"] - p["fin_root_chord"]       # the fins' trailing edge on the stabiliser's tip's
        x_ac_v = x_fin + z_mac * math.tan(math.radians(p["fin_sweep_deg"])) + 0.25 * mac_v
        R_out = p["edf_diameter"] / 2 + p["edf_wall"] + 2.0
        x_exit = p["edf_x"] + p["edf_length"]
        jet_r = p["nozzle_ratio"] * p["edf_diameter"] / 2
        # the jets' edges where they cross the tail (spreading ~5 deg half-angle: HAND)
        spread = lambda x: jet_r + max(0.0, x - x_exit) * math.tan(math.radians(5.0))
        x_tail = p["tail_le_x"] + p["htail_root_chord"]
        stations = np.array([s[:3] for s in L["stations"]])
        x_te_at_pod = float(np.interp(p["edf_y"], stations[:, 0], stations[:, 1] + stations[:, 2]))
        L.update({
            "S_v_mm2": 2 * S_fin, "x_fin": x_fin, "x_ac_v": x_ac_v, "z_ac_v": 2.0 + z_mac, "fin_y": hb2,
            "l_v": x_ac_v - L["x_ac"], "V_v": 2 * S_fin * (x_ac_v - L["x_ac"]) / (L["S_mm2"] * p["span"]),
            "props": [], "x_prop": None, "prop_d": 0.0, "camera": (p["camera_diameter"] / 2, 0.0, 0.0), "prop_below_belly": None,
            "edf": [(p["edf_y"], p["edf_z"]), (-p["edf_y"], p["edf_z"])], "edf_R_out": R_out, "x_exit": x_exit,
            "fsa_mm2": math.pi * (p["edf_diameter"] / 2) ** 2, "exit_area_mm2": math.pi * jet_r ** 2,
            # clearances
            "intake_behind_wing_te": p["edf_x"] - x_te_at_pod,          # >0: the lip behind the wing's trailing edge (A-10)
            "nacelle_gap_to_fuselage": math.hypot(p["edf_y"], p["edf_z"]) - R_out - L["R"],
            "nacelle_gap_between": 2 * p["edf_y"] - 2 * R_out,
            "jet_above_stabiliser": p["edf_z"] - spread(x_tail) - 2.0,      # >0: the jet clears the stabiliser at its trailing edge
            "jet_inside_fins": hb2 - (p["edf_y"] + spread(x_tail)),         # >0: the jet passes between the fins
        })
        return L

    # -------------------------------------------------------------------------------------------- tail: H
    def _tail(self, p):
        L = self.layout(p)
        h = self._surface(p["tail_le_x"], p["htail_root_chord"], p["htail_taper"], p["htail_span"], p["htail_sweep_deg"],
                          p["tail_thickness"], "y", z0=2.0)
        out = h
        for s in (1, -1):
            fin = self._surface(L["x_fin"], p["fin_root_chord"], p["fin_taper"], p["fin_height"], p["fin_sweep_deg"],
                                p["tail_thickness"], "z", z0=-0.12 * p["fin_height"])            # a short stub below the stabiliser, kept above the belly
            out = out.union(fin.translate((0, s * L["fin_y"], 2.0)))
        return out

    # -------------------------------------------------------------------------------------------- fans
    def _nacelle(self, p):
        """One nacelle along +x from its intake lip at the origin: a duct with a rounded lip, the fan's hub and
        stator body inside, a nozzle narrowing to ``nozzle_ratio``; and a tail cone in the jet (the motor's)."""
        Ri = p["edf_diameter"] / 2
        Ro = Ri + p["edf_wall"]
        n = p["edf_length"]
        r_exit = p["nozzle_ratio"] * Ri
        outer = [(0.0, Ri + 0.5 * p["edf_wall"]), (6.0, Ro + 1.5), (0.3 * n, Ro + 2.0), (0.75 * n, Ro + 1.0), (n, r_exit + p["edf_wall"])]
        inner = [(n, r_exit), (0.6 * n, Ri), (0.12 * n, Ri), (0.0, Ri + 0.5 * p["edf_wall"])]
        prof = outer + inner
        duct = cq.Workplane("XY").polyline(prof).close().revolve(360.0, (0, 0, 0), (1, 0, 0))
        hub = (cq.Workplane("YZ").circle(0.36 * Ri).extrude(0.62 * n).translate((0.18 * n, 0, 0))
               .faces("<X").edges().fillet(0.3 * Ri))
        stators = None
        for k in range(3):
            v = cq.Workplane("XY").box(0.25 * n, 1.6, 0.70 * Ri).translate((0.4 * n, 0, 0.30 * Ri + 0.35 * Ri))             # into the duct wall
            v = v.rotate((0, 0, 0), (1, 0, 0), 120.0 * k)
            stators = v if stators is None else stators.union(v)
        return duct.union(hub).union(stators)

    def _motors(self, p):
        """Both nacelles on swept pylons from the fuselage's top."""
        L = self.layout(p)
        out = None
        for y, z in L["edf"]:
            s = 1 if y > 0 else -1
            nac = self._nacelle(p).rotate((0, 0, 0), (0, 0, 1), s * p["edf_toe_deg"]).translate((p["edf_x"], y, z))
            # the pylon: a thin streamlined plate from the fuselage's crown to the nacelle's underside
            r = math.hypot(y, z)
            reach = r - 0.5 * L["edf_R_out"]                  # from the fuselage's axis into the nacelle (unions cleanly)
            up, lo = naca4(0.0, 0.4, 0.10, n=16)
            c = 0.55 * p["edf_length"]
            sec = [(x * c, zz * c) for x, zz in up + lo[::-1][1:]]
            pyl = (cq.Workplane("XY").polyline(sec).close().extrude(reach)                      # along +z ...
                   .rotate((0, 0, 0), (1, 0, 0), -math.degrees(math.atan2(y, z)))              # ... turned to point at the nacelle
                   .translate((p["edf_x"] + 0.30 * p["edf_length"], 0, 0)))
            out = (nac.union(pyl)) if out is None else out.union(nac).union(pyl)
        return out

    def _props(self, p):
        """The two jets as cones (pictures only)."""
        L = self.layout(p)
        out = None
        r0 = p["nozzle_ratio"] * p["edf_diameter"] / 2
        for y, z in L["edf"]:
            j = cq.Solid.makeCone(r0, r0 + 300.0 * math.tan(math.radians(5.0)), 300.0, cq.Vector(L["x_exit"], y, z), cq.Vector(1, 0, 0))
            j = cq.Workplane("XY").add(j)
            out = j if out is None else out.union(j)
        return out

    def parts(self, p) -> dict:
        out = {"fuselage": self._fuselage(p), "wing": self._wing(p), "tail": self._tail(p), "camera": self._camera(p), "motors": self._motors(p)}
        if p["show_jets"]:
            out["props"] = self._props(p)
        if p["perch_hook"]:
            out["hook"] = self._hook(p)
        return out

    def build(self, p):
        if p["part"] == "tail":
            return self._tail(p)
        if p["part"] in ("wing", "fuselage"):
            return super().build(p)
        ac = self._fuselage(p).union(self._wing(p)).union(self._tail(p)).union(self._camera(p)).union(self._motors(p))
        if p["perch_hook"]:
            ac = ac.union(self._hook(p))
        if p["show_jets"]:
            ac = ac.union(self._props(p))
        return ac


def variant(**over) -> dict:
    q = dict(_OVERRIDE)
    q["tail_le_x"] = q["fuselage_length"] - 10.0 - q["htail_root_chord"]
    return {**q, **over}


def resolve(q=None) -> dict:
    return Arigo20().resolve(**dict(q or variant()))


def checks(q=None) -> dict:
    p = resolve(q)
    L = Arigo20.layout(p)
    keys = ("S_mm2", "AR", "mac", "V_h", "V_v", "fsa_mm2", "exit_area_mm2", "intake_behind_wing_te", "nacelle_gap_to_fuselage",
            "nacelle_gap_between", "jet_above_stabiliser", "jet_inside_fins")
    return {k: round(float(L[k]), 3 if k in ("AR", "V_h", "V_v") else 1) for k in keys}
