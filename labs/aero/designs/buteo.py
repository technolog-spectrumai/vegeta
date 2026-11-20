"""BUTEO — the buzzard: a lift + cruise VTOL that guards one site with nobody there (RAPTORS product line; the
concept in ``reports/arigo_catalog/PRODUCT_SPACE.md``, accepted 2026-11-20).

**The job.** Two to four units sit on charging pads around a substation, a solar farm, a prison. On the site sensor's
cue one takes off vertically, turns into a wing-borne aircraft, meets the intruder inside ~1.5 km, comes back, lands
on its pad and recharges — for weeks, with nobody touching it. The requirement that sizes it: **the unattended
cycle** (vertical launch and a landing on a 1.2 m pad in the site's wind), not the flight in between.

**The layout, and why.**

- A **pod-and-twin-boom pusher**. The pod carries the camera in the nose, the pack and the avionics, and the pusher
  motor at its end; the wing sits on top of it. Two carbon booms under the wing carry everything else.
- **Four lift rotors** (9 x 5, two blades) on the booms, two ahead of the wing and two behind it, at equal arms from
  the CG. In the hover the wing is level: a gust meets the pod's and the fins' side, not the wing's whole area (the
  tailsitter's weakness). In wing-borne flight the rotors stop, blades along the booms.
- An **H tail on the booms**: the stabiliser spans from boom to boom behind the rear rotors and the pusher's disc,
  a fin on each boom. The pusher turns between the booms, clear of the rear rotors by 29 mm.
- A **moderate, unswept wing** (1.1 m, aspect ratio 6, the family's 12 % section): it flies for an hour if it has
  to, but the job needs minutes; the area is set by a stall the transition can reach quickly.
- **Four feet** under the rotor mounts with sprung charging contacts in their soles (GAVILAN's perch idea,
  ``gavilan_perch``): any two diagonal feet are +, the other two -, through an ideal-diode bridge.
- A **4S2P Li-ion pack**: the lift rotors' bursts (4 x 300 W) and the pusher (800 W) stay inside its 90 A; it lives
  on a charger for weeks better than a LiPo pack.

Built on ARIGO-1's parametric wing, pod and surface generators (imported, not changed) with BUTEO's own layout,
parts, mass budget and pack. Frame: x aft from the nose tip, y right, z up; mm.
"""
from __future__ import annotations

import dataclasses
import math

import cadquery as cq

import arigo1 as a1
from vegeta.dedalus import Parameter

PAD_SIZE_M = 1.2                   # the charging pad: a square plate; all four feet (a 0.54 m square) must land on it
PAD_TOLERANCE_M = 0.22            # how far off the centre the touchdown may be (pad half-size - feet half-diagonal)

_OVERRIDE = {
    "propulsion": "pusher",
    # wing: S = 0.202 m2, AR 6.0, unswept, on top of the pod
    "span": 1100.0, "root_chord": 230.0, "taper": 0.60, "le_sweep_deg": 4.0, "dihedral_deg": 3.0, "washout_deg": 2.0,
    "incidence_deg": 2.0, "thickness": 0.12, "camber": 0.02, "wing_x": 200.0, "wing_z": 30.0,
    # the pod
    "fuselage_length": 560.0, "fuselage_diameter": 80.0, "nose_length": 150.0, "tail_cone": 150.0, "tail_end_diameter": 40.0,
    # H tail on the booms: the stabiliser boom to boom, a fin on each
    "htail_span": 540.0, "htail_root_chord": 100.0, "htail_taper": 1.0, "htail_sweep_deg": 0.0, "htail_z": 18.0,
    "tail_le_x": 705.0,
    "fin_height": 130.0, "fin_root_chord": 110.0, "fin_taper": 0.65, "fin_sweep_deg": 25.0, "tail_thickness": 0.09,
    "prop_diameter": 254.0, "camera_diameter": 40.0, "skid": False, "perch_hook": False,
    "static_margin": 0.12, "cg_x": 0.0,
}
_NEW = [
    Parameter("boom_y", 270.0, "mm", min=150.0, description="each boom's axis from the centre plane"),
    Parameter("boom_z", 8.0, "mm", description="the booms' axis above the pod's axis (under the wing)"),
    Parameter("boom_diameter", 16.0, "mm", min=8.0, description="carbon tube"),
    Parameter("boom_x0", 5.0, "mm", description="the booms' front end from the nose tip"),
    Parameter("lift_prop_diameter", 228.6, "mm", min=120.0, description="each lift rotor (9 inch)"),
    Parameter("lift_x_front", 40.0, "mm", description="the front rotors' axis from the nose tip"),
    Parameter("lift_x_rear", 575.0, "mm", description="the rear rotors' axis from the nose tip"),
    Parameter("lift_z", 34.0, "mm", min=15.0, description="the rotor planes above the booms' axis"),
    Parameter("foot_length", 72.0, "mm", min=30.0, description="the feet below the booms' axis"),
    Parameter("show_lift", True, description="the lift rotors as thin discs (pictures)"),
]

# ---------------------------------------------------------------------------------------------- mass (HAND, marked)
WING_KG_M2, TAIL_KG_M2, POD_KG_M, BOOM_KG_M = 1.55, 0.90, 0.28, 0.045
ITEMS = [  # (item, grams, source)
    ("boom mounts and rotor mounts", 30.0, "HAND (+-10)"),
    ("4 lift motors (2212-class, 300 W)", 4 * 55.0, "HAND (+-20)"),
    ("lift ESC, 4-in-1, 40 A", 40.0, "HAND (+-10)"),
    ("4 lift propellers 9x5", 4 * 9.0, "HAND (+-6)"),
    ("pusher motor (2814-class, 800 W)", 110.0, "HAND (+-20)"),
    ("pusher ESC 60 A", 45.0, "HAND (+-10)"),
    ("pusher propeller 10 in + spinner, braked level", 35.0, "HAND (+-10)"),
    ("5 servos, 12 g (2 ailerons, elevator, 2 rudders)", 60.0, "HAND (+-10)"),
    ("flight controller + GPS/compass (RTK for the pad)", 50.0, "HAND (+-10)"),
    ("vision computer (Jetson Orin Nano class, module + light carrier, heatsink)", 125.0, "HAND (+-25)"),
    ("nose camera + a downward pad camera", 45.0, "HAND (+-10)"),
    ("radio + telemetry + antennas", 25.0, "HAND (+-10)"),
    ("wiring, connectors, power module", 65.0, "HAND (+-20)"),
    ("4 feet, sprung charging contacts, ideal-diode bridge", 45.0, "HAND (+-15)"),
]
PACKS = {  # name -> (grams, Wh nominal, max continuous A, source)
    "Li-ion 4S2P P42A": (round(8 * 70.0 * 1.07), 8 * 3.6 * 4.2, 90.0, "cell x count + 7 % (HAND); 45 A per cell"),
    "Li-ion 4S3P P42A": (round(12 * 70.0 * 1.07), 12 * 3.6 * 4.2, 135.0, "cell x count + 7 % (HAND); 45 A per cell"),
    "4S 5000 mAh LiPo": (500.0, 74.0, 150.0, "HAND: 100 g per Ah at 4S, 30C (+-10 %)"),
}
DEFAULT_PACK = "Li-ion 4S2P P42A"
LIFT_W, PUSHER_W = 300.0, 800.0


class Buteo(a1.Arigo1):
    """BUTEO: a pod, a wing on top, two booms with four lift rotors and an H tail, a pusher between the booms, four feet."""

    parameters = [dataclasses.replace(q, default=_OVERRIDE[q.name]) if q.name in _OVERRIDE else q
                  for q in a1.Arigo1.parameters] + _NEW

    @staticmethod
    def layout(p) -> dict:
        L = a1.Arigo1.layout(p)
        fct = p["fin_root_chord"] * p["fin_taper"]
        S_fin = 0.5 * (p["fin_root_chord"] + fct) * p["fin_height"]
        mac_v = 2 / 3 * (p["fin_root_chord"] ** 2 + p["fin_root_chord"] * fct + fct ** 2) / (p["fin_root_chord"] + fct)
        z_mac = p["fin_height"] / 3 * (p["fin_root_chord"] + 2 * fct) / (p["fin_root_chord"] + fct)
        x_fin = p["tail_le_x"] + p["htail_root_chord"] - p["fin_root_chord"]
        x_ac_v = x_fin + z_mac * math.tan(math.radians(p["fin_sweep_deg"])) + 0.25 * mac_v
        boom_x1 = p["tail_le_x"] + p["htail_root_chord"]
        rl = p["lift_prop_diameter"] / 2
        te_at_boom = p["wing_x"] + p["boom_y"] * math.tan(math.radians(p["le_sweep_deg"])) + p["root_chord"] * (1 - (1 - p["taper"]) * p["boom_y"] / (p["span"] / 2))
        le_at_boom = p["wing_x"] + p["boom_y"] * math.tan(math.radians(p["le_sweep_deg"]))
        L.update({
            "S_v_mm2": 2 * S_fin, "x_fin": x_fin, "x_ac_v": x_ac_v, "z_ac_v": p["boom_z"] + z_mac, "fin_y": p["boom_y"],
            "l_v": x_ac_v - L["x_ac"], "V_v": 2 * S_fin * (x_ac_v - L["x_ac"]) / (L["S_mm2"] * p["span"]),
            "boom_x1": boom_x1,
            "lift": [(p["lift_x_front"], -p["boom_y"]), (p["lift_x_front"], p["boom_y"]), (p["lift_x_rear"], -p["boom_y"]), (p["lift_x_rear"], p["boom_y"])],
            "lift_disc_mm2": 4 * math.pi * rl ** 2,
            "front_rotor_clear_of_wing_le": le_at_boom - (p["lift_x_front"] + rl),
            "rear_rotor_clear_of_wing_te": (p["lift_x_rear"] - rl) - te_at_boom,
            "rear_rotor_clear_of_stabiliser": p["tail_le_x"] - (p["lift_x_rear"] + rl),
            "rotor_clear_of_pod": p["boom_y"] - rl - p["fuselage_diameter"] / 2,
            "rear_rotor_clear_of_pusher_disc": p["boom_y"] - rl - p["prop_diameter"] / 2,
            "pusher_clear_of_stabiliser": p["tail_le_x"] - (p["fuselage_length"] + 18.0),
            "foot_below_pod_belly": (p["foot_length"] - p["boom_z"]) - p["fuselage_diameter"] / 2,
            "feet_square_m": (abs(p["lift_x_rear"] - p["lift_x_front"]) / 1e3, 2 * p["boom_y"] / 1e3),
        })
        return L

    # ------------------------------------------------------------------------------------------------ parts
    def _booms(self, p):
        L = self.layout(p)
        out = None
        for s in (1, -1):
            b = (cq.Workplane("YZ").circle(p["boom_diameter"] / 2).extrude(L["boom_x1"] - p["boom_x0"])
                 .translate((p["boom_x0"], s * p["boom_y"], p["boom_z"])))
            # the pylon from the boom up into the wing
            py = cq.Workplane("XY").box(90.0, 10.0, max(4.0, p["wing_z"] - p["boom_z"])).translate(
                (L["x_ac"], s * p["boom_y"], (p["wing_z"] + p["boom_z"]) / 2))
            b = b.union(py)
            out = b if out is None else out.union(b)
        return out

    def _lift_motors(self, p):
        L = self.layout(p)
        out = None
        for x, y in L["lift"]:
            m = cq.Workplane("XY").circle(14.0).extrude(p["lift_z"] - 4.0).translate((x, y, p["boom_z"]))
            f = (cq.Workplane("XY").circle(5.0).extrude(p["foot_length"]).translate((x, y, p["boom_z"] - p["foot_length"]))
                 .union(cq.Workplane("XY").circle(11.0).extrude(4.0).translate((x, y, p["boom_z"] - p["foot_length"]))))
            s = m.union(f)
            out = s if out is None else out.union(s)
        return out

    def _lift_discs(self, p):
        L = self.layout(p)
        out = None
        for x, y in L["lift"]:
            d = cq.Workplane("XY").circle(p["lift_prop_diameter"] / 2).extrude(1.5).translate((x, y, p["boom_z"] + p["lift_z"]))
            out = d if out is None else out.union(d)
        return out

    def _tail(self, p):
        L = self.layout(p)
        h = self._surface(p["tail_le_x"], p["htail_root_chord"], p["htail_taper"], p["htail_span"], p["htail_sweep_deg"],
                          p["tail_thickness"], "y", z0=p["htail_z"])
        out = h
        for s in (1, -1):
            f = self._surface(L["x_fin"], p["fin_root_chord"], p["fin_taper"], p["fin_height"], p["fin_sweep_deg"],
                              p["tail_thickness"], "z", z0=p["boom_z"]).translate((0, s * p["boom_y"], 0))
            out = out.union(f)
        return out

    def parts(self, p) -> dict:
        out = {"fuselage": self._fuselage(p), "wing": self._wing(p), "tail": self._tail(p), "camera": self._camera(p),
               "motors": self._motors(p).union(self._lift_motors(p)), "booms": self._booms(p), "props": self._props(p)}
        if p["show_lift"]:
            out["lift_rotors"] = self._lift_discs(p)
        return out

    def build(self, p):
        if p["part"] in ("wing", "fuselage"):
            return super().build(p)
        if p["part"] == "tail":
            return self._tail(p)
        ac = self._fuselage(p).union(self._wing(p)).union(self._tail(p)).union(self._camera(p)).union(self._motors(p))
        ac = ac.union(self._booms(p)).union(self._lift_motors(p))
        if p["show_props"]:
            ac = ac.union(self._props(p)).union(self._lift_discs(p))
        return ac


def variant(**over) -> dict:
    return {**_OVERRIDE, **over}


def resolve(q=None) -> dict:
    return Buteo().resolve(**dict(q or variant()))


def budget(p=None, L=None, pack: str | None = None) -> dict:
    p = p or resolve()
    L = L or Buteo.layout(p)
    S_w, S_t = L["S_mm2"] / 1e6, (L["S_h_mm2"] + L["S_v_mm2"]) / 1e6
    boom_m = 2 * (L["boom_x1"] - p["boom_x0"]) / 1e3
    items = [("wing", 1000 * WING_KG_M2 * S_w, f"geometry: {S_w:.3f} m2 x {WING_KG_M2} kg/m2 (HAND +-20 %)"),
             ("tail surfaces", 1000 * TAIL_KG_M2 * S_t, f"geometry: {S_t:.3f} m2 x {TAIL_KG_M2} kg/m2 (HAND +-20 %)"),
             ("pod", 1000 * POD_KG_M * p["fuselage_length"] / 1e3, f"geometry: {p['fuselage_length']:.0f} mm x {POD_KG_M} kg/m (HAND +-20 %)"),
             ("2 booms (16 mm carbon tube)", 1000 * BOOM_KG_M * boom_m, f"geometry: {boom_m:.2f} m x {BOOM_KG_M} kg/m (HAND +-20 %)")]
    items += list(ITEMS)
    items.append(("growth margin (10 % of the items above)", 0.10 * sum(x[1] for x in items), "HAND: a preliminary budget's usual allowance"))
    pk = pack or DEFAULT_PACK
    g, wh, amps, src = PACKS[pk]
    empty = sum(x[1] for x in items)
    return {"items": [{"item": n, "g": round(v, 1), "source": s} for n, v, s in items], "empty_g": round(empty, 1),
            "pack": {"name": pk, "g": g, "Wh_nominal": round(wh, 1), "max_A": amps, "source": src},
            "total_g": round(empty + g, 1), "pack_share": round(g / (empty + g), 3)}


def checks(q=None) -> dict:
    p = resolve(q)
    L = Buteo.layout(p)
    keys = ("S_mm2", "AR", "mac", "x_ac", "V_h", "V_v", "front_rotor_clear_of_wing_le", "rear_rotor_clear_of_wing_te",
            "rear_rotor_clear_of_stabiliser", "rotor_clear_of_pod", "rear_rotor_clear_of_pusher_disc", "pusher_clear_of_stabiliser",
            "foot_below_pod_belly")
    out = {k: round(float(L[k]), 3 if k in ("AR", "V_h", "V_v") else 1) for k in keys}
    out["feet_square_m"] = L["feet_square_m"]
    return out
