"""ASTOR — the goshawk: a propeller dart that reaches a fast threat first (RAPTORS product line; the concept in
``reports/arigo_catalog/PRODUCT_SPACE.md``, accepted 2026-11-20).

**The job.** It waits in a launcher at a site or on a vehicle. On a cue it is launched from a rail, dashes at 60-70
m/s to meet a fixed-wing drone flying 30-50 m/s some kilometres out, and comes down under a parachute; a fresh one
is loaded in minutes. The requirement that sizes it: **catch a 50 m/s target from behind** (top speed above
1.15 x 50 = 57.5 m/s, with minutes of full power, not seconds).

**The layout, and why.**

- A **short, swept, thin wing**: 0.8 m span, aspect ratio 4, 28 deg of leading-edge sweep, a 9 % section with little
  camber. At 65 m/s the lift coefficient is ~0.05: the wing is there for the launch, the turn and the stall, so it is
  kept small (drag at the dash is almost all zero-lift drag) and stiff (dynamic pressure ~2.6 kPa at 65 m/s, 1.5 x
  ARIGO-1H's top-speed load). Mid-mounted: no pitch change with speed from a wing above or below the thrust line.
- A **slim fuselage** (70 mm) on the thrust line: the camera in the nose (nothing turns ahead of it), the pack and the
  avionics, a parachute bay on the back between the wing and the tail, the motor at the end.
- A **1.4 kW pusher on a folding 9 x 11 propeller**: a propeller is ~70 % efficient at 65 m/s where a 70 mm ducted fan
  is ~55 % (run 1: an 800 mm ARIGO-1H with 1.2 kW matched ARIGO-20's speed with twice the full-power time). 9 inches
  holds the same thrust at the dash as 10 and leaves room for the wing-fold growth version (a tube).
- A **6S pack**: 1.4 kW from 3S is ~125 A, at the limit of a 3S3P Li-ion pack; from 6S it is ~65 A.
- A **cruciform tail**, the stabiliser raised 40 mm out of the mid wing's wake, swept; the fin swept, the rudder for
  the turn's coordination. Sized for a pitch-agile aircraft: tail volume 0.46, static margin ~10 % MAC. No dihedral
  (the sweep gives the roll-with-sideslip). The Dutch roll is lightly damped (zeta ~0.07 on the lattice model, the
  family's trait; a larger fin moves it little): ASTOR relies on the autopilot's yaw damper.
- **Rail shoes** under the belly and a **parachute hatch** on the back. No skid: it does not belly-land.

Built on ARIGO-1's parametric airframe (its wing, fuselage and tail generators, imported, not changed) with ASTOR's
own sizes, parts, mass budget and packs. Frame: x aft from the nose tip, y right, z up; mm.

Growth version (not built here): the same aircraft with the wing and fins folding along the body for a sealed tube.
"""
from __future__ import annotations

import dataclasses

import cadquery as cq

import arigo1 as a1
from vegeta.dedalus import Parameter

TARGET_M_S = 50.0                 # the requirement: catch this from behind
CHASE_MARGIN = 1.15               # top speed / target speed for a tail chase

_OVERRIDE = {
    "propulsion": "pusher",
    # wing: S = 0.160 m2 (AR 4.0), swept and thin
    "span": 800.0, "root_chord": 276.0, "taper": 0.45, "le_sweep_deg": 28.0, "dihedral_deg": 0.0, "washout_deg": 2.5,
    "incidence_deg": 0.5, "thickness": 0.09, "camber": 0.012, "tip_round": 0.10, "wing_x": 235.0, "wing_z": 0.0,
    "aileron_span": 0.55, "aileron_chord": 0.28,
    # fuselage: slim, the thrust line through it
    "fuselage_length": 790.0, "fuselage_diameter": 70.0, "nose_length": 170.0, "tail_cone": 280.0, "tail_end_diameter": 40.0,
    # cruciform tail
    "htail_span": 380.0, "htail_root_chord": 165.0, "htail_taper": 0.60, "htail_sweep_deg": 25.0, "htail_z": 40.0,
    "fin_height": 175.0, "fin_root_chord": 190.0, "fin_taper": 0.50, "fin_sweep_deg": 35.0, "tail_thickness": 0.08,
    "elevator_chord": 0.38, "rudder_chord": 0.35,
    # propulsion: 9 x 11 folding pusher, a 1.4 kW motor
    "prop_diameter": 228.6, "motor_diameter": 40.0, "motor_length": 42.0,
    "camera_diameter": 36.0, "skid": False, "perch_hook": False,
    "static_margin": 0.10, "cg_x": 0.0,          # the CG by the lattice's static margin until the CFD fixes it
}
_NEW = [
    Parameter("parachute_x", 450.0, "mm", min=200.0, description="the parachute hatch's front edge from the nose tip"),
    Parameter("parachute_length", 95.0, "mm", min=40.0, description="the parachute bay along the fuselage (a ~0.8 m2 canopy, packed)"),
    Parameter("rail_shoe_x1", 250.0, "mm", min=80.0, description="front rail shoe from the nose tip"),
    Parameter("rail_shoe_x2", 470.0, "mm", min=150.0, description="rear rail shoe from the nose tip"),
]

# ---------------------------------------------------------------------------------------------- mass (HAND, marked)
WING_KG_M2 = 1.90        # +-0.35: a stiffer wing than ARIGO-1's 1.55 (carbon spar caps for ~2.6 kPa), servos' bays, skin
TAIL_KG_M2 = 1.05        # +-0.2
FUSE_KG_M = 0.30         # +-0.06 per metre at 70 mm
ITEMS = [  # (item, grams, source)
    ("motor (3536-class outrunner, 1.4 kW peak)", 175.0, "HAND (+-25)"),
    ("ESC 80 A, 6S", 65.0, "HAND (+-10)"),
    ("folding propeller 9x11 + spinner", 30.0, "HAND (+-8)"),
    ("4 servos, metal gear, 15 g", 60.0, "HAND (+-10)"),
    ("flight controller + GPS/compass", 40.0, "HAND (+-10)"),
    ("vision computer (Jetson Orin Nano class, module + light carrier, heatsink)", 125.0, "HAND (+-25)"),
    ("nose camera (global shutter, M12 lens)", 30.0, "HAND (+-10)"),
    ("radio + telemetry + antennas", 25.0, "HAND (+-10)"),
    ("wiring, connectors, power module (6S)", 60.0, "HAND (+-20)"),
    ("parachute, canopy bay, release servo", 85.0, "HAND (+-20)"),
    ("rail shoes", 12.0, "HAND (+-4)"),
]
# name -> (grams, Wh nominal, max continuous A, source)
PACKS = {
    "6S 3000 mAh LiPo": (450.0, 66.6, 90.0, "HAND: 150 g per Ah at 6S, 30C (+-10 %)"),
    "6S 4000 mAh LiPo": (600.0, 88.8, 120.0, "HAND: 150 g per Ah at 6S, 30C (+-10 %)"),
    "6S 5000 mAh LiPo": (750.0, 111.0, 150.0, "HAND: 150 g per Ah at 6S, 30C (+-10 %)"),
    "Li-ion 6S2P P42A": (round(12 * 70.0 * 1.07), 12 * 3.6 * 4.2, 90.0, "cell x count + 7 % (HAND); 45 A per cell"),
}
DEFAULT_PACK = "6S 4000 mAh LiPo"
MOTOR_W = 1400.0


class Astor(a1.Arigo1):
    """ASTOR: a slim fuselage, a short swept mid wing, a cruciform tail, a folding pusher; parachute hatch, rail shoes."""

    parameters = [dataclasses.replace(q, default=_OVERRIDE[q.name]) if q.name in _OVERRIDE else q
                  for q in a1.Arigo1.parameters] + _NEW

    def _parachute(self, p):
        R = p["fuselage_diameter"] / 2
        hatch = cq.Workplane("XY").box(p["parachute_length"], 0.62 * p["fuselage_diameter"], 3.0)
        return hatch.translate((p["parachute_x"] + p["parachute_length"] / 2, 0, R + 0.5))

    def _rail(self, p):
        R = p["fuselage_diameter"] / 2
        out = None
        for x in (p["rail_shoe_x1"], p["rail_shoe_x2"]):
            stem = cq.Workplane("XY").box(16.0, 6.0, 9.0).translate((x, 0, -R - 3.5))
            foot = cq.Workplane("XY").box(16.0, 14.0, 3.0).translate((x, 0, -R - 9.0))
            s = stem.union(foot)
            out = s if out is None else out.union(s)
        return out

    def parts(self, p) -> dict:
        out = {"fuselage": self._fuselage(p), "wing": self._wing(p), "tail": self._tail(p), "camera": self._camera(p),
               "motors": self._motors(p), "props": self._props(p), "parachute": self._parachute(p), "rail": self._rail(p)}
        return out

    def build(self, p):
        if p["part"] in ("wing", "tail", "fuselage"):
            return super().build(p)
        ac = self._fuselage(p).union(self._wing(p)).union(self._tail(p)).union(self._camera(p)).union(self._motors(p))
        ac = ac.union(self._parachute(p)).union(self._rail(p))
        if p["show_props"]:
            ac = ac.union(self._props(p))
        return ac


def variant(**over) -> dict:
    q = dict(_OVERRIDE)
    q["tail_le_x"] = q["fuselage_length"] - 10.0 - q["htail_root_chord"]
    return {**q, **over}


def resolve(q=None) -> dict:
    return Astor().resolve(**dict(q or variant()))


def budget(p=None, L=None, pack: str | None = None) -> dict:
    """ASTOR's mass budget: geometry x densities, the items, a 10 % growth margin, the pack."""
    p = p or resolve()
    L = L or Astor.layout(p)
    S_w, S_t = L["S_mm2"] / 1e6, (L["S_h_mm2"] + L["S_v_mm2"]) / 1e6
    items = [("wing", 1000 * WING_KG_M2 * S_w, f"geometry: {S_w:.3f} m2 x {WING_KG_M2} kg/m2 (HAND +-20 %)"),
             ("tail surfaces", 1000 * TAIL_KG_M2 * S_t, f"geometry: {S_t:.3f} m2 x {TAIL_KG_M2} kg/m2 (HAND +-20 %)"),
             ("fuselage", 1000 * FUSE_KG_M * p["fuselage_length"] / 1000, f"geometry: {p['fuselage_length']:.0f} mm x {FUSE_KG_M} kg/m (HAND +-20 %)")]
    items += list(ITEMS)
    items.append(("growth margin (10 % of the items above)", 0.10 * sum(x[1] for x in items), "HAND: a preliminary budget's usual allowance"))
    pk = pack or DEFAULT_PACK
    g, wh, amps, src = PACKS[pk]
    empty = sum(x[1] for x in items)
    return {"items": [{"item": n, "g": round(v, 1), "source": s} for n, v, s in items], "empty_g": round(empty, 1),
            "pack": {"name": pk, "g": g, "Wh_nominal": round(wh, 1), "max_A": amps, "source": src},
            "total_g": round(empty + g, 1), "pack_share": round(g / (empty + g), 3)}


def checks(q=None) -> dict:
    """The layout's numbers: areas, aspect ratio, tail volumes, where things sit."""
    p = resolve(q)
    L = Astor.layout(p)
    out = {k: round(float(L[k]), 3 if k in ("AR", "V_h", "V_v") else 1) for k in ("S_mm2", "AR", "mac", "x_ac", "V_h", "V_v", "l_h")}
    out["parachute_clear_of_wing_te"] = round(p["parachute_x"] - (p["wing_x"] + p["root_chord"]), 1)
    out["parachute_clear_of_tail_le"] = round(p["tail_le_x"] - (p["parachute_x"] + p["parachute_length"]), 1)
    out["prop_tip_below_axis_mm"] = round(p["prop_diameter"] / 2, 1)
    return out
