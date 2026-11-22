"""ASTOR-T — ASTOR-V redesigned to leave a tube on a spring (2026-11-22).

**Why a redesign and not a launcher.** ASTOR-V stowed is 931 x 309 x 250 mm: the folded tail fan is 309 mm wide (a
180 mm root chord swings out when the panel folds) and the fixed fin hangs 160 mm under the tail. It would need a
350 mm tube. Everything that sets the tube's bore is changed here; the body, the motor, the folding propeller, the
net tube and the cameras are ASTOR-V's.

- **Wing**: two narrow panels (165 -> 105 mm chord, 600 mm long) on ONE pivot axis on the carriage, one above the
  other (ASTOR-V's pivots were 52 mm apart, its root chord 270 mm). Stowed they lie along the body's back, one on
  the other, 165 mm wide. The carriage, the sweep range (12 / 60 deg in flight) and the ailerons stay.
- **Tail**: two slender blades (95 -> 60 mm, 250 mm) in place of the broad fan panels: folded they trail along the
  tail cone inside the bore. Spread they are the stabiliser and, moved apart, help the ailerons.
- **Fins**: two ventral fins on hinges along the belly, 84 mm high. Folded they lie flat under the tail cone, one to
  each side; springs swing them down to 25 deg from the vertical (an inverted V under the tail) as the tail leaves
  the tube. Together they have the side area of ASTOR-V's fin and half its depth under the body.
- **Launch keys**: the two pairs of lugs on the fuselage's sides ride in the shuttle's slots; the shuttle carries the
  aircraft by them (the tail, with the net tube's mouth and the folded blades, is not pushed on).

**The launcher**: a tube, a shuttle on two rails inside it, a compression spring behind the shuttle, a latch.
``launcher()`` sizes the spring for an exit speed (energy, peak acceleration, a steel spring's mass);
``astor_tube_launch`` flies the first seconds after the muzzle.

Mechanism masses and the spring's energy per kilogram are HAND estimates, marked. Frame: x aft from the nose tip, y
right, z up; mm.
"""
from __future__ import annotations

import dataclasses
import math

import cadquery as cq
import numpy as np

import astor as A
import astor_v as V
from vegeta.dedalus import Parameter

FIN_SPLAY_DEG = 25.0
_DEFAULTS = {"pivot_y": 0.0, "panel_span": 600.0, "panel_root_chord": 165.0, "panel_tip_chord": 105.0, "panel_thickness": 0.09,
             "wing_plane_z": 46.0, "panel_stack": 16.0,
             "panel_pivot_chord": 0.5, "pivot_x": 335.0,          # the pivot at mid chord: stowed, each panel lies centred on the body
             # the lattice's balance (astor_v_model on these panels): 10 / 11 / 12 % of the reference chord at 12 / 35 / 60 deg
             # with this travel and the tail blades 30 % spread in the dash (ASTOR-V's CFD found 8 % more than its
             # lattice at 60 deg: the CFD of ASTOR-T decides)
             "slide_35": 100.0, "slide_60": 165.0, "dash_tail_fan": 0.3,
             "tail_anhedral_deg": 30.0,                           # the two fins alone left it without directional stability (Cnb -0.003)
             "tail_panel_span": 250.0, "tail_panel_root": 95.0, "tail_panel_tip": 60.0, "tail_pivot_y": 18.0,
             "fin_height": 84.0, "fin_root_chord": 230.0, "fin_taper": 0.45, "fin_sweep_deg": 40.0}
_MIN = {"pivot_y": 0.0}
_NEW = [
    Parameter("fin_fold", 0.0, "", min=0.0, max=1.0, description="the two ventral fins: 0 down (flight), 1 flat under the tail (in the tube)"),
    Parameter("fin_splay_deg", FIN_SPLAY_DEG, "deg", min=0.0, max=45.0, description="each fin from the vertical when down"),
    Parameter("fin_hinge_y", 9.0, "mm", min=3.0, description="each fin's hinge line from the centre plane"),
]

EXTRA_ITEMS = list(V.EXTRA_ITEMS) + [
    ("second ventral fin", 0.0, "geometry (set in budget)"),
    ("2 fin hinges, springs and down-locks", 18.0, "HAND (+-6)"),
    ("wing and tail opening springs, stowed latches", 24.0, "HAND (+-8)"),
    ("launch keys reinforced (4 lugs, doublers)", 16.0, "HAND (+-6)"),
]
# stowed, each panel's mid-chord line lies along the body: its sweep is 90 deg, so the leading edge's is a little more
STOWED_DEG = round(V.SPREAD_DEG + 90.0 - math.degrees(math.atan(math.tan(math.radians(V.SPREAD_DEG)) + 0.5 * (_DEFAULTS["panel_tip_chord"] - _DEFAULTS["panel_root_chord"]) / _DEFAULTS["panel_span"])), 1)
STOWED = dict(wing_sweep_deg=STOWED_DEG, tail_fan=0.0, prop_fold_deg=82.0, fin_fold=1.0)
CLEARANCE_MM = 4.0                 # between the stowed aircraft and the bore, all round


class AstorT(V.AstorV):
    """ASTOR-T: ASTOR-V's body; narrow wing panels on one pivot; tail blades; two folding ventral fins."""

    parameters = [dataclasses.replace(q, default=_DEFAULTS.get(q.name, q.default), min=_MIN.get(q.name, q.min))
                  for q in V.AstorV.parameters] + _NEW

    def _one_fin(self, p, side):
        """One fin: built pointing down from its hinge line, then swung about the hinge (a line along x)."""
        fin = self._surface(p["fuselage_length"] - 24.0 - p["fin_root_chord"], p["fin_root_chord"], p["fin_taper"], p["fin_height"],
                            p["fin_sweep_deg"], p["tail_thickness"], "z", z0=0.0).mirror("XY")
        ang = p["fin_splay_deg"] + p["fin_fold"] * (90.0 - p["fin_splay_deg"])
        zh = -(p["tail_end_diameter"] / 2 + 2.0)                          # the hinge line: just under the tail cone's end
        return fin.rotate((0, 0, 0), (1, 0, 0), side * ang).translate((0, side * p["fin_hinge_y"], zh))

    def _fin(self, p):
        return self._one_fin(p, 1).union(self._one_fin(p, -1))


def variant(**over) -> dict:
    q = {**V._OVERRIDE, **_DEFAULTS}
    q["tail_le_x"] = q["fuselage_length"] - 10.0 - q["htail_root_chord"]
    return {**q, **over}


def resolve(q=None) -> dict:
    return AstorT().resolve(**dict(q or variant()))


def fin_area_m2(p) -> float:
    return 0.5 * p["fin_root_chord"] * (1 + p["fin_taper"]) * p["fin_height"] / 1e6


def fin_equivalent(p) -> dict:
    """The single upright fin that gives the two splayed ones' side force (their side area x cos^2 of the splay), at
    their own height: its root chord, for the hand formulas."""
    k = 2.0 * math.cos(math.radians(p["fin_splay_deg"])) ** 2
    return {"fin_height": p["fin_height"] * math.cos(math.radians(p["fin_splay_deg"])),
            "fin_root_chord": p["fin_root_chord"] * k / math.cos(math.radians(p["fin_splay_deg"])), "side_area_factor": round(k, 2)}


def budget(p=None, pack=None) -> dict:
    p = p or resolve()
    extra = [(n, (1000 * A.TAIL_KG_M2 * fin_area_m2(p) if n == "second ventral fin" else g),
              (f"geometry: {fin_area_m2(p):.4f} m2 x {A.TAIL_KG_M2} kg/m2 (HAND +-20 %)" if n == "second ventral fin" else src))
             for n, g, src in EXTRA_ITEMS]
    return V.budget(p, pack, extra=extra)


# ------------------------------------------------------------------------------------------------ the bore it needs
def _yz(shape, tol=1.5):
    v, _ = (shape.val() if hasattr(shape, "val") else shape).tessellate(tol)
    return np.array([[q.x, q.y, q.z] for q in v])


def bore(p=None) -> dict:
    """The smallest circle round the stowed aircraft's cross-section (its centre anywhere on the centre plane), from
    the solid's own surface points; the tube's bore is that plus the clearance."""
    q = variant(**STOWED) if p is None else p
    P = _yz(AstorT().generate(**q).shape)
    best = None
    for zc in np.arange(-40.0, 60.0, 0.5):
        r = float(np.max(np.hypot(P[:, 1], P[:, 2] - zc)))
        if best is None or r < best[0]:
            best = (r, float(zc))
    r, zc = best
    i = int(np.argmax(np.hypot(P[:, 1], P[:, 2] - zc)))
    return {"radius_mm": round(r, 1), "centre_above_axis_mm": round(zc, 1), "bore_mm": round(2 * (r + CLEARANCE_MM), 0),
            "length_mm": round(float(P[:, 0].max() - P[:, 0].min()), 0), "x_min_mm": round(float(P[:, 0].min()), 0),
            "width_mm": round(float(P[:, 1].max() - P[:, 1].min()), 0), "height_mm": round(float(P[:, 2].max() - P[:, 2].min()), 0),
            "what_touches": {"x": round(float(P[i, 0])), "y": round(float(P[i, 1])), "z": round(float(P[i, 2]))}}


# ------------------------------------------------------------------------------------------------ the launcher
SPRING_J_PER_KG = 90.0             # HAND: a steel compression spring, music wire worked to ~700 MPa and half of the ideal
                                   # tau^2 / (4 G rho) = 197 J/kg used over its life (+-30)
SHUTTLE_KG = 0.60                  # HAND: the cradle: a ring at the keys, two arms, a sleeve back past the tail to the spring (+-0.15)
END_FORCE_RATIO = 0.4              # the spring still pushes with 40 % of its first force when the cradle stops
KEYS_X = 500.0                     # the rear launch keys [mm from the nose tip]: where the cradle holds the aircraft
V_EXIT, STROKE_M, ELEVATION_DEG = 18.0, 0.9, 40.0     # the design point (astor_tube_launch: flies away with an 8 m/s tailwind or a motor 1 s late)


def launcher(v_exit=V_EXIT, stroke_m=STROKE_M, mass_kg=None, elevation_deg=ELEVATION_DEG, b=None) -> dict:
    """The spring for an exit speed: the work it must do on the aircraft and the cradle along the stroke (their
    weight's component along the tube included), its first and last force, the peak acceleration, a steel spring's
    mass; and the tube: the cradle must stop inside it, so the muzzle is the stroke ahead of the keys."""
    m = (mass_kg if mass_kg is not None else budget()["total_g"] / 1000.0)
    mm = m + SHUTTLE_KG
    g_along = 9.80665 * math.sin(math.radians(elevation_deg))
    work = 0.5 * mm * v_exit ** 2 + mm * g_along * stroke_m
    f_mean = work / stroke_m
    f0 = 2 * f_mean / (1 + END_FORCE_RATIO)
    f1 = END_FORCE_RATIO * f0
    rate = (f0 - f1) / stroke_m
    stored = 0.5 * f0 ** 2 / rate                                     # in the spring at the latch (it keeps f1 at the stop)
    b = b or bore()
    closed = 0.35 * stroke_m / (1 - END_FORCE_RATIO)                  # HAND: the cocked spring's length, ~35 % of its whole travel
    x_muzzle = min(b["x_min_mm"] - 20.0, KEYS_X - 30.0 - stroke_m * 1000.0 - 40.0)
    x_tail = b["x_min_mm"] + b["length_mm"]
    x_breech = x_tail + 20.0 + closed * 1000.0 + 20.0
    tube_len = (x_breech + 12.0 - x_muzzle) / 1000.0
    return {"exit_speed_m_s": v_exit, "stroke_m": stroke_m, "elevation_deg": elevation_deg, "aircraft_kg": round(m, 3), "cradle_kg": SHUTTLE_KG,
            "work_J": round(work, 0), "force_first_N": round(f0, 0), "force_last_N": round(f1, 0), "rate_N_per_mm": round(rate / 1000.0, 3),
            "peak_acceleration_g": round(f0 / mm / 9.80665 - math.sin(math.radians(elevation_deg)), 1),
            "time_in_tube_s": round(2 * stroke_m / v_exit * 1.05, 3),
            "energy_stored_J": round(stored, 0), "steel_spring_kg": round(stored / SPRING_J_PER_KG, 1),
            "spring_note": f"{SPRING_J_PER_KG:.0f} J/kg (HAND +-30); cocking it needs {f0:.0f} N: a winch or a lever, not a hand",
            "tube": {"bore_mm": b["bore_mm"], "length_m": round(tube_len, 2), "wall_mm": 3.0, "muzzle_x_mm": round(x_muzzle), "breech_x_mm": round(x_breech),
                     "spring_cocked_mm": round(closed * 1000.0), "muzzle_ahead_of_the_nose_mm": round(b["x_min_mm"] - x_muzzle),
                     "mass_kg": round(math.pi * (b["bore_mm"] / 1000.0 + 0.003) * 0.003 * tube_len * 1600.0, 1),
                     "note": "glass/carbon tube 1600 kg/m3 (HAND)"},
            "system_kg": round(stored / SPRING_J_PER_KG + SHUTTLE_KG + math.pi * (b["bore_mm"] / 1000.0 + 0.003) * 0.003 * tube_len * 1600.0 + 1.5, 1),
            "system_note": "spring + cradle + tube + 1.5 kg of latch, breech and winch (HAND); without the stand"}


def launcher_parts(v_exit=V_EXIT, stroke_m=STROKE_M) -> dict:
    """The tube (its lower half: the picture shows what is in it), the cradle, the cocked spring and the breech."""
    b = bore()
    L = launcher(v_exit, stroke_m, b=b)
    Rb, zc = b["bore_mm"] / 2, b["centre_above_axis_mm"]
    x0, x1 = L["tube"]["muzzle_x_mm"], L["tube"]["breech_x_mm"]
    x_tail = b["x_min_mm"] + b["length_mm"]
    tube = cq.Workplane("YZ").circle(Rb + 3.0).circle(Rb).extrude(x1 + 12.0 - x0).translate((x0, 0, zc))
    half = cq.Workplane("XY").box(x1 - x0 + 40.0, 2 * Rb + 20.0, Rb + 10.0).translate(((x0 + x1) / 2, 0, zc + (Rb + 10.0) / 2))
    tube = tube.cut(half)
    Rf = A._OVERRIDE["fuselage_diameter"] / 2
    cradle = cq.Workplane("YZ").circle(Rb - 1.5).circle(Rb - 8.0).extrude(40.0).translate((KEYS_X - 20.0, 0, zc)).cut(half)   # the ring at the rear keys
    for s in (1, -1):
        cradle = cradle.union(cq.Workplane("XY").box(310.0, 8.0, 20.0).translate((365.0, s * (Rf + 10.0), -8.0)))            # the arms with the key slots
        cradle = cradle.union(cq.Workplane("XY").box(x_tail + 20.0 - KEYS_X, 8.0, 14.0).translate(((x_tail + 20.0 + KEYS_X) / 2, s * (Rb - 10.0), zc - 30.0)))   # back past the tail
    cradle = cradle.union(cq.Workplane("YZ").circle(Rb - 1.5).circle(Rb - 22.0).extrude(10.0).translate((x_tail + 20.0, 0, zc)).cut(half))  # the spring's seat
    spring, n = None, 12
    xs0, xs1 = x_tail + 32.0, x1 - 10.0
    for k in range(n):
        ring = cq.Workplane("YZ").circle(Rb - 6.0).circle(Rb - 18.0).extrude(8.0).translate((xs0 + (xs1 - xs0) * k / (n - 1), 0, zc))
        spring = ring if spring is None else spring.union(ring)
    breech = cq.Workplane("YZ").circle(Rb + 3.0).extrude(12.0).translate((x1, 0, zc))
    return {"tube": tube, "cradle": cradle, "spring": spring, "breech": breech, "numbers": L, "bore": b}


def checks() -> dict:
    p = resolve()
    mb = budget(p)
    b = bore()
    out = {"wing": [V.wing_numbers(p, s) for s in (V.SPREAD_DEG, 35.0, V.DASH_DEG)], "stowed": b,
           "tube_bore_mm": b["bore_mm"], "clearance_mm": CLEARANCE_MM,
           "astor_v_stowed_for_comparison_mm": {"length": 931, "width": 309, "height": 250},
           "fins": {"each_height_mm": p["fin_height"], "splay_deg": p["fin_splay_deg"], "side_area_of_both_m2": round(2 * fin_area_m2(p) * math.cos(math.radians(p["fin_splay_deg"])) ** 2, 4),
                    "astor_v_fin_m2": round(0.5 * 230.0 * 1.45 * 160.0 / 1e6, 4), "depth_under_axis_mm": round(p["tail_end_diameter"] / 2 + 2 + p["fin_height"] * math.cos(math.radians(p["fin_splay_deg"])))},
           "mass": {k: mb[k] for k in ("empty_g", "total_g", "without_nets_g")},
           "balance_by_mass": V.balance(p, mb=mb),
           "launcher": launcher(b=b)}
    return out
