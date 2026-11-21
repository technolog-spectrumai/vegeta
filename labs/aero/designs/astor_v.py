"""ASTOR-V — the goshawk's ASTOR: variable-sweep wings, a fanning tail, and nets it drops in the bandit's path
(RAPTORS product line; the direction agreed 2026-11-20 after the bird study and ``astor_hawk``'s experiments).

**How it hunts.** Launched from a rail (or thrown from a perch: spread, it stalls near 12.5 m/s), it tucks its wings
and dashes; it overtakes the bandit, flies 20-40 m ahead of it and 2-3 m above its track, and pushes a net out of its tail. The net stops in the air and sinks; the bandit
flies into it at its own speed. ASTOR-V never touches the bandit, carries two nets, and comes home.

**The layout, and why.**

- **Two wing panels on pivots** on a carriage that slides along the fuselage's back (forward as the wing sweeps, so
  the lift stays over the CG: ``SLIDE``), one panel a little above the other, so
  they cross like scissors. ``wing_sweep_deg``: 12 (spread: launch, turns, braking, landing), ~60 (the dash: the
  panels' inner parts lie over the fuselage, less wing in the air, half the gust load), 98 (stowed: the panels along
  the body, for a canister). Sweeping both pitches the aircraft (the lift moves aft); sweeping them differently
  rolls it — the bird's controls. The panels' section is symmetric and thin (8 %).
- **A fanning tail**: two panels on pivots at the tail's root. Folded (85 deg) they are a narrow dart's tail;
  spread (35 deg) a broad delta that gives pitch authority at low speed and, raised, a brake (``astor_hawk``:
  65 -> 20 m/s in ~120 m instead of ~780 m). A low fixed fin under the tail keeps the nose straight (above, the tucked panels cross).
- **A tractor propeller and a tail net tube**: the 1.4 kW motor and its folding 9 x 11 propeller are on the nose, so
  nothing turns behind the tail; two net cartridges lie in tandem in a tube in the tail cone and are pushed straight
  aft, at full power. (The first ASTOR-V was a pusher with a belly bay: the motor had to be cut and the blades
  folded for each drop.)
- **Two cameras**: the chase camera low in the nose looks through the propeller disc (ARIGO-0's arrangement: ~9 % of
  a frame blocked, ~2 % after two frames); the drop camera under the belly looks down and back, where the bandit is
  when the net leaves.
- The body, the 6S pack and the parachute hatch are ASTOR's (``astor``); rail lugs on the fuselage's sides.

**What is and is not designed here.** This file is the geometry (CAD), the mass budget and the geometric checks
(spans, the stowed envelope, the net's path past the propeller). The flight model for a wing that changes sweep
(lift, drag and balance over the sweep range; sweep as a control) and the net-drop simulation are the next steps;
``astor_hawk`` holds the first experiments. Mechanism masses are HAND estimates.
Frame: x aft from the nose tip, y right, z up; mm.
"""
from __future__ import annotations

import dataclasses
import math

import cadquery as cq

import arigo1 as a1
import astor as A
from vegeta.dedalus import Parameter

SPREAD_DEG, DASH_DEG, STOWED_DEG = 12.0, 60.0, 98.0

_OVERRIDE = {**A._OVERRIDE, "propulsion": "tractor_lowcam",        # the motor and the folding propeller on the nose
             "fuselage_length": 790.0, "tail_cone": 300.0, "tail_end_diameter": 58.0,   # the tail ends in the net tube's mouth
             "lowcam_drop": 50.0, "camera_diameter": 32.0,
             # a bigger fin than the pusher's: a propeller ahead of the CG pulls the nose off line
             "fin_height": 115.0, "fin_root_chord": 195.0, "fin_taper": 0.45, "fin_sweep_deg": 45.0}
_NEW = list(A._NEW) + [
    Parameter("wing_sweep_deg", SPREAD_DEG, "deg", min=5.0, max=105.0, description="the wing panels' leading-edge sweep: 12 spread, 60 dash, 98 stowed (along the body)"),
    Parameter("wing_sweep_left_deg", -1.0, "deg", min=-1.0, max=105.0, description="the left panel's sweep if different (roll control); -1: as the right"),
    Parameter("pivot_x", 320.0, "mm", min=150.0, description="the wing pivots from the nose tip, wing spread"),
    Parameter("pivot_slides", True, description="the pivots' carriage slides forward with the sweep (SLIDE); False: fixed pivots"),
    Parameter("pivot_y", 26.0, "mm", min=10.0, description="each pivot from the centre plane"),
    Parameter("panel_span", 520.0, "mm", min=150.0, description="each wing panel along its own span, from its pivot"),
    Parameter("panel_root_chord", 270.0, "mm", min=60.0),
    Parameter("panel_tip_chord", 120.0, "mm", min=30.0),
    Parameter("panel_thickness", 0.08, "", min=0.05, max=0.14),
    Parameter("panel_pivot_chord", 0.30, "", min=0.1, max=0.6, description="the pivot's place along the root chord"),
    Parameter("panel_stack", 9.0, "mm", min=4.0, description="the right panel above, the left below the wing plane (they cross)"),
    Parameter("wing_plane_z", 44.0, "mm", description="the wing plane above the fuselage axis (on the shoulder block)"),
    Parameter("tail_fan", 1.0, "", min=0.0, max=1.0, description="the tail: 0 folded (85 deg), 1 spread (35 deg)"),
    Parameter("tail_panel_span", 200.0, "mm", min=80.0), Parameter("tail_panel_root", 180.0, "mm", min=60.0),
    Parameter("tail_panel_tip", 60.0, "mm", min=20.0),
    Parameter("tail_pivot_x", 655.0, "mm", min=300.0), Parameter("tail_pivot_y", 14.0, "mm", min=5.0),
    Parameter("net_count", 2, "", min=0, max=2, description="net cartridges in the tail tube, in tandem"),
    Parameter("prop_fold_deg", -1.0, "deg", min=-1.0, max=88.0, description="draw the folding propeller (astor_prop): 0 open ... 82 folded; -1: not drawn"),
    Parameter("nets_station_mm", 0.0, "mm", min=0.0, description="the nets' centre for the balance; 0: in the tail tube"),
    Parameter("belly_camera_x", 520.0, "mm", min=200.0, description="the drop camera (looks down and back) under the belly"),
    Parameter("net_diameter", 44.0, "mm", min=25.0), Parameter("net_length", 150.0, "mm", min=60.0),
    Parameter("net_bay_x", 300.0, "mm", min=150.0, description="the belly bay's front from the nose tip"),
]

# ---------------------------------------------------------------------------------------------- mass (HAND, marked)
EXTRA_ITEMS = [
    ("2 wing pivots (bearings, pins) on a sliding carriage, its rail and lead screw", 130.0, "HAND (+-40)"),
    ("wing sweep actuators (2 servos + linkages: asymmetric sweep)", 60.0, "HAND (+-20)"),
    ("tail fan pivots and actuator", 40.0, "HAND (+-15)"),
    ("net tube in the tail cone, ejector spring and latches", 40.0, "HAND (+-15)"),
    ("belly drop camera", 20.0, "HAND (+-8)"),
    ("nose camera blister", 15.0, "HAND (+-5)"),
]
NET_G = 110.0                      # HAND: one net cartridge: a 2 m mesh, corner weights, its sleeve (+-30)


# The pivots ride on a carriage that slides forward as the wing sweeps (the Bell X-5's answer): with fixed pivots the
# neutral point runs 116 mm aft between 12 and 60 deg (astor_v_model: the tail would carry a download equal to the
# weight at the dash); with this travel the static margin stays 8-16 % MAC. (sweep deg, carriage travel forward mm)
SLIDE = ((12.0, 0.0), (35.0, 61.0), (60.0, 117.0), (98.0, 117.0))


def pivot_x(p, sweep=None) -> float:
    """The pivots' station at a sweep: ``pivot_x`` (spread) less the carriage's travel."""
    sw = p["wing_sweep_deg"] if sweep is None else sweep
    if not p.get("pivot_slides", True):
        return p["pivot_x"]
    xs, ys = zip(*SLIDE)
    k = max(0, min(len(xs) - 2, next((i for i in range(len(xs) - 1) if sw <= xs[i + 1]), len(xs) - 2)))
    f = max(0.0, min(1.0, (sw - xs[k]) / (xs[k + 1] - xs[k])))
    return p["pivot_x"] - (ys[k] + f * (ys[k + 1] - ys[k]))


def _panel(root_c, tip_c, span, sweep_deg, t):
    """One right-hand panel, its root's leading edge at the origin, the span along +y, a symmetric section."""
    tr = tip_c / root_c
    both = a1.Arigo1._surface(None, 0.0, root_c, tr, 2 * span, sweep_deg, t, "y", 0.0)
    return both.intersect(cq.Workplane("XY").box(4000.0, span + 5.0, 400.0).translate((0, (span + 5.0) / 2, 0)))


class AstorV(a1.Arigo1):
    """ASTOR-V: ASTOR's body; two pivoting wing panels that cross over its back; a fanning tail; a belly net bay."""

    parameters = [dataclasses.replace(q, default=_OVERRIDE[q.name]) if q.name in _OVERRIDE else q
                  for q in a1.Arigo1.parameters] + _NEW

    # ------------------------------------------------------------------------------------------------ wing
    def _one_wing(self, p, side, sweep):
        pan = _panel(p["panel_root_chord"], p["panel_tip_chord"], p["panel_span"], SPREAD_DEG, p["panel_thickness"])
        pan = pan.translate((-p["panel_pivot_chord"] * p["panel_root_chord"], 0, 0))            # the pivot at the origin
        pan = pan.rotate((0, 0, 0), (0, 0, 1), -(sweep - SPREAD_DEG))                              # aft sweep: +y toward +x
        z = p["wing_plane_z"] + side * p["panel_stack"] / 2
        if side < 0:
            pan = pan.mirror("XZ")
        return pan.translate((pivot_x(p, min(sweep, DASH_DEG) if sweep <= DASH_DEG else sweep), side * p["pivot_y"], z))

    def _wing(self, p):
        sl = p["wing_sweep_deg"] if p["wing_sweep_left_deg"] < 0 else p["wing_sweep_left_deg"]
        return self._one_wing(p, 1, p["wing_sweep_deg"]).union(self._one_wing(p, -1, sl))

    def _shoulder(self, p):
        """The block on the fuselage's back that carries the pivots, and the pivot pins."""
        R = p["fuselage_diameter"] / 2
        h = p["wing_plane_z"] + p["panel_stack"] - R + 6.0
        px = pivot_x(p)
        blk = (cq.Workplane("XY").box(110.0, 2 * p["pivot_y"] + 44.0, h).edges("|Z").fillet(18.0)
               .translate((px + 10.0, 0, R + h / 2 - 6.0)))                      # the carriage
        rail = cq.Workplane("XY").box(SLIDE[-1][1] + 130.0, 22.0, 8.0).translate((p["pivot_x"] - SLIDE[-1][1] / 2 + 10.0, 0, R + 1.0))
        blk = blk.union(rail)
        for s in (1, -1):
            pin = cq.Workplane("XY").circle(7.0).extrude(p["panel_stack"] + 16.0).translate(
                (px, s * p["pivot_y"], p["wing_plane_z"] - p["panel_stack"] / 2 - 8.0))
            blk = blk.union(pin)
        return blk

    # ------------------------------------------------------------------------------------------------ tail
    def _tail(self, p):
        sweep = 85.0 - 50.0 * p["tail_fan"]                       # folded 85 deg (narrow), spread 35 deg (a broad delta)
        out = None
        for s in (1, -1):
            pan = _panel(p["tail_panel_root"], p["tail_panel_tip"], p["tail_panel_span"], 35.0, 0.07)
            pan = pan.translate((-0.25 * p["tail_panel_root"], 0, 0)).rotate((0, 0, 0), (0, 0, 1), -(sweep - 35.0))
            if s < 0:
                pan = pan.mirror("XZ")
            pan = pan.translate((p["tail_pivot_x"], s * p["tail_pivot_y"], 6.0 + s * 3.0))
            out = pan if out is None else out.union(pan)
        # the fin is ventral: tucked, the wing panels cross the fuselage's back where a dorsal fin would stand
        fin = self._surface(p["fuselage_length"] - 20.0 - p["fin_root_chord"], p["fin_root_chord"], p["fin_taper"], p["fin_height"],
                            p["fin_sweep_deg"], p["tail_thickness"], "z", z0=p["tail_end_diameter"] / 2 - 4.0).mirror("XY")
        return out.union(fin)

    # ------------------------------------------------------------------------------------------------ nets
    def _net_bay(self, p):
        """The net tube's mouth: a ring at the fuselage's end (the tube itself runs inside the tail cone)."""
        r = p["net_diameter"] / 2
        return (cq.Workplane("YZ").circle(r + 5.0).circle(r + 0.5).extrude(14.0).translate((p["fuselage_length"] - 4.0, 0, 0)))

    def _nets(self, p):
        """The cartridges in tandem in the tail tube; the last one's end shows at the mouth."""
        out = None
        for k in range(int(p["net_count"])):
            x1 = p["fuselage_length"] + 6.0 - k * (p["net_length"] + 8.0)
            c = cq.Workplane("YZ").circle(p["net_diameter"] / 2).extrude(p["net_length"]).translate((x1 - p["net_length"], 0, 0))
            out = c if out is None else out.union(c)
        return out

    def _propeller(self, p):
        """The folding propeller (``astor_prop``) on the nose motor's shaft."""
        import astor_prop
        P = astor_prop.FoldingProp()
        return P.build(P.resolve(diameter=p["prop_diameter"], fold_deg=p["prop_fold_deg"])).translate((self.layout(p)["x_prop"] - 6.0, 0, 0))

    def _belly_camera(self, p):
        R = p["fuselage_diameter"] / 2
        return (cq.Workplane("XY").sphere(13.0).translate((p["belly_camera_x"], 0, -R - 3.0))
                .union(cq.Workplane("XY").box(34.0, 22.0, 8.0).translate((p["belly_camera_x"], 0, -R + 2.0))))

    def _rail(self, p):
        """Rail lugs on the fuselage's sides (the belly is the net bay's, the tail's underside the fin's)."""
        R = p["fuselage_diameter"] / 2
        out = None
        for x in (230.0, 500.0):
            for s in (1, -1):
                lug = cq.Workplane("XZ").circle(5.0).extrude(12.0).translate((x, s * R + (12.0 if s > 0 else 0.0), -8.0))
                out = lug if out is None else out.union(lug)
        return out

    def _parachute(self, p):
        R = p["fuselage_diameter"] / 2
        return cq.Workplane("XY").box(p["parachute_length"], 0.62 * p["fuselage_diameter"], 3.0).translate(
            (p["parachute_x"] + 40.0 + p["parachute_length"] / 2, 0, R + 0.5))

    def parts(self, p) -> dict:
        out = {"fuselage": self._fuselage(p), "wing": self._wing(p), "shoulder": self._shoulder(p), "tail": self._tail(p),
               "camera": self._camera(p), "motors": self._motors(p), "props": self._props(p), "parachute": self._parachute(p),
               "rail": self._rail(p)}
        out["belly_camera"] = self._belly_camera(p)
        if p["prop_fold_deg"] >= 0:
            out["propeller"] = self._propeller(p)
        if int(p["net_count"]) > 0:
            out["net_bay"] = self._net_bay(p)
            out["nets"] = self._nets(p)
        return out

    def build(self, p):
        ac = self._fuselage(p)
        for k, s in self.parts(p).items():
            if k in ("fuselage", "props", "nets") and not (k == "props" and p["show_props"]):
                continue
            ac = ac.union(s)
        return ac


def variant(**over) -> dict:
    q = dict(_OVERRIDE)
    q["tail_le_x"] = q["fuselage_length"] - 10.0 - q["htail_root_chord"]
    return {**q, **over}


def resolve(q=None) -> dict:
    return AstorV().resolve(**dict(q or variant()))


def wing_numbers(p, sweep) -> dict:
    """Span, the panels' exposed area (the part outside the fuselage's width) and where their lift acts, at a sweep."""
    import numpy as np
    s = math.radians(sweep - SPREAD_DEG)
    n = 60
    eta = (np.arange(n) + 0.5) / n
    c = p["panel_root_chord"] + (p["panel_tip_chord"] - p["panel_root_chord"]) * eta
    yl = eta * p["panel_span"]
    xl = yl * math.tan(math.radians(SPREAD_DEG)) + 0.25 * c - p["panel_pivot_chord"] * p["panel_root_chord"]   # quarter chord, pivot frame
    px = pivot_x(p, sweep)
    X = xl * math.cos(s) + yl * math.sin(s)
    Y = -xl * math.sin(s) + yl * math.cos(s)
    dA = c * p["panel_span"] / n
    # the share of each strip outside the fuselage's width (the rest lies over its back), by points along the chord
    xi = (np.arange(24) + 0.5) / 24
    xc = yl[:, None] * math.tan(math.radians(SPREAD_DEG)) + xi[None, :] * c[:, None] - p["panel_pivot_chord"] * p["panel_root_chord"]
    Yc = p["pivot_y"] - xc * math.sin(s) + yl[:, None] * math.cos(s)
    out_of_body = np.mean(np.abs(Yc) > p["fuselage_diameter"] / 2, axis=1)
    A_exp = 2 * float(np.sum(dA * out_of_body))
    x_ac = px + float(np.sum(X * dA * out_of_body) / max(np.sum(dA * out_of_body), 1e-9))
    tip_y = float(np.max(Yc))
    return {"sweep_deg": sweep, "span_mm": round(2 * tip_y, 0), "exposed_area_m2": round(A_exp / 1e6, 4),
            "lift_centre_x_mm": round(x_ac, 0)}


def budget(p=None, pack: str | None = None, extra=None, default_pack=None) -> dict:
    """ASTOR's budget with the fixed wing replaced by the two panels, plus the mechanisms and the nets."""
    p = p or resolve()
    S_w = (p["panel_root_chord"] + p["panel_tip_chord"]) * p["panel_span"] / 1e6
    S_t = (p["tail_panel_root"] + p["tail_panel_tip"]) * p["tail_panel_span"] / 1e6 + 0.5 * p["fin_root_chord"] * (1 + p["fin_taper"]) * p["fin_height"] / 1e6
    items = [("2 wing panels", 1000 * A.WING_KG_M2 * S_w, f"geometry: {S_w:.3f} m2 x {A.WING_KG_M2} kg/m2 (HAND +-20 %)"),
             ("tail fan and fin", 1000 * A.TAIL_KG_M2 * S_t, f"geometry: {S_t:.3f} m2 x {A.TAIL_KG_M2} kg/m2 (HAND +-20 %)"),
             ("fuselage", 1000 * A.FUSE_KG_M * p["fuselage_length"] / 1000, f"geometry: {p['fuselage_length']:.0f} mm x {A.FUSE_KG_M} kg/m (HAND +-20 %)")]
    items += list(A.ITEMS) + list(EXTRA_ITEMS if extra is None else extra)
    items.append((f"{int(p['net_count'])} net cartridges", NET_G * int(p["net_count"]), "HAND (+-30 each)"))
    items.append(("growth margin (10 % of the items above)", 0.10 * sum(x[1] for x in items), "HAND"))
    pk = pack or default_pack or A.DEFAULT_PACK
    g, wh, amps, src = A.PACKS[pk]
    empty = sum(x[1] for x in items)
    return {"items": [{"item": n, "g": round(v, 1), "source": s} for n, v, s in items], "empty_g": round(empty, 1),
            "pack": {"name": pk, "g": g, "Wh_nominal": round(wh, 1), "max_A": amps, "source": src},
            "total_g": round(empty + g, 1), "without_nets_g": round(empty + g - NET_G * int(p["net_count"]), 1)}


def checks() -> dict:
    """The geometry's numbers: per sweep the span, the exposed wing and its lift centre; the stowed envelope; the net's
    path past the propeller."""
    p = resolve()
    out = {"wing": [wing_numbers(p, s) for s in (SPREAD_DEG, 35.0, DASH_DEG, STOWED_DEG)]}
    m = budget(p)["total_g"] / 1000.0
    for w in out["wing"]:                                            # the nominal stall on the exposed wing (CL_max 0.95, HAND)
        w["stall_nominal_m_s"] = round(math.sqrt(m * 9.80665 / (0.5 * 1.225 * w["exposed_area_m2"] * 0.95)), 1)
    sh = AstorV().generate(**variant(wing_sweep_deg=STOWED_DEG, tail_fan=0.0)).shape
    bb = (sh.val() if hasattr(sh, "val") else sh).BoundingBox()
    out["stowed_envelope_mm"] = {"length": round(bb.xlen), "width": round(bb.ylen), "height": round(bb.zlen),
                                 "tube_diameter": round(math.hypot(bb.ylen, bb.zlen) * 0.87)}
    out["net_release"] = {"from": "a tube in the tail, straight aft", "propeller_behind_the_nets": False,
                          "motor_cut_needed": False, "note": "the propeller is on the nose: nothing turns behind the tube's mouth"}
    out["balance_by_mass"] = balance(p)
    out["mass"] = {k: v for k, v in budget(p).items() if k in ("total_g", "without_nets_g", "empty_g")}
    return out


# stations [mm from the nose tip] of what the budget weighs (HAND: a first layout; the pack's is solved for)
STATIONS = {"motor": 35.0, "ESC": 110.0, "propeller": -15.0, "camera": 70.0, "blister": 70.0, "flight controller": 170.0,
            "vision computer": 200.0, "radio": 250.0, "wiring": 330.0, "servos": 520.0, "parachute": 480.0, "rail": 365.0,
            "fuselage": 395.0, "tail fan": 720.0, "pivots": 330.0, "sweep actuators": 330.0, "net tube": 700.0, "belly drop camera": 520.0}


def balance(p=None, x_cg_target=None, mb=None, stations=None) -> dict:
    """Where the pack must sit for the CG wanted (spread, nets aboard), and how far the CG moves when the nets go."""
    p = p or resolve()
    mb = mb or budget(p)
    st = {**STATIONS, **(stations or {})}
    if x_cg_target is None:
        x_cg_target = CG_MM
    mom, m = 0.0, 0.0
    nets_g = NET_G * int(p["net_count"])
    x_nets = p.get("nets_station_mm") or (p["fuselage_length"] - (p["net_length"] + 4.0) * int(p["net_count"]) / 2)
    for it in mb["items"]:
        n, g = it["item"], it["g"]
        if "net cartridges" in n:
            x = x_nets
        elif "wing panels" in n:
            x = p["pivot_x"] + 60.0
        elif "growth margin" in n:
            continue
        else:
            x = next((v for k, v in (stations or {}).items() if k in n), None)        # a variant's own stations first
            if x is None:
                x = next((v for k, v in STATIONS.items() if k in n), 395.0)
        mom += g * x
        m += g
    margin = next(it["g"] for it in mb["items"] if "growth margin" in it["item"])
    mom += margin * (mom / m)
    m += margin
    pk = mb["pack"]["g"]
    x_pack = (x_cg_target * (m + pk) - mom) / pk
    m_all = m + pk
    cg_no_nets = (x_cg_target * m_all - nets_g * x_nets) / (m_all - nets_g)
    return {"cg_target_mm": x_cg_target, "pack_station_mm": round(x_pack, 0), "pack_fits": bool(110.0 <= x_pack <= 430.0),
            "cg_after_both_nets_mm": round(cg_no_nets, 0), "cg_shift_mm": round(cg_no_nets - x_cg_target, 0),
            "note": "stations are a first layout (HAND); the carriage re-trims the shift when the nets have gone"}


CG_MM = 365.0                      # set from the lattice (astor_v_model): 8 % MAC ahead of the spread neutral point (381 mm)
