"""Kestrel-4Q-2N — Kestrel-4Q-2 fitted with GAVILAN's net launcher: the air vessel in the pod, the tube on the axis, a
catcher net out of the nose. The hunter version of the medical courier: it chases a quadcopter and nets it
(Armageddon's bandit: 6-7 m/s, 6-22 m up — ``astor_netdrop.ArmageddonBandit`` on ``dev_astor_hawk``).

**What is fitted** (``NET``: GAVILAN's launcher, ``gavilan_flight.NET`` on ``dev_astor_hawk``; the net's own test from a
fixed launcher at a hovering quadcopter: 22 captures in 24 between 4 and 10 m):

* a 2.4 m square net in a 150 g cartridge, leaving at 40 m/s, 2.95 m across when open;
* **two** tubes (GAVILAN carries one), 42.7 mm bore, 0.30 m, 1 mm aluminium, one above the other, a fast valve each;
* one air vessel, 1.3 L at 30 bar (gauge) — a third larger than GAVILAN's, for the second shot at full speed.

**Where it goes** (``layout``): the pod is stretched 140 mm (470 → 610 mm; the constant mid-body 220 → 360 mm) and the
medical capsule's bay becomes the launcher's; the electronics keep their shelf:

* the **vessel** Ø90 × 260 mm (1.3 L) lies on the pod's axis in the rear of the mid-body (x 250-510), aluminium 6061
  2 mm (hoop stress 64 MPa at 30 bar, a quarter of the yield), hemispherical ends — the mid-body is 98 mm wide and
  262 mm tall inside, the capsule used 72 × 72 of it;
* the **electronics** keep the shelf above the bay (``ELECTRONICS_MM``): the flight stack (90 mm) and the mission
  computer (60 mm) where Kestrel-4Q put them, and a 95 mm spare behind them for the tracker's radio and the
  launcher's firing circuit; nothing of the launcher reaches above the vessel;
* the **battery** slides on the pod's floor under the tube, forward of the vessel (``cg``: the launcher is nose-heavy,
  the cartridge and the tube sit ahead of the wing; the battery goes where it trims the CG to Kestrel's target);
* the **valve block** (the fast valve, its pilot, the breech) between the vessel's front dome and the tube, x 220-250;
* the **tubes** run forward on the centre plane through the nose, 26 mm above and below the bay's axis, 300 mm from
  the breech to muzzles 80 mm ahead of the nose tip; a cartridge sits in the front 100 mm of each. Fired one after the
  other (``shot``: the vessel keeps 22 bar after the first, enough for the second at full speed);
* the **camera** leaves the nose tip for a chin window at x 70 mm, 58 mm below the axis, looking ahead and a little
  down — the chase is flown in cruise, nose into the wind, and the net leaves where the camera looks.

**Unorthodox, on purpose: a closed breech, no rear venturi.** GAVILAN's launcher is recoilless (a venturi throws air
backwards to cancel the cartridge's 6 N·s). On a 2.7 kg aircraft that impulse is a 2.2 m/s step backwards: at the
chase speed (16-20 m/s) it costs 2 m/s of a 6-8 m/s margin over the stall, in the hover it is a 2.2 m/s sink the
rotors (thrust/weight ≈ 3) stop in a quarter of a second. The venturi would need a gas duct the length of the pod
to the tail cone, 60 g of parts and most of the vessel's air (19 g a shot of 36 g carried). The recoil is cheaper
than the cure; ``shot`` gives both numbers.

**Propulsion**: the launcher is 1.0 kg where the capsule was 0.4; the other Kestrel sessions' fast comparator (``kestrel_fast``)
puts hover thrust/weight at 1.9 with the 9x6 propellers, under Kestrel's own minimum of 2, and back at ~3 with 10x7
propellers on 2814 motors (+4 × 43 g). ``vehicle`` runs that comparator for the aircraft as fitted; the rules of thumb
there are estimates, not a transition simulation.

Masses are hand estimates (``NET['parts_g']``) except the vessel, which ``vessel`` sizes. ``layout`` returns the boxes;
``checks`` the clearances and the CG; ``sizing`` the numbers; ``KestrelNetter`` the CAD (Dedalus, ``part``: launcher,
aircraft, assembly); ``export`` writes STEP/STL, views and the sizing to ``exports/kestrel_netter/``.

    python kestrel_netter.py            # the sizing
    python kestrel_netter.py --export   # the CAD, views and sizing into exports/kestrel_netter/
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

import kestrel
import kestrel2 as k2
import kestrel_systems as ks
from vegeta.dedalus import Design, Parameter

__all__ = ["NET", "VESSEL", "TUBE", "params", "layout", "vessel", "shot", "masses", "cg", "checks", "vehicle", "sizing",
           "summary", "KestrelNetter", "views", "export"]

G, RHO = ks.G, ks.RHO0
R_GAS, T_AIR = 287.05, 293.15
BAR = 1e5

#: GAVILAN's net launcher (gavilan_flight.NET on dev_astor_hawk: net_drone.py results and golf_recoilless.shot, dev_net_launcher @ bb041dd)
NET = {
    "source": "gavilan_flight.NET (dev_astor_hawk): net_drone.py results and golf_recoilless.shot (dev_net_launcher @ bb041dd); masses are hand estimates",
    "range_m": (4.0, 10.0), "captures": {4: "5/6", 6: "6/6", 8: "5/6", 10: "6/6", 12: "0/6", 15: "1/6"},
    "muzzle_m_s": 40.0, "net_side_m": 2.4, "open_span_m": 2.95, "flight_s_10m": 0.51, "cartridge_g": 150.0, "cartridge_len_mm": 100.0,
    "tube_m": 0.30, "bore_mm": 42.7, "wall_mm": 1.0, "pressure_bar_g": 29.8, "tank_l": 1.0, "air_g_per_shot_recoilless": 19.1,
    "parts_g": {"tube 0.30 m, 42.7 mm bore, 1 mm aluminium": 112.0, "chamber and breech (closed, no venturi)": 60.0,
                "25 mm fast valve with its pilot": 150.0, "net cartridge": 150.0, "mounts": 40.0},
    "t_open_s": 0.15, "solidity": 0.10,
}
#: the vessel as fitted [mm]: on the axis in the rear of the stretched mid-body
VESSEL = {"d_mm": 90.0, "wall_mm": 2.0, "x0_mm": 250.0, "x1_mm": 510.0, "material": "aluminium 6061-T6", "rho": 2700.0, "yield_MPa": 240.0}
#: the tube as fitted [mm]: from the breech forward through the nose
TUBE = {"x_breech_mm": 250.0, "valve_mm": 30.0}      #: the muzzle follows: breech - valve - NET["tube_m"] = -80 mm (80 mm ahead of the nose tip)
CAMERA = {"x_mm": 70.0, "z_mm": -58.0, "window_mm": 24.0}     #: under the lower tube
POD_STRETCH_MM = 140.0
N_TUBES = 2                                   #: two tubes, one above the other, on the pod's centre plane
TUBE_DZ_MM = 26.0                             #: their axes above and below the bay's axis (3 mm between the walls)
ELECTRONICS_MM = {"flight stack": (250.0, 340.0), "mission computer": (345.0, 405.0), "electronics spare": (410.0, 505.0)}   #: along x, above the vessel
BATTERY_MM = (140.0, 44.0, 32.0)              #: the 4S 3300 (kestrel_systems.COMPONENTS["battery"])
BATTERY_Z_MM = -80.0                          #: its centre: on the pod's floor, under the tube and the vessel
BATTERY_TRAVEL_MM = (170.0, 440.0)            #: its centre's travel: from the nose's full section to the end of the mid-body


# ================================================================================================= the aircraft
def params(p=None) -> dict:
    """Kestrel-4Q-2N's CAD parameters: 4Q-2 (elevons, faired posts), the pod 90 mm longer, the feet behind it, 10 inch
    propellers; the capsule bay re-cut to the vessel's section so the pod's own bay checks guard it."""
    base = k2.params("4Q-2", p)
    d = VESSEL["d_mm"]
    return kestrel.resolve(dict(base, pod_length=base["pod_length"] + POD_STRETCH_MM, x_feet=base["x_feet"] + POD_STRETCH_MM,
                                prop_diameter=254.0,
                                bay_length=VESSEL["x1_mm"] - VESSEL["x0_mm"], bay_width=d, bay_height=d, bay_x=VESSEL["x0_mm"], bay_z=10.0))


def tube_offsets() -> list[float]:
    """The tubes' axes above the bay's axis [mm]: one on it, or two (and more) spread by ``TUBE_DZ_MM``."""
    if N_TUBES == 1:
        return [0.0]
    return [TUBE_DZ_MM * (2 * i - (N_TUBES - 1)) for i in range(N_TUBES)]


def layout(p=None, battery_x: float | None = None) -> dict:
    """The launcher's boxes in the pod [mm, (x0, x1, y0, y1, z0, z1)] beside the pod's own bays."""
    p = params(p)
    b = kestrel.Kestrel4Q.bays(p)
    r = VESSEL["d_mm"] / 2
    bz = p["bay_z"]
    ro = NET["bore_mm"] / 2 + NET["wall_mm"]
    x_breech = TUBE["x_breech_mm"]
    x_muzzle = x_breech - TUBE["valve_mm"] - NET["tube_m"] * 1e3
    boxes = {
        "vessel": (VESSEL["x0_mm"], VESSEL["x1_mm"], -r, r, bz - r, bz + r),
        "valve block": (x_breech - TUBE["valve_mm"], x_breech, -ro - 6, ro + 6, bz - TUBE_DZ_MM - ro - 6, bz + TUBE_DZ_MM + ro + 6),
        "camera (chin window)": (CAMERA["x_mm"] - 15, CAMERA["x_mm"] + 15, -CAMERA["window_mm"] / 2, CAMERA["window_mm"] / 2, CAMERA["z_mm"] - 12, CAMERA["z_mm"] + 12),
    }
    for i, dz in enumerate(tube_offsets(), 1):
        boxes[f"tube {i}"] = (x_muzzle, x_breech - TUBE["valve_mm"], -ro, ro, bz + dz - ro, bz + dz + ro)
        boxes[f"cartridge {i}"] = (x_muzzle + 5.0, x_muzzle + 5.0 + NET["cartridge_len_mm"], -NET["bore_mm"] / 2, NET["bore_mm"] / 2, bz + dz - NET["bore_mm"] / 2, bz + dz + NET["bore_mm"] / 2)
    z_top = b["flight stack"][4]                                            # the pod's own electronics shelf above the bay
    for k, (xa, xb_) in ELECTRONICS_MM.items():
        boxes[k] = (xa, xb_, -30.0, 30.0, z_top, z_top + 30.0)
    bl, bw, bh = BATTERY_MM
    xb = (battery_x if battery_x is not None else _battery_x()) * 1e3
    zb = BATTERY_Z_MM
    boxes["battery"] = (xb - bl / 2, xb + bl / 2, -bw / 2, bw / 2, zb - bh / 2, zb + bh / 2)
    return {"p": p, "boxes": boxes, "mid_body_mm": (p["nose_length"], p["pod_length"] - p["tail_length"]),
            "inside_mm": (p["pod_width"] - 2 * p["pod_wall"], 0.98 * p["gap"] - 2 * p["pod_wall"])}


# ================================================================================================= the pieces
def vessel() -> dict:
    """The air vessel: volume, wall stress, mass, the air it holds."""
    d, t, L = VESSEL["d_mm"] * 1e-3, VESSEL["wall_mm"] * 1e-3, (VESSEL["x1_mm"] - VESSEL["x0_mm"]) * 1e-3
    r_in = d / 2 - t
    L_cyl = L - d                                           # hemispherical ends inside the length
    V = math.pi * r_in ** 2 * L_cyl + 4 / 3 * math.pi * r_in ** 3
    p_g = NET["pressure_bar_g"] * BAR
    hoop = p_g * r_in / t
    m_shell = VESSEL["rho"] * (math.pi * ((d / 2) ** 2 - r_in ** 2) * L_cyl + 4 / 3 * math.pi * ((d / 2) ** 3 - r_in ** 3))
    m_fittings = 0.045                                      # the fill valve, the boss to the valve block, two saddles
    m_air = (p_g + 1.013e5) * V / (R_GAS * T_AIR)
    return {"volume_l": V * 1e3, "hoop_stress_MPa": hoop / 1e6, "yield_margin": VESSEL["yield_MPa"] * 1e6 / hoop,
            "burst_pressure_bar": VESSEL["yield_MPa"] * 1e6 * t / r_in / BAR, "shell_kg": m_shell, "fittings_kg": m_fittings,
            "mass_kg": m_shell + m_fittings, "air_g": m_air * 1e3, "material": VESSEL["material"]}


def shot(mass_aircraft_kg: float | None = None) -> dict:
    """One shot: the cartridge's energy and impulse, the recoil on the aircraft (closed breech), the gas it takes
    (isothermal, the tube's volume filled to the pressure that gives the muzzle energy at ~25 % efficiency), the
    vessel's pressure after it, and the recoilless alternative's air."""
    m_c = NET["cartridge_g"] * 1e-3
    v0 = NET["muzzle_m_s"]
    E = 0.5 * m_c * v0 ** 2
    J = m_c * v0
    A = math.pi * (NET["bore_mm"] * 1e-3 / 2) ** 2
    V_tube = A * NET["tube_m"]
    eta = 0.25                                              # gas work to cartridge energy, a pneumatic launcher's usual share
    W_gas = E / eta
    V_ves = vessel()["volume_l"] * 1e-3
    p0 = (NET["pressure_bar_g"] + 1.013) * BAR
    # isothermal expansion from the vessel into vessel + tube: W = p0 V_ves ln((V_ves + V_tube) / V_ves) at most
    W_full = p0 * V_ves * math.log((V_ves + V_tube) / V_ves)
    p_after = p0 * V_ves / (V_ves + V_tube)                 # all the gas that entered the tube is lost at the muzzle
    m_air_used = (p0 - p_after) * V_ves / (R_GAS * T_AIR)
    m = mass_aircraft_kg or cg()["mass_kg"]
    dv = J / m
    # the shots in turn from one fill: each starts at the pressure the last one left
    seq, pk = [], p0
    for i in range(1, N_TUBES + 1):
        W_i = pk * V_ves * math.log((V_ves + V_tube) / V_ves)
        p_next = pk * V_ves / (V_ves + V_tube)
        seq.append({"shot": i, "vessel_before_bar_g": pk / BAR - 1.013, "work_margin": W_i / W_gas, "full_speed": W_i >= W_gas,
                    "air_used_g": (pk - p_next) * V_ves / (R_GAS * T_AIR) * 1e3})
        pk = p_next
    return {"cartridge_J": E, "impulse_Ns": J, "recoil_m_s": dv, "recoil_g_over_0_05_s": dv / 0.05 / G,
            "gas_work_available_J": W_full, "gas_work_needed_J": W_gas, "work_margin": W_full / W_gas,
            "vessel_after_shot_bar_g": p_after / BAR - 1.013, "air_used_g": m_air_used * 1e3,
            "air_used_recoilless_g": NET["air_g_per_shot_recoilless"], "tubes": N_TUBES,
            "shots_per_fill": sum(1 for q in seq if q["full_speed"]), "shots": seq,
            "note": f"one cartridge per tube, {N_TUBES} tubes: {sum(1 for q in seq if q['full_speed'])} full-speed shots a fill; no reloading in flight"}


def masses() -> dict:
    """The launcher's parts [kg] and what the airframe changes cost."""
    v = vessel()
    parts = {}
    for k, g in NET["parts_g"].items():
        if k == "mounts":
            parts[k] = g * 1e-3 * (1 + 0.5 * (N_TUBES - 1))
        else:
            parts[f"{k} (x{N_TUBES})" if N_TUBES > 1 else k] = g * 1e-3 * N_TUBES
    parts[f"vessel Ø{VESSEL['d_mm']:.0f} × {VESSEL['x1_mm'] - VESSEL['x0_mm']:.0f} mm, {VESSEL['material']}"] = v["mass_kg"]
    parts[f"air, {v['volume_l']:.1f} L at 30 bar"] = v["air_g"] * 1e-3
    launcher = sum(parts.values())
    airframe = {f"pod stretch {POD_STRETCH_MM:.0f} mm (shell)": ks.POD_SHELL_KG * POD_STRETCH_MM / 470.0,
                "2814 motors for 2216 (4x)": 4 * (0.105 - ks.COMPONENTS["motor"]["mass_kg"]),
                "10x7 propellers for 9x6 (4x)": 4 * (0.018 - ks.COMPONENTS["propeller"]["mass_kg"]),
                "capsule and its shell, removed": -(ks.COMPONENTS["capsule"]["shell_kg"] + ks.COMPONENTS["capsule"]["contents_nominal_kg"])}
    return {"launcher": parts, "launcher_kg": launcher, "airframe_changes": airframe, "airframe_changes_kg": sum(airframe.values())}


def _launcher_rows() -> list[dict]:
    """The launcher's items as mass-table rows (metres, the aircraft's frame)."""
    L = layout(battery_x=0.0)
    b = L["boxes"]
    bz = L["p"]["bay_z"] * 1e-3
    mid = lambda k: 0.5 * (b[k][0] + b[k][1]) * 1e-3
    m = masses()["launcher"]
    rows = []
    for k, mk in m.items():
        if k.startswith("tube"):
            x, rod = mid("tube 1"), ("x", NET["tube_m"])
        elif k.startswith("chamber") or k.startswith("25 mm"):
            x, rod = mid("valve block"), None
        elif k.startswith("net cartridge"):
            x, rod = mid("cartridge 1"), None
        elif k.startswith("vessel") or k.startswith("air"):
            x, rod = mid("vessel"), ("x", (VESSEL["x1_mm"] - VESSEL["x0_mm"]) * 1e-3)
        else:
            x, rod = 0.5 * (mid("tube 1") + mid("vessel")), None
        rows.append({"item": k, "mass_kg": mk, "x": x, "y": 0.0, "z": bz, "group": "payload", "rod": rod})
    return rows


def _battery_x() -> float:
    return cg()["battery_x"]


def cg() -> dict:
    """The aircraft's CG with the launcher: the standard table without the capsule, the launcher's rows, the heavier
    motors and propellers, and the battery slid in its bay to the CG target (``kestrel_systems.X_CG_TARGET_FRAC``)."""
    p = params()
    L = kestrel.layout(p)
    target = L["x_le"] + ks.X_CG_TARGET_FRAC * L["chord"]
    rows = [dict(r) for r in ks.mass_table(p) if r["item"] != "battery" and not r["item"].startswith("capsule")]
    for r in rows:
        if r["item"].startswith("motor"):
            r["mass_kg"] = 0.105
        if r["item"].startswith("propeller"):
            r["mass_kg"] = 0.018
    rows += _launcher_rows()
    m_b = ks.COMPONENTS["battery"]["mass_kg"]
    m_rest = sum(r["mass_kg"] for r in rows)
    x_rest = sum(r["mass_kg"] * r["x"] for r in rows) / m_rest
    x_batt = (target * (m_rest + m_b) - m_rest * x_rest) / m_b           # where the battery must sit for the CG target
    lo, hi = (v * 1e-3 for v in BATTERY_TRAVEL_MM)
    x_b = min(max(x_batt, lo), hi)
    zb = BATTERY_Z_MM * 1e-3
    rows.append({"item": "battery", "mass_kg": m_b, "x": x_b, "y": 0.0, "z": zb, "group": "systems", "rod": None, "clipped": x_b != x_batt})
    ci = ks.cg_inertia(rows)
    return {"rows": rows, "mass_kg": ci["mass_kg"], "x_cg": ci["x_cg"], "z_cg": ci["z_cg"], "x_cg_target": target,
            "battery_x": x_b, "battery_x_for_target": x_batt, "battery_travel": (lo, hi), "battery_clipped": x_b != x_batt,
            "cg_off_target_mm": (ci["x_cg"] - target) * 1e3, "Ixx": ci["Ixx"], "Iyy": ci["Iyy"], "Izz": ci["Izz"]}


def checks() -> list[str]:
    """The fit: the pod's own rules and bay checks on the re-cut bay, the tube inside the nose skin, the launcher's
    boxes clear of each other and of the pod's bays, the vessel's wall, the CG on target with the battery in its bay."""
    L = layout()
    p, b = L["p"], L["boxes"]
    out = list(kestrel.Kestrel4Q.checks(p)) + list(kestrel.Kestrel4Q.bay_checks(p))
    # the tube in the nose: the ellipsoidal nose's half-width at x must exceed the tube's radius + wall
    prof = kestrel.Kestrel4Q.pod_profile(p, n_nose=30)
    ro = NET["bore_mm"] / 2 + NET["wall_mm"]
    xs = [s[0] for s in prof]
    ws = [s[1] / 2 for s in prof]
    hs = [s[2] / 2 for s in prof]
    x_enter = None
    z_out = max(abs(dz) for dz in tube_offsets()) + abs(p["bay_z"]) + ro          # the outermost tube's reach from the axis
    for x in np.arange(0.0, p["nose_length"], 1.0):
        if np.interp(x, xs, ws) - p["pod_wall"] >= ro and np.interp(x, xs, hs) - p["pod_wall"] >= z_out and x_enter is None:
            x_enter = float(x)
    if x_enter is None or x_enter > 60.0:
        out.append(f"the tubes ({2 * ro:.1f} mm, {z_out:.0f} mm off the axis) do not fit inside the nose skin until x = {x_enter} mm")
    # boxes clear of each other (the tube passes through the valve block by design: skip that pair)
    names = list(b)
    for i, a in enumerate(names):
        for c in names[i + 1:]:
            if (a.startswith("tube") and c == "valve block") or (a.startswith("tube") and c.startswith("cartridge") and a[-1] == c[-1]):
                continue
            A, C = b[a], b[c]
            if all(A[2 * k] < C[2 * k + 1] and C[2 * k] < A[2 * k + 1] for k in range(3)):
                out.append(f"the {a} and the {c} overlap")
    # the vessel inside the mid-body and clear of the skin
    x0, x1 = L["mid_body_mm"]
    w_in, h_in = L["inside_mm"]
    if b["vessel"][0] < x0 or b["vessel"][1] > x1:
        out.append("the vessel reaches out of the constant mid-body")
    if VESSEL["d_mm"] > w_in or b["vessel"][5] > h_in / 2 or -b["vessel"][4] > h_in / 2:
        out.append("the vessel does not clear the pod's skin")
    bb = b["battery"]
    for x in (bb[0], bb[1]):
        half_h = np.interp(x, xs, hs) if x < p["nose_length"] else 0.49 * p["gap"]
        half_w = np.interp(x, xs, ws) if x < p["nose_length"] else p["pod_width"] / 2
        if -bb[4] > half_h - p["pod_wall"] or bb[3] > half_w - p["pod_wall"]:
            out.append(f"the battery does not clear the pod's skin at x = {x:.0f} mm")
    v = vessel()
    if v["yield_margin"] < 2.5:
        out.append(f"the vessel's wall: yield margin {v['yield_margin']:.1f} < 2.5 at 30 bar")
    if v["volume_l"] < NET["tank_l"] - 0.05:
        out.append(f"the vessel holds {v['volume_l']:.2f} L, under GAVILAN's {NET['tank_l']:.1f}")
    c = cg()
    if c["battery_clipped"] and abs(c["cg_off_target_mm"]) > 5.0:
        out.append(f"the battery cannot reach the CG target: it would have to sit at x = {c['battery_x_for_target'] * 1e3:.0f} mm "
                   f"(its travel {c['battery_travel'][0] * 1e3:.0f}-{c['battery_travel'][1] * 1e3:.0f}); the CG is {c['cg_off_target_mm']:+.0f} mm off")
    return out


def vehicle() -> dict:
    """The aircraft as fitted through ``kestrel_fast`` (estimates): 10x7 propellers, 2814 motors, the launcher and the
    pod stretch as payload — mass, hover thrust/weight and power, stall and cruise, endurance, the static margin."""
    import kestrel_fast as kfst
    m = masses()
    extra = m["launcher_kg"] + m["airframe_changes"][f"pod stretch {POD_STRETCH_MM:.0f} mm (shell)"] - m["airframe_changes"]["capsule and its shell, removed"] - 0.39 + 0.15
    # kestrel_fast's payload_kg replaces its 0.15 kg default; the standard table it starts from carries the capsule (0.39 kg): take that out
    c = kfst.Candidate("Kestrel-4Q-2N", prop="10x7", motor="2814 1000KV", payload_kg=max(extra, 0.0),
                       note="4Q-2 with GAVILAN's net launcher in the pod; 10x7 on 2814")
    base = kfst.Candidate("Kestrel-4Q-2", note="as designed, the medical capsule")
    bigger = kfst.Candidate("Kestrel-4Q-2N, 1.2 m wing", span=1.2, chord=0.16, prop="10x7", motor="2814 1000KV", payload_kg=max(extra, 0.0),
                            note="the same launcher on the larger wing Kestrel's notes propose (kestrel3.md): the stall margin back")
    rows = {}
    for cand in (base, c, bigger):
        try:
            rows[cand.name] = kfst.evaluate(cand)
        except Exception as e:                               # NeuralFoil / Boreas not available: the table is left out
            rows[cand.name] = {"error": repr(e)}
    return rows


def sizing() -> dict:
    v, s, m, c = vessel(), shot(), masses(), cg()
    L = layout()
    return {"aircraft": "Kestrel-4Q-2N", "base": "Kestrel-4Q-2 (kestrel2.params)", "launcher_source": NET["source"],
            "net": {k: NET[k] for k in ("range_m", "captures", "muzzle_m_s", "net_side_m", "open_span_m", "flight_s_10m", "cartridge_g", "t_open_s", "solidity")},
            "vessel": v, "shot": s, "masses": m, "mass_kg": c["mass_kg"], "cg": {k: c[k] for k in c if k != "rows"},
            "layout_mm": L["boxes"], "pod_mm": {"length": L["p"]["pod_length"], "width": L["p"]["pod_width"], "mid_body": L["mid_body_mm"], "inside": L["inside_mm"]},
            "checks": checks(), "vehicle": vehicle()}


def summary() -> str:
    return json.dumps(sizing(), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))


# ================================================================================================= CAD (mm, the aircraft's frame)
def _cyl_x(r, x0, x1, y=0.0, z=0.0):
    import cadquery as cq
    return cq.Workplane("YZ").center(y, z).circle(r).extrude(x1 - x0).translate((x0, 0, 0))


class KestrelNetter(Design):
    """The launcher in the pod (``part="launcher"``: vessel, valve block, tube, cartridge, the chin camera), the aircraft
    around it (``part="aircraft"``: ``kestrel.Kestrel4Q`` at ``params()``) or both (``part="assembly"``)."""

    parameters = [Parameter("part", "launcher", choices=("launcher", "aircraft", "assembly"), description="what to build"),
                  Parameter("show_props", True, "", description="the propellers as thin discs (aircraft / assembly)")]

    def launcher(self, p):
        import cadquery as cq
        L = layout()
        b = L["boxes"]
        bz = L["p"]["bay_z"]
        r = VESSEL["d_mm"] / 2
        x0, x1 = VESSEL["x0_mm"], VESSEL["x1_mm"]
        ves = (_cyl_x(r, x0 + r, x1 - r, z=bz).union(cq.Workplane("XY").sphere(r).translate((x0 + r, 0, bz)))
               .union(cq.Workplane("XY").sphere(r).translate((x1 - r, 0, bz))))
        ro = NET["bore_mm"] / 2 + NET["wall_mm"]
        ves = ves.union(_cyl_x(ro * 0.6, x0 - 2.0, x0 + r, z=bz))           # the boss to the valve block
        vb = b["valve block"]
        valve = cq.Workplane("XY").box(vb[1] - vb[0], vb[3] - vb[2], vb[5] - vb[4]).translate((0.5 * (vb[0] + vb[1]), 0, 0.5 * (vb[4] + vb[5]))).edges("|X").fillet(4.0)
        out = ves.union(valve)
        for i, dz in enumerate(tube_offsets(), 1):
            tb, cb = b[f"tube {i}"], b[f"cartridge {i}"]
            tube = _cyl_x(ro, tb[0], tb[1], z=bz + dz).cut(_cyl_x(NET["bore_mm"] / 2, tb[0] - 1.0, tb[1] + 1.0, z=bz + dz))
            cart = _cyl_x(NET["bore_mm"] / 2 - 0.5, cb[0], cb[1], z=bz + dz)
            out = out.union(tube).union(cart)
        cam = b["camera (chin window)"]
        camera = cq.Workplane("XY").box(cam[1] - cam[0], cam[3] - cam[2], cam[5] - cam[4]).translate((0.5 * (cam[0] + cam[1]), 0, 0.5 * (cam[4] + cam[5])))
        return out.union(camera)

    def aircraft(self, p):
        q = params()
        return kestrel.Kestrel4Q().generate(**{k: v for k, v in q.items() if k not in ("part", "show_props")}, part="aircraft", show_props=p["show_props"])

    def assembly(self, p):
        import cadquery as cq
        air = self.aircraft(p)
        shape = _cq_shape(air)
        return cq.Workplane("XY").add(shape).union(self.launcher(p)) if shape is not None else self.launcher(p)

    def build(self, p):
        return getattr(self, p["part"])(p)


def _cq_shape(geom):
    """The CadQuery shape inside a Dedalus Geometry, when it carries one."""
    import cadquery as cq
    if isinstance(geom, (cq.Workplane, cq.Shape)):
        return geom
    for attr in ("_shape", "shape", "_obj", "obj", "_cq", "cq", "_geometry", "geometry", "_solid"):
        s = getattr(geom, attr, None)
        if isinstance(s, (cq.Workplane, cq.Shape)):
            return s
    return None


def _tess(geom, tol=2.0, ang=0.35):
    """Triangles [m] of a CadQuery shape or a Dedalus Geometry."""
    import cadquery as cq
    s = _cq_shape(geom)
    if s is not None:
        s = s.val() if isinstance(s, cq.Workplane) else s
        v, t = s.tessellate(tol, ang)
        v = np.array([[q.x, q.y, q.z] for q in v]) * 1e-3
        return v[np.array(t)] if len(t) else np.zeros((0, 3, 3))
    v, t = geom.tessellate(tol, ang)
    v, t = np.asarray(v, float), np.asarray(t, int)
    if v.size and v.max() > 50:                              # mm
        v = v * 1e-3
    return v[t] if len(t) else np.zeros((0, 3, 3))


def views(path, title="Kestrel-4Q-2N - the net launcher in the pod: 1.3 L vessel, valve block, two tubes and cartridges; the electronics shelf (green), the chin camera"):
    """Four shaded views (3/4 front, side, top, nose-on) of the airframe (grey, translucent) with the launcher coloured."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    d = KestrelNetter()
    p = d.resolve()
    air = _tess(d.aircraft(p), tol=3.0)
    L = layout()
    bz = L["p"]["bay_z"]
    parts = {"vessel": (0.80, 0.12, 0.12, 1.0), "valve block": (0.25, 0.27, 0.30, 1.0), "tube": (0.75, 0.75, 0.78, 1.0),
             "cartridge": (0.95, 0.70, 0.15, 1.0), "camera (chin window)": (0.15, 0.35, 0.85, 1.0)}
    import cadquery as cq
    meshes = []
    b = L["boxes"]
    r = VESSEL["d_mm"] / 2
    x0, x1 = VESSEL["x0_mm"], VESSEL["x1_mm"]
    ro = NET["bore_mm"] / 2 + NET["wall_mm"]
    solids = {"vessel": _cyl_x(r, x0 + r, x1 - r, z=bz).union(cq.Workplane("XY").sphere(r).translate((x0 + r, 0, bz))).union(cq.Workplane("XY").sphere(r).translate((x1 - r, 0, bz))),
              "valve block": cq.Workplane("XY").box(b["valve block"][1] - b["valve block"][0], b["valve block"][3] - b["valve block"][2], b["valve block"][5] - b["valve block"][4]).translate((0.5 * (b["valve block"][0] + b["valve block"][1]), 0, bz)),
              "camera (chin window)": cq.Workplane("XY").box(30, CAMERA["window_mm"], 24).translate((CAMERA["x_mm"], 0, CAMERA["z_mm"]))}
    for i, dz in enumerate(tube_offsets(), 1):
        solids[f"tube {i}"] = _cyl_x(ro, b[f"tube {i}"][0], b[f"tube {i}"][1], z=bz + dz)
        solids[f"cartridge {i}"] = _cyl_x(NET["bore_mm"] / 2 + 0.5, b[f"cartridge {i}"][0] - 1, b[f"cartridge {i}"][1] + 1, z=bz + dz)
    for k, (xa, xb_) in ELECTRONICS_MM.items():
        bx = b[k]
        solids[k] = cq.Workplane("XY").box(bx[1] - bx[0], bx[3] - bx[2], bx[5] - bx[4]).translate((0.5 * (bx[0] + bx[1]), 0, 0.5 * (bx[4] + bx[5])))
    for k, s in solids.items():
        col = parts.get(k.rstrip(" 12"), parts.get(k, (0.20, 0.65, 0.35, 0.9)))
        meshes.append((_tess(s, tol=1.5), col))
    light = np.array([0.4, -0.5, 0.8]) / np.linalg.norm([0.4, -0.5, 0.8])
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title, fontsize=14)
    cams = [("3/4 front, from below the nose", 18, -150), ("side", 0, -90), ("top", 90, -90), ("nose-on", 0, 180)]
    for i, (name, elev, azim) in enumerate(cams, 1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        for tris, col, alpha in [(air, (0.55, 0.57, 0.60), 0.18)] + [(t, c, 1.0) for t, c in meshes]:
            if not len(tris):
                continue
            n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
            sh = 0.55 + 0.45 * np.clip(n @ light, 0, 1)
            fc = np.c_[np.outer(sh, col[:3]), np.full(len(sh), alpha)]
            ax.add_collection3d(Poly3DCollection(tris, facecolors=fc, edgecolors="none"))
        ax.set_xlim(-0.1, 0.7)
        ax.set_ylim(-0.55, 0.55)
        ax.set_zlim(-0.4, 0.4)
        ax.set_box_aspect((0.8, 1.1, 0.8), zoom=1.3)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(name)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def export(out: str | Path | None = None) -> list[Path]:
    """STEP and STL of the launcher, the aircraft and the assembly, the views and the sizing into ``exports/kestrel_netter/``."""
    out = Path(out) if out else Path(__file__).resolve().parents[3] / "exports" / "kestrel_netter"
    out.mkdir(parents=True, exist_ok=True)
    import cadquery as cq
    from vegeta.dedalus import Geometry
    d = KestrelNetter()
    written = []
    for part in ("launcher", "aircraft", "assembly"):
        g = d.build(d.resolve(part=part))
        if not isinstance(g, Geometry):
            g = Geometry.from_cadquery(g)
        for ext, fn in (("step", lambda f: g.export_step(f)), ("stl", lambda f: g.export_stl(f, tolerance=0.3, angular_tolerance=0.3))):
            f = out / f"kestrel_netter_{part}.{ext}"
            fn(f)
            written.append(f)
    f = out / "kestrel_netter_views.png"
    views(f)
    written.append(f)
    f = out / "kestrel_netter_sizing.json"
    f.write_text(summary() + "\n")
    written.append(f)
    return written


if __name__ == "__main__":
    if "--export" in sys.argv:
        for f in export():
            print(f)
    else:
        print(summary())
