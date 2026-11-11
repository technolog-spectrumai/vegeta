"""The tracked machines of the land lab side by side — the Pekari Rover, Pekari-2M, Pumba (with its grab and with
the chainsaw), Perseus and Calydon (``docs/pekari_pumba.md`` §9). Every number comes from each machine's own model
(``pekari_rover_robot``, ``pekari_family`` + ``pekari_2m_robot`` / ``pumba_robot`` / ``calydon_robot``,
``perseus_robot``); the comparisons that need one rule for all are computed here the same way for every machine:

* **envelope**: the machine's CAD (Dedalus) measured with the arms folded / the boom stowed (needs CadQuery; else
  the chassis' parametric envelope without the arms);
* **drive**: ``ogma.tracks.resistance`` on gravel with the nominal payload, the two track drives' stall and rated
  forces at the sprocket, the traction limit; the steepest grade held continuously (within the drives' rated
  force and the traction); the ground pressure;
* **driving endurance**: the battery (85 % usable) over the traction power at 0.8 m/s on flat gravel (through
  80 % drive efficiency) plus the electronics — a driving-only figure, the same profile for all;
* **arms**: count, DOF of the main arm, its reach (to the TCP: the jaw tips or the middle of the saw bar), its lift
  rating; a saw's margin: stall over torque at the wrist (the weakest joint) cutting outstretched with the cutting
  loads × 1.5 (``ogma.cutting.head_loads``), Perseus's ``arm_torques`` measure;
* **stability**: the static tip angle of the loaded machine, arms folded / stowed, on level ground.

    import land_comparison as lc
    df = lc.table()               # pandas DataFrame, machines as columns
    print(lc.markdown())
"""
from __future__ import annotations

import math
import re

from ogma import actuators as act
from ogma import cutting
from ogma import tracks

__all__ = ["MACHINES", "rows", "table", "markdown", "drive_numbers"]

G = 9.81
V_ENDURANCE = 0.8
DRIVE_EFF = 0.80
USABLE = 0.85
PEKARI_ELECTRONICS_W = 25.0      # assumed: Pekari Rover's computer, controllers, stereo camera and 2D lidar
MACHINES = ("Pekari Rover", "Pekari-2M", "Pumba", "Pumba + chainsaw", "Perseus", "Calydon")


def drive_numbers(W: float, g: dict, drive: str, f0: float = 0.035) -> dict:
    """Drive figures of a tracked machine of weight ``W`` [N] with ``pekari_rover_robot.geometry`` ``g``."""
    d = act.get(drive)
    r = g["r_sprocket"]
    w0 = d.no_load_rpm * 2 * math.pi / 60
    F_rated = 2 * d.rated_Nm / r
    res20 = tracks.resistance(W, b=g["b"], L=g["L"], soil="gravel", grade_deg=20.0, v=0.3, f0=f0)["total"]
    F_stall = 2 * d.stall_Nm / r * max(0.0, 1 - (0.3 / r) / w0)
    trac20 = tracks.tractive_limit(W, b=g["b"], L=g["L"], soil="gravel", grade_deg=20.0)
    cont = 0.0
    for k in range(0, 451):
        gr = k / 10
        need = tracks.resistance(W, b=g["b"], L=g["L"], soil="gravel", grade_deg=gr, v=0.3, f0=f0)["total"]
        if need > min(F_rated, tracks.tractive_limit(W, b=g["b"], L=g["L"], soil="gravel", grade_deg=gr)):
            break
        cont = gr
    flat = tracks.resistance(W, b=g["b"], L=g["L"], soil="gravel", grade_deg=0.0, v=V_ENDURANCE, f0=f0)["total"]
    return {"v_top_kmh": w0 * r * 3.6, "climb20": F_stall >= res20 and trac20 >= res20, "grade_cont_deg": cont,
            "ground_kPa": tracks.ground_pressure(W, g["b"], g["L"]) / 1000.0, "traction_W": flat * V_ENDURANCE / DRIVE_EFF}


def _cad_dims(cls, **kw):
    try:
        import cadquery  # noqa: F401
    except ImportError:
        return None
    d = cls().generate(**kw).measure()["dimensions"]
    return tuple(v / 1000.0 for v in d)


def _short(name: str) -> str:
    n = re.sub(r"\s*\([^)]*\)", "", name)                 # drop the parentheses, then everything after a comma
    return n.split(",")[0].replace("parallel-jaw gripper", "parallel gripper")


def _family_row(pl, role, cad_cls=None, cad_kw=None, saw_mount=None):
    import pekari_arms as pa
    import pekari_family as pf

    s = pf.summary(pl)
    g = pf.geometry(pl)
    W = s["mass_loaded_kg"] * G
    dn = drive_numbers(W, g, pl.drive, pl.internal_f0)
    budget = pf.mass_budget(pl)
    chassis = sum(v for k, v in budget.items() if not k.startswith(("arm '", "stabilizer '")))
    main = pl.mounts[0].arm
    dims = _cad_dims(cad_cls, **(cad_kw or {})) if cad_cls else None
    dims = dims or s["overall_m"]
    lift = f"{main.rated_payload_kg:g} kg at {main.rated_reach:g} m" if main.rated_payload_kg else "— (a saw)"
    saw = "—"
    if main.saw:
        c = pa.cut_check(main, 0.95 * pa.reach(main), dynamic=1.5)
        sv = cutting.get_saw(main.saw)
        saw = (f"{sv.power_W / 1000:g} kW, {sv.bar_length * 100:.0f} cm bar (≤ {c['max_branch_m'] * 100:.0f} cm a pass); "
               f"wrist SF {c['sf_stall']['wrist_pitch']:.2f}, elbow {c['sf_stall']['elbow']:.2f}")
    tools = " + ".join(_short(m.arm.gripper) for m in pl.mounts)
    st = pf.stability(pl, payload_kg=pl.payload_kg)
    lo, hi = pf.bom_total(pl)
    return {
        "role": role, "empty_kg": s["mass_empty_kg"], "chassis_kg": chassis, "payload_kg": pl.payload_kg,
        "payload_range": f"{pl.payload_range_kg[0]:g}–{pl.payload_range_kg[1]:g} kg", "loaded_kg": s["mass_loaded_kg"],
        "L_m": dims[0], "W_m": dims[1], "H_m": dims[2],
        "tracks": f"{pl.chassis['track_width']:g} mm, {int(pl.chassis['road_wheels'])} wheels/side"
                  + (", rubber band" if pl.band_track_kg_m else ", PA6 links"),
        "drive": pl.drive, **dn, "battery_Wh": pl.battery_Wh,
        "drive_endurance_h": pl.battery_Wh * USABLE / (dn["traction_W"] + pl.electronics_W),
        "arms": len(pl.mounts), "main_dof": pa.arm_dof(main), "main_reach_m": pa.reach(main), "main_lift": lift,
        "tools": tools, "saw": saw, "stabilizers": len(pl.stabilizers), "tip_deg": st["tip_deg"],
        "parts_EUR": f"{lo / 1000:.0f}–{hi / 1000:.0f} k",
    }


def _pekari_rover_row():
    import pekari_rover_robot as prr

    g = prr.geometry()
    c = prr.cg()
    W = c["loaded"]["mass_kg"] * G
    dn = drive_numbers(W, g, prr.DRIVE)
    dims = None
    try:
        import pekari_rover
        dims = _cad_dims(pekari_rover.PekariRover)
    except ImportError:
        pass
    ov = g["overall"]
    dims = dims or (ov["length"], ov["width"], ov["height"])
    x, z = c["loaded"]["x"], c["loaded"]["z"]
    xs = g["wheel_x"]
    mid = sum(xs) / len(xs)
    margin = min(mid + g["L"] / 2 - x, x - (mid - g["L"] / 2), g["track_center_y"] + g["b"] / 2)
    battery = 240.0
    return {
        "role": "small tracked carrier (notebook 30), no arm", "empty_kg": c["empty"]["mass_kg"],
        "chassis_kg": c["empty"]["mass_kg"], "payload_kg": prr.PAYLOAD_KG, "payload_range": f"{prr.PAYLOAD_KG:g} kg",
        "loaded_kg": c["loaded"]["mass_kg"], "L_m": dims[0], "W_m": dims[1], "H_m": dims[2],
        "tracks": f"{prr.DESIGN['track_width']:g} mm, {int(prr.DESIGN['road_wheels'])} wheels/side, PA6 links",
        "drive": prr.DRIVE, **dn, "battery_Wh": battery,
        "drive_endurance_h": battery * USABLE / (dn["traction_W"] + PEKARI_ELECTRONICS_W),
        "arms": 0, "main_dof": 0, "main_reach_m": 0.0, "main_lift": "—", "tools": "basket", "saw": "—",
        "stabilizers": 0, "tip_deg": math.degrees(math.atan2(margin, z)), "parts_EUR": "not estimated",
    }


def _perseus_row():
    import perseus_robot as pbr
    import pekari_rover_robot as prr

    v = pbr.PERSEUS
    p = pbr.design_params()
    cp = pbr.chassis_params(p)
    g = prr.geometry(cp)
    m = pbr.total_mass(p)
    dn = drive_numbers(m * G, g, v.drive, v.internal_f0)
    arm_kg = sum(pbr.arm_masses().values())
    dims = None
    try:
        import perseus_rover
        st = v.stow
        dims = _cad_dims(perseus_rover.PerseusRover, boom_deg=st["boom"], stick_deg=st["stick"], head_deg=st["head"])
    except ImportError:
        pass
    ov = g["overall"]
    dims = dims or (ov["length"], ov["width"], ov["height"])
    tip = pbr.tip_over()
    c = pbr.cg()
    margin = min(tip["margin_front_m"], tip["margin_rear_m"], tip["margin_side_m"])
    q = pbr.elevations_to_q(0.0, 0.0, 0.0)                         # outstretched, bar level
    t = pbr.arm_torques(q, load=v.cut_loads(dynamic=1.5))
    saw = cutting.get_saw(v.saw)
    ag = pbr.arm_geometry(p)
    return {
        "role": "branch-cutting rover (notebook 37), a saw boom", "empty_kg": m, "chassis_kg": m - arm_kg,
        "payload_kg": 0.0, "payload_range": "— (no deck)", "loaded_kg": m, "L_m": dims[0], "W_m": dims[1],
        "H_m": dims[2], "tracks": f"{cp['track_width']:g} mm, {int(cp['road_wheels'])} wheels/side, PA6 links",
        "drive": v.drive, **dn, "battery_Wh": v.battery_Wh,
        "drive_endurance_h": v.battery_Wh * USABLE / (dn["traction_W"] + v.electronics_W),
        "arms": 1, "main_dof": 4, "main_reach_m": ag["Lb"] + ag["Ls"] + ag["Lh"] + ag["Lbar"] / 2,
        "main_lift": "— (a saw)", "tools": v.saw + " (fixed)",
        "saw": f"{saw.power_W / 1000:g} kW, {saw.bar_length * 100:.0f} cm bar (≤ {cutting.max_branch_for_bar(saw, 1) * 100:.0f} cm "
               f"a pass); wrist SF {t['arm_wrist']['sf_stall']:.2f}, elbow {t['arm_elbow']['sf_stall']:.2f}",
        "stabilizers": 0, "tip_deg": math.degrees(math.atan2(margin, c["z"])), "parts_EUR": "not estimated",
    }


def rows(cad: bool = True) -> dict:
    """{machine: {metric: value}} for ``MACHINES`` (``cad`` False: the parametric envelopes, no CadQuery)."""
    import calydon
    import calydon_robot as cr
    import pekari_2m
    import pekari_2m_robot as p2
    import pumba
    import pumba_robot as pu

    def c(cls):
        return cls if cad else None
    return {
        "Pekari Rover": _pekari_rover_row(),
        "Pekari-2M": _family_row(p2.PEKARI_2M, "light dual-arm response platform", c(pekari_2m.Pekari2M)),
        "Pumba": _family_row(pu.PUMBA, "WILK-class dual-arm platform, the grab", c(pumba.Pumba)),
        "Pumba + chainsaw": _family_row(pu.PUMBA_SAW, "Pumba with Perseus's saw on the main arm", c(pumba.Pumba),
                                        {"tool": "chainsaw 2 kW"}),
        "Perseus": _perseus_row(),
        "Calydon": _family_row(cr.CALYDON, "heavy-lift dual-arm platform (the first Pumba)", c(calydon.Calydon)),
    }


LABELS = [("role", "role", "{}"), ("empty_kg", "mass empty [kg]", "{:.0f}"),
          ("chassis_kg", "  without arms / tools / stabilizers [kg]", "{:.0f}"),
          ("payload_range", "deck payload", "{}"), ("loaded_kg", "mass with nominal payload [kg]", "{:.0f}"),
          ("L_m", "length [m] (arms folded / boom stowed)", "{:.2f}"), ("W_m", "width [m]", "{:.2f}"),
          ("H_m", "height [m]", "{:.2f}"), ("tracks", "tracks", "{}"), ("drive", "track drives (x2)", "{}"),
          ("v_top_kmh", "top speed [km/h]", "{:.1f}"), ("climb20", "20° grade loaded (peak, traction)", "{}"),
          ("grade_cont_deg", "steepest grade held continuously (drive rating, traction) [°]", "{:.0f}"),
          ("ground_kPa", "ground pressure loaded [kPa]", "{:.1f}"), ("battery_Wh", "battery [Wh]", "{:.0f}"),
          ("drive_endurance_h", "driving endurance, 0.8 m/s flat gravel [h]", "{:.1f}"),
          ("arms", "arms", "{}"), ("main_dof", "main arm DOF", "{}"), ("main_reach_m", "main arm reach to TCP [m]", "{:.2f}"),
          ("main_lift", "main arm lift rating (peak / 1.5)", "{}"), ("tools", "tools fitted", "{}"),
          ("saw", "saw (stall SF cutting outstretched, loads x1.5)", "{}"), ("stabilizers", "stabilizer legs", "{}"),
          ("tip_deg", "static tip angle, loaded, folded [°]", "{:.0f}"), ("parts_EUR", "parts cost (class estimate)", "{}")]


def table(cad: bool = True):
    """The comparison as a pandas DataFrame: metrics as rows, machines as columns (formatted strings)."""
    import pandas as pd

    r = rows(cad)
    data = {m: [fmt.format(r[m][k]) if not isinstance(r[m][k], bool) else ("yes" if r[m][k] else "no")
                for k, _, fmt in LABELS] for m in MACHINES}
    return pd.DataFrame(data, index=[lab for _, lab, _ in LABELS])


def markdown(cad: bool = True) -> str:
    df = table(cad)
    head = "| | " + " | ".join(df.columns) + " |\n|---|" + "---|" * len(df.columns) + "\n"
    return head + "".join(f"| {i} | " + " | ".join(df.loc[i]) + " |\n" for i in df.index)
