"""Propulsor maps — notebook 25's propellers and ducted fans as tables for the reduced flight models (MERLIN, notebooks 26, 27).

The propeller notebook (25) does the propulsion analysis; this module is how it hands the results on. Every propulsor is
solved once, over airspeed x rpm, and written to one JSON library (``data/propulsor_maps.json``); the flight models only
interpolate in it (``unit``), so a race over hundreds of designs takes seconds.

- **Open propellers** (two, six or twelve of notebook 25's blades, the planform scaled with the diameter): Boreas BEMT in the free stream —
  thrust and shaft power over (V, rpm) — plus, per layout on MERLIN, the installation from ``air_propeller``'s models
  (``merlin_flight.installation``): the effective wake fraction w (the blades work at ``V (1 - w)``) and the thrust
  deduction t (the net thrust is ``T (1 - t)``), both at MERLIN's 25 m/s cruise.
- **Ducted fans**, one, two or three stages (``boreas.ducted``, ``stages``): notebook 25's 90 mm, 12-blade fan with its
  stators, scaled to the diameter (``merlin.edf_housing_params``: hub, chords, stators, lip and duct), the pitch, the
  nozzle exit area and the duct's build quality (``duct_loss``: the catalogue fit or a well-made fan); each extra stage
  lengthens the duct by a rotor and a stator row (its outside friction) and adds an interstage loss. The table holds the
  unit's thrust less the nacelle's friction, the shaft power and the jet's exit velocity (for the jet scrubbing the
  fuselage behind it, which depends on the aircraft: ``unit``'s ``scrub_area_m2``).
- **Masses** of a propulsion unit at a rated electrical power (``unit_mass_kg``), anchored on notebook 26's units at
  1900 W: the fan's rotor + duct + stators 120 g x (D / 90 mm)^2 and 45 g x (D / 90 mm)^2 per extra stage, its motor
  200 g x (P / 1900 W)^0.75, its controller 85 g x P / 1900 W (405 g at 90 mm); a propeller unit's motor
  190 g x (P / 1900 W)^0.75, controller 65 g x P / 1900 W, propeller 18 g x (D / 10 in)^2.5 x blades / 2 (273 g).

``build_library`` (parallel) → ``save`` → ``load`` → ``unit(entry, power, layout=...)`` → ``merlin_flight.Unit``.
Regenerate the library with notebook 25 §16 or ``python scenarios/propulsor_maps.py``.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np

LIBRARY = Path(__file__).resolve().parent / "data" / "propulsor_maps.json"
SECTION_KW = dict(name="thin cambered section", cl_alpha=2 * math.pi * 0.9, alpha0_deg=-3.0, cl_max=1.1, cd0=0.018, k=0.04)
V_TABLE = np.linspace(0.0, 90.0, 19)
CATALOGUE_DUCT_LOSS = 0.3353          # fitted to a 90 mm, 12-blade hobby fan: 3.0 kgf from 1930 W (notebooks 25, 26)
WELL_MADE_DUCT_LOSS = 0.20
EDF_SPACE = {"diameter_mm": (70.0, 80.0, 90.0, 100.0, 110.0, 120.0), "pitch_ratio": (1.4, 1.78, 2.2),
             "exit_area_ratio": (0.65, 0.8, 0.9), "duct_loss": (CATALOGUE_DUCT_LOSS, WELL_MADE_DUCT_LOSS), "stages": (1, 2, 3)}
PROP_SPACE = {"diameter_in": (9.0, 10.0, 11.0, 12.0), "pitch_ratio": (0.6, 0.8, 1.0, 1.2), "blades": (2, 6, 12)}
# the upper bound, not a design: one-stage fans whose duct loses nothing (no inlet, wall or nozzle loss at all)
EDF_BOUND = {"diameter_mm": EDF_SPACE["diameter_mm"], "pitch_ratio": EDF_SPACE["pitch_ratio"], "exit_area_ratio": (0.65, 0.8),
             "duct_loss": (0.0,), "stages": (1,)}
EDF_TIP_SPEED = 240.0                 # m/s at the blade tips: the fan's rpm limit
PROP_TIP_MACH = 0.75
INSTALLATION_SPEED = 25.0             # m/s: MERLIN's cruise, where w and t are taken
DRIVE_EFFICIENCY = 0.85


def section():
    from vegeta import boreas
    return boreas.Airfoil(**SECTION_KW)


# ------------------------------------------------------------------------------------------------- the objects
def edf_params(diameter_mm, pitch_ratio, exit_area_ratio):
    import merlin as m
    return m.Merlin().resolve(propulsion="edf", edf_diameter=float(diameter_mm), edf_pitch=float(pitch_ratio * diameter_mm),
                              edf_exit_area_ratio=float(exit_area_ratio))


def fan_object(diameter_mm, pitch_ratio, exit_area_ratio, duct_loss, stages):
    """The ``boreas.ducted.DuctedFan`` of one fan design (no solving), with its housing geometry."""
    import ducted_fan as df
    import merlin as m
    from vegeta import boreas
    from vegeta.boreas import ducted
    hp = m.edf_housing_params(edf_params(diameter_mm, pitch_ratio, exit_area_ratio))
    hg = df.housing_geometry(hp)
    stage_len = (hp["hub_height"] + hp["stator_gap"] + hp["stator_chord"]) / 1000          # m: one more rotor + stator row
    extra = (stages - 1) * stage_len
    blade = boreas.Propeller.from_pitch(f"EDF {hp['diameter']:.0f} mm, 12 blades", hp["diameter"] / 1000, hp["pitch"] / 1000, blades=12,
                                        chord_root_m=hp["chord_root"] / 1000, chord_max_m=hp["chord_max"] / 1000,
                                        chord_tip_m=hp["chord_tip"] / 1000, hub_radius_m=hp["hub_diameter"] / 2000)
    fan = ducted.DuctedFan(blade.name, blade, tip_clearance_m=hp["tip_clearance"] / 1000, exit_area_ratio=hg["exit_area_ratio"],
                           stator_vanes=int(hp["stator_vanes"]), stator_loss=0.10, duct_loss=float(duct_loss),
                           external_wetted_area_m2=hg["external_wetted_area"] * 1e-6 + 2 * math.pi * hg["nacelle_radius"] / 1000 * extra,
                           duct_length_m=hg["total_length"] / 1000 + extra, stages=int(stages))
    return fan, hp, hg


def prop_object(diameter_in, pitch_ratio, blades=2):
    """An open propeller: notebook 25's blade (planform scaled with the diameter), ``blades`` of it on one hub."""
    from vegeta import boreas
    D = diameter_in * 0.0254
    return boreas.Propeller.from_pitch(f"{diameter_in:.0f}x{diameter_in * pitch_ratio:.0f}", D, pitch_ratio * D, blades=int(blades),
                                       chord_root_m=0.018 * D / 0.254, chord_max_m=0.026 * D / 0.254, chord_tip_m=0.010 * D / 0.254,
                                       hub_radius_m=0.011)


# ------------------------------------------------------------------------------------------------- the entries
def quality(duct_loss):
    """The name of a duct's build quality: ``catalogue`` (the fitted hobby fan), ``well-made``, ``lossless`` (the bound)."""
    if abs(duct_loss - CATALOGUE_DUCT_LOSS) < 1e-6:
        return "catalogue"
    if abs(duct_loss - WELL_MADE_DUCT_LOSS) < 1e-6:
        return "well-made"
    return "lossless" if duct_loss == 0 else f"duct loss {duct_loss:g}"



def edf_entry(diameter_mm, pitch_ratio, exit_area_ratio, duct_loss, stages, V=V_TABLE, n_rpm=15) -> dict:
    """One fan solved over (V, rpm): ``thrust`` (the unit's, less the nacelle's friction), ``power`` (shaft), ``exit_velocity``."""
    from vegeta.boreas import ducted
    fan, hp, hg = fan_object(diameter_mm, pitch_ratio, exit_area_ratio, duct_loss, stages)
    sec = section()
    rpm_max = EDF_TIP_SPEED / (math.pi * hp["diameter"] / 1000) * 60
    rpm = rpm_max * (0.1 + 0.9 * np.linspace(0.0, 1.0, n_rpm) ** 1.3)   # from 10 % (a low cruise) up, denser at the low end
    T, P, Ve = (np.zeros((len(V), len(rpm))) for _ in range(3))
    for i, v in enumerate(V):
        for j, n in enumerate(rpm):
            e = ducted.solve(fan, sec, n, v)
            T[i, j], P[i, j], Ve[i, j] = e.thrust - ducted.nacelle_drag(fan, v), e.power, e.exit_velocity
    q = quality(duct_loss)
    return {"kind": "edf", "id": f"edf-{diameter_mm:.0f}-p{pitch_ratio:.2f}-e{exit_area_ratio:.2f}-{q}-s{stages}",
            "label": f"EDF {diameter_mm:.0f} mm, {stages} stage{'s' if stages > 1 else ''}, pitch {pitch_ratio:.2f} D, exit {exit_area_ratio:.2f}, {q}",
            "diameter_mm": diameter_mm, "pitch_ratio": pitch_ratio, "exit_area_ratio": exit_area_ratio, "duct_loss": duct_loss,
            "quality": q, "stages": stages, "rpm_max": rpm_max, "fan_area_m2": fan.fan_area, "duct_length_m": fan.duct_length_m,
            "V": V, "rpm": rpm, "thrust": T, "power": P, "exit_velocity": Ve}


def prop_entry(diameter_in, pitch_ratio, blades=2, V=V_TABLE, n_rpm=20, merlin=None) -> dict:
    """One propeller in the free stream over (V, rpm), and its installation on MERLIN as tractor and pusher."""
    import merlin_flight as mf
    from vegeta import boreas
    prop, sec = prop_object(diameter_in, pitch_ratio, blades), section()
    rpm_max = PROP_TIP_MACH * 340.0 / (math.pi * prop.diameter) * 60
    rpm = np.linspace(0.1 * rpm_max, rpm_max, n_rpm)
    T, P = np.zeros((len(V), len(rpm))), np.zeros((len(V), len(rpm)))
    for i, v in enumerate(V):
        for j, n in enumerate(rpm):
            op = boreas.solve(prop, sec, n, v)
            T[i, j], P[i, j] = op.thrust, op.power
    inst = {}
    for layout in ("tractor", "pusher"):
        af = merlin_airframe(layout, 1900.0, {"kind": "propeller", "diameter_in": diameter_in, "blades": blades}, merlin)
        r = mf.installation(layout, dict(merlin or {}), prop, sec, INSTALLATION_SPEED, float(af.drag(INSTALLATION_SPEED)))
        inst[layout] = {"w": r["w"], "t": r["t"]}
    return {"kind": "propeller", "id": f"prop-{diameter_in:.0f}x{diameter_in * pitch_ratio:.0f}-b{int(blades)}",
            "label": f"{prop.name} in, {int(blades)} blades", "diameter_in": diameter_in, "pitch_ratio": pitch_ratio, "blades": int(blades), "rpm_max": rpm_max, "V": V, "rpm": rpm,
            "thrust": T, "power": P, "installation": inst}


def _build(job):
    kind, args = job
    try:
        return edf_entry(*args) if kind == "edf" else prop_entry(*args)
    except Exception as exc:                                     # recorded, not raised: one bad design must not stop the rest
        return {"kind": kind, "id": f"{kind}-{args}", "error": repr(exc)}


def build_library(edf_space=EDF_SPACE, prop_space=PROP_SPACE, processes=None, progress=False, edf_bound=EDF_BOUND) -> dict:
    """Every fan and propeller of the spaces (and the lossless bound fans, ``edf_bound``; None leaves them out), solved in
    parallel: ``{"meta": ..., "entries": [...]}``."""
    import multiprocessing as mp
    jobs = [("edf", c) for c in itertools.product(*edf_space.values())] + [("prop", c) for c in itertools.product(*prop_space.values())]
    if edf_bound:
        jobs += [("edf", c) for c in itertools.product(*edf_bound.values()) if ("edf", c) not in jobs]
    with mp.get_context("fork").Pool(processes) as pool:
        it = pool.imap_unordered(_build, jobs)
        if progress:
            from tqdm.auto import tqdm
            it = tqdm(it, total=len(jobs), desc="propulsor maps")
        entries = sorted(it, key=lambda e: e["id"])
    meta = {"source": "notebook 25 (propulsor_maps.build_library)", "section": SECTION_KW, "edf_space": edf_space, "edf_bound": edf_bound,
            "prop_space": prop_space, "edf_tip_speed_m_s": EDF_TIP_SPEED, "prop_tip_mach": PROP_TIP_MACH,
            "installation_speed_m_s": INSTALLATION_SPEED, "rho": 1.225,
            "units": "V m/s, rpm, thrust N (EDF: net of the nacelle's friction; propeller: free stream), power W (shaft), exit_velocity m/s"}
    return {"meta": meta, "entries": entries}


def _round(a, sig=5):
    a = np.asarray(a, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where(a != 0, 10 ** (sig - 1 - np.floor(np.log10(np.abs(np.where(a != 0, a, 1))))), 1.0)
    return (np.round(a * mag) / mag).tolist()


def save(lib: dict, path=LIBRARY) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    out = {"meta": lib["meta"], "entries": []}
    for e in lib["entries"]:
        out["entries"].append({k: (_round(v) if isinstance(v, np.ndarray) else v) for k, v in e.items()})
    path.write_text(json.dumps(out, separators=(",", ":"), default=float))
    return path


def load(path=LIBRARY) -> dict:
    """The library with its tables as arrays; ``entries`` keyed by id in ``by_id``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run notebook 25 §16 or `python scenarios/propulsor_maps.py`")
    lib = json.loads(path.read_text())
    for e in lib["entries"]:
        for k in ("V", "rpm", "thrust", "power", "exit_velocity"):
            if k in e:
                e[k] = np.asarray(e[k], float)
    lib["by_id"] = {e["id"]: e for e in lib["entries"] if "error" not in e}
    return lib


# ------------------------------------------------------------------------------------------------- using an entry
def unit_mass_kg(entry: dict, power_w: float) -> float:
    """The propulsion unit's mass [kg] at a rated electrical power (the module docstring's model)."""
    f = power_w / 1900.0
    if entry["kind"] == "edf":
        k2 = (entry["diameter_mm"] / 90.0) ** 2
        return 0.120 * k2 + 0.045 * k2 * (entry.get("stages", 1) - 1) + 0.200 * f ** 0.75 + 0.085 * f
    return 0.190 * f ** 0.75 + 0.065 * f + 0.018 * (entry["diameter_in"] / 10.0) ** 2.5 * entry.get("blades", 2) / 2


def merlin_airframe(layout, power_w, entry, merlin=None):
    """MERLIN carrying ``entry`` as ``layout`` (edf / tractor / pusher) at ``power_w``: the common masses plus the unit's,
    the drag build-up of that nose (``merlin.drag_buildup``) plus the skid/intake allowance."""
    import merlin as m
    import merlin_flight as mf
    p = dict(merlin or {}, propulsion=layout)
    if entry["kind"] == "edf":
        p.update(edf_diameter=entry["diameter_mm"], edf_pitch=entry["pitch_ratio"] * entry["diameter_mm"], edf_exit_area_ratio=entry["exit_area_ratio"])
    b = m.drag_buildup(p, INSTALLATION_SPEED)
    mass = sum(mf.COMMON_MASS_KG.values()) + unit_mass_kg(entry, power_w)
    return mf.Airframe(f"MERLIN ({entry.get('label', layout)})", mass, b["planform"], b["aspect_ratio"],
                       (b["cd_area_m2"] + mf.EXTRA_CD_AREA_M2) / b["planform"])


def edf_scrub_area(entry, merlin=None) -> float:
    """The fuselage area [m^2] MERLIN's annular jet scrubs behind the nozzle: the fairing and 150 mm of the cylinder."""
    import merlin as m
    p = m.Merlin().resolve(**dict(merlin or {}))
    k = entry["diameter_mm"] / 90.0
    hub = 40.0 * k
    return math.pi * 0.5 * (hub + p["fuselage_diameter"]) / 1000 * p["fairing_length"] / 1000 + math.pi * p["fuselage_diameter"] / 1000 * 0.15 * k


def unit(entry: dict, power_w: float, *, layout=None, scrub_area_m2=None, drive_efficiency=DRIVE_EFFICIENCY, merlin=None, nu=1.5e-5,
         rho=1.225):
    """A ``merlin_flight.Unit`` from a library entry at the electrical limit ``power_w`` — no solving, only interpolation:

    - a propeller as ``layout`` (tractor or pusher): its free-stream table read at ``V (1 - w)``, the thrust times ``1 - t``;
    - a fan: its table less the jet scrubbing ``scrub_area_m2`` (default: MERLIN's, ``edf_scrub_area``) at the jet's extra
      dynamic pressure ``1/2 rho (V_exit^2 - V^2)`` with turbulent friction on 0.3 m (as ``merlin_flight.ducted_fan_unit``)."""
    import merlin_flight as mf
    V, rpm, T, P = entry["V"], entry["rpm"], np.array(entry["thrust"]), np.array(entry["power"])
    if entry["kind"] == "propeller":
        if layout not in ("tractor", "pusher"):
            raise ValueError("a propeller needs layout='tractor' or 'pusher'")
        w, t = entry["installation"][layout]["w"], entry["installation"][layout]["t"]
        Va = V * (1 - w)
        T = np.array([[np.interp(va, V, T[:, j]) for j in range(len(rpm))] for va in Va]) * (1 - t)
        P = np.array([[np.interp(va, V, P[:, j]) for j in range(len(rpm))] for va in Va])
        notes = f"library map {entry['id']}, {layout}: w = {w:.4f}, t = {t:.4f}"
        name = f"{layout} propeller {entry['label'].split(' in')[0]}, {entry.get('blades', 2)} blades"
    else:
        area = edf_scrub_area(entry, merlin) if scrub_area_m2 is None else scrub_area_m2
        vj = np.maximum(entry["exit_velocity"], V[:, None])
        cf = 0.455 / np.log10(np.maximum(vj * 0.3 / nu, 1e4)) ** 2.58
        T = T - cf * area * 0.5 * rho * (vj ** 2 - V[:, None] ** 2)
        notes = f"library map {entry['id']}, scrub area {area:.4f} m^2"
        name = entry["label"]
    T = np.maximum.accumulate(T, axis=1) + np.arange(len(rpm)) * 1e-9          # monotone in rpm (invertible)
    return mf.Unit(name, np.asarray(V, float), np.asarray(rpm, float), T, P, unit_mass_kg(entry, power_w), power_w,
                   drive_efficiency, notes)


def find(lib, **kw):
    """The entries whose fields match ``kw`` (floats to 1e-6)."""
    def ok(e):
        return all(abs(e.get(k, math.nan) - v) < 1e-6 if isinstance(v, float) else e.get(k) == v for k, v in kw.items())
    return [e for e in lib["entries"] if "error" not in e and ok(e)]


__all__ = ["LIBRARY", "SECTION_KW", "V_TABLE", "EDF_SPACE", "EDF_BOUND", "PROP_SPACE", "quality", "CATALOGUE_DUCT_LOSS", "WELL_MADE_DUCT_LOSS",
           "fan_object", "prop_object", "edf_entry", "prop_entry", "build_library", "save", "load", "unit_mass_kg",
           "merlin_airframe", "edf_scrub_area", "unit", "find"]
