"""MERLIN's propulsor race (notebook 27, branch ``dev_rave``): can a reasonable ducted fan beat the propellers on the same
battery?

Two races, both from the launch stand, still air, the same 6S 8000 mAh pack and its 15 % reserve for everyone:

- **5 km and back** (``round_trip``): climb to 150 m, dash out, a 60-degree-bank turn over the point, dash back; the
  clock stops over the launch site (the landing that follows is the same for all and is left out);
- **10 km** (``merlin_flight.race``): the reach — launch to overhead a fire 10 km away — keeping the energy to sample for
  three minutes and fly home; the 10 km out-and-back is reported as well.

Each propulsor may change what a designer would reasonably change, inside ``EDF_SPACE`` / ``PROP_SPACE``:

- the **ducted fan**: diameter 70–120 mm (the rotor, hub, stators, lip and duct scale with it: ``merlin.edf_housing_params``),
  blade pitch 1.4–2.2 diameters, nozzle exit 65–90 % of the fan annulus, and the duct's build quality — the hobby
  catalogue fan's fitted loss (0.34, notebook 25) or a well-made fan (0.20: a smooth bell-mouth, tight tips, a measured
  unit);
- the **propellers**: tractor or pusher, 9–12 inch, pitch 0.8–1.2 diameters (the installation's wake and thrust
  deduction recomputed for each, ``merlin_flight.installation``);
- **both**: the electrical power the pack is asked for — 1900, 2700 or 3500 W (about 11, 15 and 20 C) — with motors and
  controllers that grow with it (``unit_mass_kg``).

A ``Unit`` table does not depend on the power limit (the limit picks the full-throttle point on it), so every fan and
propeller is tabulated once and raced at all three powers. The airframe stays MERLIN's: its mass is the common items plus
the propulsion unit's.

Mass models (``unit_mass_kg``), anchored on notebook 26's units at 1900 W: the EDF's rotor + duct + stators 120 g x
(D / 90 mm)^2, its motor 200 g x (P / 1900 W)^0.75, its controller 85 g x P / 1900 W (405 g at 90 mm and 1900 W); a
propeller unit's motor 190 g x (P / 1900 W)^0.75, controller 65 g x P / 1900 W, propeller 18 g x (D / 10 in)^2.5 (273 g).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace

import numpy as np

import merlin_flight as mf

POWERS_W = (1900.0, 2700.0, 3500.0)
CATALOGUE_DUCT_LOSS = 0.3353          # notebook 25 / 26: fitted to 3.0 kgf from 1930 W (a hobby 90 mm, 12-blade fan)
WELL_MADE_DUCT_LOSS = 0.20
EDF_SPACE = {"diameter_mm": (70.0, 80.0, 90.0, 100.0, 110.0, 120.0), "pitch_ratio": (1.4, 1.78, 2.2),
             "exit_area_ratio": (0.65, 0.8, 0.9), "duct_loss": (CATALOGUE_DUCT_LOSS, WELL_MADE_DUCT_LOSS)}
PROP_SPACE = {"layout": ("tractor", "pusher"), "diameter_in": (9.0, 10.0, 11.0, 12.0), "pitch_ratio": (0.8, 1.0, 1.2)}
EDF_TIP_SPEED = 240.0                  # m/s: the fan's rpm limit (hobby EDFs run 200-250 m/s at the tips)
PROP_TIP_MACH = 0.75
V_TABLE = np.linspace(0.0, 90.0, 19)


@dataclass(frozen=True)
class Config:
    kind: str                          # "edf", "tractor", "pusher"
    diameter: float                    # mm (EDF) or inch (propeller)
    pitch_ratio: float
    exit_area_ratio: float = 0.0       # EDF only
    duct_loss: float = 0.0             # EDF only

    @property
    def label(self):
        if self.kind == "edf":
            q = "catalogue" if abs(self.duct_loss - CATALOGUE_DUCT_LOSS) < 1e-6 else "well-made"
            return f"EDF {self.diameter:.0f} mm, pitch {self.pitch_ratio:.2f} D, exit {self.exit_area_ratio:.2f}, {q}"
        return f"{self.kind} {self.diameter:.0f}x{self.diameter * self.pitch_ratio:.0f} in"


def unit_mass_kg(cfg: Config, power_w: float) -> float:
    """The propulsion unit's mass [kg] at an electrical power rating (see the module docstring)."""
    f = power_w / 1900.0
    if cfg.kind == "edf":
        return 0.120 * (cfg.diameter / 90.0) ** 2 + 0.200 * f ** 0.75 + 0.085 * f
    return 0.190 * f ** 0.75 + 0.065 * f + 0.018 * (cfg.diameter / 10.0) ** 2.5


def merlin_params(cfg: Config) -> dict:
    """``merlin.Merlin`` parameters of the aircraft that carries ``cfg``."""
    if cfg.kind == "edf":
        return {"propulsion": "edf", "edf_diameter": cfg.diameter, "edf_pitch": cfg.pitch_ratio * cfg.diameter,
                "edf_exit_area_ratio": cfg.exit_area_ratio}
    return {"propulsion": cfg.kind}


def build_unit(cfg: Config, V=V_TABLE, n_rpm=10):
    """The ``merlin_flight.Unit`` of ``cfg`` (its power limit is set later, per race), and notes."""
    import merlin as m
    import ducted_fan as df
    from vegeta import boreas
    from vegeta.boreas import ducted
    section = boreas.Airfoil(**mf.SECTION_KW)
    p = m.Merlin().resolve(**merlin_params(cfg))
    if cfg.kind == "edf":
        hp = m.edf_housing_params(p)
        hg = df.housing_geometry(hp)
        k = hp["diameter"] / 90.0
        blade = boreas.Propeller.from_pitch(cfg.label, hp["diameter"] / 1000, hp["pitch"] / 1000, blades=12,
                                            chord_root_m=hp["chord_root"] / 1000, chord_max_m=hp["chord_max"] / 1000,
                                            chord_tip_m=hp["chord_tip"] / 1000, hub_radius_m=hp["hub_diameter"] / 2000)
        fan = ducted.DuctedFan(blade.name, blade, tip_clearance_m=hp["tip_clearance"] / 1000, exit_area_ratio=hg["exit_area_ratio"],
                               stator_vanes=int(hp["stator_vanes"]), stator_loss=0.10, duct_loss=cfg.duct_loss,
                               external_wetted_area_m2=hg["external_wetted_area"] * 1e-6, duct_length_m=hg["total_length"] / 1000)
        scrub = math.pi * 0.5 * (hp["hub_diameter"] + p["fuselage_diameter"]) / 1000 * p["fairing_length"] / 1000 \
            + math.pi * p["fuselage_diameter"] / 1000 * 0.15 * k
        rpm_max = EDF_TIP_SPEED / (math.pi * hp["diameter"] / 1000) * 60
        u = mf.ducted_fan_unit(cfg.label, fan, section, mass_kg=0.0, max_electrical_w=1900.0, rpm_max=rpm_max,
                               scrub_area_m2=scrub, V=V, n_rpm=n_rpm)
        return u, {"rpm_max": rpm_max, "exit_area_ratio_built": hg["exit_area_ratio"], "nacelle_radius_mm": hg["nacelle_radius"]}
    D = cfg.diameter * 0.0254
    prop = boreas.Propeller.from_pitch(cfg.label, D, cfg.pitch_ratio * D, blades=2, chord_root_m=0.018 * D / 0.254,
                                       chord_max_m=0.026 * D / 0.254, chord_tip_m=0.010 * D / 0.254, hub_radius_m=0.011)
    af = airframe(cfg, 1900.0)
    inst = mf.installation(cfg.kind, p, prop, section, 25.0, float(af.drag(25.0)))
    rpm_max = PROP_TIP_MACH * 340.0 / (math.pi * D) * 60
    u = mf.open_propeller(cfg.label, prop, section, wake_fraction=inst["w"], thrust_deduction=inst["t"], mass_kg=0.0,
                          max_electrical_w=1900.0, rpm_max=rpm_max, V=V, n_rpm=max(n_rpm, 16))
    return u, {"rpm_max": rpm_max, "w": inst["w"], "t": inst["t"]}


def airframe(cfg: Config, power_w: float) -> mf.Airframe:
    """MERLIN carrying ``cfg`` at ``power_w``: the common masses plus the unit's, the drag build-up of that nose."""
    import merlin as m
    b = m.drag_buildup(merlin_params(cfg), 25.0)
    mass = sum(mf.COMMON_MASS_KG.values()) + unit_mass_kg(cfg, power_w)
    return mf.Airframe(f"MERLIN ({cfg.label})", mass, b["planform"], b["aspect_ratio"],
                       (b["cd_area_m2"] + mf.EXTRA_CD_AREA_M2) / b["planform"])


def round_trip(af: mf.Airframe, unit: mf.Unit, distance_m: float, mission: mf.Mission = mf.Mission(), *, bank_deg=60.0,
               rho=mf.RHO) -> dict:
    """Out to a point ``distance_m`` away and back over the launch site: the launch and climb as ``merlin_flight.race``,
    a full-power acceleration, the dash out, a turn at ``bank_deg`` (radius ``V^2 / (g tan bank)``, load factor
    ``1 / cos bank``, at full power), the dash back. The dash speed is the highest (up to the top speed) the energy
    allows with the reserve kept. Returns the time [s] (launch to over the launch site), the dash speed, what limited it
    and the energy."""
    pf = mf.performance(af, unit, rho)
    W = af.mass_kg * mf.G
    v_c = pf["v_climb"]
    T_c, P_c, _ = unit.full(v_c)
    roc = (T_c - af.drag(v_c, rho)) * v_c / W
    if roc <= 0:
        return {"reachable": False, "why": "cannot climb"}
    t_climb = mission.sampling_agl_m / roc
    x_climb, e_climb = v_c * t_climb, P_c * t_climb / 3600
    t_l, x_l, e_l = mf._accelerate(af, unit, mission.launch_speed, v_c, rho)
    x_dash = 2 * distance_m - x_l - x_climb               # out from the end of the climb, and all the way back
    budget = mission.battery_wh * mission.usable_fraction * (1 - mission.reserve_fraction)
    tb = math.tan(math.radians(bank_deg))

    def lap(v):
        t_a, x_a, e_a = mf._accelerate(af, unit, v_c, v, rho, max_distance=x_dash)
        x = max(x_dash - x_a, 0.0)
        P = unit.electrical_power(v, float(af.drag(v, rho)))
        t_turn = math.pi * v / (mf.G * tb)
        P_turn = min(unit.electrical_power(v, float(af.drag(v, rho, 1.0 / math.cos(math.radians(bank_deg))))), unit.max_electrical_w)
        return t_l + t_climb + t_a + x / v + t_turn, e_l + e_climb + e_a + (P * x / v + P_turn * t_turn) / 3600

    v_hi = 0.995 * pf["v_top"]
    v_lo = min(pf["v_range"], v_hi)
    if lap(v_lo)[1] > budget:
        return {"reachable": False, "why": "not enough energy"}
    if lap(v_hi)[1] <= budget:
        v, limit = v_hi, "thrust"
    else:
        lo, hi = v_lo, v_hi
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if lap(mid)[1] <= budget else (lo, mid)
        v, limit = lo, "energy"
    t, e = lap(v)
    return {"reachable": True, "time_s": t, "dash_speed": v, "top_speed": pf["v_top"], "limit": limit, "energy_wh": e,
            "budget_wh": budget, "turn_s": math.pi * v / (mf.G * tb)}


def configs(edf_space=EDF_SPACE, prop_space=PROP_SPACE):
    out = [Config("edf", d, pr, ex, dl) for d, pr, ex, dl in itertools.product(*edf_space.values())]
    out += [Config(lay, d, pr) for lay, d, pr in itertools.product(*prop_space.values())]
    return out


def _build(cfg):
    try:
        return cfg, build_unit(cfg)
    except Exception as exc:                                    # a configuration that cannot run (recorded, not raised)
        return cfg, (None, {"error": repr(exc)})


def tabulate(cfgs, processes=None, progress=False):
    """``{Config: (Unit, notes)}`` for every configuration, built in parallel (the fan tables take a few seconds each)."""
    import multiprocessing as mp
    with mp.get_context("fork").Pool(processes) as pool:
        it = pool.imap_unordered(_build, cfgs)
        if progress:
            from tqdm.auto import tqdm
            it = tqdm(it, total=len(cfgs), desc="propulsor tables")
        return dict(it)


def study(units, powers=POWERS_W, mission: mf.Mission = mf.Mission()):
    """Every configuration at every power through the races: a table (pandas) with one row per (configuration, power)."""
    import pandas as pd
    rows = []
    for cfg, (u, notes) in units.items():
        if u is None:
            continue
        for P in powers:
            unit = replace(u, max_electrical_w=P, mass_kg=unit_mass_kg(cfg, P))
            af = airframe(cfg, P)
            pf = mf.performance(af, unit)
            r5 = round_trip(af, unit, 5000.0, mission)
            r10 = mf.race(af, unit, 10000.0, mission)
            r10b = round_trip(af, unit, 10000.0, mission)
            rows.append({"kind": cfg.kind, "config": cfg.label, "power [W]": P, "mass [kg]": af.mass_kg,
                         "static thrust [N]": unit.full(0.0)[0], "top speed [m/s]": pf["v_top"],
                         "energy [Wh/km]": pf["wh_per_km_best"],
                         "5 km and back [s]": r5["time_s"] if r5["reachable"] else math.nan,
                         "5 km and back: dash [m/s]": r5.get("dash_speed", math.nan), "5 km and back: limit": r5.get("limit", r5.get("why")),
                         "10 km reach [s]": r10["time_to_fire_s"] if r10["reachable"] else math.nan,
                         "10 km reach: limit": r10.get("limit", r10.get("why")),
                         "10 km and back [s]": r10b["time_s"] if r10b["reachable"] else math.nan,
                         "diameter": cfg.diameter, "pitch ratio": cfg.pitch_ratio, "exit area ratio": cfg.exit_area_ratio or math.nan,
                         "duct loss": cfg.duct_loss or math.nan, **{f"note {k}": v for k, v in notes.items()}})
    return pd.DataFrame(rows)


RACES = ("5 km and back [s]", "10 km reach [s]", "10 km and back [s]")


def winners(table, race, by=("kind", "power [W]")):
    """The fastest configuration of each kind (and power) in ``race``."""
    t = table.dropna(subset=[race])
    return t.loc[t.groupby(list(by))[race].idxmin()].set_index(list(by)).sort_index()


__all__ = ["Config", "POWERS_W", "EDF_SPACE", "PROP_SPACE", "CATALOGUE_DUCT_LOSS", "WELL_MADE_DUCT_LOSS", "RACES", "unit_mass_kg",
           "merlin_params", "build_unit", "airframe", "round_trip", "configs", "tabulate", "study", "winners"]
