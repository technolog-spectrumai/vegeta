"""FALCO's mountain missions in MuJoCo: NISUS+'s six conditions (``nisus_plus_scenario``: calm, the ridge lift, the lee
downdraft, a hot day, the storm escape, the Jetson failure) on FALCO's pieces (``SIM``: its parameters, robot, drive,
aerodynamics, the ``FalcoAero`` hook with the parked propeller, the ``FalcoController``), the same bungee launch, the
same tables and movie. The judge adds the propeller: a touchdown with the blades not parked is a blade strike (a
failure), and the time from the park command to the parked state is reported.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import falco
import falco_controller as fcc
import falco_flight as fl
import falco_robot as frb
import falco_systems as fsy
import nisus_plus_scenario as fsc
from nisus_plus_scenario import (Scenario, CONDITIONS, Sim, timeseries, phase_table, energy_table, summary, compare, render_movie)  # noqa: F401

CRAFT = "Falco-Zero"
SLUG = "falco"

SIM = Sim(resolve=falco.resolve, robot=frb.falco_robot, drive=fsy.drive, aero=fl.aero, derivatives=fl.derivatives, cg_inertia=fsy.cg_inertia,
          aero_cls=frb.FalcoAero, controller_cls=fcc.FalcoController, phase_power=fsy.phase_power, pack=fsy.pack, jetson_w=fsy.JETSON_INSTALLATION_W,
          craft=CRAFT, body=frb.BODY, massif_cls=frb.Massif, wind_cls=frb.MountainWind)


def scenario(name: str = "calm", **kw) -> Scenario:
    """A FALCO scenario: NISUS+'s condition ``name`` with FALCO's name, slug and pack."""
    base = dict(CONDITIONS.get(name, {}))
    base.update(kw)
    return Scenario(name, craft=CRAFT, slug_prefix=SLUG, battery_key=base.pop("battery_key", fsy.DEFAULT_PACK), **base)


def standard_scenarios() -> list:
    return [scenario(name) for name in CONDITIONS]


def make_lab(scn: Scenario, **kw):
    return fsc.make_lab(scn, sim=SIM, **kw)


def controller(lab, scn: Scenario, **kw):
    return fsc.controller(lab, scn, **kw)


def run(lab, scn: Scenario, **kw):
    """NISUS+'s run (the bungee launch, the mission until it has landed and stopped) with FALCO's judge."""
    ep = fsc.run(lab, scn, **kw)
    fc = ep.controller
    ep.log["parked_at_touchdown"] = fc.parked_at_touchdown
    ep.log["t_park_cmd"] = fc.t_park_cmd
    t_parked = next((e[0] for e in ep.log["events"] if e[1] == "propeller" and "parked horizontal" in e[2]), None)
    ep.log["t_parked"] = t_parked
    ep.outcome = outcome(ep, scn.land_radius)
    return ep


def outcome(ep, land_radius: float = 150.0) -> dict:
    """NISUS+'s judge plus the propeller: landed with the blades parked, else a blade strike."""
    out = fsc.outcome(ep, land_radius)
    parked = ep.log.get("parked_at_touchdown")
    if ep.log.get("touchdown") is not None and parked is False:
        return {"success": False, "reason": "blade strike", "t_end": float(ep.log["touchdown"]),
                "detail": "the propeller was not parked at touchdown (the brake had not stopped it in time): a blade in the meadow"}
    if out["success"]:
        t_cmd, t_p = ep.log.get("t_park_cmd"), ep.log.get("t_parked")
        if t_cmd is not None and t_p is not None:
            out["park_time_s"] = float(t_p - t_cmd)
            out["detail"] += f"; propeller parked {t_p - t_cmd:.1f} s after the command, {ep.log['touchdown'] - t_p:.1f} s before touchdown"
    return out


def park_table(eps: list) -> pd.DataFrame:
    """Per episode: when the park was commanded, when the blades stopped, the height and speed then, the touchdown."""
    rows = {}
    for ep in eps:
        ts = timeseries(ep)
        t_cmd, t_p, td = ep.log.get("t_park_cmd"), ep.log.get("t_parked"), ep.log.get("touchdown")
        at = lambda tt, col: float(np.interp(tt, ts["t"], ts[col])) if tt is not None else np.nan
        rows[ep.log["scenario"]["name"]] = {"park commanded [s]": t_cmd, "agl then [m]": at(t_cmd, "agl"), "EAS then [m/s]": at(t_cmd, "EAS"),
                                            "parked [s]": t_p, "stop time [s]": (t_p - t_cmd) if (t_cmd is not None and t_p is not None) else np.nan,
                                            "agl when parked [m]": at(t_p, "agl"), "touchdown [s]": td, "parked at touchdown": ep.log.get("parked_at_touchdown"),
                                            "outcome": ep.outcome["reason"]}
    return pd.DataFrame(rows).T


__all__ = ["CRAFT", "SLUG", "SIM", "Scenario", "CONDITIONS", "scenario", "standard_scenarios", "make_lab", "controller", "run", "outcome", "park_table",
           "timeseries", "phase_table", "energy_table", "summary", "compare", "render_movie"]
