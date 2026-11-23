"""Design requirements from race runs: valid laps only, three separate views.

* ``split``      — the runs into valid laps (status finished) and failed or incomplete ones (DNF, with the reason).
                   A DNF lap never feeds a requirement: its peaks belong to a crash or a reset, not to the course.
* ``per_course`` — measured statistics of each valid lap (no margins): percentiles are taken over the moving part
                   of the lap (speed > 1 m/s), weighted by time, as ``races.analysis.lap_report`` does.
* ``pooled``     — the valid laps put end to end and weighted by time: what a pilot flying all of them spends their
                   time doing (a time-weighted percentile across courses, not the worst course).
* ``envelope``   — the worst case across valid courses (the maximum of each demand, the minimum of each allowance)
                   times a stated margin (``MARGIN``): the design requirement. Rows the runs cannot support are
                   ``unavailable`` with the reason, never a number.

Definitions: load factor ``n`` = thrust / weight carried while manoeuvring (it is a manoeuvre demand, not a hover
demand; a hover needs n = 1 plus control authority, which no run measures); along-track acceleration and braking in g
along the velocity; lateral g across it; body rates from the recorded attitude in MuJoCo, estimated from the thrust
vector's slew (roll and pitch) and the heading rate (yaw) in the mock; endurance is the reference pack's usable
energy over the lap's mean power (the time the pack would last flying that lap continuously)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import refquad

MARGIN = {"speed": 1.1, "acceleration": 1.2, "lateral g": 1.2, "load factor": 1.3, "body rate": 1.3, "vertical rate": 1.2,
          "power": 1.2, "energy": 1.25, "clearance": 1.0, "turn radius": 1.0}
RACE_LAPS = 3

# row: (how it is read from a lap report, worst-case direction, margin key or None)
ROWS = {
    "lap time [s]": ("max", None), "distance [m]": ("max", None),
    "speed p50 [m/s]": ("max", None), "speed p90 [m/s]": ("max", None), "speed max [m/s]": ("max", "speed"),
    "acceleration p95 [g]": ("max", "acceleration"), "braking p95 [g]": ("max", "acceleration"), "hardest braking [g]": ("max", "acceleration"),
    "lateral g p95": ("max", "lateral g"), "lateral g max": ("max", None),
    "least turn radius (p5) [m]": ("min", "turn radius"), "turn rate p99 [deg/s]": ("max", None),
    "roll / pitch rate p99 [deg/s]": ("max", "body rate"), "yaw rate p99 [deg/s]": ("max", "body rate"),
    "climb rate max [m/s]": ("max", "vertical rate"), "descent rate max [m/s]": ("max", "vertical rate"),
    "manoeuvre load factor p99 (thrust / weight)": ("max", "load factor"), "manoeuvre load factor max": ("max", None),
    "mean power [W]": ("max", None), "power p90 [W]": ("max", "power"), "peak power [W]": ("max", "power"),
    "energy per lap [Wh]": ("max", "energy"), "Wh per km": ("max", None),
    "endurance at the lap's mean power [min]": ("min", None),
    "closest approach flown [m]": ("min", "clearance"), "smallest opening [m]": ("min", "clearance"),
}


def split(runs: list[dict]) -> tuple[list[dict], pd.DataFrame]:
    """(valid runs, a table of the failed or incomplete ones with their reason)."""
    valid = [r for r in runs if r.get("valid_lap")]
    bad = pd.DataFrame([{"course": r["course"], "run": r.get("dir"), "simulator": r["provenance"]["simulator"], "status": r["status"],
                         "reason": r.get("dnf_reason"), "collisions": r.get("collisions"), "elapsed sim time [s]": r.get("elapsed_sim_time_s"),
                         "wall time [s]": r.get("wall_time_s")} for r in runs if not r.get("valid_lap")])
    return valid, bad


def _pc(R, row, col):
    pc = R["percentiles"]
    return float(pc.loc[row, col]) if row in pc.index else np.nan


def course_stats(R: dict) -> dict:
    """The measured row values of one lap report (``races.analysis.lap_report``)."""
    k, pw, vt = R["kinematics"], R["power"], R["vertical"]
    usable = refquad.summary()["usable energy [Wh]"]
    radius = k["turn_radius_m"].dropna()
    return {
        "lap time [s]": float(k["t"].iloc[-1] - k["t"].iloc[0]), "distance [m]": pw["distance [m]"],
        "speed p50 [m/s]": _pc(R, "speed", "p50"), "speed p90 [m/s]": _pc(R, "speed", "p90"), "speed max [m/s]": _pc(R, "speed", "max"),
        "acceleration p95 [g]": _pc(R, "a_along_g", "p95"), "braking p95 [g]": -_pc(R, "a_along_g", "p5"),
        "hardest braking [g]": float(R["braking"]["peak [g]"].max()) if len(R["braking"]) else np.nan,
        "lateral g p95": _pc(R, "a_lat_g", "p95"), "lateral g max": _pc(R, "a_lat_g", "max"),
        "least turn radius (p5) [m]": float(np.percentile(radius, 5)) if len(radius) else np.nan, "turn rate p99 [deg/s]": _pc(R, "turn_rate_deg_s", "p99"),
        "roll / pitch rate p99 [deg/s]": max(_pc(R, "roll_rate_deg_s", "p99"), _pc(R, "pitch_rate_deg_s", "p99")), "yaw rate p99 [deg/s]": _pc(R, "yaw_rate_deg_s", "p99"),
        "climb rate max [m/s]": vt["peak climb rate [m/s]"], "descent rate max [m/s]": vt["peak descent rate [m/s]"],
        "manoeuvre load factor p99 (thrust / weight)": _pc(R, "load_g", "p99"), "manoeuvre load factor max": _pc(R, "load_g", "max"),
        "mean power [W]": pw["mean power [W]"], "power p90 [W]": pw["p90 power [W]"], "peak power [W]": pw["peak power [W]"],
        "energy per lap [Wh]": pw["energy [Wh]"], "Wh per km": pw["Wh per km"],
        "endurance at the lap's mean power [min]": usable / pw["mean power [W]"] * 60.0 if pw["mean power [W]"] > 0 else np.nan,
        "closest approach flown [m]": float(R["closest approaches"]["distance [m]"].min()) if len(R["closest approaches"]) else np.nan,
        "smallest opening [m]": R["course"].get("smallest opening [m]", np.nan),
    }


def per_course(reports: dict) -> pd.DataFrame:
    """One column per valid lap: measured, no margins."""
    return pd.DataFrame({name: course_stats(R) for name, R in reports.items()}).reindex(list(ROWS))


def _wpct(x, w, q):
    i = np.argsort(x)
    x, w = x[i], w[i]
    c = np.cumsum(w) / w.sum()
    return float(np.interp(q / 100.0, c, x))


def pooled(reports: dict) -> pd.Series:
    """The valid laps end to end, time-weighted: percentiles over every moving second of every lap, totals summed."""
    if not reports:
        return pd.Series(dtype=float)
    K = pd.concat([R["kinematics"] for R in reports.values()], ignore_index=True)
    m = K[K["speed"] > 1.0]
    w = m["dt"].to_numpy()
    out = {"laps": len(reports), "flying time [s]": float(sum(R["kinematics"]["dt"].sum() for R in reports.values())),
           "distance [m]": float(sum(R["power"]["distance [m]"] for R in reports.values())),
           "energy [Wh]": float(sum(R["power"]["energy [Wh]"] for R in reports.values()))}
    out["mean power [W]"] = out["energy [Wh]"] * 3600.0 / out["flying time [s]"]
    for col, label in (("speed", "speed [m/s]"), ("a_lat_g", "lateral g"), ("load_g", "manoeuvre load factor"), ("a_along_g", "along-track acceleration [g]"),
                       ("roll_rate_deg_s", "roll rate [deg/s]"), ("yaw_rate_deg_s", "yaw rate [deg/s]"), ("climb_rate", "climb rate [m/s]"), ("power_W", "power [W]")):
        v = m[col].to_numpy(dtype=float)
        ok = np.isfinite(v)
        if ok.sum():
            for q in (5, 50, 95, 99):
                out[f"{label} p{q} (pooled)"] = _wpct(v[ok], w[ok], q)
    out["time share below 3 m/s (pooled)"] = float(K.loc[K["speed"] < 3.0, "dt"].sum() / K["dt"].sum())
    out["time share below 12 m/s (pooled)"] = float(K.loc[K["speed"] < 12.0, "dt"].sum() / K["dt"].sum())
    return pd.Series(out)


def envelope(table: pd.DataFrame, *, mock_only: bool, race_laps: int = RACE_LAPS) -> pd.DataFrame:
    """The worst case across the valid laps of ``per_course`` and the requirement (worst × margin)."""
    rows = {}
    for row, (how, mkey) in ROWS.items():
        vals = table.loc[row].astype(float) if row in table.index else pd.Series(dtype=float)
        vals = vals[np.isfinite(vals)]
        if not len(vals):
            rows[row] = {"worst case": np.nan, "from course": None, "margin": None, "requirement": "unavailable", "note": "no valid lap measured it"}
            continue
        worst = vals.max() if how == "max" else vals.min()
        frm = vals.idxmax() if how == "max" else vals.idxmin()
        mg = MARGIN[mkey] if mkey else None
        note = ""
        if "rate" in row and "deg/s" in row and mock_only:
            note = "estimated from the mock (thrust-vector slew / heading rate), not body rates"
        rows[row] = {"worst case": float(worst), "from course": frm, "margin": mg,
                     "requirement": float(worst * mg) if mg else "descriptive", "note": note}
    e = pd.DataFrame(rows).T
    pack = e.loc["energy per lap [Wh]", "worst case"]
    e.loc[f"pack for {race_laps} laps [Wh usable]"] = {"worst case": pack * race_laps if np.isfinite(pack) else np.nan, "from course": e.loc["energy per lap [Wh]", "from course"],
                                                        "margin": MARGIN["energy"], "requirement": pack * race_laps * MARGIN["energy"] if np.isfinite(pack) else "unavailable",
                                                        "note": f"{race_laps} laps of the most energetic course"}
    e.loc["hover thrust / weight"] = {"worst case": np.nan, "from course": None, "margin": None, "requirement": "unavailable",
                                      "note": "a hover needs n = 1 plus control authority; no run measures the authority margin (the manoeuvre load factor is not a hover demand)"}
    e.loc["endurance requirement [min]"] = {"worst case": np.nan, "from course": None, "margin": None, "requirement": "unavailable",
                                            "note": "no race duration is specified; see the pack for N laps and the endurance row (descriptive)"}
    return e
