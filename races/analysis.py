"""Deep statistics of race laps and the design factors they imply (the race notebooks in notebooks/races/).

From every lap's trajectory (``ingest_race.json`` or the run tree): the regimes a lap is flown in (hover, slow,
cruise, fast: time and distance shares), the distributions of speed, acceleration, lateral g, load factor, power,
tilt and body rates (percentiles, not just peaks), the braking and acceleration episodes (how often, how hard, how
long), the climbs and descents, the corners (turn radius against speed, lateral g against speed), the straights
(length and the speed reached), the power duty (sustained against peak, energy per km by regime), the control quality
(tracking error, saturation), the collisions and the closest approaches, the openings the course imposes. Then
``design_factors``: what an aircraft built for these courses must do — the hover thrust-to-weight, the lateral g and
the turn radius at the speeds used, the body rates, the top and cruise speeds, the share of a lap below wing-borne
speed (how much a winged raptor must fly as a rotorcraft), the energy per lap and the pack for a race, the sustained
and peak power, the footprint, the braking and the vertical speeds, the stall-speed ceiling — per course and over all,
with the margin convention stated. The output is the input of a raptor design notebook (not connected yet).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import courses, metrics, refquad

G = 9.81
REGIMES = [("hover (< 3 m/s)", 0.0, 3.0), ("slow (3–8 m/s)", 3.0, 8.0), ("cruise (8–15 m/s)", 8.0, 15.0), ("fast (15–25 m/s)", 15.0, 25.0), ("very fast (> 25 m/s)", 25.0, 1e9)]
PCTS = (5, 25, 50, 75, 90, 95, 99)
MARGIN = {"thrust / weight": 1.3, "lateral g": 1.2, "body rate": 1.3, "power": 1.2, "energy": 1.25, "speed": 1.1}


# ================================================================================================= one lap, deep
def kinematics(H: np.ndarray, events: list | None = None, *, blank_s: float = 0.25) -> pd.DataFrame:
    """The lap as a frame: t, position, velocity, speed, acceleration (smoothed), along-path and lateral
    acceleration, load factor, climb rate, heading, turn rate, turn radius, thrust, power, tilt and body rates.
    The samples within ``blank_s`` of a collision or a reset (a teleport, not a manoeuvre) have their derived
    quantities blanked."""
    t = H[:, 0]
    dt = np.maximum(np.gradient(t), 1e-3)
    v = H[:, 4:7]
    a = np.gradient(v, axis=0) / dt[:, None]
    a = np.column_stack([metrics._smooth(a[:, i], 5) for i in range(3)])
    speed = H[:, 9]
    e_t = np.where(speed[:, None] > 0.5, v / np.maximum(speed, 1e-6)[:, None], 0.0)
    a_along = np.einsum("ij,ij->i", a, e_t)
    a_lat = np.linalg.norm(a - a_along[:, None] * e_t, axis=1)
    load = np.linalg.norm(a + np.array([0.0, 0.0, G]), axis=1) / G
    heading = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    turn_rate = np.where(speed > 1.0, np.abs(metrics._smooth(np.gradient(heading) / dt, 5)), 0.0)
    r_turn = np.where((speed > 3.0) & (a_lat > 1.0), speed ** 2 / np.maximum(a_lat, 1.0), np.nan)
    df = pd.DataFrame({"t": t, "x": H[:, 1], "y": H[:, 2], "z": H[:, 3], "speed": speed, "a_along_g": a_along / G, "a_lat_g": a_lat / G, "load_g": load, "climb_rate": v[:, 2],
                       "turn_rate_deg_s": np.degrees(turn_rate), "turn_radius_m": r_turn, "thrust_N": H[:, 7], "power_W": H[:, 8], "dt": dt})
    if H.shape[1] >= 17:
        df["tilt_deg"] = [math.degrees(math.acos(max(-1.0, min(1.0, metrics._quat_to_R(q)[2, 2])))) for q in H[:, 10:14]]
        df["roll_rate_deg_s"], df["pitch_rate_deg_s"], df["yaw_rate_deg_s"] = np.degrees(np.abs(H[:, 14])), np.degrees(np.abs(H[:, 15])), np.degrees(np.abs(H[:, 16]))
    else:
        f = a + np.array([0.0, 0.0, G])
        n = f / np.maximum(np.linalg.norm(f, axis=1), 1e-6)[:, None]
        df["tilt_deg"] = np.where(load > 0.5, np.degrees(np.arccos(np.clip(n[:, 2], -1, 1))), np.nan)
        slew = np.degrees(np.linalg.norm(np.gradient(n, axis=0) / dt[:, None], axis=1))
        df["roll_rate_deg_s"] = df["pitch_rate_deg_s"] = metrics._smooth(slew, 7)
        df["yaw_rate_deg_s"] = df["turn_rate_deg_s"]
    df["regime"] = pd.cut(df["speed"], [-1, 3, 8, 15, 25, 1e9], labels=[r[0] for r in REGIMES])
    bad = np.zeros(len(df), dtype=bool)
    for te, s_ in events or []:
        s_ = str(s_)
        if "collision" in s_ or "reset" in s_:
            bad |= np.abs(t - float(te)) <= blank_s
    if bad.any():
        for c in ("a_along_g", "a_lat_g", "load_g", "turn_rate_deg_s", "turn_radius_m", "roll_rate_deg_s", "pitch_rate_deg_s", "yaw_rate_deg_s", "climb_rate"):
            df.loc[bad, c] = np.nan
        df["blanked"] = bad
    return df


def regimes(k: pd.DataFrame) -> pd.DataFrame:
    """Time, distance and energy shares of each speed regime, and the mean power and specific energy in it."""
    rows = []
    T, D = k["dt"].sum(), (k["speed"] * k["dt"]).sum()
    E = (k["power_W"] * k["dt"]).sum()
    for name, lo, hi in REGIMES:
        m = (k["speed"] >= lo) & (k["speed"] < hi)
        dt, dist, en = k.loc[m, "dt"].sum(), (k.loc[m, "speed"] * k.loc[m, "dt"]).sum(), (k.loc[m, "power_W"] * k.loc[m, "dt"]).sum()
        rows.append({"regime": name, "time share": dt / T if T else 0, "distance share": dist / D if D else 0, "energy share": en / E if E else 0, "mean power [W]": en / dt if dt else np.nan,
                     "Wh per km": (en / 3600) / (dist / 1e3) if dist > 1 else np.nan, "mean load [g]": float(np.nanmean(k.loc[m, "load_g"])) if m.any() and k.loc[m, "load_g"].notna().any() else np.nan})
    return pd.DataFrame(rows).set_index("regime")


def percentiles(k: pd.DataFrame, cols=("speed", "a_along_g", "a_lat_g", "load_g", "power_W", "tilt_deg", "roll_rate_deg_s", "pitch_rate_deg_s", "yaw_rate_deg_s", "turn_rate_deg_s", "climb_rate")) -> pd.DataFrame:
    moving = k[k["speed"] > 1.0]
    return pd.DataFrame({c: {f"p{p}": float(np.nanpercentile(moving[c], p)) for p in PCTS} | {"max": float(np.nanmax(moving[c]))} for c in cols if c in k and moving[c].notna().any()}).T


def episodes(k: pd.DataFrame, col: str = "a_along_g", threshold: float = 0.5, sign: int = -1, min_s: float = 0.2) -> pd.DataFrame:
    """Braking (sign −1) or acceleration (+1) episodes: contiguous stretches where the along-path acceleration
    exceeds ``threshold`` g in that sense, each with its duration, speed change, distance, peak and mean g."""
    x = np.nan_to_num(sign * k[col].values, nan=0.0)
    on = x > threshold
    rows, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            seg = k.iloc[i:j]
            dur = seg["dt"].sum()
            if dur >= min_s:
                rows.append({"t": float(seg["t"].iloc[0]), "duration [s]": dur, "from [m/s]": float(seg["speed"].iloc[0]), "to [m/s]": float(seg["speed"].iloc[-1]), "distance [m]": float((seg["speed"] * seg["dt"]).sum()),
                             "peak [g]": float(x[i:j].max()), "mean [g]": float(x[i:j].mean())})
            i = j
        else:
            i += 1
    return pd.DataFrame(rows)


def corners(k: pd.DataFrame, min_lat_g: float = 0.5) -> pd.DataFrame:
    """The corners: contiguous stretches of lateral acceleration above ``min_lat_g``, each with its duration, heading
    change, mean speed, least radius, peak lateral g and peak tilt."""
    on = (k["a_lat_g"].fillna(0.0) > min_lat_g) & (k["speed"] > 3.0)
    rows, i = [], 0
    hd = np.degrees(np.unwrap(np.arctan2(np.gradient(k["y"]), np.gradient(k["x"]))))
    while i < len(on):
        if on.iloc[i]:
            j = i
            while j < len(on) and on.iloc[j]:
                j += 1
            seg = k.iloc[i:j]
            if seg["dt"].sum() >= 0.2:
                rows.append({"t": float(seg["t"].iloc[0]), "duration [s]": float(seg["dt"].sum()), "heading change [deg]": float(abs(hd[j - 1] - hd[i])), "mean speed [m/s]": float(seg["speed"].mean()),
                             "least radius [m]": float(np.nanmin(seg["turn_radius_m"])) if seg["turn_radius_m"].notna().any() else np.nan, "peak lateral [g]": float(seg["a_lat_g"].max()),
                             "peak tilt [deg]": float(np.nanmax(seg["tilt_deg"])), "altitude change [m]": float(seg["z"].iloc[-1] - seg["z"].iloc[0])})
            i = j
        else:
            i += 1
    return pd.DataFrame(rows)


def straights(k: pd.DataFrame, max_lat_g: float = 0.3, min_m: float = 5.0) -> pd.DataFrame:
    """The straights: stretches of little lateral acceleration, each with its length, duration, entry, peak and exit speed."""
    on = (k["a_lat_g"].fillna(9.0) < max_lat_g) & (k["speed"] > 3.0)
    rows, i = [], 0
    while i < len(on):
        if on.iloc[i]:
            j = i
            while j < len(on) and on.iloc[j]:
                j += 1
            seg = k.iloc[i:j]
            L = float((seg["speed"] * seg["dt"]).sum())
            if L >= min_m:
                rows.append({"t": float(seg["t"].iloc[0]), "length [m]": L, "duration [s]": float(seg["dt"].sum()), "entry [m/s]": float(seg["speed"].iloc[0]), "peak [m/s]": float(seg["speed"].max()), "exit [m/s]": float(seg["speed"].iloc[-1])})
            i = j
        else:
            i += 1
    return pd.DataFrame(rows)


def vertical(k: pd.DataFrame) -> dict:
    up, down = k["climb_rate"].fillna(0.0).clip(lower=0), (-k["climb_rate"].fillna(0.0)).clip(lower=0)
    return {"total climb [m]": float((up * k["dt"]).sum()), "total descent [m]": float((down * k["dt"]).sum()), "peak climb rate [m/s]": float(np.nanpercentile(up, 99)), "peak descent rate [m/s]": float(np.nanpercentile(down, 99)),
            "altitude band [m]": (float(k["z"].min()), float(k["z"].max())), "share of time climbing or descending > 1 m/s": float((k.loc[np.abs(k["climb_rate"]) > 1.0, "dt"].sum()) / k["dt"].sum())}


def power_duty(k: pd.DataFrame) -> dict:
    E = float((k["power_W"] * k["dt"]).sum() / 3600)
    D = float((k["speed"] * k["dt"]).sum())
    P = k["power_W"]
    return {"energy [Wh]": E, "distance [m]": D, "Wh per km": E / (D / 1e3) if D > 1 else np.nan, "mean power [W]": float(P.mean()), "p90 power [W]": float(np.percentile(P, 90)), "p99 power [W]": float(np.percentile(P, 99)),
            "peak power [W]": float(P.max()), "share above the sustainable power": float((k.loc[P > refquad.P_SUSTAIN_W, "dt"].sum()) / k["dt"].sum()), "longest burst above sustainable [s]": _longest(P.values > refquad.P_SUSTAIN_W, k["dt"].values)}


def _longest(mask, dt):
    best = cur = 0.0
    for m, d in zip(mask, dt):
        cur = cur + d if m else 0.0
        best = max(best, cur)
    return float(best)


def course_demands(course) -> dict:
    """What the course itself asks, before any aircraft: the openings (smallest zone), the corner speeds the schedule
    allows at a 2.5 g lateral capability, the straights' lengths, the vertical content, the total length."""
    from . import mock
    p = mock.schedule(course)
    zones = course.zones
    half = np.array([min(z.half) for z in zones])
    turns = np.degrees(p.turn[1:-1])
    legs = p.L
    dz = np.abs(np.diff(course.points[:, 2]))
    return {"length [m]": course.length(), "zones": len(zones), "smallest opening [m]": float(2 * half.min()), "openings under 2 m across": int((2 * half < 2.0).sum()),
            "corners over 60°": int((turns > 60).sum()), "corners over 120°": int((turns > 120).sum()), "sharpest corner [deg]": float(turns.max()) if len(turns) else 0.0,
            "longest leg [m]": float(legs.max()), "median leg [m]": float(np.median(legs)), "legs under 10 m": int((legs < 10).sum()), "total vertical [m]": float(dz.sum()),
            "schedule top speed [m/s]": float(p.v.max()), "schedule median speed [m/s]": float(np.median(p.v)), "schedule lap [s]": p.time_estimate(), "kind": course.kind}


def lap_report(H: np.ndarray, course=None, events: list | None = None) -> dict:
    """Everything above for one lap (``events``: the referee's, to blank the resets)."""
    k = kinematics(H, events)
    out = {"kinematics": k, "regimes": regimes(k), "percentiles": percentiles(k), "braking": episodes(k, sign=-1), "accelerating": episodes(k, sign=+1), "corners": corners(k), "straights": straights(k),
           "vertical": vertical(k), "power": power_duty(k)}
    if course is not None:
        out["course"] = course_demands(course)
        near = []
        for row in H[::max(len(H) // 500, 1)]:
            d, name = course.nearest_obstacle(row[1:4])
            near.append((d, name, row[0], row[9]))
        near.sort()
        out["closest approaches"] = pd.DataFrame(near[:5], columns=["distance [m]", "obstacle", "t", "speed"])
    return out


# ================================================================================================= the design factors
def design_factors(reports: dict, *, wing_borne_speed: float = 12.0, race_laps: int = 3) -> pd.DataFrame:
    """One column per course and an ``all courses`` column: what an aircraft for these races must do. The margins
    (``MARGIN``) turn the measured demand into the requirement; ``wing_borne_speed`` is the speed below which a
    winged raptor would be hovering or in transition (its stall speed ceiling is set at the 10th-percentile speed)."""
    cols = {}
    for name, r in reports.items():
        k, pc, rg, pw, vt = r["kinematics"], r["percentiles"], r["regimes"], r["power"], r["vertical"]
        moving = k[k["speed"] > 1.0]
        below = k.loc[k["speed"] < wing_borne_speed, "dt"].sum() / k["dt"].sum()
        cols[name] = {
            "lap time [s]": float(k["t"].iloc[-1] - k["t"].iloc[0]), "distance [m]": pw["distance [m]"],
            "cruise design speed (p50 while moving) [m/s]": float(pc.loc["speed", "p50"]), "fast design speed (p90) [m/s]": float(pc.loc["speed", "p90"]), "top speed needed (max × margin) [m/s]": float(pc.loc["speed", "max"]) * MARGIN["speed"],
            "stall speed ceiling (p10 of moving speed) [m/s]": float(np.percentile(moving["speed"], 10)), "share of time below wing-borne speed": float(below),
            "hover time share": float(rg.loc["hover (< 3 m/s)", "time share"]), "energy share below 8 m/s": float(rg.loc["hover (< 3 m/s)", "energy share"] + rg.loc["slow (3–8 m/s)", "energy share"]),
            "lateral g required (p95 × margin)": float(pc.loc["a_lat_g", "p95"]) * MARGIN["lateral g"], "lateral g peak seen": float(pc.loc["a_lat_g", "max"]),
            "load factor required (p99 × margin)": float(pc.loc["load_g", "p99"]) * MARGIN["thrust / weight"], "hover thrust / weight required": max(2.0, float(pc.loc["load_g", "p99"]) * MARGIN["thrust / weight"]),
            "deceleration required (p95 braking) [g]": float(-np.nanpercentile(k["a_along_g"], 5)) if len(k) else np.nan, "acceleration required (p95) [g]": float(np.nanpercentile(k["a_along_g"], 95)),
            "least turn radius needed [m]": float(np.nanpercentile(k["turn_radius_m"], 5)) if k["turn_radius_m"].notna().any() else np.nan, "typical corner speed [m/s]": float(r["corners"]["mean speed [m/s]"].median()) if len(r["corners"]) else np.nan,
            "roll / pitch rate required (p99 × margin) [deg/s]": float(max(pc.loc["roll_rate_deg_s", "p99"], pc.loc["pitch_rate_deg_s", "p99"])) * MARGIN["body rate"], "yaw rate required (p99) [deg/s]": float(pc.loc["yaw_rate_deg_s", "p99"]),
            "tilt used (p95) [deg]": float(pc.loc["tilt_deg", "p95"]) if "tilt_deg" in pc.index else np.nan,
            "climb rate required (p99) [m/s]": vt["peak climb rate [m/s]"], "descent rate required (p99) [m/s]": vt["peak descent rate [m/s]"], "vertical per lap [m]": vt["total climb [m]"] + vt["total descent [m]"],
            "sustained power required (p90 × margin) [W]": pw["p90 power [W]"] * MARGIN["power"], "peak power required [W]": pw["peak power [W]"] * MARGIN["power"], "mean power [W]": pw["mean power [W]"],
            "energy per lap [Wh]": pw["energy [Wh]"], f"pack for {race_laps} laps + margin [Wh usable]": pw["energy [Wh]"] * race_laps * MARGIN["energy"], "Wh per km": pw["Wh per km"],
            "braking episodes per lap": int(len(r["braking"])), "hardest braking [g]": float(r["braking"]["peak [g]"].max()) if len(r["braking"]) else np.nan,
            "corners per lap": int(len(r["corners"])), "straights per lap": int(len(r["straights"])), "longest straight [m]": float(r["straights"]["length [m]"].max()) if len(r["straights"]) else np.nan,
            "smallest opening on the course [m]": r.get("course", {}).get("smallest opening [m]", np.nan), "footprint allowed (opening − 0.3 m) [m]": r.get("course", {}).get("smallest opening [m]", np.nan) - 0.3 if r.get("course") else np.nan,
            "closest approach flown [m]": float(r["closest approaches"]["distance [m]"].iloc[0]) if "closest approaches" in r else np.nan,
        }
    df = pd.DataFrame(cols)
    if len(df.columns) > 1:
        agg_max = {"top speed needed (max × margin) [m/s]", "fast design speed (p90) [m/s]", "lateral g required (p95 × margin)", "lateral g peak seen", "load factor required (p99 × margin)", "hover thrust / weight required",
                   "deceleration required (p95 braking) [g]", "acceleration required (p95) [g]", "roll / pitch rate required (p99 × margin) [deg/s]", "yaw rate required (p99) [deg/s]", "tilt used (p95) [deg]",
                   "climb rate required (p99) [m/s]", "descent rate required (p99) [m/s]", "sustained power required (p90 × margin) [W]", "peak power required [W]", "energy per lap [Wh]", f"pack for {race_laps} laps + margin [Wh usable]",
                   "hardest braking [g]", "share of time below wing-borne speed", "hover time share", "vertical per lap [m]", "lap time [s]", "distance [m]", "longest straight [m]", "braking episodes per lap", "corners per lap", "straights per lap", "Wh per km", "energy share below 8 m/s", "mean power [W]"}
        agg_min = {"stall speed ceiling (p10 of moving speed) [m/s]", "least turn radius needed [m]", "smallest opening on the course [m]", "footprint allowed (opening − 0.3 m) [m]", "closest approach flown [m]", "typical corner speed [m/s]"}
        allc = {}
        for idx in df.index:
            row = df.loc[idx].astype(float)
            allc[idx] = row.max() if idx in agg_max else row.min() if idx in agg_min else row.median()
        df["all courses"] = pd.Series(allc)
    return df


def factors_for_raptors(df: pd.DataFrame) -> dict:
    """The compact hand-over: the all-courses column as a dict, plus the per-course columns, the margins and the
    reading of what they mean for a winged raptor."""
    a = df["all courses"] if "all courses" in df else df.iloc[:, 0]
    reading = []
    if a["share of time below wing-borne speed"] > 0.5:
        reading.append("more than half of a lap is below wing-borne speed: a raptor for these courses is a rotorcraft that sometimes flies on its wing, not a plane that sometimes hovers; its hover efficiency and hover authority matter most")
    else:
        reading.append("most of a lap is above wing-borne speed: the wing pays; the transitions and the hover authority decide the technical sections")
    reading.append(f"the stall speed ceiling is {a['stall speed ceiling (p10 of moving speed) [m/s]']:.1f} m/s: below it the raptor must hover or fly post-stall with thrust")
    reading.append(f"the openings allow a footprint of {a['footprint allowed (opening − 0.3 m) [m]']:.2f} m: the span of a winged raptor must be clipped or it must pass sideways")
    reading.append(f"hover thrust / weight {a['hover thrust / weight required']:.1f}, lateral {a['lateral g required (p95 × margin)']:.1f} g, roll/pitch rates {a['roll / pitch rate required (p99 × margin) [deg/s]']:.0f} °/s: the four-motor cells meet this, a twin-motor plank does not without big surfaces")
    reading.append(f"energy per lap {a['energy per lap [Wh]']:.2f} Wh, pack for a race {a[[i for i in a.index if i.startswith('pack for')][0]]:.1f} Wh usable: a 6S 1300 (23 Wh) covers it; a winged raptor's economy buys laps, not a smaller pack")
    return {"all_courses": {k: (None if isinstance(v, float) and not np.isfinite(v) else float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in a.items()},
            "per_course": {c: {k: (None if isinstance(v, float) and not np.isfinite(v) else float(v)) for k, v in df[c].items()} for c in df.columns if c != "all courses"},
            "margins": MARGIN, "reading": reading}


# ================================================================================================= drawings
def plot_path_coloured(course, k: pd.DataFrame, ax=None, by: str = "speed", view: str = "plan", title: str | None = None):
    """The flown path over the course map, coloured by a quantity (speed, load factor, power, tilt)."""
    import matplotlib.pyplot as plt
    from . import replay
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 6))
    a, b = (0, 1) if view == "plan" else (0, 2)
    replay._patches(course, ax, a, b)
    xy = k[["x", "y", "z"]].values
    sc = ax.scatter(xy[:, a], xy[:, b], c=k[by], s=4, cmap="viridis", zorder=5)
    plt.colorbar(sc, ax=ax, label=by, fraction=0.03)
    x0, x1, y0, y1 = course.extent
    ax.set_xlim(x0, x1)
    ax.set_ylim(*((y0, y1) if view == "plan" else (-1, max(o.pos[2] + o.half_extents()[2] for o in course.obstacles) + 4)))
    ax.set_aspect("equal"); ax.grid(alpha=0.2); ax.set_title(title or f"{course.name}: {view}, coloured by {by}")
    return ax


def plot_distributions(reports: dict, cols=("speed", "a_lat_g", "load_g", "power_W", "tilt_deg", "pitch_rate_deg_s")):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, c in zip(axes.flat, cols):
        for name, r in reports.items():
            k = r["kinematics"]
            k = k[k["speed"] > 1.0]
            if c in k:
                ax.hist(k[c].dropna(), bins=40, histtype="step", density=True, label=name, lw=1.3)
        ax.set_title(c); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_corner_envelope(reports: dict):
    """Lateral g against speed and turn radius against speed for every corner sample, with the thrust-vector envelope."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for name, r in reports.items():
        k = r["kinematics"]
        m = (k["speed"] > 3) & (k["a_lat_g"] > 0.3)
        ax[0].scatter(k.loc[m, "speed"], k.loc[m, "a_lat_g"], s=4, alpha=0.4, label=name)
        ax[1].scatter(k.loc[m, "speed"], k.loc[m, "turn_radius_m"], s=4, alpha=0.4, label=name)
    V = np.linspace(3, 30, 50)
    for tilt in (45, 60, 70):
        g_lat = math.tan(math.radians(tilt))
        ax[0].axhline(g_lat, ls=":", lw=0.8, color="k"); ax[0].text(29, g_lat, f"{tilt}° tilt", fontsize=7, va="bottom", ha="right")
        ax[1].plot(V, V ** 2 / (G * g_lat), "k:", lw=0.8); ax[1].text(V[-1], V[-1] ** 2 / (G * g_lat), f"{tilt}°", fontsize=7)
    ax[0].set(xlabel="speed [m/s]", ylabel="lateral [g]", title="corner demand against the tilt envelope", ylim=(0, 4)); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7)
    ax[1].set(xlabel="speed [m/s]", ylabel="turn radius [m]", title="turn radius against speed", ylim=(0, 60)); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_timeline(k: pd.DataFrame, events: list | None = None, title: str = ""):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(k["t"], k["speed"], color="#111"); axes[0].set_ylabel("speed [m/s]")
    axes[1].plot(k["t"], k["a_lat_g"], label="lateral"); axes[1].plot(k["t"], k["a_along_g"], label="along the path"); axes[1].plot(k["t"], k["load_g"], label="load factor", lw=0.8); axes[1].set_ylabel("[g]"); axes[1].legend(fontsize=7)
    axes[2].plot(k["t"], k["power_W"], color="#d62828"); axes[2].axhline(refquad.P_SUSTAIN_W, ls=":", color="k", lw=0.8); axes[2].set_ylabel("power [W]")
    axes[3].plot(k["t"], k["tilt_deg"], label="tilt"); axes[3].plot(k["t"], k["pitch_rate_deg_s"], lw=0.7, label="pitch rate"); axes[3].plot(k["t"], k["roll_rate_deg_s"], lw=0.7, label="roll rate"); axes[3].set_ylabel("deg, deg/s"); axes[3].legend(fontsize=7)
    for t, s in events or []:
        s = str(s)
        col = "#d62828" if ("collision" in s or "reset" in s) else "#3377dd"
        for a in axes:
            a.axvline(float(t), color=col, lw=0.5, alpha=0.6)
    for a in axes:
        a.grid(alpha=0.2)
    axes[-1].set_xlabel("t [s]"); axes[0].set_title(title)
    fig.tight_layout()
    return fig
