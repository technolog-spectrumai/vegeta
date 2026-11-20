"""Charts and KPI scores from an ARIGO run file (or a merged file): the performance cards' properties as bar charts,
KPI scores, and the turn-rate and climb curves over speed.

    python labs/aero/sims/arigo_charts.py labs/aero/sims/output/runs/arigo_run_<time>_<host>.yaml

Writes next to it ``<stem>_properties.png``, ``<stem>_kpi.png``, ``<stem>_curves.png`` and adds a ``kpi`` section to
the file. ``arigo_stability.sh`` calls it at the end of every run.

KPIs: each property scored 0-100 across the configurations in the file (min-max; "lower is better" ones inverted),
then averaged into a general score and a chase score:

- general: max speed, min speed (low), max climb, best sustained turn, tightest sustained radius (low), roll rate,
  pitch rate (first 0.3 s), yaw rate, endurance, range, takeoff time (low: hand launch to 10 m at 1.3 x stall;
  a failed hand launch counts as the catapult's time + 5 s, no launch at all as 30 s);
- chase: run-down of a 14 m/s bandit (low), zig-zag PN chase time to 5 m (low) and time within 10 m, best
  instantaneous turn, max speed, roll rate;
- urban chase: the urban PN chase's time to 5 m (low) and time within 10 m (a crash scores 0), tightest sustained
  radius (low), best instantaneous turn, min speed (low), roll rate;
- open-field long chase: the open-field PN chase's time to 5 m (low) and time within 10 m, max speed, run-down of an
  18 m/s bandit (low), climb, range, endurance.
A chase never closed to 5 m counts as 1.5 x its scenario's length.

Scores are relative to the configurations present: add or drop one and the others move.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "designs"))     # arigo_products (the product scores)

# reference categorical palette (dataviz skill, light mode), in fixed order: one colour per aircraft
COLOUR = {"ARIGO-0": "#2a78d6", "ARIGO-1": "#eb6834", "ARIGO-2": "#1baf7a", "ARIGO-20": "#eda100", "ARIGO-0 strut": "#e87ba4"}
RANK_HATCH = ["xxxx", "///", "/"]           # winner (densest), runner-up, second runner-up
WING_COLOUR = {"goshawk": "#2a78d6", "zero": "#eb6834", "hybrid": "#1baf7a", "": "#eda100"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

PROPS = [  # (summary key, label, unit, higher is better)
    ("max_speed_m_s", "Max level speed", "m/s", True),
    ("min_speed_m_s", "Min speed (stall)", "m/s", False),
    ("max_climb_m_s", "Max climb rate", "m/s", True),
    ("best_sustained_turn_deg_s", "Best sustained turn", "deg/s", True),
    ("tightest_sustained_radius_m", "Tightest sustained radius", "m", False),
    ("best_instant_turn_deg_s", "Best instantaneous turn", "deg/s", True),
    ("roll_rate_deg_s", "Roll rate (full aileron)", "deg/s", True),
    ("pitch_rate_deg_s", "Pitch rate (first 0.3 s)", "deg/s", True),
    ("yaw_rate_deg_s", "Yaw rate (full rudder)", "deg/s", True),
    ("run_down_14_m_s_bandit_s", "Run down a 14 m/s bandit", "s", False),
    ("zigzag_time_to_5m_s", "Zig-zag chase (PN): time to 5 m", "s", False),
    ("urban_time_to_5m_s", "Urban chase (PN): time to 5 m", "s", False),
    ("urban_share_within_10m", "Urban chase: time within 10 m", "share", True),
    ("open_field_time_to_5m_s", "Open-field chase (PN): time to 5 m", "s", False),
    ("open_field_share_within_10m", "Open-field chase: time within 10 m", "share", True),
    ("endurance_min", "Endurance (best speed)", "min", True),
    ("takeoff_time_s", "Takeoff: time to safe flight", "s", False),
    ("mass_kg", "Take-off mass (budget, default pack)", "kg", False),
    ("range_km", "Range (still air)", "km", True),
]
GENERAL = ["max_speed_m_s", "min_speed_m_s", "max_climb_m_s", "best_sustained_turn_deg_s", "tightest_sustained_radius_m",
           "roll_rate_deg_s", "pitch_rate_deg_s", "yaw_rate_deg_s", "endurance_min", "range_km", "takeoff_time_s"]
CHASE = ["run_down_14_m_s_bandit_s", "zigzag_time_to_5m_s", "zigzag_share_within_10m", "best_instant_turn_deg_s", "max_speed_m_s",
         "roll_rate_deg_s"]
URBAN = ["urban_time_to_5m_s", "urban_share_within_10m", "tightest_sustained_radius_m", "best_instant_turn_deg_s",
         "min_speed_m_s", "roll_rate_deg_s"]
OPEN_FIELD = ["open_field_time_to_5m_s", "open_field_share_within_10m", "max_speed_m_s", "run_down_18_m_s_bandit_s",
              "max_climb_m_s", "range_km", "endurance_min"]
# a chase that never closed to 5 m scores as this many seconds (its scenario's length x 1.5); a crash scores 0
NOT_CAUGHT_S = {"zigzag_time_to_5m_s": 30.0, "urban_time_to_5m_s": 45.0, "open_field_time_to_5m_s": 90.0,
                "run_down_14_m_s_bandit_s": 180.0, "run_down_18_m_s_bandit_s": 180.0, "takeoff_time_s": 30.0}
EXTRA = [("zigzag_share_within_10m", True), ("run_down_18_m_s_bandit_s", False)]


# ------------------------------------------------------------------------------------------------ mission KPIs
# Absolute, for redesign: each metric scores 0 at its THRESHOLD (the least the client accepts) and 100 at its
# OBJECTIVE (where more stops paying), linear between, clipped. A score moves only when the aircraft changes, not when
# the set of configurations does. All thresholds and objectives are HAND (the client's assumptions): tune them here.
# (summary key, label, threshold, objective) — lower-is-better when objective < threshold; booleans 0/100.
MISSIONS = {   # mirrors reports/arigo_catalog/KPIS.md (the source: IDs, evidence and the gates are there)
    "buteo": ("BUTEO: point defence from a perch or launch station", [
        ("cue_to_safe_s", "B1 readiness: cue to safe flight (s)", 12, 3),
        ("standby_h", "B2 readiness: watch on station (h)", 2, 24),
        ("perch_margin_m", "B3 deployment: perch height to spare (m)", 0, 6),
        ("launch_success_pct", "B4 deployment: launch success, 0-12 m/s + gusts (%)", 95, 99.5),
        # B5 (the quasi-steady gust stall) is reported, not scored: a very conservative bound; B6's turbulence model
        # measures gust tolerance in flight (decision 2026-11-20)
        ("operating_wind_m_s", "B6 weather: operating wind, gusts 1.5x (m/s)", 12, 15),
        ("perch_lift_off_wind_m_s", "B7 weather: wind that lifts it off the perch (m/s)", 15, 30),
        ("keep_out_30_m_s_m", "B8 effect: keep-out vs a 30 m/s drone (m)", 1000, 2000),
        ("urban_time_to_5m_s", "B9 effect: urban chase, time to 5 m (s)", 15, 3),
        ("touchdown_energy_J", "B10 recovery: touchdown energy (J)", 300, 40),
        ("landing_miss_m", "B11 recovery: miss from the aim point (m)", 30, 3),
        ("turnaround_s", "B12 recovery: landed to ready (s)", 600, 120)]),
    "arpella": ("ARPELLA: area patrol out to 10 km", [
        ("endurance_min", "A1 endurance (min)", 45, 120),
        ("reserve_10km_pct", "A2 pack left after a 10 km mission (%)", 15, 50),
        ("wind_limit_10km_m_s", "A3 wind for the 10 km mission (m/s)", 8, 15),
        ("coverage_km2", "A4 coverage per sortie (km2)", 4, 15),
        ("time_to_10km_s", "A5 time to 10 km (s)", 600, 250),
        ("link_margin_10km_db", "A6 link margin at 10 km (dB)", 10, 20),
        ("open_field_time_to_5m_s", "A8 open-field chase, time to 5 m (s)", 20, 8),
        ("touchdown_energy_J", "A9 recovery: touchdown energy (J)", 300, 60),
        ("operating_wind_m_s", "A10 weather: operating wind, gusts 1.5x (m/s)", 10, 15),
        ("calm_success_pct", "A11 calm air: launch and landing success (%)", 90, 99)]),
    "astor": ("ASTOR: fast response from a prepared launcher", [
        ("cue_to_safe_s", "S1 cue to safe flight (s)", 12, 3),
        ("time_to_3km_s", "S2 time to 3 km (s)", 90, 35),
        ("time_to_5km_s", "S3 time to 5 km (s)", 150, 60),
        ("time_to_150m_alt_s", "S4 time to 150 m height (s)", 20, 6),
        ("keep_out_50_m_s_m", "S5 keep-out vs a 50 m/s fixed wing (m)", 1000, 2000),
        ("tail_chase_50_m_s", "S6 can tail-chase a 50 m/s target", 0, 1),
        ("full_power_min", "S7 full power on one pack (min)", 1, 6),
        ("launch_success_pct", "S8 launch success from the rail, 0-12 m/s + gusts (%)", 98, 99.9),
        ("turnaround_s", "S9 landed to ready (s)", 900, 180),
        ("landing_miss_m", "S10 recovery miss (m)", 30, 3),
        ("touchdown_energy_J", "S11 touchdown energy (J)", 300, 40),
        ("fast_50_time_to_5m_s", "S12 tail chase of a 50 m/s target from 300 m: time to 5 m (s)", 45, 20),
        ("fast_30_time_to_5m_s", "S13 tail chase of a 30 m/s target from 300 m: time to 5 m (s)", 25, 10)]),
    "sparfell": ("SPARFELL: one person, a backpack, a throw", [
        ("mass_kg", "P1 mass (kg)", 1.8, 0.9),
        ("packed_length_mm", "P2 longest piece to carry (mm)", 900, 500),
        ("assembly_s", "P3 assembly (s)", 180, 30),
        ("throw_margin", "P4 throw speed / stall speed", 0.9, 1.3),
        ("takeoff_time_s", "P5 throw to safe flight (s)", 5, 1.5),
        ("keep_out_close_m", "P6 keep-out vs a 15 m/s quad seen at 1 km (m)", 200, 700),
        ("urban_time_to_5m_s", "P7 urban chase, time to 5 m (s)", 15, 3),
        ("endurance_min", "P8 endurance (min)", 15, 40),
        ("touchdown_energy_J", "P9 rugged: touchdown energy (J)", 200, 40),
        ("belly_safety", "P10 rugged: propulsor in a belly landing (0-100)", 0, 100),
        ("launch_success_pct", "P12 throw success, 0-12 m/s + gusts (%)", 90, 99),
        ("calm_success_pct", "P13 calm air: throw and landing success (%)", 90, 99)]),
}
# the gates (pass/fail; KPIS.md): a spec failing its line's gate scores 0 for that line
GATES = {"buteo": [("launch", lambda v: v in ("perch", "site launcher"))],
         "arpella": [("wind_limit_10km_m_s", lambda v: v is not None and v >= 10.0)],
         "astor": [("launch", lambda v: v == "site launcher"),
                   ("tail_chase_50_m_s", lambda v: bool(v))],     # decided 2026-11-20: it must catch 50 m/s from behind
         "sparfell": [("throwable", lambda v: bool(v))]}


def mission_score(v, thr, obj):
    if v is None:
        return 0.0                                              # not caught / not reached / no landing: no credit
    v = float(v)
    s = (v - thr) / (obj - thr) if obj != thr else 1.0
    return round(100.0 * max(0.0, min(1.0, s)), 1)


def mission_kpi(cards: dict) -> dict:
    """Per configuration: each product mission's score (the mean of its metrics) and the best fit."""
    out = {}
    for k, c in cards.items():
        s = c.get("summary") or {}
        if "keep_out_30_m_s_m" not in s:                        # a card from before the mission metrics
            continue
        per, mis = {}, {}
        for m, (_, items) in MISSIONS.items():
            sc = []
            for key, _, thr, obj in items:
                if key not in s:                                # not computed in this run (e.g. robustness): left out
                    continue
                v = s.get(key)
                if key.startswith("urban") and s.get("urban_crashed"):
                    v = None
                if isinstance(v, bool):
                    v = 1.0 if v else 0.0
                per[key] = mission_score(v, thr, obj)
                sc.append(per[key])
            gated = all(f(s.get(k_)) for k_, f in GATES.get(m, []))
            mis[m] = round(float(np.mean(sc)), 1) if gated else 0.0
        out[k] = {"name": c["aircraft"], **mis, "per_metric": per,
                  "best_at": max(MISSIONS, key=lambda m: mis[m])}
    return out


def cards_of(doc: dict) -> dict:
    r = doc.get("results", {})
    perf = r.get("arigo_perf")
    if isinstance(perf, dict) and "data" in perf:                 # a merged file
        perf = perf["data"]
    return (perf or {}).get("cards", {}) if isinstance(perf, dict) else {}


def kpi(cards: dict) -> dict:
    ok = {k: c for k, c in cards.items() if isinstance(c, dict) and "summary" in c}
    scores = {k: {} for k in ok}
    for key, _, _, hi in PROPS + [(k_, None, None, h_) for k_, h_ in EXTRA]:
        vals = {}
        for k, c in ok.items():
            v = c["summary"].get(key)
            if v is None and key in NOT_CAUGHT_S:
                v = NOT_CAUGHT_S[key]
            if key.startswith("urban") and c["summary"].get("urban_crashed"):
                v = NOT_CAUGHT_S["urban_time_to_5m_s"] if key == "urban_time_to_5m_s" else 0.0
            if v is not None:
                vals[k] = float(v)
        if len(vals) < 2:
            continue
        lo_, hi_ = min(vals.values()), max(vals.values())
        for k, v in vals.items():
            s = 50.0 if hi_ == lo_ else 100.0 * (v - lo_) / (hi_ - lo_)
            scores[k][key] = round(s if hi else 100.0 - s, 1)
    out = {}
    for k, s in scores.items():
        mean = lambda keys: (round(float(np.mean([s[x] for x in keys if x in s])), 1) if any(x in s for x in keys) else None)
        out[k] = {"name": ok[k]["aircraft"], "general": mean(GENERAL), "chase": mean(CHASE), "urban_chase": mean(URBAN),
                  "open_field_chase": mean(OPEN_FIELD), "per_property": s}
    return out


def _style(ax, title, unit=None):
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if unit:
        ax.set_xlabel(unit, fontsize=8, color=INK2)


def _family(name):
    return "ARIGO-0 strut" if "strut" in name else name.split()[0]


def _bars(ax, names, vals, unit, fmt="{:.0f}", higher=True):
    """Horizontal bars, coloured by aircraft; the best three hatched (winner densest), by ``higher`` is better."""
    y = np.arange(len(names))
    cols = [COLOUR.get(_family(n), "#888888") for n in names]
    ok = [(i, v) for i, v in enumerate(vals) if v is not None and v == v]
    ranked = [i for i, _ in sorted(ok, key=lambda x: -x[1] if higher else x[1])]
    hatch = [None] * len(names)
    for r, i in enumerate(ranked[:3]):
        hatch[i] = RANK_HATCH[r]
    for yi, v, c, h in zip(y, vals, cols, hatch):
        if v is None or v != v:
            ax.text(0, yi, "  n/a", va="center", fontsize=7, color=INK2)
            continue
        ax.barh(yi, v, height=0.7, color=c, edgecolor=SURFACE, linewidth=2)
        if h:
            ax.barh(yi, v, height=0.7, color="none", edgecolor=INK, linewidth=0, hatch=h, alpha=0.55)
        ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=7, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7, color=INK)
    ax.invert_yaxis()
    ax.margins(x=0.18)


def _legend(fig):
    import matplotlib.patches as mp
    h = [mp.Patch(color=c, label=n.replace("strut", "chin strut")) for n, c in COLOUR.items()]
    h += [mp.Patch(facecolor="white", edgecolor=INK2, hatch=hh, label=lab) for hh, lab in zip(RANK_HATCH, ("winner", "runner-up", "2nd runner-up"))]
    fig.legend(handles=h, loc="upper right", ncol=8, fontsize=8, frameon=False)


def charts(path: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    doc = yaml.safe_load(path.read_text()) or {}
    cards = {k: c for k, c in cards_of(doc).items() if isinstance(c, dict) and "summary" in c}
    if not cards:
        print("no performance cards in", path)
        return []
    K = kpi(cards)
    MK = mission_kpi(cards)
    for k, v in MK.items():
        K.setdefault(k, {})["missions"] = v
    doc["kpi"] = K
    try:
        import arigo_products
        PS = arigo_products.product_scores(cards, MK)
        doc["products"] = PS
    except Exception as exc:                                    # noqa: BLE001  older card files have no products
        PS = {}
        print("no product scores:", exc)
    doc["kpi_missions_rule"] = {"metrics": {m: [{"key": a, "label": b, "threshold": t, "objective": o}
                                                                         for a, b, t, o in it] for m, (_, it) in MISSIONS.items()},
                                "note": "0 at the threshold, 100 at the objective, linear, clipped; HAND values"}
    path.write_text(yaml.safe_dump(json.loads(json.dumps(doc, default=str)), sort_keys=False, width=140))
    keys = list(cards)
    names = [cards[k]["aircraft"] for k in keys]
    stem = path.with_suffix("")
    out = []
    run = doc.get("run") or doc.get("merged") or {}
    sub = f"run {run.get('started', run.get('when', ''))} on {run.get('host', 'several machines')}"

    # 1. properties
    n = len(PROPS)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(15, 3.2 + 0.28 * len(keys) * rows), facecolor=SURFACE)
    for ax, (key, lab, unit, hi) in zip(axs.flat, PROPS):
        _bars(ax, names, [cards[k]["summary"].get(key) for k in keys], unit, "{:.0%}" if unit == "share" else ("{:.1f}" if unit in ("s", "m", "km", "kg") else "{:.0f}"), higher=hi)
        _style(ax, f"{lab} ({'higher' if hi else 'lower'} is better)", unit)
    for ax in list(axs.flat)[n:]:
        ax.set_visible(False)
    fig.suptitle(f"ARIGO family: flight properties — {sub}", x=0.01, ha="left", fontsize=12, color=INK)
    _legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.append(stem.parent / f"{stem.name}_properties.png")
    fig.savefig(out[-1], dpi=110, facecolor=SURFACE)
    plt.close(fig)

    # 2. KPI scores: general and chase, then the per-property scores as a table-like grid of bars
    fig, axs = plt.subplots(2, 2, figsize=(13, 3.2 + 0.6 * len(keys)), facecolor=SURFACE)
    for ax, (which, title) in zip(axs.flat, (("general", "General"), ("chase", "Chase"), ("urban_chase", "Urban chase"),
                                             ("open_field_chase", "Open-field long chase"))):
        order = sorted(keys, key=lambda k: -(K[k][which] or 0))
        _bars(ax, [K[k]["name"] for k in order], [K[k][which] for k in order], "score")
        _style(ax, f"{title} KPI (0-100, relative to this set)", "score")
        ax.set_xlim(0, 115)
    fig.suptitle(f"ARIGO family: KPI scores — {sub}", x=0.01, ha="left", fontsize=12, color=INK)
    _legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out.append(stem.parent / f"{stem.name}_kpi.png")
    fig.savefig(out[-1], dpi=110, facecolor=SURFACE)
    plt.close(fig)

    # 2b. mission fit: configurations x missions (absolute scores), and the client score
    if MK:
        mk = sorted(MK, key=lambda k: -max(MK[k][m] for m in MISSIONS))
        cols_ = list(MISSIONS)
        M = np.array([[MK[k][c] for c in cols_] for k in mk])
        fig, ax = plt.subplots(figsize=(13, 1.6 + 0.42 * len(mk)), facecolor=SURFACE)
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("seq", ["#f4f3f1", "#9ec5f0", "#2a78d6", "#123f78"])
        ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > 55 else INK, fontweight="bold" if M[i, j] == M[i].max() else None)
        ax.set_xticks(range(len(cols_)))
        ax.set_xticklabels([MISSIONS[c][0].split(":")[0] for c in cols_], fontsize=9)
        ax.set_yticks(range(len(mk)))
        ax.set_yticklabels([MK[k]["name"] for k in mk], fontsize=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title("Product fit: each configuration on each product's mission (absolute: 0 = the client's threshold, "
                     "100 = the objective; bold = its best fit)", fontsize=9, loc="left", color=INK)
        fig.suptitle(f"ARIGO family: which product for what — {sub}", x=0.01, ha="left", fontsize=12, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out.append(stem.parent / f"{stem.name}_missions.png")
        fig.savefig(out[-1], dpi=110, facecolor=SURFACE)
        plt.close(fig)

    # 2c. products: each product's candidates on its own mission score
    PS = {k: v for k, v in PS.items() if v["candidates"]}
    if PS:
        fig, axs = plt.subplots(1, len(PS), figsize=(4.6 * len(PS), 1.8 + 0.5 * max(len(v["candidates"]) for v in PS.values())),
                                facecolor=SURFACE, squeeze=False)
        for ax, (pk, pr) in zip(axs[0], PS.items()):
            rows = pr["candidates"]
            _bars(ax, [r["name"] for r in rows], [r["score"] for r in rows], "score")
            _style(ax, f"{pr['title']} ({pr['role']})\n{pr['priorities']}", "score")
            ax.title.set_fontsize(8)
            ax.set_xlim(0, 115)
        fig.suptitle(f"ARIGO products: the candidates for each product (absolute mission scores) — {sub}", x=0.01, ha="left",
                     fontsize=12, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        out.append(stem.parent / f"{stem.name}_products.png")
        fig.savefig(out[-1], dpi=110, facecolor=SURFACE)
        plt.close(fig)

    # 3. curves over speed: sustained turn rate and climb rate, one panel per aircraft, one line per wing
    fams = [f for f in ("ARIGO-0", "ARIGO-1", "ARIGO-2", "ARIGO-20") if any(cards[k]["aircraft"].split()[0] == f for k in keys)]
    fig, axs = plt.subplots(3, len(fams), figsize=(4.2 * len(fams), 10.5), facecolor=SURFACE, squeeze=False)
    for j, fam in enumerate(fams):
        for k in keys:
            nm = cards[k]["aircraft"]
            if nm.split()[0] != fam:
                continue
            wing = next((w for w in WING_COLOUR if w and w in nm), "")
            col = COLOUR["ARIGO-0 strut"] if "strut" in nm else WING_COLOUR[wing]
            ls = "-"
            pm = cards[k]["point_mass"]
            t = [(x["V_m_s"], x["sustained_rate_deg_s"]) for x in pm["turns"] if x["sustained_rate_deg_s"]]
            if t:
                axs[0, j].plot(*zip(*t), color=col, lw=2, ls=ls, marker="o", ms=4, label=nm.replace(fam, "").strip() or fam)
            c = [(x["V_m_s"], x["rate_m_s"]) for x in pm.get("climb_curve", [])]
            if c:
                axs[1, j].plot(*zip(*c), color=col, lw=2, ls=ls, label=nm.replace(fam, "").strip() or fam)
            e = [(x["V_m_s"], x["power_W"]) for x in (pm.get("energy") or {}).get("power_curve", [])]
            if e:
                axs[2, j].plot(*zip(*e), color=col, lw=2, ls=ls, label=nm.replace(fam, "").strip() or fam)
        for i, (lab, unit) in enumerate((("Sustained turn rate", "deg/s"), ("Climb rate", "m/s"), ("Battery power, level flight", "W"))):
            ax = axs[i, j]
            _style(ax, f"{fam}: {lab}")
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_ylabel(unit, fontsize=8, color=INK2)
            ax.set_xlabel("airspeed (m/s)", fontsize=8, color=INK2)
            if ax.lines:
                ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"ARIGO family: turn, climb and power over speed (point mass) — {sub}", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.append(stem.parent / f"{stem.name}_curves.png")
    fig.savefig(out[-1], dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path, help="a run file (output/runs/arigo_run_*.yaml) or a merged file")
    a = ap.parse_args(argv)
    for p in charts(a.file):
        print("->", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
