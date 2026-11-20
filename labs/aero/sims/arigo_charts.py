"""Charts and KPI scores from an ARIGO run file (or a merged file): the performance cards' properties as bar charts,
KPI scores, and the turn-rate and climb curves over speed.

    python labs/aero/sims/arigo_charts.py labs/aero/sims/output/runs/arigo_run_<time>_<host>.yaml

Writes next to it ``<stem>_properties.png``, ``<stem>_kpi.png``, ``<stem>_curves.png`` and adds a ``kpi`` section to
the file. ``arigo_all.py`` calls it at the end of every run.

KPIs: each property scored 0-100 across the configurations in the file (min-max; "lower is better" ones inverted),
then averaged into a general score and a chase score:

- general: max speed, min speed (low), max climb, best sustained turn, tightest sustained radius (low), roll rate,
  pitch rate (first 0.3 s), yaw rate, endurance, range;
- chase: run-down time on a 14 m/s bandit (low), zig-zag chase closest approach (low), best instantaneous turn,
  max speed, roll rate.

Scores are relative to the configurations present: add or drop one and the others move.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

# reference categorical palette (dataviz skill, light mode), in fixed order: one colour per aircraft
COLOUR = {"ARIGO-0": "#2a78d6", "ARIGO-1": "#eb6834", "ARIGO-2": "#1baf7a", "ARIGO-20": "#eda100"}
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
    ("zigzag_closest_m", "Zig-zag chase: closest", "m", False),
    ("endurance_min", "Endurance (best speed)", "min", True),
    ("range_km", "Range (still air)", "km", True),
]
GENERAL = ["max_speed_m_s", "min_speed_m_s", "max_climb_m_s", "best_sustained_turn_deg_s", "tightest_sustained_radius_m",
           "roll_rate_deg_s", "pitch_rate_deg_s", "yaw_rate_deg_s", "endurance_min", "range_km"]
CHASE = ["run_down_14_m_s_bandit_s", "zigzag_closest_m", "best_instant_turn_deg_s", "max_speed_m_s", "roll_rate_deg_s"]


def cards_of(doc: dict) -> dict:
    r = doc.get("results", {})
    perf = r.get("arigo_perf")
    if isinstance(perf, dict) and "data" in perf:                 # a merged file
        perf = perf["data"]
    return (perf or {}).get("cards", {}) if isinstance(perf, dict) else {}


def kpi(cards: dict) -> dict:
    ok = {k: c for k, c in cards.items() if isinstance(c, dict) and "summary" in c}
    scores = {k: {} for k in ok}
    for key, _, _, hi in PROPS:
        vals = {k: c["summary"].get(key) for k, c in ok.items()}
        vals = {k: float(v) for k, v in vals.items() if v is not None}
        if len(vals) < 2:
            continue
        lo_, hi_ = min(vals.values()), max(vals.values())
        for k, v in vals.items():
            s = 50.0 if hi_ == lo_ else 100.0 * (v - lo_) / (hi_ - lo_)
            scores[k][key] = round(s if hi else 100.0 - s, 1)
    out = {}
    for k, s in scores.items():
        g = [s[x] for x in GENERAL if x in s]
        c = [s[x] for x in CHASE if x in s]
        out[k] = {"name": ok[k]["aircraft"], "general": round(float(np.mean(g)), 1) if g else None,
                  "chase": round(float(np.mean(c)), 1) if c else None, "per_property": s}
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
    return name.split()[0]


def _bars(ax, names, vals, unit, fmt="{:.0f}"):
    y = np.arange(len(names))
    cols = [COLOUR.get(_family(n), "#888888") for n in names]
    hatch = ["////" if "strut" in n else None for n in names]
    for yi, v, c, h in zip(y, vals, cols, hatch):
        if v is None or v != v:
            ax.text(0, yi, "  n/a", va="center", fontsize=7, color=INK2)
            continue
        ax.barh(yi, v, height=0.7, color=c, edgecolor=SURFACE, linewidth=2, hatch=h)
        ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=7, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7, color=INK)
    ax.invert_yaxis()
    ax.margins(x=0.18)


def _legend(fig):
    import matplotlib.patches as mp
    h = [mp.Patch(color=c, label=n) for n, c in COLOUR.items()] + [mp.Patch(facecolor="white", edgecolor=INK2, hatch="////", label="ARIGO-0 chin strut")]
    fig.legend(handles=h, loc="upper right", ncol=5, fontsize=8, frameon=False)


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
    doc["kpi"] = K
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
        _bars(ax, names, [cards[k]["summary"].get(key) for k in keys], unit, "{:.1f}" if unit in ("s", "m", "km") else "{:.0f}")
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
    fig, axs = plt.subplots(1, 2, figsize=(13, 1.6 + 0.3 * len(keys)), facecolor=SURFACE)
    for ax, which in zip(axs, ("general", "chase")):
        order = sorted(keys, key=lambda k: -(K[k][which] or 0))
        _bars(ax, [K[k]["name"] for k in order], [K[k][which] for k in order], "score")
        _style(ax, f"{which.capitalize()} KPI score (0-100, relative to this set)", "score")
        ax.set_xlim(0, 115)
    fig.suptitle(f"ARIGO family: KPI scores — {sub}", x=0.01, ha="left", fontsize=12, color=INK)
    _legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out.append(stem.parent / f"{stem.name}_kpi.png")
    fig.savefig(out[-1], dpi=110, facecolor=SURFACE)
    plt.close(fig)

    # 3. curves over speed: sustained turn rate and climb rate, one panel per aircraft, one line per wing
    fams = [f for f in COLOUR if any(_family(cards[k]["aircraft"]) == f for k in keys)]
    fig, axs = plt.subplots(3, len(fams), figsize=(4.2 * len(fams), 10.5), facecolor=SURFACE, squeeze=False)
    for j, fam in enumerate(fams):
        for k in keys:
            nm = cards[k]["aircraft"]
            if _family(nm) != fam:
                continue
            wing = next((w for w in WING_COLOUR if w and w in nm), "")
            ls = "--" if "strut" in nm else "-"
            pm = cards[k]["point_mass"]
            t = [(x["V_m_s"], x["sustained_rate_deg_s"]) for x in pm["turns"] if x["sustained_rate_deg_s"]]
            if t:
                axs[0, j].plot(*zip(*t), color=WING_COLOUR[wing], lw=2, ls=ls, marker="o", ms=4, label=nm.replace(fam, "").strip() or fam)
            c = [(x["V_m_s"], x["rate_m_s"]) for x in pm.get("climb_curve", [])]
            if c:
                axs[1, j].plot(*zip(*c), color=WING_COLOUR[wing], lw=2, ls=ls, label=nm.replace(fam, "").strip() or fam)
            e = [(x["V_m_s"], x["power_W"]) for x in (pm.get("energy") or {}).get("power_curve", [])]
            if e:
                axs[2, j].plot(*zip(*e), color=WING_COLOUR[wing], lw=2, ls=ls, label=nm.replace(fam, "").strip() or fam)
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
