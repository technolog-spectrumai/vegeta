"""kestrel_benchmark — the Kestrel family on one table, read from what the design notebooks exported (notebook 47).

No physics here: ``load`` reads notebook 43's report (Kestrel-4Q-1/2/3), ``kestrel_x_design.json`` (X-A, X-B) and
``kestrel_tandem_design.json`` (the tandems) and maps their differently named columns onto one set of ``MEASURES``;
a missing file leaves its rows out with a note, a missing column leaves a blank. ``hover pitch`` of the 4Q variants is
the main variant's from the report's hover-authority table (the others carry none).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
SOURCES = {"43": DATA / "reports" / "43_kestrel_study.json", "45": DATA / "kestrel_x_design.json", "46": DATA / "kestrel_tandem_design.json", "48": DATA / "kestrel_xc_design.json"}
MEASURES = ["family", "notebook", "mass [kg]", "wing area [m²]", "span [m]", "Cd0", "induced k", "CL max", "stall speed [m/s]", "L/D at 16 m/s", "power at 16 m/s [W]",
            "best endurance [min]", "hover T/W", "hover pitch [rad/s²]", "hover heading [rad/s²]", "transitions passed", "bracing", "ground handling"]
BETTER = {"mass [kg]": "low", "Cd0": "low", "induced k": "low", "CL max": "high", "stall speed [m/s]": "low", "L/D at 16 m/s": "high", "power at 16 m/s [W]": "low",
          "best endurance [min]": "high", "hover T/W": "high", "hover pitch [rad/s²]": "high", "hover heading [rad/s²]": "high"}


def _rows(section: dict) -> dict:
    return {r["_index"]: {k: v for k, v in r.items() if k != "_index"} for r in section.get("rows", [])}


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load(sources: dict | None = None) -> tuple[pd.DataFrame, list[str]]:
    """The benchmark table (one row per airframe, ``MEASURES`` as columns) and the notes on what was missing."""
    src = {**SOURCES, **(sources or {})}
    rows, notes = {}, []
    # ---- notebook 43: the biplanes
    p = Path(src["43"])
    if p.exists():
        try:
            doc = json.loads(p.read_text())
            s = doc["sections"]
            var = _rows(s["variants"])
            hov = _rows(s.get("hover authority", {}))
            pitch = next((r.get("angular acceleration [rad/s²]") for k, r in hov.items() if k.startswith("about y")), None)
            head = next((r.get("angular acceleration [rad/s²]") for k, r in hov.items() if k.startswith("about the thrust")), None)
            trans = _rows(s.get("transitions", {}))
            lay = _rows(s.get("layout [m, m²]", {}))
            for name, v in var.items():
                mine = [r for k, r in trans.items() if k.startswith(name + " ")]
                main = name.endswith("4Q-2")
                rows[name] = {"family": "biplane (tail-sitter)", "notebook": "43", "mass [kg]": _num(v.get("take-off mass [kg]")), "wing area [m²]": 0.28, "span [m]": 1.0,
                              "Cd0": _num(v.get("Cd0")), "induced k": 0.0599, "CL max": _num(v.get("CL max (polar)")), "stall speed [m/s]": _num(v.get("stall speed [m/s]")),
                              "L/D at 16 m/s": _num(v.get("L/D at 16 m/s")), "power at 16 m/s [W]": _num(v.get("power at 16 m/s [W]")), "best endurance [min]": _num(v.get("best endurance [min]")),
                              "hover T/W": _num(v.get("hover T/W")), "hover pitch [rad/s²]": _num(pitch) if main else None, "hover heading [rad/s²]": _num(head) if main else None,
                              "transitions passed": f"{sum(bool(r.get('pass')) for r in mine)} of {len(mine)}" if mine else "", "bracing": v.get("bracing", ""),
                              "ground handling": "stands on its feet"}
            notes.append(f"43: {p.name} exported {doc.get('exported', '?')}; the hover authority is the main variant's; induced k is the lattice's 0.060 for all three")
        except (KeyError, ValueError, TypeError) as exc:
            notes.append(f"43: {p.name} unreadable ({type(exc).__name__}: {exc})")
    else:
        notes.append(f"43: {p} missing — run notebook 43_kestrel_study")
    # ---- notebook 45: the X-wings (notebook 41's export stands in when 45 has not run)
    p = Path(src["45"])
    fallback = DATA / "xwing_design.json"
    if not p.exists() and fallback.exists() and "45" not in (sources or {}):
        p = fallback
        notes.append("45: kestrel_x_design.json missing — notebook 41's xwing_design.json (the same model, an earlier run) stands in; run 45_kestrel_x")
    if p.exists():
        try:
            doc = json.loads(p.read_text())
            for name, c in doc["comparison"].items():
                if not name.startswith("X-"):
                    continue
                short = name.split(" ")[0]
                rows[f"Kestrel-{short}"] = {"family": "X-wing (tail-sitter)", "notebook": "45", "mass [kg]": _num(c.get("mass [kg]")), "wing area [m²]": _num(c.get("wing area [m²]")),
                                            "span [m]": _num(c.get("projected span [m]")), "Cd0": _num(c.get("Cd0")), "induced k": _num(c.get("induced k")), "CL max": None,
                                            "stall speed [m/s]": None, "L/D at 16 m/s": None, "power at 16 m/s [W]": _num(c.get("power at 16 m/s [W]")), "best endurance [min]": _num(c.get("endurance best [min]")),
                                            "hover T/W": None, "hover pitch [rad/s²]": _num(c.get("hover pitch [rad/s²]")), "hover heading [rad/s²]": _num(c.get("hover heading [rad/s²]")),
                                            "transitions passed": c.get("transitions passed", ""), "bracing": c.get("bracing", "none"), "ground handling": "stands on the four tips"}
            dv = doc.get("drag_vs_speed", {})
            for name, r in dv.items():
                key = f"Kestrel-{name}" if name.startswith("X-") else None
                if key in rows:
                    rows[key]["stall speed [m/s]"] = _num(r.get("V stall [m/s]"))
                    rows[key]["L/D at 16 m/s"] = None
                    rows[key]["best L/D"] = _num(r.get("best L/D"))
            notes.append(f"45: {p.name}; the X's stall speed from its drag-vs-speed table (panel cl max), no L/D at 16 (the panels fly at 10.8° there), CL max per panel only")
        except (KeyError, ValueError, TypeError) as exc:
            notes.append(f"45: {p.name} unreadable ({type(exc).__name__}: {exc})")
    else:
        notes.append(f"45: {p} missing — run notebook 45_kestrel_x")
    # ---- notebook 46: the tandems
    p = Path(src["46"])
    if p.exists():
        try:
            doc = json.loads(p.read_text())
            for name, c in doc["comparison"].items():
                if name.startswith("Kestrel-4Q"):
                    continue
                rows[f"Kestrel-{name}"] = {"family": "tandem (horizontal start)", "notebook": "46", "mass [kg]": _num(c.get("mass [kg]")), "wing area [m²]": _num(c.get("wing area [m²]")),
                                          "span [m]": _num(c.get("span [m]")), "Cd0": _num(c.get("Cd0")), "induced k": _num(c.get("induced k")), "CL max": _num(c.get("CL max (aircraft)")),
                                          "stall speed [m/s]": _num(c.get("stall speed [m/s]")), "L/D at 16 m/s": _num(c.get("L/D at 16 m/s")), "power at 16 m/s [W]": _num(c.get("power at 16 m/s [W]")),
                                          "best endurance [min]": _num(c.get("endurance best [min]")), "hover T/W": None, "hover pitch [rad/s²]": _num(c.get("hover pitch [rad/s²]")),
                                          "hover heading [rad/s²]": _num(c.get("hover heading [rad/s²]")), "transitions passed": c.get("transitions passed", ""), "bracing": c.get("bracing", "none"),
                                          "ground handling": f"skids, {_num(c.get('ground roll [m]')) or float('nan'):.1f} m ground roll",
                                          "balloon into the hover [m]": _num(c.get("balloon into the hover, 3 s [m]"))}
            notes.append(f"46: {p.name}; main variant {doc.get('main')}")
        except (KeyError, ValueError, TypeError) as exc:
            notes.append(f"46: {p.name} unreadable ({type(exc).__name__}: {exc})")
    else:
        notes.append(f"46: {p} missing — run notebook 46_kestrel_tandem")
    # ---- notebook 48: the X-C (and X-C45)
    p = Path(src["48"])
    if p.exists():
        try:
            doc = json.loads(p.read_text())
            for name, c in doc["comparison"].items():
                if not name.startswith("Kestrel-X-C"):
                    continue
                rows[name] = {"family": "X-wing (tail-sitter)", "notebook": "48", "mass [kg]": _num(c.get("mass [kg]")), "wing area [m²]": _num(c.get("wing area [m²]")), "span [m]": _num(c.get("projected span [m]")),
                              "Cd0": _num(c.get("Cd0")), "induced k": _num(c.get("induced k")), "CL max": None, "stall speed [m/s]": _num(c.get("stall speed [m/s]")), "L/D at 16 m/s": _num(c.get("L/D at 16 m/s")),
                              "power at 16 m/s [W]": _num(c.get("power at 16 m/s [W]")), "best endurance [min]": _num(c.get("endurance best [min]")), "hover T/W": None,
                              "hover pitch [rad/s²]": _num(c.get("hover pitch [rad/s²]")), "hover heading [rad/s²]": _num(c.get("hover heading [rad/s²]")), "transitions passed": c.get("transitions passed", ""),
                              "bracing": "none", "ground handling": "stands on the tip pods", "payload score": 1.0 if c.get("capsule fits") else 0.3}
            notes.append(f"48: {p.name}; the X-C's payload constant is 1.0 where its module takes the capsule")
        except (KeyError, ValueError, TypeError) as exc:
            notes.append(f"48: {p.name} unreadable ({type(exc).__name__}: {exc})")
    else:
        notes.append(f"48: {p} missing — run notebook 48_kestrel_xc")
    cols = MEASURES + [c for c in ("best L/D", "balloon into the hover [m]") if any(c in r for r in rows.values())]
    df = pd.DataFrame(rows).T.reindex(columns=cols) if rows else pd.DataFrame(columns=cols)
    return df, notes


def ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Each measure's rank across the airframes (1 = best, by ``BETTER``), blanks where the notebook gave none."""
    out = {}
    for m, how in BETTER.items():
        if m in df.columns:
            s = pd.to_numeric(df[m], errors="coerce")
            out[m] = s.rank(ascending=(how == "low"), method="min")
    r = pd.DataFrame(out)
    r["measures ranked"] = r.notna().sum(axis=1)
    r["mean rank"] = r.drop(columns=["measures ranked"]).mean(axis=1)
    return r.sort_values("mean rank")


def plot(df: pd.DataFrame, ax=None):
    """Two bars per airframe: the cruise power at 16 m/s (lower better) and the hover pitch authority (higher better)."""
    import matplotlib.pyplot as plt
    import numpy as np
    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(14, 0.45 * len(df) + 1.5), sharey=True)
    names = list(df.index)
    y = np.arange(len(names))
    for a, col, title in zip(ax, ("power at 16 m/s [W]", "hover pitch [rad/s²]"), ("cruise power at 16 m/s [W] — lower is better", "hover pitch authority [rad/s²] — higher is better")):
        v = pd.to_numeric(df[col], errors="coerce").values.astype(float)
        a.barh(y, np.nan_to_num(v), color=["#4682b4" if "Q4T1N-10" in n else "#c8c8c8" for n in names])
        for yi, vi in zip(y, v):
            a.text(vi + 0.5, yi, "—" if np.isnan(vi) else f"{vi:.0f}", va="center", fontsize=9)
        a.set_title(title, fontsize=10)
        a.set_yticks(y); a.set_yticklabels(names); a.invert_yaxis(); a.grid(axis="x", alpha=0.3)
    return ax


# ================================================================================================= the scoring methodology
# Each category is scored 1–10 from one measure of the table by a linear scale between two anchors (``worst`` → 1,
# ``best`` → 10, clipped), so a score is reproducible from the table and the anchors, never a rank. Where a notebook
# exported no number for an airframe the category takes the family's constant from the notebooks' text (``FAMILY``),
# and the table says so. The anchors are design targets of the Kestrel briefs, not the family's extremes.
CATEGORIES = {
    "cruise economy": {"measure": "power at 16 m/s [W]", "worst": 120.0, "best": 50.0, "why": "electrical power in level flight at the 16 m/s survey speed"},
    "endurance": {"measure": "best endurance [min]", "worst": 20.0, "best": 60.0, "why": "best endurance on 80 % of the 4S 3300 pack"},
    "slow flight": {"measure": "stall speed [m/s]", "worst": 14.0, "best": 8.0, "why": "the stall speed: the launch, the landing and the entry into the hover"},
    "clean airframe": {"measure": "Cd0", "worst": 0.07, "best": 0.018, "why": "zero-lift drag on the wing area: struts, pods, junctions"},
    "hover authority": {"measure": "hover pitch [rad/s²]", "worst": 0.0, "best": 90.0, "why": "pitch acceleration available in the hover (the weakest axis of a quad tail-sitter)"},
    "transitions": {"measure": "transition pass fraction", "worst": 0.0, "best": 1.0, "why": "share of the modelled hover↔cruise transitions that pass as is"},
    "structure": {"measure": "structure score", "worst": 0.0, "best": 1.0, "why": "no bracing and a spar reserve ≥ 2 score full; posts 0.6, an X truss 0.3; a reserve below 2 halves it"},
    "ground handling": {"measure": "ground score", "worst": 0.0, "best": 1.0, "why": "tail-sitters: tip-over angle and the wind that topples them; the tandem: skids and a short roll but a strip needed"},
    "payload": {"measure": "payload score", "worst": 0.0, "best": 1.0, "why": "does the medical capsule (72×72×160 mm) fit the pod near the CG"},
    "evidence": {"measure": "evidence score", "worst": 0.0, "best": 1.0, "why": "how much of the row rests on solver results (CFD, FEA) rather than the model"},
    "confined launch and recovery": {"measure": "launch score", "worst": 0.0, "best": 1.0, "why": "can it leave and come back to a vessel's deck or a shore without a strip: vertical take-off and landing, the wind it stands in"},
}
FAMILY = {   # the family constants the notebooks' text gives (0–1), with their source
    "biplane (tail-sitter)": {"ground score": 0.4, "payload score": 1.0, "evidence score": 0.3, "launch score": 0.75,
                              "notes": "ground: tips at 39° and 8.8 m/s broadside (43 § 8); launch: vertical from a deck, but it topples in 8.8 m/s of wind; payload: the capsule stacks in the 274 mm pod (43 Part II); evidence: an imported screening CFD sweep, 1 of 13 converged, FEA NOT RUN"},
    "X-wing (tail-sitter)": {"ground score": 0.85, "payload score": 0.3, "evidence score": 0.15, "launch score": 0.95,
                             "notes": "ground: tips at 60° and 16 m/s (45 § 5); launch: vertical from a deck and stands in 16 m/s; payload: the capsule fits the section but not the pod's length, 462 of 430 mm (45 § 4c); evidence: model only, no CAD yet"},
    "tandem (horizontal start)": {"ground score": 0.7, "payload score": 0.8, "evidence score": 0.15, "launch score": 0.25,
                                  "notes": "ground: skids, a 3 m roll, no wind limit but a strip or a hand launch (46 § 5); launch: a hand launch from a deck is possible, a landing back on it is not (belly landing on a strip or a net); payload: the pod is 0.82 m long, the capsule bay is not drawn yet; evidence: model only"},
}
MISSIONS = {   # weights per category (they sum to 1); the first four are the Kestrel briefs', the last three X-Wing-U's (one aircraft, three payload modules)
    "mountain survey (cruise)": {"cruise economy": 0.25, "endurance": 0.25, "slow flight": 0.10, "clean airframe": 0.05, "hover authority": 0.05, "transitions": 0.10, "structure": 0.05, "ground handling": 0.05, "payload": 0.05, "evidence": 0.05, "confined launch and recovery": 0.0},
    "medical delivery (hover at the site)": {"cruise economy": 0.05, "endurance": 0.05, "slow flight": 0.10, "clean airframe": 0.0, "hover authority": 0.20, "transitions": 0.20, "structure": 0.05, "ground handling": 0.10, "payload": 0.20, "evidence": 0.05, "confined launch and recovery": 0.0},
    "pursuit (fast response)": {"cruise economy": 0.15, "endurance": 0.10, "slow flight": 0.05, "clean airframe": 0.20, "hover authority": 0.10, "transitions": 0.15, "structure": 0.10, "ground handling": 0.10, "payload": 0.0, "evidence": 0.05, "confined launch and recovery": 0.0},
    "field trials (build it first)": {"cruise economy": 0.10, "endurance": 0.05, "slow flight": 0.05, "clean airframe": 0.0, "hover authority": 0.10, "transitions": 0.15, "structure": 0.25, "ground handling": 0.10, "payload": 0.0, "evidence": 0.20, "confined launch and recovery": 0.0},
    # X-Wing-U: maritime rescue is the primary mission — reach the coordinates, then search, deliver flotation, hold contact until the boat arrives;
    # the defining requirement is endurance left after the transit, then the launch from a vessel or the shore and holding position in wind
    "3-civilian-missions: maritime rescue (primary)": {"cruise economy": 0.15, "endurance": 0.25, "slow flight": 0.10, "clean airframe": 0.0, "hover authority": 0.10, "transitions": 0.10, "structure": 0.0, "ground handling": 0.0, "payload": 0.05, "evidence": 0.05, "confined launch and recovery": 0.20},
    # a small mission-critical part between known endpoints, often offshore: the cargo pod, the transit, the deck at both ends
    "3-civilian-missions: urgent industrial logistics": {"cruise economy": 0.15, "endurance": 0.15, "slow flight": 0.05, "clean airframe": 0.05, "hover authority": 0.05, "transitions": 0.10, "structure": 0.05, "ground handling": 0.0, "payload": 0.20, "evidence": 0.05, "confined launch and recovery": 0.15},
    # medicine, a diagnostic kit or a sample in a secured container with its handling conditions: the payload first, then arriving and leaving where there is no strip
    "3-civilian-missions: emergency medical delivery": {"cruise economy": 0.15, "endurance": 0.10, "slow flight": 0.05, "clean airframe": 0.0, "hover authority": 0.10, "transitions": 0.10, "structure": 0.05, "ground handling": 0.0, "payload": 0.25, "evidence": 0.05, "confined launch and recovery": 0.15},
}


# the X-Wing-U set as one score: one aircraft flies all three with different payload modules, the rescue being primary
MISSION_GROUPS = {"3-civilian-missions": {"3-civilian-missions: maritime rescue (primary)": 0.5, "3-civilian-missions: urgent industrial logistics": 0.25,
                                          "3-civilian-missions: emergency medical delivery": 0.25}}
CIVILIAN = {
    "maritime rescue": {"situation": "a person is reported overboard or in difficulty offshore", "payload": "compact flotation equipment, search camera, optional thermal camera",
                        "outcome": "locate the person, deliver accessible flotation and help rescuers maintain their position",
                        "what the score weighs": "endurance left after the transit (0.25), cruise economy (0.15), launch and recovery from a vessel or shore (0.20), holding position over the casualty in wind (hover 0.10), slow search flight (0.10), transitions (0.10)"},
    "urgent industrial logistics": {"situation": "a vessel, offshore installation or remote facility needs a small part to resume work", "payload": "protected cargo pod with a spare part, tool or electronic component",
                                    "outcome": "deliver the correct item intact, avoiding a longer boat or road journey",
                                    "what the score weighs": "the cargo pod (payload 0.20), the transit (cruise 0.15, endurance 0.15), a deck at both ends (launch and recovery 0.15), transitions (0.10)"},
    "emergency medical delivery": {"situation": "a clinic, vessel or remote team urgently needs medical supplies or sample transport", "payload": "secured medical container, insulated and temperature-monitored where required",
                                   "outcome": "deliver the supplies within their required time and handling conditions",
                                   "what the score weighs": "the secured container near the CG (payload 0.25), the transit (cruise 0.15, endurance 0.10), arriving where there is no strip (launch and recovery 0.15, hover 0.10), transitions (0.10)"},
}


def group_scores(wt: pd.DataFrame, groups: dict | None = None) -> pd.DataFrame:
    """A mission group's one score per airframe: the weighted mean of its missions' totals (``MISSION_GROUPS``)."""
    groups = groups or MISSION_GROUPS
    out = {}
    for g, w in groups.items():
        tot = sum(w.values())
        out[g] = sum(pd.to_numeric(wt[m], errors="coerce") * wm for m, wm in w.items()) / tot
        for m in w:
            out[f"  {m.split(': ', 1)[-1]}"] = pd.to_numeric(wt[m], errors="coerce")
    return pd.DataFrame(out).sort_values(list(groups)[0], ascending=False)


def _scale(x, worst, best):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    t = (float(x) - worst) / (best - worst)
    return float(min(max(1 + 9 * t, 1.0), 10.0))


def _pass_fraction(txt) -> float | None:
    try:
        a, b = str(txt).split(" of ")
        return int(a.split()[-1]) / int(b.split()[0]) if int(b.split()[0]) else None
    except (ValueError, AttributeError):
        return None


def derived(df: pd.DataFrame) -> pd.DataFrame:
    """The table with the derived measures the categories score: the transition pass fraction, the structure, ground,
    payload and evidence scores (0–1)."""
    out = df.copy()
    out["transition pass fraction"] = [_pass_fraction(t) for t in out["transitions passed"]]
    st = []
    for _, r in out.iterrows():
        base = {"none": 1.0, "posts": 0.6, "x": 0.3, "x, round": 0.3, "posts, faired": 0.6}.get(str(r.get("bracing", "")).strip(), 0.6)
        res = _num(r.get("spar reserve")) if "spar reserve" in out.columns else None
        st.append(base * (0.5 if (res is not None and res < 2) else 1.0))
    out["structure score"] = st
    for key in ("ground score", "payload score", "evidence score", "launch score"):
        own = out[key] if key in out.columns else [None] * len(out)
        out[key] = [o if (o is not None and not (isinstance(o, float) and math.isnan(o))) else FAMILY.get(f, {}).get(key) for o, f in zip(own, out["family"])]
    return out


def scores(df: pd.DataFrame) -> pd.DataFrame:
    """Every category 1–10 per airframe (``CATEGORIES``' anchors on ``derived``'s measures); blank where no number exists."""
    d = derived(df)
    out = {}
    for cat, c in CATEGORIES.items():
        col = c["measure"]
        out[cat] = [_scale(_num(v), c["worst"], c["best"]) if col in d.columns else None for v in (d[col] if col in d.columns else [None] * len(d))]
    return pd.DataFrame(out, index=df.index)


def weighted(sc: pd.DataFrame, missions: dict | None = None) -> pd.DataFrame:
    """The weighted total 1–10 per mission; a blank category is left out and the weights renormalised, the ``coverage``
    column says how much of the weight was scored."""
    missions = missions or MISSIONS
    rows = {}
    for name, w in missions.items():
        tot, cov = {}, {}
        for af, r in sc.iterrows():
            num = den = 0.0
            for cat, wt in w.items():
                v = r.get(cat)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    num += wt * v
                    den += wt
            tot[af] = num / den if den else None
            cov[af] = den
        rows[name] = tot
        rows[name + " · coverage"] = cov
    return pd.DataFrame(rows)


def anchors_table() -> pd.DataFrame:
    return pd.DataFrame({k: {"measure": v["measure"], "1 point at": v["worst"], "10 points at": v["best"], "why": v["why"]} for k, v in CATEGORIES.items()}).T


def plot_scores(sc: pd.DataFrame, ax=None, *, highlight: str = "Q4T1N-10"):
    """Grouped bars: the ten categories per airframe."""
    import matplotlib.pyplot as plt
    import numpy as np
    if ax is None:
        _, ax = plt.subplots(figsize=(15, 5))
    cats = list(sc.columns)
    n = len(sc)
    x = np.arange(len(cats))
    w = 0.8 / max(n, 1)
    for i, (af, r) in enumerate(sc.iterrows()):
        vals = [np.nan if v is None else v for v in r.values.astype(float)]
        ax.bar(x + (i - n / 2 + 0.5) * w, np.nan_to_num(vals), w, label=af, alpha=0.95 if highlight in af else 0.6)
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=20, ha="right"); ax.set_ylim(0, 10.5); ax.set_ylabel("points (1–10)")
    ax.set_title("category scores: 1 at the brief's worst anchor, 10 at its target (kestrel_benchmark.CATEGORIES)")
    ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=0.3)
    return ax


def plot_missions(wt: pd.DataFrame, ax=None):
    """One panel per mission: the weighted total per airframe, sorted, the coverage printed beside the bar."""
    import matplotlib.pyplot as plt
    import numpy as np
    names = [c for c in wt.columns if not c.endswith("coverage")]
    if ax is None:
        _, ax = plt.subplots(1, len(names), figsize=(4.2 * len(names), 0.45 * len(wt) + 1.8), sharey=False)
    ax = np.atleast_1d(ax).ravel()
    for a, m in zip(ax, names):
        s = pd.to_numeric(wt[m], errors="coerce").sort_values(ascending=True)
        cov = wt[m + " · coverage"].reindex(s.index)
        colors = ["#4682b4" if "Q4T1N-10" in i else ("#e0995a" if "4Q-2" in i else ("#7bb37b" if "X-A" in i else "#c8c8c8")) for i in s.index]
        a.barh(range(len(s)), s.values, color=colors)
        for k, (v, c) in enumerate(zip(s.values, cov.values)):
            a.text(v + 0.1, k, f"{v:.1f}" + (f" ({c * 100:.0f} %)" if c < 0.999 else ""), va="center", fontsize=8)
        a.set_yticks(range(len(s))); a.set_yticklabels([i.replace("Kestrel-", "") for i in s.index], fontsize=8)
        a.set_xlim(0, 11); a.set_title(m, fontsize=10); a.grid(axis="x", alpha=0.3)
    return ax
