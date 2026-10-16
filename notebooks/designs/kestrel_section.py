"""Kestrel-4Q — the wing section study on Zephyr: NACA 0012 against reflexed sections, and the elevon's section data
(notebook 38, Part II). Needs ``vegeta.zephyr`` (branch ``dev_zephyr``) and, to recompute, NeuralFoil.

A tailless aircraft trims its pitch with the wing alone. With the CG ahead of the neutral point (static margin SM),
the moment about the CG at a lift coefficient CL is Cm_cg = cm_section − SM·CL: a symmetric section leaves a nose-down
moment that something must cancel — the thrust split (Part I: it closes only above 14 m/s and never with faired
struts), the elevons turned trailing edge up (drag), or a section whose own moment is positive: a **reflexed** camber
line. ``CANDIDATES``: NACA 0012, the plain cambered 2412 (the wrong way — a nose-down moment — for the contrast), the
standard reflexed 5-digit lines (Cm ≈ 0) and the same lines with more aft reflex (``reflex_k21``: Cm > 0), 12 % and 10 %
thick. For each: Zephyr's polar (NeuralFoil) at the cruise and the transition Reynolds numbers, the thin-airfoil numbers,
the check of the surrogate against theory, and the plain-flap sweep of the elevon's chord. ``SECTIONS_JSON`` keeps the
results so the notebook runs without NeuralFoil; ``table(recompute=True, save=True)`` refreshes them.

``trim_table``: for each candidate, the elevon deflection and its drag that trim the aircraft at the cruise CL with the
static margin — the number that says whether the reflex pays. ``cfd_specs``: Aeromant cases of the extruded section (an
AR-8 slab, ``rans_ksst_external``) for the finalists: the CFD check of the surrogate, prepared by
``kestrel_cfd.section_case``; NOT RUN without OpenFOAM.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from vegeta import zephyr

SECTIONS_JSON = Path(__file__).resolve().parent / "data" / "kestrel_sections.json"
RE_CRUISE = 1.5e5                # 16 m/s on the 140 mm chord
RE_TRANSITION = 7e4              # ~7 m/s: the middle of the transitions
CL_CRUISE = 0.38                 # the wings' lift coefficient at 16 m/s (Part I)
N_CRIT = 6.0                     # a built-up small-UAV wing in turbulent air (9 is a clean tunnel)
ALPHAS = np.arange(-8.0, 22.0, 1.0)
ELEVON_CF = 0.28
CANDIDATES = (("NACA 0012", None), ("NACA 2412", None), ("NACA 22112", None), ("NACA 23112", None), ("NACA 24112", None),
              ("NACA 23112", 0.015), ("NACA 23112", 0.02), ("NACA 23112", 0.025), ("NACA 23112", 0.03),
              ("NACA 24112", 0.06), ("NACA 24112", 0.08), ("NACA 23110", 0.02), ("NACA 23110", 0.025))


def section(name: str = "NACA 0012", reflex_k21: float | None = None) -> zephyr.Section:
    return zephyr.naca(name, reflex_k21=reflex_k21)


def key(name: str, reflex_k21: float | None = None) -> str:
    return section(name, reflex_k21).name


def study_one(name: str, reflex_k21: float | None = None, *, n_crit: float = N_CRIT) -> dict:
    s = section(name, reflex_k21)
    pc = zephyr.analyze(s, ALPHAS, RE_CRUISE, n_crit=n_crit)
    pt = zephyr.analyze(s, ALPHAS, RE_TRANSITION, n_crit=n_crit)
    return {"name": name, "reflex_k21": reflex_k21, "section": s.name, "thickness": s.thickness, "thin": zephyr.thin_airfoil(s),
            "cruise": pc.summary(CL_CRUISE), "transition": pt.summary(CL_CRUISE), "check": zephyr.check_against_thin_airfoil(pc, s),
            "polar_cruise": pc.to_dict(), "elevon": zephyr.flap_sweep(s, ELEVON_CF, RE_CRUISE, alpha_deg=3.0, n_crit=n_crit)}


def table(recompute: bool = False, save: bool = False) -> dict:
    """{section name: study_one(...)} for ``CANDIDATES``: read from ``SECTIONS_JSON`` when it matches, else computed with
    Zephyr / NeuralFoil (a minute)."""
    cands = [[n, k] for n, k in CANDIDATES]
    if not recompute and SECTIONS_JSON.exists():
        doc = json.loads(SECTIONS_JSON.read_text())
        if doc.get("candidates") == cands and doc.get("elevon_cf") == ELEVON_CF and doc.get("n_crit") == N_CRIT:
            return doc["sections"]
    if not zephyr.backends()["neuralfoil"]:
        raise RuntimeError("NeuralFoil is not installed (pip install neuralfoil) and data/kestrel_sections.json is missing or stale")
    out = {key(n, k): study_one(n, k) for n, k in CANDIDATES}
    if save:
        SECTIONS_JSON.parent.mkdir(parents=True, exist_ok=True)
        SECTIONS_JSON.write_text(json.dumps({"candidates": cands, "elevon_cf": ELEVON_CF, "n_crit": N_CRIT, "Re": [RE_CRUISE, RE_TRANSITION],
                                             "source": "vegeta.zephyr / NeuralFoil (large)", "sections": out}, indent=1))
    return out


def properties(name: str = "NACA 0012", reflex_k21: float | None = None) -> dict:
    t = table()
    k = key(name, reflex_k21)
    if k not in t:
        raise KeyError(f"{k} is not among the stored candidates: {list(t)}")
    return t[k]


def summary_frame(t: dict | None = None):
    """One row per candidate: the numbers the choice turns on."""
    import pandas as pd
    t = t or table()
    rows = {}
    for k, v in t.items():
        c, tr, e, th = v["cruise"], v["transition"], v["elevon"], v["thin"]
        rows[k] = {"t/c": v["thickness"], "α₀ [deg]": c["alpha0_deg"], "cm₀": c["cm0"], "cm (thin)": th["cm_c4"],
                   "cl max (cruise Re)": c["cl_max"], "α cl max": c["alpha_cl_max_deg"], "cl max (transition Re)": tr["cl_max"],
                   "cd at cruise cl": c["cd_design"], "L/D at cruise cl": c["LD_design"], "L/D max": c["LD_max"],
                   "cl kept 4° past stall": c["cl_4deg_past_stall_ratio"], "x_ac": c["x_ac"], "elevon τ": e["tau"], "elevon x_cp": e["x_cp_flap"],
                   "confidence (attached)": c["confidence_min_attached"]}
    return pd.DataFrame(rows).T


def trim_table(t: dict | None = None, *, static_margin: float, cl: float = CL_CRUISE):
    """The elevon that trims each candidate at ``cl`` with the wing's ``static_margin`` (chords; the propellers' and
    pod's shares already in it): the moment the wing must carry about the CG, what the section gives, the elevon
    deflection that makes up the rest (from the flap sweep's slopes), its section-drag increment and the trim drag as
    a fraction of the cruise drag. Negative deflection = trailing edge up."""
    import pandas as pd
    t = t or table()
    rows = {}
    for k, v in t.items():
        c, e = v["cruise"], v["elevon"]
        cm_need = static_margin * cl
        cm_have = c["cm_design"]
        dcm = cm_need - cm_have
        delta = dcm / e["dcm_ddelta_per_rad"] if e["dcm_ddelta_per_rad"] else math.nan
        d_deg = math.degrees(delta)
        cd_inc = float(np.interp(d_deg, e["deltas_deg"], e["cd_increment"]))
        rows[k] = {"cm needed": cm_need, "cm of the section": cm_have, "elevon [deg]": d_deg, "Δcd (elevon)": cd_inc,
                   "cd cruise": c["cd_design"], "trim drag / section drag": cd_inc / c["cd_design"],
                   "cl lost to the elevon": e["dcl_ddelta_per_rad"] * delta, "trims within ±10°": abs(d_deg) <= 10.0}
    return pd.DataFrame(rows).T


def cfd_specs(finalists, *, chord_m: float = 0.14, V: float = 16.0, alphas=(0.0, 4.0, 8.0, 12.0), AR: float = 8.0) -> dict:
    """Aeromant cases of each finalist as an extruded slab of aspect ratio ``AR`` (``rans_ksst_external``): the
    surrogate's polar against a RANS run (3-D: the slab's lift slope is AR/(AR+2) of the section's; the drag at zero
    lift the section's plus the tips')."""
    out = {}
    for name, k in finalists:
        for a in alphas:
            out[f"{key(name, k)}, α {a:g}°, {V:g} m/s, AR {AR:g} slab"] = {"name": name, "reflex_k21": k, "alpha_deg": a, "V": V, "chord_m": chord_m, "AR": AR}
    return out
