"""kestrel_study — the unified Kestrel notebook's calculations (notebook 43, ``43_kestrel_study``).

The notebook is an engineering narrative; the arithmetic lives here so that each cell stays a few readable lines.
Everything is **inexpensive**: the existing design modules (mass table, Boreas drive tables, the vortex lattice, the
drag build-up, the plane frame, the reduced-order transition model) and the reading of workflow results from
``workflows/runs/`` by explicit path. **Nothing here launches a solver** — the CFD (``workflows/kestrel/airframe_cfd.py``)
and the FEA (``workflows/kestrel/structure_fea.py``) run on their own, and ``prerequisites`` says whether what they
produced exists, is finished and is compatible with the design as the notebook builds it.

Every number is labelled with how it was obtained (``METHODS``): *analytic* (closed-form physics), *empirical* (a
correlation or a surrogate, with its source and range in ``CORRELATIONS``), *solver* (a Vegeta tool's numerical
solution on its mesh or lattice) or *validated* (a converged solver result checked against a reference). An
attached-flow formula is never presented as a stall or transition model, and an unconverged CFD case never as a
validation.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

import kestrel
import kestrel_campaigns as kcp
import kestrel_cfd as kcfd
import kestrel_control as kc
import kestrel_fea as kfe
import kestrel_flight as kf
import kestrel_loop as kl
import kestrel_structure as kst
import kestrel_systems as ks
import kestrel_transition as kt
import kestrel2 as k2

DESIGNS = Path(__file__).resolve().parent
ROOT = DESIGNS.parents[1]
NAME = "43_kestrel_study"
VARIANTS = ("4Q-1", "4Q-2", "4Q-3")
MAIN = "4Q-2"
V_CRUISE = kf.V_CRUISE
HEADWIND = 6.0
WORKFLOW_THRESHOLD_MIN = (15, 30)

METHODS = {
    "analytic": "closed-form physics: momentum theory, beam and truss statics, thin-airfoil theory, the energy balance",
    "empirical": "a correlation or a surrogate with a stated source and applicability range (CORRELATIONS); its uncertainty is the table's",
    "solver": "a Vegeta tool's numerical solution — Boreas BEMT, the vortex lattice, the reduced-order transition model, RANS, FEA — on its discretisation: a model's answer, not a measurement",
    "validated": "a converged solver result checked against a reference or a second method (a converged CFD sweep against the lattice, an FEA case against the plane frame)",
    "NOT RUN": "the workflow that produces it has not been run for this design: the analytical estimate stands in, labelled",
}

CORRELATIONS = [
    ("Prandtl's biplane factor", "induced drag of the two wings against one monoplane of the same span", "Prandtl, NACA TR 116 (1924); Munk's stagger theorem",
     "equal spans, zero stagger, gap/span 0.1–0.5 (here 0.28); the lattice checks it", "±5 % on the induced drag", "§ 5"),
    ("momentum theory with a figure of merit", "hover power per rotor: P = T^{3/2} / sqrt(2 ρ A) / FM", "Glauert; FM from Boreas BEMT (generic blade section, prop_lab calibration when on file)",
     "uniform inflow, no interference between the four discs, no ground effect", "±15 % on the hover power (the FM of a 9-inch propeller is 0.5–0.65)", "§ 4"),
    ("drag build-up", "Cd0 from wetted areas: turbulent flat-plate skin friction with form factors, cylinders in cross-flow, an interference allowance",
     "Hoerner, Fluid-Dynamic Drag; Prandtl–Schlichting Cf; Cd 1.1 for a tube at Re ~5e3, 0.12 in a 3:1 fairing", "Re 5e4–5e5, attached flow; no gaps, no protuberances",
     "±20 % on Cd0 (the campaign's bare sweep read 0.8 × the model's)", "§ 5"),
    ("Oswald efficiency", "e = 0.90 × the lattice's inviscid span efficiency", "a rule of thumb for built-up small wings", "attached flow", "±10 % on the induced drag", "§ 5"),
    ("Viterna post-stall extension", "the wing's forces beyond the stall, blended to a flat plate at 90°", "Viterna & Corrigan (1981), the flat-plate normal force with cd90 from the aspect ratio",
     "time-averaged post-stall loads of a straight wing; **not** a model of stall onset, hysteresis or unsteady transition aerodynamics", "±30 % beyond 20°; the transition verdicts inherit it", "§ 5, § 9"),
    ("section polars (NeuralFoil)", "cl_max (×0.9 for a built-up wing), α₀, the elevon's flap effectiveness τ, the section drag ratio", "NeuralFoil surrogate of XFOIL (Sharpe, 2024) at n_crit 6",
     "Re 5e4–1e6, smooth sections; the reflexed line's confidence 0.84–0.91, the 0012's 0.96", "±0.1 on cl_max, ±15 % on cd; XFOIL or a slab CFD is the check", "§ 5"),
    ("De Young's propeller normal force", "the propellers' destabilising normal-force slope in the neutral point", "De Young, J. Aircraft 2 (1965); prop_lab's cn_factor when on file",
     "advance ratio below ~1.5, small inflow angles", "±30 %; it moves the static margin by ~0.01 chord", "§ 6"),
    ("Glauert's actuator disk", "the slipstream velocity the blown spans see in the transition model", "momentum theory, uniform jet, fully developed at the wing", "a wing close behind the disk sees less than the fully developed Δv", "±20 % on the blown lift", "§ 9"),
    ("plane frame and Euler buckling", "spar stress and deflection, strut forces, buckling margins", "beam and truss statics; Euler columns pin-ended", "linear elastic, the joints ideal; the FEA case checks it", "the joint stiffness is not in it: ±25 % on the deflection", "§ 8"),
    ("material allowables", "carbon tube 400 MPa (compression, knock-down 0.8), PETG 45 MPa in plane / 27 MPa across the layers (knock-down 0.7)", "supplier data and printed-coupon literature; no coupon test of these parts",
     "room temperature, dry; printed parts at 100 % infill", "±25 % until coupons are tested", "§ 8"),
    ("vortex lattice (PEREGRINE)", "the lift slope, the span efficiency, the neutral point of the wing pair", "a first-order lattice, inviscid", "attached flow, small angles: it knows nothing of the stall", "±5 % on the lift slope (the campaign's sweep read within 2 %)", "§ 5, § 6"),
]


def correlations_table():
    import pandas as pd
    return pd.DataFrame(CORRELATIONS, columns=["correlation", "used for", "source", "applicability", "uncertainty", "section"]).set_index("correlation")


# ================================================================================================= prerequisites
def drive_table_status() -> dict:
    """Whether ``ks.drive_table()`` will read ``data/kestrel_drive.json`` or compute (Boreas, ~40 s): the notebook says
    so before it happens — no expensive work hides in a loader."""
    on_file = False
    if ks.DRIVE_JSON.exists():
        doc = json.loads(ks.DRIVE_JSON.read_text())
        on_file = doc.get("components") == json.loads(json.dumps(ks._drive_key()))
    return {"on_file": on_file, "path": str(ks.DRIVE_JSON.relative_to(ROOT)), "note": "on file for this motor, propeller, battery and calibration" if on_file else "will be computed here with Boreas (~40 s) and saved"}


def sections_status() -> dict:
    try:
        import kestrel_section as ksec
        ok = ksec.SECTIONS_JSON.exists()
        return {"on_file": ok, "path": str(ksec.SECTIONS_JSON.relative_to(ROOT)), "note": "NeuralFoil polars on file" if ok else "missing: the 0012's numbers stand in (kestrel2.section_numbers)"}
    except Exception as exc:
        return {"on_file": False, "path": "designs/data/kestrel_sections.json", "note": f"vegeta.zephyr not importable ({exc}); the 0012's numbers stand in"}


def runs_from_gate(gate) -> tuple[dict, set]:
    """The notebook's Yanus gate (``yanus.enter("43_kestrel_study")``: the runs pinned in
    ``notebooks/pins/43_kestrel_study.json``) as the ``RUNS`` / ``REQUIRED`` pair ``prerequisites`` reads —
    ``{"airframe_cfd": {variant: {campaign: run dir | None}}, "structure_fea": {variant: run dir | None}}`` and the
    required ``("airframe_cfd", variant, campaign)`` / ``("structure_fea", variant)``. Every run take of the
    contract appears (an unpinned one as ``None``: NOT RUN, or the error when it is required)."""
    from vegeta.yanus import Run
    pins_ = __import__("vegeta.yanus.pins", fromlist=["read_pins"]).read_pins(gate.flow.root, gate.unit.name)
    runs, required = {"airframe_cfd": {}, "structure_fea": {}}, set()
    for t in gate.unit.takes:
        if not isinstance(t, Run):
            continue
        ch = gate.flow.chapter(t.chapter)
        sel = pins_.get(t.key())
        wf, v = ch.kind.workflow, t.variant
        if wf == "airframe_cfd":
            runs[wf].setdefault(v, {})[ch.campaign] = sel
            if t.required and not gate.screening:
                required.add((wf, v, ch.campaign))
        elif wf == "structure_fea":
            runs[wf][v] = sel
            if t.required and not gate.screening:
                required.add((wf, v))
    return runs, required


def prerequisites(runs: dict, required: set, variants=VARIANTS, *, fingerprints: dict | None = None):
    """The prerequisite table and the usable summaries. ``runs`` =
    ``{"airframe_cfd": {variant: {campaign: run dir | None}}, "structure_fea": {variant: run dir | None}}``;
    ``required`` = a set of ``("airframe_cfd", variant, campaign)`` / ``("structure_fea", variant)``. Every row is
    checked (existence, state, fingerprint) before a single exception lists **all** the missing required inputs with
    their commands — never a substitute, never a solver launched here."""
    import pandas as pd
    fps = fingerprints or {v: kcp.fingerprint(v) for v in variants}
    rows, summaries, missing = [], {"airframe_cfd": {}, "structure_fea": {}}, []
    pl = ks.prop_lab_calibration()
    rows.append({"input": "prop_lab calibration (notebook 38)", "role": "optional", "selection": str(ks.PROP_LAB_JSON.relative_to(ROOT)), "status": (
        f"on file: cl ×{pl['cl_scale']:.3f}, cd ×{pl['cd_scale']:.3f}, CNα ×{pl['cn_factor']:.3f} ({pl['meta'].get('date', '')})" if pl["on_file"]
        else "NOT RUN or rejected: Boreas's generic blade section and De Young's normal force" + (f" (rejected: {'; '.join(pl['rejected'])[:120]})" if pl.get("rejected") else "")),
        "provenance": pl["meta"].get("source", ""), "compatibility": "gates in prop_lab.GATES" if pl["on_file"] else "", "usable": pl["on_file"]})
    ds = drive_table_status()
    rows.append({"input": "drive tables (Boreas)", "role": "required", "selection": ds["path"], "status": ds["note"], "provenance": "kestrel_systems.compute_drive_table", "compatibility": "keyed on motor, propeller, battery, calibration", "usable": True})
    ss = sections_status()
    rows.append({"input": "section polars (Zephyr / NeuralFoil)", "role": "optional", "selection": ss["path"], "status": ss["note"], "provenance": "kestrel_section.table", "compatibility": "keyed on candidates, flap chord, n_crit", "usable": ss["on_file"]})
    for v in variants:
        for camp, rd in (runs.get("airframe_cfd", {}).get(v) or {}).items():
            req = ("airframe_cfd", v, camp) in required
            ck = kcp.check_run(rd, workflow="airframe_cfd", variant=v, campaign=camp, expected=fps[v], blown=camp in ("blown", "cruise", "transitions", "all"))
            if rd and not ck["ok"] and "legacy_" in str(rd) and ck["status"].startswith("not found"):
                ck["command"] = f"{kcp.PYTHON} {kcp.SCRIPTS['import_legacy']}"
            rows.append({"input": f"airframe CFD, {v}, {camp}", "role": "required" if req else "optional", "selection": str(rd) if rd else "—", "status": ck["status"],
                         "provenance": ck["provenance"], "compatibility": (ck["compat"] + (": " + "; ".join(ck["reasons"]) if ck["reasons"] else "")) if ck["compat"] else "", "usable": ck["ok"]})
            if ck["ok"]:
                summaries["airframe_cfd"].setdefault(v, {})[camp] = ck["summary"]
            elif req:
                missing.append((f"airframe CFD of Kestrel-{v}, campaign {camp}", ck))
        rd = runs.get("structure_fea", {}).get(v)
        if rd is not None or ("structure_fea", v) in required:
            req = ("structure_fea", v) in required
            ck = kcp.check_run(rd, workflow="structure_fea", variant=v, campaign="all", expected=fps[v])
            rows.append({"input": f"structural FEA, {v}", "role": "required" if req else "optional", "selection": str(rd) if rd else "—", "status": ck["status"],
                         "provenance": ck["provenance"], "compatibility": (ck["compat"] + (": " + "; ".join(ck["reasons"]) if ck["reasons"] else "")) if ck["compat"] else "", "usable": ck["ok"]})
            if ck["ok"]:
                summaries["structure_fea"][v] = ck["summary"]
            elif req:
                missing.append((f"structural FEA of Kestrel-{v}", ck))
    table = pd.DataFrame(rows).set_index("input")
    if missing:
        lines = ["required workflow results are missing or unusable — this notebook never runs a solver in their place:"]
        for what, ck in missing:
            lines.append(f"  • {what}: {ck['status']}\n      run:  {ck['command']}\n      then pin its run directory: yn pin 43_kestrel_study <chapter code> --variant <v> <run dir>")
        lines.append("  For an analytical screening pass without them: KESTREL_SCREENING=1 (the final assessment is then labelled unvalidated).")
        raise kcp.PrerequisiteError("\n".join(lines))
    return table, summaries


# ================================================================================================= the variants
def build(variant: str, *, aero: dict | None = None, sec: dict | None = None) -> dict:
    """One variant's cheap numbers from the design modules: parameters, layout, mass table, CG and inertia, the
    polar, the neutral point, the hover point, the level-flight table, the drag build-up, the transition vehicle."""
    p = k2.params(variant)
    L = kestrel.layout(p)
    a = aero or kf.aero(p)
    sec = sec or k2.section_numbers()
    table = ks.mass_table(p)
    ci = ks.cg_inertia(table)
    pol = k2.polar_for(variant, p, a, sec)
    veh = kt.make_vehicle(p, aero=a, pol=pol, name=f"Kestrel-{variant}")
    veh.elevon_tau = sec["tau"]
    np_ = kf.neutral_point(p, x_cg=ci["x_cg"], a=a)
    hp = ks.hover_point(ci["mass_kg"])
    d = kf.drag_buildup(p, faired=bool(p["strut_fairing"]))
    cru = kf.cruise(p, mass_kg=ci["mass_kg"], x_cg=ci["x_cg"], z_cg=ci["z_cg"], pol=pol)
    return {"variant": variant, "p": p, "L": L, "aero": a, "sec": sec, "table": table, "ci": ci, "pol": pol, "veh": veh, "np": np_, "hover": hp, "drag": d, "cruise": cru,
            "V_stall": kf.stall_speed(p, mass_kg=ci["mass_kg"], pol=pol), "layout_rules": kestrel.Kestrel4Q.checks(p)}


def build_all(variants=VARIANTS) -> dict:
    a = kf.aero(k2.params(variants[0]))               # the same wings in every variant: one lattice
    sec = k2.section_numbers()
    return {v: build(v, aero=a, sec=sec) for v in variants}


def hover_analytic(s: dict, *, FM: float = kl.ENVELOPE_ASSUMPTIONS["figure of merit in the hover"]) -> dict:
    """Momentum theory beside Boreas: P_ideal = T^{3/2} / sqrt(2 ρ A) per rotor, the electrical power at the assumed
    FM and a 0.8 drive efficiency."""
    T = s["ci"]["mass_kg"] * ks.G / 4
    A = s["L"]["A_disc"]
    P_i = T ** 1.5 / math.sqrt(2 * ks.RHO0 * A)
    return {"thrust per rotor [N]": T, "disc loading [N/m²]": T / A, "induced velocity [m/s]": math.sqrt(T / (2 * ks.RHO0 * A)), "ideal power, one rotor [W]": P_i,
            f"electrical at FM {FM:g}, η 0.8, four rotors [W]": 4 * P_i / FM / 0.8, "Boreas FM": s["hover"]["figure_of_merit"], "Boreas electrical, four rotors [W]": s["hover"]["total_W"]}


def variants_table(sets: dict):
    """The variants side by side: what each costs and gives (the methods in the last column)."""
    import pandas as pd
    rows = {}
    for v, s in sets.items():
        c16 = s["cruise"].loc[V_CRUISE] if V_CRUISE in s["cruise"].index else None
        feas = s["cruise"][s["cruise"]["feasible"]]
        rows[f"Kestrel-{v}"] = {"take-off mass [kg]": s["ci"]["mass_kg"], "CG x [mm]": s["ci"]["x_cg"] * 1e3, "static margin [chords]": s["np"]["static_margin"],
                               "Cd0": s["pol"].cd0, "CL max (polar)": s["pol"].cl_max, "stall speed [m/s]": s["V_stall"], "stall / cruise": s["V_stall"] / V_CRUISE,
                               "hover T/W": s["hover"]["T_W"], "hover electrical [W]": s["hover"]["total_W"],
                               f"L/D at {V_CRUISE:g} m/s": float(c16["L/D"]) if c16 is not None and c16["feasible"] else np.nan,
                               f"power at {V_CRUISE:g} m/s [W]": float(c16["electrical [W]"]) if c16 is not None and c16["feasible"] else np.nan,
                               f"endurance at {V_CRUISE:g} m/s [min]": float(c16["endurance [min]"]) if c16 is not None and c16["feasible"] else np.nan,
                               "best endurance [min]": float(feas["endurance [min]"].max()) if len(feas) else np.nan,
                               "trim by thrust alone at the cruise": bool(c16["trim by thrust"]) if c16 is not None and c16["feasible"] else False,
                               "section": s["pol"].source if s["pol"].source != "model" else "NACA 0012", "elevons": bool(s["p"]["elevons"]), "bracing": s["p"]["bracing"],
                               "struts": "faired" if s["p"]["strut_fairing"] else "round"}
    df = pd.DataFrame(rows).T
    df.attrs["method"] = "mass table (estimate), Boreas BEMT (solver), lattice + drag build-up (solver + empirical), NeuralFoil cl_max (empirical)"
    return df


def mass_groups(sets: dict):
    import pandas as pd
    return pd.DataFrame({f"Kestrel-{v}": ks.groups(s["table"]) for v, s in sets.items()}).T


def envelope_vs_tools(s: dict) -> "pd.DataFrame":
    """Iteration 0's envelope (spec + layout + five assumptions, analytic) against the tools, for the main variant."""
    import pandas as pd
    env = kl.envelope(s["p"], mass_kg=2.0)
    c16 = s["cruise"].loc[V_CRUISE]
    W = s["ci"]["mass_kg"] * ks.G
    CL16 = W / (0.5 * ks.RHO0 * V_CRUISE ** 2 * s["L"]["S_ref"])
    d = {"envelope [analytic]": {"take-off mass [kg]": env["take-off mass [kg]"], "cruise CL": env["cruise CL"], "CL max": env["CL max assumed"], "stall speed [m/s]": env["stall speed [m/s]"],
                                 "hover power [W]": env["hover power, momentum theory [W]"], f"cruise power at {V_CRUISE:g} m/s [W]": env["cruise power [W]"], "L/D at the cruise": env["assumptions"]["L/D at the cruise"],
                                 "endurance at the cruise [min]": env["endurance at the cruise [min]"], "static margin": env["static margin [chords]"]},
         "tools [solver + empirical]": {"take-off mass [kg]": s["ci"]["mass_kg"], "cruise CL": CL16, "CL max": s["pol"].cl_max, "stall speed [m/s]": s["V_stall"], "hover power [W]": s["hover"]["total_W"],
                                        f"cruise power at {V_CRUISE:g} m/s [W]": c16["electrical [W]"], "L/D at the cruise": c16["L/D"], "endurance at the cruise [min]": c16["endurance [min]"],
                                        "static margin": s["np"]["static_margin"]}}
    df = pd.DataFrame(d)
    df["tools / envelope"] = df.iloc[:, 1] / df.iloc[:, 0]
    return df


# ================================================================================================= aerodynamics and the stall
def biplane_check(a: dict, L: dict) -> dict:
    """Prandtl's factor from gap/span against the lattice's: σ ≈ (1 − 0.66 g/b) / (1.05 + 3.7 g/b) (Prandtl's curve fitted), the
    induced-drag ratio (1 + σ)/2."""
    gb = L["gap"] / L["span"]
    sigma = (1 - 0.66 * gb) / (1.05 + 3.7 * gb)
    return {"gap / span": gb, "Prandtl σ": sigma, "induced drag / monoplane, Prandtl": (1 + sigma) / 2, "induced drag / monoplane, lattice": a["factors"]["induced drag vs the monoplane"],
            "lift slope / two isolated wings, lattice": a["factors"]["lift slope vs two isolated wings"]}


def cfd_sweep(summary: dict | None, s: dict) -> dict:
    """What a CFD run says about the bare airframe: the comparison frame, the numbers, the convergence count and
    the label the notebook must print (``validated`` only when the sweep converged)."""
    if not summary:
        return {"label": "NOT RUN", "cmp": None, "numbers": {}, "n_bare": 0, "n_converged": 0, "rt": None}
    rt = kcp.records_frame(summary)
    bare = rt[~rt["rotor disks"].astype(bool)] if len(rt) else rt
    cmp_ = kcfd.sweep_comparison(rt, s["veh"]) if len(bare) else None
    numbers = kcfd.sweep_numbers(cmp_) if cmp_ is not None and len(cmp_) else {}
    n_conv = int(bare["converged"].fillna(False).astype(bool).sum()) if len(bare) else 0
    label = "validated" if len(bare) and n_conv == len(bare) else ("solver (unconverged: indicative)" if len(bare) else "NOT RUN")
    return {"label": label, "cmp": cmp_, "numbers": numbers, "n_bare": int(len(bare)), "n_converged": n_conv, "rt": rt, "quality": summary.get("quality"),
            "provenance": summary.get("provenance", {})}


def stall_decision(sets: dict, cfd: dict) -> "pd.DataFrame":
    """CL max and the stall speed per variant — the polar's (NeuralFoil ×0.9, empirical) and the CFD sweep's (the largest
    CL below 20° on the sweep's angles: a lower bound on a 5° grid) — and what each means for the cruise margin and for
    the wing area that would give 1.3 × stall at the cruise."""
    import pandas as pd
    rows = {}
    for v, s in sets.items():
        W, S = s["ci"]["mass_kg"] * ks.G, s["L"]["S_ref"]
        row = {"CL max, polar": s["pol"].cl_max, "stall speed, polar [m/s]": s["V_stall"], "cruise / stall, polar": V_CRUISE / s["V_stall"]}
        c = cfd.get(v) or {}
        num = c.get("numbers") or {}
        if num:
            clc = num["CFD"]["CL max below 20°"]
            vs = math.sqrt(2 * W / (ks.RHO0 * S * clc))
            row.update({"CL max, CFD sweep (< 20°)": clc, "at α [deg]": num["CFD"]["α of it [deg]"], "stall speed, CFD [m/s]": vs, "cruise / stall, CFD": V_CRUISE / vs,
                        "CFD label": c["label"], "CFD converged": f"{c['n_converged']} of {c['n_bare']}",
                        "wing area for cruise = 1.3 × stall on the CFD's CL max [m²]": 2 * W / (ks.RHO0 * (V_CRUISE / 1.3) ** 2 * clc), "wing area now [m²]": S})
        else:
            row.update({"CL max, CFD sweep (< 20°)": np.nan, "CFD label": "NOT RUN"})
        rows[f"Kestrel-{v}"] = row
    return pd.DataFrame(rows).T


# ================================================================================================= trim, control, structure
def trim_table(s: dict) -> "pd.DataFrame":
    cru = s["cruise"]
    keep = [v for v in (10.0, 12.0, 14.0, 16.0, 20.0, 25.0) if v in cru.index]
    cols = [c for c in ("CL", "L/D", "upper rotor [N]", "lower rotor [N]", "trim by thrust", "electrical [W]", "endurance [min]", "range [km]", "feasible") if c in cru.columns]
    return cru.loc[keep, cols]


def control_table(s: dict) -> dict:
    cg = {"x_cg": s["ci"]["x_cg"], "y_cg": s["ci"]["y_cg"], "z_cg": s["ci"]["z_cg"]}
    d = s["veh"].drive
    rank = kc.rank_table(s["p"], kappa=float(d.kap(0.0)), cg=cg)
    ha = kc.hover_authority(s["p"], mass_kg=s["ci"]["mass_kg"], cg=cg, inertia=s["ci"]["I"], drive=d)
    out = {"rank": rank, "hover authority": ha}
    if s["p"]["elevons"]:
        out["elevon hinge"] = kc.elevon_hinge(s["p"], V=V_CRUISE)
        out["elevon authority"] = kc.elevon_authority(s["p"], mass_kg=s["ci"]["mass_kg"], cg=cg, tau=s["veh"].elevon_tau, cl_alpha=s["pol"].cl_alpha)
    return out


def structure_screen(s: dict) -> dict:
    """The plane frame (beam and truss statics) at 4 g, even lift and the 75 % wing, the stiffness and first modes per
    bracing, the ground stance — all analytic, the FEA's reference."""
    p, m = s["p"], s["ci"]["mass_kg"]
    return {"even lift": kst.compare_bracing(p, mass_kg=m), "upper wing 75 %": kst.compare_bracing(p, mass_kg=m, upper_share=0.75),
            "modes": kst.stiffness_and_modes(p), "stance": kst.ground_stance(p, mass_kg=m, x_cg=s["ci"]["x_cg"], z_cg=s["ci"]["z_cg"]),
            "reference": kfe.plane_frame_reference(p, mass_kg=m)}


def fea_table(summary: dict | None) -> "pd.DataFrame | None":
    """A structural run's cases as a frame (status, margins, frequencies) — ``None`` when there is no run."""
    import pandas as pd
    if not summary:
        return None
    rows = {}
    for n, c in summary["cases"].items():
        rows[n] = {k: v for k, v in c.items() if k not in ("artifacts", "duration_s", "process", "strength_MPa", "strength_z_MPa")}
    return pd.DataFrame(rows).T


def fea_against_frame(summary: dict | None, s: dict) -> "pd.DataFrame | None":
    """The solved static frame cases beside the plane frame's ultimate spar stress (the number the FEA is read against)."""
    import pandas as pd
    if not summary:
        return None
    ref = summary.get("plane_frame_reference", {})
    rows = {}
    for n, c in summary["cases"].items():
        if c["status"] != "solved" or c["kind"] != "static" or c["part"] != "frame":
            continue
        r = ref.get(n, {})
        rows[n] = {"FEA nominal von Mises [MPa]": c.get("nominal von Mises [MPa]"), "FEA peak [MPa]": c.get("peak von Mises [MPa]"), "plane frame spar stress, ultimate [MPa]": r.get("spar stress, ultimate [MPa]"),
                   "FEA max displacement [mm]": c.get("max displacement [mm]"), "plane frame tip deflection, ultimate [mm]": r.get("tip deflection, ultimate [mm]"), "margin (knocked-down allowable)": c.get("margin")}
    return pd.DataFrame(rows).T if rows else None


# ================================================================================================= transitions
def transitions(sets: dict, *, main: str = MAIN, cfd_correction=None, headwind: float = HEADWIND, t_tr: float = 3.0, remedies: dict | None = None) -> dict:
    """The decision-relevant transitions only (each episode is ~8 s of integration): every variant forward and back
    at the nominal schedule in still air, the back transition into a headwind; the main variant's back transition
    with its remedies (reverse thrust, the crow brake with it); the main variant again on the CFD's correction when
    one is given. Returns the metrics table (a ``method`` column) and the episodes."""
    import pandas as pd
    remedies = remedies if remedies is not None else {"reverse thrust 3 N per rotor": {"reverse_N": k2.REVERSE_N},
                                                      f"crow brake ±{k2.AIRBRAKE_DEG:g}° + reverse thrust": {"airbrake_deg": k2.AIRBRAKE_DEG, "reverse_N": k2.REVERSE_N}}
    rows, eps = {}, {}
    for v, s in sets.items():
        for kind in ("forward", "back"):
            ep = kt.simulate(s["veh"], kind, t_tr=t_tr)
            rows[(f"Kestrel-{v}", kind, "as is, still air")] = dict(ep.metrics, method="reduced-order model (solver)")
            eps[(v, kind, "as is, still air")] = ep
        ep = kt.simulate(s["veh"], "back", t_tr=t_tr, headwind=headwind)
        rows[(f"Kestrel-{v}", "back", f"as is, headwind {headwind:g} m/s")] = dict(ep.metrics, method="reduced-order model (solver)")
        eps[(v, "back", f"as is, headwind {headwind:g} m/s")] = ep
    for v in [vv for vv in sets if vv != "4Q-1"]:
        for name, kw in remedies.items():
            ep = kt.simulate(sets[v]["veh"], "back", t_tr=t_tr, **kw)
            rows[(f"Kestrel-{v}", "back", name)] = dict(ep.metrics, method="reduced-order model (solver)")
            eps[(v, "back", name)] = ep
    if cfd_correction is not None:
        s = sets[main]
        vc = kt.make_vehicle(s["p"], aero=s["aero"], pol=s["pol"], airframe=cfd_correction, name=f"Kestrel-{main}, CFD-corrected")
        vc.elevon_tau = s["veh"].elevon_tau
        for kind in ("forward", "back"):
            ep = kt.simulate(vc, kind, t_tr=t_tr)
            rows[(f"Kestrel-{main}", kind, f"as is, CFD {cfd_correction.source} correction")] = dict(ep.metrics, method="reduced-order model on the CFD's airframe correction")
            eps[(main, kind, f"cfd {cfd_correction.source}")] = ep
    df = pd.DataFrame(rows).T
    df.index.names = ["vehicle", "direction", "case"]
    return {"table": df, "episodes": eps}


TRANSITION_COLS = ["height loss [m]", "height gain [m]", "pitch error max [deg]", "time stalled, free stream [s]", "least pitch margin", "peak rotor thrust / T_max",
                   "elevon max [deg]", "roll demand / authority", "energy [Wh]", "pass"]


def transition_columns(df):
    cols = [c for c in TRANSITION_COLS if c in df.columns]
    out = df[cols].copy()
    for c in cols:
        if c != "pass":
            out[c] = out[c].astype(float)
    out["method"] = df["method"]
    return out


def deceleration(sets: dict, *, V0: float = V_CRUISE, V1: float = 13.0) -> "pd.DataFrame":
    import pandas as pd
    rows = {}
    for v, s in sets.items():
        rows[(f"Kestrel-{v}", "rotors at idle")] = k2.decelerate_level(s["veh"], V0, V1)
        if s["p"]["elevons"]:
            rows[(f"Kestrel-{v}", f"crow brake ±{k2.AIRBRAKE_DEG:g}°")] = k2.decelerate_level(s["veh"], V0, V1, airbrake_deg=k2.AIRBRAKE_DEG)
        if v != "4Q-1":
            rows[(f"Kestrel-{v}", f"reverse thrust {k2.REVERSE_N:g} N")] = k2.decelerate_level(s["veh"], V0, V1, reverse_N=k2.REVERSE_N)
    df = pd.DataFrame(rows).T
    df.index.names = ["vehicle", "means"]
    return df


# ================================================================================================= performance and energy
def performance(sets: dict) -> dict:
    perfs = {f"Kestrel-{v}": k2.performance(s["veh"]) for v, s in sets.items()}
    return {"perfs": perfs, "pursuit": k2.pursuit(perfs)}


def energy_budget(s: dict) -> dict:
    """The battery's usable energy and where a sortie spends it: the hover (take-off, landing, two transitions'
    worth), the cruise at 16 m/s, the loiter at the best-endurance speed."""
    b = ks.COMPONENTS["battery"]
    Ewh = b["cells"] * 3.7 * b["capacity_ah"] * b["usable"]
    c16 = s["cruise"].loc[V_CRUISE]
    feas = s["cruise"][s["cruise"]["feasible"]]
    best = feas["endurance [min]"].idxmax()
    hover_min = 1.5
    E_hover = s["hover"]["total_W"] * hover_min / 60
    return {"usable energy [Wh]": Ewh, f"hover, {hover_min:g} min (take-off, landing, transitions) [Wh]": E_hover, "hover power [W]": s["hover"]["total_W"],
            f"cruise power at {V_CRUISE:g} m/s [W]": float(c16["electrical [W]"]), f"cruise time on the rest [min]": (Ewh - E_hover) / float(c16["electrical [W]"]) * 60,
            "best-endurance speed [m/s]": float(best), "power there [W]": float(feas.loc[best, "electrical [W]"]), "loiter time on the rest [min]": (Ewh - E_hover) / float(feas.loc[best, "electrical [W]"]) * 60,
            f"range at {V_CRUISE:g} m/s on the rest [km]": (Ewh - E_hover) / float(c16["electrical [W]"]) * V_CRUISE * 3.6}


# ================================================================================================= case selection and the assessment
def recommended_campaigns(sets: dict, cfd: dict, fea_summaries: dict, *, main: str = MAIN) -> "pd.DataFrame":
    """From the screening: which workflow campaigns settle a decision, in order, with their commands. The engineer
    launches them; the notebook only lists them."""
    import pandas as pd
    s = sets[main]
    rows = []
    c = cfd.get(main) or {}
    a_stall = math.degrees(s["pol"].alpha_stall) if hasattr(s["pol"], "alpha_stall") else 11.0
    if not c.get("numbers"):
        rows.append(("1", "airframe_cfd", main, "sweep", "screening", "no bare sweep on file: the lattice's lift slope, Cd0 and Cmα against RANS", kcp.command("airframe_cfd", variant=main, campaign="sweep", quality="screening")))
    if c.get("label") != "validated":
        rows.append(("2", "airframe_cfd", main, "stall", "fine",
                     f"CL max decides the stall speed and the wing area: the polar stalls at {a_stall:.0f}°; the screening sweep (if any) was unconverged on a 5° grid", kcp.command("airframe_cfd", variant=main, campaign="stall", quality="fine")))
    rows.append(("3", "airframe_cfd", main, "cruise", "screening", "the level cruise and the stall point with the rotor disks on: the blown spans' lift and the trim", kcp.command("airframe_cfd", variant=main, campaign="cruise", quality="screening")))
    if main not in fea_summaries:
        rows.append(("4", "structure_fea", main, "all", "—", "the frame's margins at ultimate and the joint's across-layer tension: no FEA on file", kcp.command("structure_fea", variant=main, campaign="all")))
    rows.append(("5", "airframe_cfd", main, "transitions", "screening", "the transition points with the disks (the model's least-known regime: blown, stalled); only after the bare sweep converged", kcp.command("airframe_cfd", variant=main, campaign="transitions", quality="screening")))
    df = pd.DataFrame(rows, columns=["order", "workflow", "variant", "campaign", "quality", "why", "command"]).set_index("order")
    return df


def assessment(sets: dict, cfd: dict, fea_summaries: dict, trans: dict, perf: dict, *, main: str = MAIN, screening: bool = False) -> dict:
    """The final checks of the main variant with the method behind each, and the verdict lines."""
    s = sets[main]
    c16 = s["cruise"].loc[V_CRUISE]
    tt = trans["table"]
    mine = tt.xs(f"Kestrel-{main}", level=0)
    asis = mine[mine.index.get_level_values("case").str.startswith("as is")]
    fwd_ok = bool(asis.xs("forward", level=0)["pass"].astype(bool).all())
    back_asis = asis.xs("back", level=0)
    back_asis_ok = bool(back_asis["pass"].astype(bool).all())
    rem = mine.xs("back", level=0)
    rem = rem[~rem.index.str.startswith("as is")]
    rem_ok = bool(rem["pass"].astype(bool).any()) if len(rem) else False
    cf = cfd.get(main) or {}
    num = cf.get("numbers") or {}
    checks = [kl.check("take-off mass [kg]", s["ci"]["mass_kg"], note="mass table: component data + assumed structure densities (estimate)"),
              kl.check("hover thrust / weight", s["hover"]["T_W"], note="Boreas BEMT, generic section unless prop_lab is on file (solver)"),
              kl.check("static margin [chords]", s["np"]["static_margin"], note="lattice + De Young propellers (solver + empirical)"),
              kl.check("stall speed / cruise speed", s["V_stall"] / V_CRUISE, note=f"polar's CL max {s['pol'].cl_max:.2f} (NeuralFoil ×0.9, empirical)"),
              kl.check("endurance at the cruise [min]", float(c16["endurance [min]"]), note="drag build-up + BEMT (solver + empirical, ±20 % on Cd0)"),
              kl.check("forward transitions pass (nominal, still air)", fwd_ok, "==", True, note="reduced-order model (solver); Viterna post-stall"),
              kl.check("back transitions pass as is (still air and headwind)", back_asis_ok, "==", True, note="reduced-order model (solver)"),
              kl.check("a back-transition remedy passes", rem_ok, "==", True, note="reverse thrust / crow brake in the model (solver); the drives' reverse thrust is an assumption until prop_lab measures it")]
    if num:
        checks.append(kl.check("CFD sweep: lift slope / lattice", num["CFD"]["CLα [1/rad]"] / num["model"]["CLα [1/rad]"], "between", (0.85, 1.15), note=f"{cf['label']}; {cf['n_converged']} of {cf['n_bare']} converged"))
        checks.append(kl.check("CFD sweep: CL max (< 20°) / polar", num["CFD"]["CL max below 20°"] / s["pol"].cl_max, "≥", 0.8, note=f"{cf['label']}: a 5° grid's lower bound; the fine stall campaign decides"))
    else:
        checks.append(kl.check("CFD sweep on file", False, "==", True, note="NOT RUN: run workflows/kestrel/airframe_cfd.py --campaign sweep"))
    fs = fea_summaries.get(main)
    if fs:
        solved = [c for c in fs["cases"].values() if c["status"] == "solved" and c["kind"] == "static"]
        if solved:
            checks.append(kl.check("least FEA margin, frame and joint (ultimate, knocked-down allowable)", min(float(c["margin"]) for c in solved if c.get("margin") is not None), "≥", 0.0, note="Talos / CalculiX (solver)"))
            lay = [c for c in solved if c.get("margin (layers)") is not None]
            if lay:
                checks.append(kl.check("joint across-layer margin", min(float(c["margin (layers)"]) for c in lay), "≥", 0.0, note="PETG 27 MPa across layers ×0.7 (assumed allowable)"))
        else:
            checks.append(kl.check("FEA cases solved", 0, "≥", 1, note=f"the run completed with {fs['counts']}"))
    else:
        checks.append(kl.check("structural FEA on file", False, "==", True, note="NOT RUN: run workflows/kestrel/structure_fea.py; the plane frame's margins stand in (analytic)"))
    tab = kl.check_table(checks, f"Kestrel-{main} — final checks")
    held, of = tab.attrs["passes"], tab.attrs["of"]
    pur = perf["pursuit"]
    lines = [f"Kestrel-{main}: {held} of {of} checks hold" + (" (screening pass: unvalidated — CFD and FEA labelled NOT RUN where missing)" if screening else "") + ".",
             f"Mass {s['ci']['mass_kg']:.2f} kg, static margin {s['np']['static_margin']:.3f} c, hover T/W {s['hover']['T_W']:.2f} at {s['hover']['total_W']:.0f} W; "
             f"cruise {V_CRUISE:g} m/s at {float(c16['electrical [W]']):.0f} W for {float(c16['endurance [min]']):.0f} min (L/D {float(c16['L/D']):.1f}); stall {s['V_stall']:.1f} m/s on the polar's CL max {s['pol'].cl_max:.2f}.",
             (f"The CFD sweep ({cf.get('label')}) reads CL max {num['CFD']['CL max below 20°']:.2f} below 20° against the polar's {s['pol'].cl_max:.2f}: if it holds, the stall is "
              f"{math.sqrt(2 * s['ci']['mass_kg'] * ks.G / (ks.RHO0 * s['L']['S_ref'] * num['CFD']['CL max below 20°'])):.1f} m/s and the cruise only {V_CRUISE / math.sqrt(2 * s['ci']['mass_kg'] * ks.G / (ks.RHO0 * s['L']['S_ref'] * num['CFD']['CL max below 20°'])):.2f} × it — the fine stall campaign decides the wing area." if num else
              "No CFD sweep on file: the stall speed stands on NeuralFoil's cl_max and the Viterna blend, which are not a stall model."),
             f"Transitions (reduced-order model): forward {'pass' if fwd_ok else 'FAIL'}; back as is {'pass' if back_asis_ok else 'FAIL'}; with a remedy {'pass' if rem_ok else 'FAIL'}.",
             (f"FEA: {fs['counts']['solved']} of {fs['counts']['n']} cases solved." if fs else "FEA: NOT RUN — the plane frame's spar reserve stands in."),
             (f"Pursuit: V max {float(pur.loc['V max, thrust = drag [m/s]', f'Kestrel-{main}']):.1f} m/s (thrust-limited), corner radius {float(pur.loc['turn radius at the corner [m]', f'Kestrel-{main}']):.1f} m, "
              f"best climb {float(pur.loc['best climb rate [m/s]', f'Kestrel-{main}']):.1f} m/s." if "V max, thrust = drag [m/s]" in pur.index else "")]
    return {"checks": checks, "table": tab, "lines": [l for l in lines if l], "held": held, "of": of}


class Stopwatch:
    """The notebook's measured runtime per section (printed at the end and exported: a runtime is reported, never claimed)."""

    def __init__(self):
        self.t0 = time.monotonic()
        self.laps = {}
        self._last = self.t0

    def lap(self, name: str) -> float:
        now = time.monotonic()
        self.laps[name] = round(now - self._last, 1)
        self._last = now
        return self.laps[name]

    def total(self) -> float:
        return round(time.monotonic() - self.t0, 1)

    def frame(self):
        import pandas as pd
        return pd.Series({**self.laps, "total": self.total()}, name="seconds").to_frame()
