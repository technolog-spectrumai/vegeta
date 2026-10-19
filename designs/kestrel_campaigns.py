"""kestrel_campaigns — what the Kestrel workflows run and what the Kestrel notebook reads back.

The engineering content of the thin scripts in ``side_labs/kestrel/`` and of the unified notebook's prerequisite
checks, kept here so that both read the same definitions:

- the **variants** (``4Q-1``, ``4Q-2``, ``4Q-3``: ``kestrel2.params``) and each one's **fingerprint** — the resolved
  CAD parameters and their hash, the CAD source's hash, the mass and CG the cases are set up for, the propeller
  calibration the rotor disks and the drive tables carry — written into every run's manifest and compared by the
  notebook before a run is used (``compare``: file existence is not compatibility);
- the **CFD campaigns** of the airframe with the propellers as rotor disks (``kestrel_cfd``: the bare sweeps, the
  decision-relevant stall angles, the level cruise and the stall, the transition points, the section slabs) and the
  **FEA campaigns** (``kestrel_fea``: the carbon frame of the variant's bracing, the printed joint): each a named list of
  cases an engineer launches knowingly — a script never widens it;
- the **results contract**: one case's outcome from the tool's result (status, process, convergence, the numbers), the
  run's ``summary.json`` (scalars with their units and reference quantities, the case statuses, the validity domain of
  the sampled maps, CSV files for the curves) and the legacy import of the first campaign's results file;
- the notebook side: ``load_cfd`` / ``load_fea`` by explicit run path, ``require_*`` raising ``PrerequisiteError`` with
  the exact command that produces what is missing. Nothing here runs a solver.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

import kestrel
import kestrel_cfd as kcfd
import kestrel_fea as kfe
import kestrel_systems as ks
import kestrel_transition as kt
import kestrel2 as k2

DESIGNS = Path(__file__).resolve().parent
ROOT = DESIGNS.parent
import os as _os
RUNS_BASE = Path(_os.environ.get("VEGETA_WORKFLOW_RUNS") or ROOT / "side_labs" / "runs")
BOOK = "kestrel"                                                                  # the book: side_labs/runs/kestrel/<chapter>/


def _flow():
    """The one chapter map: ``side_labs/flow.py`` (``side_labs/registry.py``), the same one ``side_labs/library.py`` reads."""
    import sys
    if str(ROOT / "side_labs") not in sys.path:
        sys.path.insert(0, str(ROOT / "side_labs"))
    from registry import load_flow
    return load_flow(ROOT / "side_labs" / "flow.py")


_BOOK = _flow().books[BOOK]
CHAPTERS = {(ch.kind.workflow, ch.campaign): ch.name for ch in _BOOK.chapters.values()}         # (workflow, campaign) → chapter folder
KIND = {k.workflow: k.name for k in _BOOK.kinds.values()}
SCRIPTS = {**{k.workflow: k.script for k in _BOOK.kinds.values()}, **{name: imp.script for name, imp in _flow().imports.items()}}
PYTHON = ".venv/bin/python"
SCHEMA = 1

VARIANTS = {"4Q-1": "Kestrel-4Q-1: round struts, X bracing, rotors as the only effectors, NACA 0012",
            "4Q-2": "Kestrel-4Q-2: elevons, faired posts, no diagonals, the reflexed section",
            "4Q-3": "Kestrel-4Q-3: 4Q-2's airframe with wider elevons (crow brake) and reverse-capable drives"}
TOLERANCE = {"x_cg_mm": 2.0, "mass_frac": 0.02}


# ================================================================================================= the variants
def params(variant: str) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r}: one of {list(VARIANTS)}")
    return k2.params(variant)


def vehicle(variant: str) -> kt.Vehicle:
    """The transition model's vehicle of the variant, as notebook 40 / 42 built it (the mass table, the lattice, the
    polar with the section's numbers): the mass, CG and polar every case is set up for."""
    return k2.vehicles(which=(variant,))[variant]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def design_params(p: dict) -> dict:
    return {k: v for k, v in p.items() if k not in kestrel.PER_CALL}


def fingerprint(variant: str, veh: kt.Vehicle | None = None) -> dict:
    """The design identity a run is made for. ``params_sha256`` decides compatibility; the mass and CG within
    ``TOLERANCE``; the source hashes and the propeller calibration warn (the drive tables and the disks' rpm follow
    the calibration, the bare airframe does not)."""
    veh = veh or vehicle(variant)
    p = design_params(params(variant))
    cal = ks.prop_lab_calibration()
    mods = {m: _sha((DESIGNS / f"{m}.py").read_text()) for m in ("kestrel", "kestrel_systems", "kestrel_cfd", "kestrel_fea", "kestrel_structure")}
    return {"schema": SCHEMA, "name": veh.name, "variant": variant, "description": VARIANTS[variant], "params": p, "params_sha256": _sha(json.dumps(p, sort_keys=True)),
            "mass_kg": float(veh.mass), "x_cg_m": float(veh.x_cg), "z_cg_m": float(veh.z_cg),
            "polar": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in veh.pol.__dict__.items()},
            "components_sha256": _sha(json.dumps(ks.COMPONENTS, sort_keys=True, default=str)),
            "prop_lab": {"on_file": cal["on_file"], "cl_scale": cal["cl_scale"], "cd_scale": cal["cd_scale"], "cn_factor": cal["cn_factor"]},
            "sources_sha256": mods}


def compare(expected: dict, found: dict | None, *, blown: bool = False) -> dict:
    """``{"level": "compatible" | "warnings" | "incompatible", "reasons": [...]}`` — ``expected`` the notebook's
    fingerprint, ``found`` the run's. The CAD parameters must match; the mass and CG within tolerance; a different
    propeller calibration is incompatible for blown (rotor-disk) cases and a warning for the bare airframe; changed
    sources are warnings (the run may still be right, the engineer decides)."""
    if not found:
        return {"level": "incompatible", "reasons": ["the run has no design fingerprint"]}
    bad, warn = [], []
    if found.get("params_sha256") != expected.get("params_sha256"):
        fp, ep = found.get("params", {}), expected.get("params", {})
        diff = sorted(k for k in set(fp) | set(ep) if fp.get(k) != ep.get(k))
        bad.append(f"CAD parameters differ: {', '.join(f'{k} {fp.get(k)} → {ep.get(k)}' for k in diff[:8])}" + (" …" if len(diff) > 8 else ""))
    if found.get("variant") != expected.get("variant"):
        bad.append(f"variant {found.get('variant')} is not {expected.get('variant')}")
    dx = abs(float(found.get("x_cg_m", np.nan)) - float(expected.get("x_cg_m", np.nan))) * 1000
    if not (dx <= TOLERANCE["x_cg_mm"]):
        bad.append(f"CG differs by {dx:.1f} mm (tolerance {TOLERANCE['x_cg_mm']:g} mm: the moments are about it)")
    m0, m1 = float(expected.get("mass_kg", np.nan)), float(found.get("mass_kg", np.nan))
    if not (abs(m1 - m0) <= TOLERANCE["mass_frac"] * m0):
        bad.append(f"mass differs: run {m1:.3f} kg, now {m0:.3f} kg (tolerance {TOLERANCE['mass_frac']:.0%}: the thrusts and rpm follow it)")
    if found.get("prop_lab") != expected.get("prop_lab"):
        (bad if blown else warn).append(f"propeller calibration differs: run {found.get('prop_lab')}, now {expected.get('prop_lab')}")
    if found.get("components_sha256") != expected.get("components_sha256"):
        warn.append("COMPONENTS (motor, propeller, battery, masses) changed since the run")
    fs, es = found.get("sources_sha256", {}), expected.get("sources_sha256", {})
    changed = [m for m in es if fs.get(m) != es[m]]
    if changed:
        warn.append(f"source changed since the run: {', '.join(changed)}")
    if found.get("assumed"):
        warn.append(found["assumed"])
    return {"level": "incompatible" if bad else ("warnings" if warn else "compatible"), "reasons": bad + warn}


# ================================================================================================= the CFD campaigns
STALL_ALPHAS = (8.0, 10.0, 12.0, 14.0, 16.0)
CFD_CAMPAIGNS = {
    "sweep": f"the bare airframe at {len(kcfd.SWEEP_ALPHAS)} angles 0–90° at {kcfd.SWEEP_V:g} m/s: the model polar's reference (lift slope, Cd0, Cmα, the stall deep in)",
    "sweep-fine": f"the bare airframe at {len(kcfd.SWEEP_ALPHAS_FINE)} angles (8–16° resolved) at {kcfd.SWEEP_V:g} m/s",
    "stall": f"the bare airframe at {', '.join(f'{a:g}' for a in STALL_ALPHAS)}°: CL max and the stall speed — the decision the wing area hangs on",
    "cruise": "the level cruise at 16 m/s and 1.05 × the stall speed with the four rotor disks at their trimmed thrust (2 cases)",
    "transitions": "the points the nominal 3 s transitions fly (α 80 → 10° forward and back) with the rotor disks, plus the hover",
    "blown": "cruise + transitions: every rotor-disk case",
    "all": "sweep + blown",
    "sections": "the two finalist sections (NACA 0012, NACA 23112 k₂/k₁ 0.03) as extruded slabs at 0, 4, 8, 12° (Zephyr on file)",
    "interference": f"the wing interference measured: one wing alone, the wing pair and the equal-span equal-area monoplane (chord 2c), bare, at "
                    f"{', '.join(f'{a:g}' for a in kcfd.INTERFERENCE_ALPHAS)}° and {kcfd.SWEEP_V:g} m/s — the lift lost between the wings and the induced drag against the monoplane",
}


def cfd_specs(variant: str, campaign: str, *, veh: kt.Vehicle | None = None) -> dict:
    """``{case name: spec}`` of the campaign for the variant — ``kestrel_cfd``'s specs (``props`` says rotor disks or
    bare; ``slab`` marks a section case). The transition points come from the model's nominal 3 s transitions of this
    vehicle (seconds of integration, no solver)."""
    if campaign not in CFD_CAMPAIGNS:
        raise ValueError(f"campaign {campaign!r}: one of {list(CFD_CAMPAIGNS)}")
    if campaign == "sections":
        import kestrel_section as ksec
        return {k: dict(v, props=False, slab=True, source="section", alpha_deg=v["alpha_deg"], V=v["V"], T_up=0.0, T_lo=0.0)
                for k, v in ksec.cfd_specs([("NACA 0012", None), ("NACA 23112", 0.03)]).items()}
    if campaign == "interference":
        return {k: dict(v, props=False) for k, v in kcfd.interference_specs().items()}
    veh = veh or vehicle(variant)
    out = {}
    if campaign in ("sweep", "all"):
        out.update({k: dict(v, props=False) for k, v in kcfd.sweep_specs(alphas=kcfd.SWEEP_ALPHAS).items()})
    if campaign == "sweep-fine":
        out.update({k: dict(v, props=False) for k, v in kcfd.sweep_specs(alphas=kcfd.SWEEP_ALPHAS_FINE).items()})
    if campaign == "stall":
        out.update({k: dict(v, props=False) for k, v in kcfd.sweep_specs(alphas=STALL_ALPHAS).items()})
    pts = []
    if campaign in ("cruise", "blown", "all"):
        pts += kcfd.flight_points(veh)
    if campaign in ("transitions", "blown", "all"):
        eps = [kt.simulate(veh, k, t_tr=3.0) for k in ("forward", "back")]
        pts += kcfd.points(eps, mass_kg=veh.mass)
    out.update({kcfd.point_name(pt): dict(pt, props=True) for pt in pts})
    return out


def cfd_case(name: str, spec: dict, workdir, *, variant: str, quality: str, veh: kt.Vehicle, env=None):
    """The prepared-not-run Aeromant case of one spec in ``workdir`` (``kestrel_cfd.case`` / ``section_case``)."""
    if spec.get("slab"):
        return kcfd.section_case(workdir, spec, quality=quality, env=env)
    if spec.get("planform"):
        return kcfd.interference_case(workdir, spec, x_cg_m=veh.x_cg, quality=quality, env=env, p=params(variant))
    pt = {k: v for k, v in spec.items() if k not in ("props", "slab")}
    return kcfd.case(workdir, pt, x_cg_m=veh.x_cg, z_cg_m=veh.z_cg, quality=quality, props=bool(spec["props"]), env=env, p=params(variant))


def cfd_case_key(case, quality: str) -> str:
    """What makes a solved case reusable: the template, every value and the geometry file's content."""
    from side_labs_runkit import sha256_file
    return _sha(json.dumps({"template": case.template.name, "parameters": case.parameters, "units": case.geometry_units, "quality": quality,
                            "stl": sha256_file(case.geometry)}, sort_keys=True, default=str))


def cfd_outcome(name: str, spec: dict, result, p=None) -> dict:
    """One case's outcome for the run record, through ``kestrel_cfd.table`` (the same statuses the notebooks use):
    ``status`` solved / diverged / FAILED, ``process`` ok / failed / cancelled, ``converged``, the coefficients."""
    if spec.get("slab"):
        t = kcfd.section_table({name: result}, {name: spec})
    else:
        t = kcfd.table({name: result}, {name: spec}, p)
    row = t.loc[name].to_dict()
    st = str(row["status"])
    status = getattr(result, "status", "failed")
    process = "cancelled" if status == "cancelled" else ("ok" if getattr(result, "ok", False) else "failed")
    metrics = {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else (v.item() if hasattr(v, "item") else v))
               for k, v in row.items() if k not in ("status", "note")}
    arts = {k: str(v) for k, v in (getattr(result, "artifacts", {}) or {}).items() if k in ("case", "summary", "log_solver", "log_snappyHexMesh", "log_checkMesh", "forces")}
    return {"status": "cancelled" if process == "cancelled" else st, "process": process, "converged": metrics.get("converged"), "note": str(row.get("note", "") or ""),
            "metrics": metrics, "artifacts": arts}


def cfd_summary(*, variant: str, campaign: str, quality: str, specs: dict, records: dict, veh: kt.Vehicle, run_root: Path, provenance: dict | None = None) -> dict:
    """The run's ``summary.json`` for the airframe CFD: scalars with their units, the reference quantities and the
    coordinate frame of the coefficients, every case's status, the solved cases as ``kestrel_cfd`` records (what
    ``CFDAirframe.from_records`` reads), the sampled maps as CSV files and their validity domain."""
    import pandas as pd
    L = kestrel.layout(params(variant))
    rows, recs = {}, []
    for name, spec in specs.items():
        r = records.get(name)
        m = (r.metrics if r is not None else {}) or {}
        row = {"status": r.status if r is not None else "NOT RUN", "process": r.process if r is not None else None, "converged": m.get("converged"),
               "iterations": m.get("iterations"), "cells": m.get("cells"), "alpha_deg": float(spec["alpha_deg"]), "V_m_s": float(max(spec["V"], 1.0)),
               "T_up_N": float(spec.get("T_up", 0.0)), "T_lo_N": float(spec.get("T_lo", 0.0)), "rotor_disks": bool(spec.get("props")), "slab": bool(spec.get("slab")),
               "planform": spec.get("planform"),
               "source": spec.get("source", ""), "CL": m.get("CL"), "CD": m.get("CD"), "Cm": m.get("Cm (CG, chord)", m.get("Cm (c/4, chord)")),
               "Cd_scatter_last50": m.get("Cd scatter (last 50)"), "note": r.note if r is not None else "", "duration_s": r.duration_s if r is not None else None,
               "dir": (r.artifacts.get("case") if r is not None else None)}
        rows[name] = row
        if row["status"] == "solved" and not spec.get("slab") and not spec.get("planform") and row["CL"] is not None:
            recs.append({"case": name, "alpha_deg": row["alpha_deg"], "V": row["V_m_s"], "T_up": row["T_up_N"], "T_lo": row["T_lo_N"], "rotor_disks": row["rotor_disks"],
                         "source": row["source"], "CL": float(row["CL"]), "CD": float(row["CD"]), "Cm": float(row["Cm"]), "converged": row["converged"], "iterations": row["iterations"]})
    df = pd.DataFrame(rows).T if rows else pd.DataFrame()
    maps = {}
    if len(df):
        bare = df[(df["status"] == "solved") & (~df["rotor_disks"].astype(bool)) & (~df["slab"].astype(bool)) & (df["planform"].isna())]
        blown = df[(df["status"] == "solved") & (df["rotor_disks"].astype(bool))]
        for key, sub in (("bare_sweep", bare), ("blown_points", blown)):
            if len(sub):
                path = run_root / f"{key}.csv"
                cols = ["alpha_deg", "V_m_s", "T_up_N", "T_lo_N", "CL", "CD", "Cm", "converged", "iterations", "cells"]
                sub[cols].sort_values("alpha_deg").to_csv(path, index_label="case")
                dom = {"alpha_deg": [float(sub["alpha_deg"].min()), float(sub["alpha_deg"].max())], "V_m_s": [float(sub["V_m_s"].min()), float(sub["V_m_s"].max())],
                       "n": int(len(sub)), "converged": int(sub["converged"].fillna(False).astype(bool).sum())}
                if key == "blown_points":
                    mus = [kcfd.model_coefficients(veh, {"alpha_deg": a, "V": v, "T_up": tu, "T_lo": tl})["mu"]
                           for a, v, tu, tl in zip(sub["alpha_deg"], sub["V_m_s"], sub["T_up_N"], sub["T_lo_N"])]
                    dom["mu"] = [float(min(mus)), float(max(mus))]
                maps[key] = {"file": path.name, "columns": cols, "valid_domain": dom,
                             "interpolation": ("linear in α inside the domain, the end values beyond" if key == "bare_sweep" else
                                               "linear in (α, μ) inside the convex hull of the points, the nearest point outside (kestrel_cfd.CFDAirframe)")}
    counts = {k: int((df["status"] == k).sum()) if len(df) else 0 for k in ("solved", "diverged", "FAILED", "cancelled", "NOT RUN")}
    counts["n"] = int(len(df))
    counts["solved with rotor disks"] = int(((df["status"] == "solved") & df["rotor_disks"].astype(bool)).sum()) if len(df) else 0
    counts["converged"] = int(df["converged"].fillna(False).astype(bool).sum()) if len(df) else 0
    interference = kcfd.interference_factors(rows) if campaign == "interference" else None
    return {"schema": SCHEMA, "workflow": "airframe_cfd", "campaign": campaign, "description": CFD_CAMPAIGNS.get(campaign, ""), "variant": variant, "quality": quality,
            "quality_settings": kcfd.QUALITY.get(quality), "design": fingerprint(variant, veh), **({"interference": interference} if interference else {}),
            "reference": {"S_ref_m2": L["S_ref"], "chord_m": L["chord"], "l_ref_m": kcfd.l_ref(params(variant)), "x_cg_m": veh.x_cg, "z_cg_m": veh.z_cg,
                          "density_kg_m3": kcfd.RHO, "kinematic_viscosity_m2_s": kcfd.NU,
                          "frame": "wind axes; CL and CD on S_ref (both wings); Cm about the CG on the chord, nose-up positive; α the free-stream angle of attack",
                          "units": {"alpha_deg": "deg", "V_m_s": "m/s", "T_up_N": "N per upper rotor", "T_lo_N": "N per lower rotor", "CL": "-", "CD": "-", "Cm": "-"}},
            "counts": counts, "cases": rows, "records": recs, "maps": maps, "provenance": provenance or {"source": "solver run"},
            "validation_note": ("a case marked converged reached the residual target; a solved, unconverged case is an indicative steady-state "
                                "average, not a validated coefficient; diverged and FAILED cases carry no coefficients")}


# ================================================================================================= the FEA campaigns
FEA_CAMPAIGNS = {"frame": "the carbon frame of the variant's bracing (X for 4Q-1, posts for 4Q-2 / 4Q-3): A the 4 g pull-up, B the 75 % wing, C the heading manoeuvre, the modes",
                 "joint": "the printed spar–post joint under the post's largest axial force (ultimate), across-layer tension read against the layer strength",
                 "all": "frame + joint"}


def fea_cases(variant: str, campaign: str, workdir, *, veh: kt.Vehicle | None = None, element_size: float = 2.0) -> list:
    """The ``kestrel_fea`` cases of the campaign for the variant (prepared models, no mesh, no solve)."""
    if campaign not in FEA_CAMPAIGNS:
        raise ValueError(f"campaign {campaign!r}: one of {list(FEA_CAMPAIGNS)}")
    veh = veh or vehicle(variant)
    p = params(variant)
    out = []
    if campaign in ("frame", "all"):
        out += kfe.frame_cases(workdir, p, mass_kg=veh.mass, bracing=p["bracing"], element_size=element_size, drive=veh.drive)
    if campaign in ("joint", "all"):
        out += kfe.joint_cases(workdir, p, mass_kg=veh.mass)
    return out


def fea_case_key(case) -> str:
    return _sha(json.dumps({"model": case.model.key, "kind": case.kind}, sort_keys=True))


def fea_outcome(case, result) -> dict:
    t = kfe.table([case], {case.name: result})
    row = t.loc[case.name].to_dict()
    st = str(row["status"])
    status = getattr(result, "status", "failed")
    process = "cancelled" if status == "cancelled" else ("ok" if getattr(result, "ok", False) else "failed")
    metrics = {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else (v.item() if hasattr(v, "item") else v))
               for k, v in row.items() if k not in ("status", "note", "description")}
    arts = {k: str(v) for k, v in (getattr(result, "artifacts", {}) or {}).items() if k in ("frd", "dat", "inp", "mesh", "log", "summary", "vtu")}
    return {"status": "cancelled" if process == "cancelled" else st, "process": process, "converged": (True if st == "solved" else None),
            "note": str(row.get("note", "") or ""), "metrics": metrics, "artifacts": arts}


def fea_summary(*, variant: str, campaign: str, cases: list, records: dict, veh: kt.Vehicle, run_root: Path) -> dict:
    p = params(variant)
    rows = {}
    for c in cases:
        r = records.get(c.name)
        m = (r.metrics if r is not None else {}) or {}
        rows[c.name] = {"status": r.status if r is not None else "NOT RUN", "process": r.process if r is not None else None, "part": c.part, "bracing": c.bracing or "",
                        "kind": c.kind, "description": c.description, "strength_MPa": c.strength, "strength_z_MPa": c.strength_z,
                        **{k: m.get(k) for k in ("peak von Mises [MPa]", "nominal von Mises [MPa]", "99.5th percentile [MPa]", "max displacement [mm]", "allowable [MPa]",
                                                 "margin", "reaction [N]", "across layers [MPa]", "margin (layers)", "f1 [Hz]", "f2 [Hz]", "f3 [Hz]")},
                        "note": r.note if r is not None else "", "duration_s": r.duration_s if r is not None else None,
                        "artifacts": r.artifacts if r is not None else {}}
    ref = kfe.plane_frame_reference(p, mass_kg=veh.mass)
    counts = {k: sum(1 for v in rows.values() if v["status"] == k) for k in ("solved", "FAILED", "cancelled", "NOT RUN")}
    counts["n"] = len(rows)
    return {"schema": SCHEMA, "workflow": "structure_fea", "campaign": campaign, "description": FEA_CAMPAIGNS.get(campaign, ""), "variant": variant,
            "design": fingerprint(variant, veh),
            "reference": {"units": "mm-N-MPa (Talos); frequencies Hz", "ultimate_factor": kfe.ULTIMATE, "knockdown": kfe.KNOCKDOWN, "materials": {"carbon": kfe.CARBON, "petg": kfe.PETG},
                          "mass_kg": veh.mass, "load_factor_g": 4.0, "frame": "aircraft axes: x aft, y right, z up; the pod's spar saddles fixed",
                          "stress_reading": "peak = the mesh's largest von Mises (support and load edges included); nominal = away from the clamps and the load introductions (what the margin is read on)"},
            "counts": counts, "cases": rows, "plane_frame_reference": {str(k): {kk: (None if isinstance(vv, float) and not np.isfinite(vv) else vv) for kk, vv in r.items()} for k, r in ref.to_dict("index").items()},
            "fields": "the native CalculiX .frd / .dat of each case (displacement, stress) stay in its directory; ParaView reads the .frd, talos.read_frd the arrays",
            "validation_note": "a static case that solved is a linear elastic solution on the given mesh, not a validated strength; the margins are against the knocked-down allowable at ultimate load"}


# ================================================================================================= the legacy import
def import_legacy(results_json=kcfd.RESULTS_JSON, out_base=RUNS_BASE, *, keys=("4Q-1", "4Q-2")) -> list[Path]:
    """The first campaign's results file (notebook 39, ``data/kestrel_cfd_results.json``) as run directories of the
    contract, one per campaign key, so that the unified notebook reads them like any run. The provenance says what
    they are: screening quality, 400 iterations, imported — and the fingerprint is the variant's **as defined now**
    (the file carries none), which the compatibility check reports as an assumption, not a fact."""
    from side_labs_runkit import write_json_atomic, utc_now, sha256_file, tool_versions
    doc = kcfd.load_records(results_json)
    if not doc:
        raise FileNotFoundError(f"{results_json} does not exist or holds no campaign")
    out = []
    for key in keys:
        camp = doc["campaigns"].get(key)
        if not camp:
            continue
        variant = key
        veh = vehicle(variant)
        meta = camp.get("meta", {})
        quality = meta.get("quality", "screening")
        root = Path(out_base) / BOOK / CHAPTERS[("airframe_cfd", "sweep")] / f"legacy_{meta.get('date', 'undated').replace('-', '')}_{variant}_{meta.get('source', 'import')}"
        root.mkdir(parents=True, exist_ok=True)
        specs, records = {}, {}
        from side_labs_runkit import CaseRecord
        byname = {r["case"]: r for r in camp.get("records", [])}
        for name, st in camp.get("statuses", {}).items():
            rec = byname.get(name)
            specs[name] = {"alpha_deg": st["α [deg]"], "V": st["V [m/s]"], "T_up": rec["T_up"] if rec else 0.0, "T_lo": rec["T_lo"] if rec else 0.0,
                           "props": bool(st.get("rotor disks")), "source": rec.get("source", "") if rec else ("sweep" if not st.get("rotor disks") else "")}
            metrics = ({"CL": rec["CL"], "CD": rec["CD"], "Cm (CG, chord)": rec["Cm"], "converged": rec.get("converged"), "iterations": rec.get("iterations")} if rec else {})
            records[name] = CaseRecord(name=name, key="legacy", status=st["status"], process="ok" if st["status"] in ("solved", "diverged") else "failed",
                                       converged=metrics.get("converged"), note=st.get("note", ""), metrics=metrics)
        fp = fingerprint(variant, veh)
        fp["assumed"] = "fingerprint assumed from the current source: the legacy results file recorded none (the CAD parameters of the campaign were notebook 39's of that day)"
        config = {"workflow": "airframe_cfd", "book": BOOK, "kind": "cfd", "chapter": CHAPTERS[("airframe_cfd", "sweep")], "campaign": "legacy-import", "variant": variant, "quality": quality, "source_file": str(results_json), "key": key}
        manifest = {"schema": 1, "run_id": root.name, "workflow": "airframe_cfd", "book": BOOK, "kind": "cfd", "chapter": CHAPTERS[("airframe_cfd", "sweep")], "campaign": "legacy-import", "created_at": utc_now(), "imported_at": utc_now(),
                    "command": f"{PYTHON} {SCRIPTS['import_legacy']}", "config": config, "config_hash": _sha(json.dumps(config, sort_keys=True)),
                    "inputs": {"results_json": {"path": str(results_json), "sha256": sha256_file(results_json)}}, "design": fp, "tool_versions": tool_versions(),
                    "planned_cases": list(specs), "resumed": [], "artifacts": {"run_log": "run.log", "status": "status.json", "cases": None, "summary": "summary.json"},
                    "provenance": {"imported_from": str(results_json), "notebook": meta.get("source"), "date": meta.get("date"), "quality": quality,
                                   "iterations": meta.get("iterations"), "native_cases": "not imported (the OpenFOAM directories were not kept)"}}
        write_json_atomic(root / "manifest.json", manifest)
        summary = cfd_summary(variant=variant, campaign="legacy-import", quality=quality, specs=specs, records=records, veh=veh, run_root=root,
                              provenance=manifest["provenance"])
        summary["cases"] = {n: {**summary["cases"][n]} for n in summary["cases"]}
        summary["state"] = "completed"
        write_json_atomic(root / "summary.json", summary)
        write_json_atomic(root / "status.json", {"state": "completed", "run_id": root.name, "updated_at": utc_now(), "finished_at": utc_now(),
                                                  "cases": {"planned": len(specs), "done": len(specs), "reused": 0, "failed": summary["counts"]["FAILED"] + summary["counts"]["diverged"], "running": None},
                                                  "note": "imported from the legacy results file"})
        (root / "run.log").write_text(f"{utc_now()} imported {len(specs)} case record(s) of campaign {key} from {results_json} ({meta})\n")
        out.append(root)
    return out


# ================================================================================================= what the notebook reads
class PrerequisiteError(RuntimeError):
    """A required workflow result is missing, unfinished, failed or incompatible; the message names the command."""


def command(workflow: str, *, variant: str, campaign: str, quality: str | None = None, extra: str = "") -> str:
    q = f" --quality {quality}" if quality and workflow == "airframe_cfd" else ""
    return f"{PYTHON} {SCRIPTS[workflow]} --variant {variant} --chapter {campaign}{q}{(' ' + extra) if extra else ''}"


def _load(run_dir, workflow: str) -> dict:
    from side_labs_runkit import load_run
    if run_dir is None:
        return {"root": None, "manifest": None, "status": None, "summary": None, "partial": None}
    p = Path(run_dir)
    if not p.is_absolute():
        p = ROOT / p
    d = load_run(p)
    if d["manifest"] and d["manifest"].get("workflow") != workflow:
        raise PrerequisiteError(f"{p} is a {d['manifest'].get('workflow')} run, not {workflow}")
    return d


def check_run(run_dir, *, workflow: str, variant: str, campaign: str, expected: dict, blown: bool = False, quality: str | None = None,
              states=("completed",)) -> dict:
    """Everything the prerequisite table shows for one input: ``ok`` (usable), ``status`` text, the run's provenance,
    the compatibility level and reasons, the summary when usable, and the ``command`` that would produce it."""
    cmd = command(workflow, variant=variant, campaign=campaign, quality=quality)
    out = {"run": None if run_dir is None else str(run_dir), "ok": False, "status": "not selected", "provenance": "", "compat": "", "reasons": [], "summary": None, "command": cmd, "manifest": None}
    if run_dir is None:
        return out
    try:
        d = _load(run_dir, workflow)
    except PrerequisiteError as exc:
        out["status"] = str(exc)
        return out
    m, st, s = d["manifest"], d["status"], d["summary"]
    if m is None:
        out["status"] = f"not found: {Path(run_dir)} has no manifest.json"
        return out
    out["manifest"] = m
    prov = m.get("provenance") or {}
    out["provenance"] = (f"imported from {Path(prov['imported_from']).name} ({prov.get('notebook')}, {prov.get('date')}, {prov.get('quality')}, {prov.get('iterations')} it)" if prov.get("imported_from")
                         else f"{m.get('campaign')} run {m.get('run_id')} on {m.get('host')}, {m.get('created_at', '')[:10]}, {(m.get('config') or {}).get('quality', '')} "
                              f"(git {(m.get('tool_versions') or {}).get('git')}{' dirty' if (m.get('tool_versions') or {}).get('git_dirty') else ''})")
    state = (st or {}).get("state")
    if state not in states or s is None:
        out["status"] = f"run state '{state}' (no completed summary.json): " + ("still running or interrupted — resume it with --resume" if state in ("running", "prepared", "cancelled") else "failed — read run.log")
        return out
    cmp_ = compare(expected, m.get("design"), blown=blown)
    out["compat"], out["reasons"] = cmp_["level"], cmp_["reasons"]
    if cmp_["level"] == "incompatible":
        out["status"] = "incompatible: " + "; ".join(cmp_["reasons"])
        return out
    c = s.get("counts", {})
    out["summary"] = s
    out["ok"] = True
    out["status"] = f"completed: {c.get('solved', 0)} of {c.get('n', 0)} solved" + (f", {c.get('converged', 0)} converged" if "converged" in c else "") + \
        (f", {c.get('FAILED', 0)} failed" if c.get("FAILED") else "") + (f", {c.get('diverged', 0)} diverged" if c.get("diverged") else "")
    return out


def require(check: dict, *, what: str, required: bool = True) -> dict | None:
    """Raise ``PrerequisiteError`` for a required input that is not usable; return its summary otherwise (``None``
    for an optional input that is not there: the notebook shows NOT RUN and its analytical estimate)."""
    if check["ok"]:
        return check["summary"]
    if not required:
        return None
    raise PrerequisiteError(f"{what}: {check['status']}.\n  Produce it with:\n    {check['command']}\n  then pin the run directory it prints (yn pin 43_kestrel_study <chapter> --variant <v> <run dir>)."
                            + ("" if check["run"] is None else ""))


def records_frame(summary: dict):
    """The solved cases of a CFD summary as ``kestrel_cfd``'s table frame (for ``sweep_comparison`` and the polar figure)."""
    return kcfd.records_table(summary.get("records", []))


def airframe_correction(summary: dict, veh: kt.Vehicle, *, source: str = "sweep", converged_only: bool = False):
    """The ``CFDAirframe`` from a run's records (``None`` when nothing usable): ``source`` ``sweep`` (the bare
    airframe, in α) or ``disks`` (the blown cases, in α and μ); ``converged_only`` keeps the cases that reached the
    residual target — the unconverged screening cases are indicative, and the caller says so."""
    recs = summary.get("records", [])
    if converged_only:
        recs = [r for r in recs if r.get("converged")]
    return kcfd.CFDAirframe.from_records(recs, veh, source=source) if recs else None
