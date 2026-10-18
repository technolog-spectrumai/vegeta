"""kestrel_xc_campaigns — the book ``kestrel-x`` (code X) behind the shared workflow scripts: the same interface as
``kestrel_campaigns`` (variants, the vehicle, the fingerprint, the CFD and FEA campaigns, case builders, outcomes and
summaries), on ``kestrel_xc``. ``workflows/kestrel/airframe_cfd.py`` and ``structure_fea.py`` take the book by name
and import this module for it (``workflows/kestrel_x/*.py`` are the one-line entry points)."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

import kestrel_fea as kfe
import kestrel_systems as ks
import kestrel_xc as xc

BOOK = "kestrel-x"
SCHEMA = 1
VARIANTS = {v: f"Kestrel-{v}: {d['title']}" for v, d in xc.VARIANTS.items()}
DEFAULT_VARIANT = xc.MAIN
TOLERANCE = {"x_cg_mm": 2.0, "mass_frac": 0.02}
CFD_CAMPAIGNS = dict(xc.CFD_CAMPAIGNS)
FEA_CAMPAIGNS = dict(xc.FEA_CAMPAIGNS)
PYTHON = ".venv/bin/python"
_VEH: dict = {}


def params(variant: str) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r}: one of {list(VARIANTS)}")
    return xc.cad_params(variant)


def vehicle(variant: str):
    if variant not in _VEH:
        _VEH[variant] = xc.make_vehicle(variant)
    return _VEH[variant]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def fingerprint(variant: str, veh=None) -> dict:
    veh = veh or vehicle(variant)
    p = params(variant)
    return {"schema": SCHEMA, "name": f"Kestrel-{variant}", "book": BOOK, "variant": variant, "params_sha256": _sha(json.dumps(p, sort_keys=True)), "params": p,
            "mass_kg": round(veh.mass, 4), "x_cg_m": round(veh.x_cg, 4), "z_cg_m": round(veh.z_cg, 4), "S_ref_m2": veh.L["S_ref"], "phi_deg": veh.L["phi_deg"],
            "sources": {m: _sha(Path(xc.__file__).with_name(m).read_text()) for m in ("kestrel_xc.py", "kestrel_xwing.py")}}


def compare(expected: dict, found: dict | None, *, blown: bool = False) -> dict:
    if not found:
        return {"level": "incompatible", "reasons": ["the run carries no design fingerprint"]}
    reasons, level = [], "compatible"
    if found.get("params_sha256") != expected.get("params_sha256"):
        return {"level": "incompatible", "reasons": ["the CAD parameters differ"]}
    if abs(found.get("mass_kg", 0) - expected.get("mass_kg", 0)) > TOLERANCE["mass_frac"] * expected.get("mass_kg", 1):
        return {"level": "incompatible", "reasons": [f"mass {found.get('mass_kg')} against {expected.get('mass_kg')} kg"]}
    if abs(found.get("x_cg_m", 0) - expected.get("x_cg_m", 0)) * 1e3 > TOLERANCE["x_cg_mm"]:
        return {"level": "incompatible", "reasons": [f"CG {found.get('x_cg_m')} against {expected.get('x_cg_m')} m"]}
    for m, h in expected.get("sources", {}).items():
        if found.get("sources", {}).get(m) not in (None, h):
            level = "warnings"
            reasons.append(f"{m} changed since the run")
    return {"level": level, "reasons": reasons}


# ------------------------------------------------------------------------------------------------- CFD
def cfd_specs(variant: str, campaign: str, *, veh=None) -> dict:
    if campaign not in CFD_CAMPAIGNS:
        raise ValueError(f"campaign {campaign!r}: one of {list(CFD_CAMPAIGNS)}")
    return xc.cfd_specs(variant, campaign)


def cfd_case(name: str, spec: dict, workdir, *, variant: str, quality: str, veh, env=None):
    return xc.cfd_case(workdir, spec, variant=variant, x_cg_m=veh.x_cg, quality=quality, env=env)


def cfd_case_key(case, quality: str) -> str:
    from workflows_runkit import sha256_file
    return _sha(json.dumps({"template": case.template.name, "parameters": case.parameters, "units": case.geometry_units, "quality": quality, "stl": sha256_file(case.geometry)},
                           sort_keys=True, default=str))


SANE = 10.0


def cfd_outcome(name: str, spec: dict, result, p=None) -> dict:
    """As ``kestrel_campaigns.cfd_outcome``: solved / diverged / FAILED with the coefficients (CL, CD on the panels'
    area; Cm about the CG on the X-C's own reference length, turned to the mean chord)."""
    L = xc.layout(xc.MAIN)
    k_cm = xc.l_ref(L) / L["chord"]
    ok = getattr(result, "ok", False)
    if result is None or not ok:
        msg = "NOT RUN" if result is None or any("NOT RUN" in m for m in getattr(result, "messages", [])) else "FAILED"
        status = getattr(result, "status", "failed") if result is not None else "failed"
        return {"status": "cancelled" if status == "cancelled" else msg, "process": "cancelled" if status == "cancelled" else ("ok" if ok else "failed"), "converged": None,
                "note": (result.messages[-1][:160] if result is not None and getattr(result, "messages", None) else ""), "metrics": {}, "artifacts": {}}
    m = result.metrics
    coefs = [np.nan if m.get(k) is None else float(m.get(k)) for k in ("Cl", "Cd", "Cm")]
    sane = all(np.isfinite(v) and abs(v) <= SANE for v in coefs)
    note = "" if sane else f"diverged: |coefficients| up to {np.nanmax(np.abs(coefs)):.3g}"
    if sane and m.get("converged") is False:
        note = f"not converged in {m.get('iterations')} iterations"
    metrics = {"CL": coefs[0], "CD": coefs[1], "Cm (CG, chord)": coefs[2] * k_cm, "iterations": m.get("iterations"), "converged": m.get("converged"), "Cd scatter (last 50)": m.get("Cd_std_lastN"),
               "cells": m.get("mesh_cells"), "α [deg]": spec["alpha_deg"], "V [m/s]": spec["V"]}
    metrics = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in metrics.items()}
    arts = {k: str(v) for k, v in (getattr(result, "artifacts", {}) or {}).items() if k in ("case", "summary", "log_solver", "log_snappyHexMesh", "log_checkMesh", "forces")}
    return {"status": "solved" if sane else "diverged", "process": "ok", "converged": m.get("converged"), "note": note, "metrics": metrics, "artifacts": arts}


def cfd_summary(*, variant: str, campaign: str, quality: str, specs: dict, records: dict, veh, run_root: Path, provenance: dict | None = None) -> dict:
    import kestrel_cfd as kcfd
    import pandas as pd
    L = veh.L
    rows, recs = {}, []
    for name, spec in specs.items():
        r = records.get(name)
        m = (r.metrics if r is not None else {}) or {}
        row = {"status": r.status if r is not None else "NOT RUN", "process": r.process if r is not None else None, "converged": m.get("converged"), "iterations": m.get("iterations"),
               "cells": m.get("cells"), "alpha_deg": float(spec["alpha_deg"]), "V_m_s": float(spec["V"]), "rotor_disks": False, "slab": False, "planform": None, "source": spec.get("source", ""),
               "CL": m.get("CL"), "CD": m.get("CD"), "Cm": m.get("Cm (CG, chord)"), "note": r.note if r is not None else "", "duration_s": r.duration_s if r is not None else None,
               "dir": (r.artifacts.get("case") if r is not None else None)}
        rows[name] = row
        if row["status"] == "solved" and row["CL"] is not None:
            recs.append({"case": name, "alpha_deg": row["alpha_deg"], "V": row["V_m_s"], "T_up": 0.0, "T_lo": 0.0, "rotor_disks": False, "source": "sweep", "CL": float(row["CL"]), "CD": float(row["CD"]),
                         "Cm": float(row["Cm"]), "converged": row["converged"], "iterations": row["iterations"]})
    df = pd.DataFrame(rows).T if rows else pd.DataFrame()
    maps = {}
    if len(df):
        bare = df[df["status"] == "solved"]
        if len(bare):
            path = run_root / "bare_sweep.csv"
            cols = ["alpha_deg", "V_m_s", "CL", "CD", "Cm", "converged", "iterations", "cells"]
            bare[cols].sort_values("alpha_deg").to_csv(path, index_label="case")
            maps["bare_sweep"] = {"file": path.name, "columns": cols, "valid_domain": {"alpha_deg": [float(bare["alpha_deg"].min()), float(bare["alpha_deg"].max())], "n": int(len(bare)),
                                                                                       "converged": int(bare["converged"].fillna(False).astype(bool).sum())},
                                  "interpolation": "linear in α inside the domain"}
    counts = {k: int((df["status"] == k).sum()) if len(df) else 0 for k in ("solved", "diverged", "FAILED", "cancelled", "NOT RUN")}
    counts["n"] = int(len(df))
    counts["converged"] = int(df["converged"].fillna(False).astype(bool).sum()) if len(df) else 0
    # the lattice's prediction beside the sweep: the slope ratio the notebook reads
    a = xc.aero(L)
    sl = {}
    sol = [r for r in recs if r["alpha_deg"] <= 10.0]
    if len(sol) >= 2:
        xs, ys = np.radians([r["alpha_deg"] for r in sol]), [r["CL"] for r in sol]
        k_, b_ = np.polyfit(xs, ys, 1)
        sl = {"CLa_cfd [1/rad]": float(k_), "CLa_lattice [1/rad]": float(a["CLa"]), "ratio": float(k_ / a["CLa"]), "CD at the lowest α": float(min(sol, key=lambda r: r["alpha_deg"])["CD"]),
              "CL max below 20°": float(max([r["CL"] for r in recs if r["alpha_deg"] < 20.0] or [np.nan]))}
    return {"schema": SCHEMA, "workflow": "airframe_cfd", "book": BOOK, "campaign": campaign, "description": CFD_CAMPAIGNS.get(campaign, ""), "variant": variant, "quality": quality,
            "quality_settings": kcfd.QUALITY.get(quality), "design": fingerprint(variant, veh),
            "reference": {"S_ref_m2": L["S_ref"], "chord_m": L["chord"], "l_ref_m": xc.l_ref(L), "x_cg_m": veh.x_cg, "z_cg_m": veh.z_cg, "density_kg_m3": xc.RHO, "kinematic_viscosity_m2_s": xc.NU,
                          "frame": "wind axes; CL and CD on the four panels' area; Cm about the CG on the mean chord, nose-up positive"},
            "counts": counts, "cases": rows, "records": recs, "maps": maps, "lattice_check": sl, "provenance": provenance or {"source": "solver run"},
            "validation_note": "a case marked converged reached the residual target; a solved, unconverged case is indicative, not validated"}


# ------------------------------------------------------------------------------------------------- FEA
def fea_cases(variant: str, campaign: str, workdir, *, veh=None, element_size: float = 2.0) -> list:
    if campaign not in FEA_CAMPAIGNS:
        raise ValueError(f"campaign {campaign!r}: one of {list(FEA_CAMPAIGNS)}")
    veh = veh or vehicle(variant)
    return xc.fea_cases(variant, campaign, workdir, mass_kg=veh.mass, element_size=element_size)


def fea_case_key(case) -> str:
    return _sha(json.dumps({"model": case.model.key, "kind": case.kind}, sort_keys=True))


def fea_outcome(case, result) -> dict:
    import kestrel_campaigns as kc
    return kc.fea_outcome(case, result)


def fea_summary(*, variant: str, campaign: str, cases: list, records: dict, veh, run_root: Path) -> dict:
    rows = {}
    for c in cases:
        r = records.get(c.name)
        m = (r.metrics if r is not None else {}) or {}
        rows[c.name] = {"status": r.status if r is not None else "NOT RUN", "process": r.process if r is not None else None, "part": c.part, "kind": c.kind, "description": c.description,
                        "strength_MPa": c.strength, **{k: m.get(k) for k in ("peak von Mises [MPa]", "nominal von Mises [MPa]", "99.5th percentile [MPa]", "max displacement [mm]", "allowable [MPa]",
                                                                             "margin", "reaction [N]", "f1 [Hz]", "f2 [Hz]", "f3 [Hz]")},
                        "note": r.note if r is not None else "", "duration_s": r.duration_s if r is not None else None}
    L = veh.L
    ref = xc.cantilever(L, mass_kg=veh.mass)
    counts = {k: sum(1 for r in rows.values() if r["status"] == k) for k in ("solved", "FAILED", "NOT RUN", "cancelled")}
    counts["n"] = len(rows)
    return {"schema": SCHEMA, "workflow": "structure_fea", "book": BOOK, "campaign": campaign, "description": FEA_CAMPAIGNS.get(campaign, ""), "variant": variant, "design": fingerprint(variant, veh),
            "materials": {"carbon": kfe.CARBON, "knockdown": kfe.KNOCKDOWN, "ultimate": kfe.ULTIMATE}, "counts": counts, "cases": rows,
            "beam_reference": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in ref.items()},
            "validation_note": "the beam rule's stress and deflection stand beside the FEA's nominal (away from the clamp and the load bands)"}
