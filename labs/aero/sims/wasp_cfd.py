"""WASP's CFD — the airframe's drag and normal force (Aeromant / OpenFOAM). RUN BY YOU, on a machine with OpenFOAM.

Two campaigns, ÉMERILLON's (``emerillon_campaigns``), on a WASP build:

- ``drag``:  the solid aircraft (no propellers) in axial flow at 50, 75 and 100 % of the build's top speed -> the drag
  area CdA, beside the hand build-up's (clean, and with its x1.6 margin: the margin is a guess until this runs);
- ``alpha``: the body at 0, 3, 6, 10, 15, 30, 60 and 90 degrees at 60 % of the top speed -> the normal force the
  level-flight trim leans on, and the pitching moment about the CG that ``wasp_stability`` reads.

Steady RANS k-omega SST with wall functions, no transition model, no propellers, incompressible (WASP-1F tops out at
Mach 0.23). At these sizes the body's Reynolds number is about 1.5e6 at top speed and the arms' about 1.5e5: partly
laminar in reality, fully turbulent here. Trends, and an upper estimate of the friction drag.

Writes ``labs/aero/sims/output/<build>_cfd.yaml`` after every case: stop and restart freely, a solved case is kept.

    labs/aero/sims/wasp_cfd.sh --list                     # the cases, nothing run
    labs/aero/sims/wasp_cfd.sh                            # WASP-1F, both campaigns, screening mesh, 4 processors
    labs/aero/sims/wasp_cfd.sh --campaign drag --quality fine --processors 8
"""
import argparse
import math
import os
import time
from pathlib import Path

import wasp_common as wc
from wasp_common import wasp

import emerillon_campaigns as ec  # noqa: E402
import emerillon_flight as ef  # noqa: E402


def specs(veh, campaigns) -> dict:
    out = {}
    if "drag" in campaigns:
        for f in ec.DRAG_FRACTIONS:
            out[f"drag_{int(f * 100)}"] = {"campaign": "drag", "alpha_deg": 0.0, "V": round(f * veh.V_top, 2)}
    if "alpha" in campaigns:
        V = round(ec.ALPHA_SPEED * veh.V_top, 2)
        for a in ec.ALPHAS:
            out[f"alpha_{int(a):02d}"] = {"campaign": "alpha", "alpha_deg": a, "V": V}
    return out


def export_stl(folder: Path, build: str, alpha_deg: float) -> Path:
    """The solid aircraft pitched nose-up by ``alpha_deg`` about the nose tip (the flow stays along +x) [mm]."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{wc.slug(build)}_a{alpha_deg:04.1f}.stl"
    if not path.exists():
        wasp.cad(wasp.VARIANTS[build], angle_of_attack_deg=float(alpha_deg)).export_stl(str(path), tolerance=0.03, angular_tolerance=0.1)
    return path


def case(workdir: Path, build: str, veh, spec: dict, quality: str):
    """One ``rans_ksst_external`` case: coefficients on the body's frontal area, the moment about the CG on its length."""
    from vegeta import aeromant
    a = float(spec["alpha_deg"])
    stl = export_stl(workdir / "stl", build, a)
    ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
    prm = dict(velocity=max(spec["V"], 1.0), kinematic_viscosity=ef.NU, density=ef.RHO, reference_area=veh.geo["frontal"],
               reference_length=veh.geo["length"], center_of_rotation=(veh.x_cg * ca, 0.0, -veh.x_cg * sa), **ec.QUALITY[quality])
    prefix = os.environ.get("FOAM_PREFIX")
    env = aeromant.OpenFOAMEnvironment.conda(prefix) if prefix else aeromant.OpenFOAMEnvironment.detect()
    return aeromant.CFDCase("rans_ksst_external", str(stl), prm, workdir=str(workdir / f"{wc.slug(build)}_a{a:04.1f}_V{spec['V']:.1f}_{quality}"),
                            geometry_units="mm", environment=env)


def hand(veh, spec: dict) -> dict:
    """What the flight model assumes for the same case: the drag build-up (no slipstream) and the normal-force area."""
    with wasp.small():
        db = ef.drag_buildup(veh.config, spec["V"], slip=1.0)
        cna = ef.normal_force_area(veh.config, math.radians(spec["alpha_deg"]))
    return {"CdA_clean_cm2": wc.num(db["clean"] * 1e4, 3), "CdA_with_margin_cm2": wc.num(db["CdA"] * 1e4, 3), "normal_force_area_cm2": wc.num(cna * 1e4, 3)}


def row(veh, spec: dict, result) -> dict:
    m = result.metrics
    cl, cd, cm = (float(m[f"{k}_mean_last50"]) for k in ("Cl", "Cd", "Cm"))
    a, S = math.radians(spec["alpha_deg"]), veh.geo["frontal"]
    cn = cl * math.cos(a) + cd * math.sin(a)
    return {**spec, "status": "solved", "Cl": wc.num(cl), "Cd": wc.num(cd), "Cm_cg_length": wc.num(cm), "CN": wc.num(cn),
            "CdA_cm2": wc.num(cd * S * 1e4, 3), "normal_force_area_cm2": wc.num(cn * S * 1e4, 3), "cells": m.get("mesh_cells"),
            "iterations": m.get("iterations"), "converged": m.get("converged"), "Cd_std_last50": wc.num(m.get("Cd_std_last50"), 6), "hand": hand(veh, spec)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", default=wc.DEFAULT, choices=wc.BUILDS)
    ap.add_argument("--campaign", nargs="*", default=["drag", "alpha"], choices=("drag", "alpha"))
    ap.add_argument("--quality", default="screening", choices=tuple(ec.QUALITY))
    ap.add_argument("--processors", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=3 * 3600.0, help="seconds per case")
    ap.add_argument("--list", action="store_true", help="print the cases and stop: nothing is meshed or solved")
    ap.add_argument("--fresh", action="store_true", help="ignore the saved results and start over")
    a = ap.parse_args(argv)
    veh = wasp.vehicle(wasp.VARIANTS[a.build])
    S = specs(veh, a.campaign)
    print(f"{a.build}: {veh.label}\n  {veh.mass * 1e3:.0f} g, CG {veh.x_cg * 1e3:.0f} mm from the nose, top speed {veh.V_top:.1f} m/s (Mach {veh.V_top / ef.A_SOUND:.2f}), "
          f"frontal area {veh.geo['frontal'] * 1e4:.2f} cm2, length {veh.geo['length'] * 1e3:.0f} mm")
    if a.list:
        for n, s in S.items():
            print(f"  {n:10s} alpha {s['alpha_deg']:4.0f} deg  V {s['V']:5.1f} m/s  Re(length) {s['V'] * veh.geo['length'] / ef.NU:9.0f}  hand {hand(veh, s)}")
        print(f"  {len(S)} cases, quality {a.quality}: NOT RUN")
        return 0
    path = wc.OUT / f"{wc.slug(a.build)}_cfd.yaml"
    doc = wc.load(path, a.fresh)
    if doc and doc.get("quality") != a.quality:
        print("  the saved results are for another quality: starting over"); doc = None
    doc = doc or {"aircraft": a.build, "config": veh.label, "quality": a.quality, "template": "rans_ksst_external",
                  "reference": {"frontal_area_m2": veh.geo["frontal"], "length_m": veh.geo["length"], "x_cg_m": veh.x_cg, "mass_kg": veh.mass, "V_top_m_s": veh.V_top},
                  "note": "airframe without propellers; coefficients on the frontal area, Cm about the CG on the length", "cases": {}}
    todo = [n for n in S if doc["cases"].get(n, {}).get("status") != "solved"]
    print(f"  {len(S) - len(todo)} cases already solved (kept), {len(todo)} to run")
    t0, bad = time.time(), 0
    for k, n in enumerate(todo, 1):
        t1 = time.time()
        try:
            res = case(wc.RUNS / "cfd", a.build, veh, S[n], a.quality).ensure(run=True, processors=a.processors, timeout=a.timeout)
            doc["cases"][n] = row(veh, S[n], res)
        except Exception as exc:
            bad += 1
            doc["cases"][n] = {**S[n], "status": "failed", "message": f"{type(exc).__name__}: {exc}"[:300]}
        r = doc["cases"][n]
        r["seconds"] = round(time.time() - t1, 1)
        left = (time.time() - t0) / k * (len(todo) - k)
        print(f"  [{k}/{len(todo)}, ~{left / 60:.0f} min left] {n}: {r['status']}" + (f"  CdA {r['CdA_cm2']} cm2 (hand {r['hand']['CdA_clean_cm2']} clean, {r['hand']['CdA_with_margin_cm2']} with margin)" if r["status"] == "solved" else f"  {r.get('message', '')}"))
        wc.save(path, doc)
    drag = [r for r in doc["cases"].values() if r.get("campaign") == "drag" and r.get("status") == "solved"]
    if drag:
        top = max(drag, key=lambda r: r["V"])
        doc["summary"] = {"CdA_at_top_speed_cm2": top["CdA_cm2"], "hand_clean_cm2": top["hand"]["CdA_clean_cm2"],
                          "margin_the_cfd_supports": wc.num(top["CdA_cm2"] / top["hand"]["CdA_clean_cm2"], 2), "margin_assumed": ef.DRAG_MARGIN}
    wc.save(path, doc)
    print(f"-> {path}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
