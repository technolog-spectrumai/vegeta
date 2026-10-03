"""AGUYA's wing at the dash: Pratt's gust load, the wing in FEA at the gust and at the 6 g limit (Talos), and its
free-vibration modes with the reduced frequency at the dash (the flutter flag).

Promoted from notebook 29 section 9. The structure is a solid-equivalent glass/carbon skin over a foam core: an
assumption for a first look; replace it with a laminate calculation.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
from vegeta import talos

from ..vida import Assembly, digest
from . import aguya as ag

COMPOSITE = talos.Material("glass/carbon skin on foam (solid-equivalent)", youngs_modulus=6000.0, poissons_ratio=0.3,
                           density=0.45e-9, yield_strength=45.0, source="assumed; replace with a laminate calculation")
U_GUST, N_LIMIT, RHO, G = 7.5, 6.0, 1.225, 9.80665
ELEMENT_MM = {"smoke": 10.0, "quick": 6.0, "full": 4.0}
N_MODES = 6


def gust(P: dict, af, v_dash: float) -> dict:
    """Pratt's gust formula at the dash: load factor, alleviation factor, gust penetration speed for the 6 g limit."""
    W = (af.dry_mass_kg + af.tank_kg) * G
    S, c_mean = af.wing_area_m2, P["root_chord"] * (1 + P["taper"]) / 2000
    a = 2 * math.pi * af.aspect_ratio / (af.aspect_ratio + 2)
    mu = 2 * W / S / (RHO * c_mean * a * G)
    k_g = 0.88 * mu / (5.3 + mu)
    return {"weight_N": W, "wing_area_m2": S, "mean_chord_m": c_mean, "lift_slope": a, "mass_ratio": mu, "alleviation": k_g,
            "v_dash": v_dash, "gust_m_s": U_GUST, "load_factor": 1 + k_g * RHO * v_dash * U_GUST * a * S / (2 * W),
            "limit_load_factor": N_LIMIT, "penetration_speed": (N_LIMIT - 1) * 2 * W / (k_g * RHO * U_GUST * a * S)}


def assembly(P: dict, af, v_dash: float, fidelity: str) -> Assembly:
    if fidelity not in ELEMENT_MM:
        raise ValueError(f"fidelity must be one of {tuple(ELEMENT_MM)}")
    return Assembly("wing", "wing_fea", params=dict(geometry=P, gust=gust(P, af, v_dash), material=asdict(COMPOSITE),
                                                    element_mm=ELEMENT_MM[fidelity], n_modes=N_MODES, fidelity=fidelity))


def step(P: dict, out: Path) -> Path:
    """The wing as STEP in ``out`` (kept when made from the same parameters: a new STEP would change the mesh key)."""
    out = Path(out)
    stamp, want = out / "wing.params", digest({"geometry": P, "part": "wing"})
    steps = sorted(out.glob("*.step")) if out.is_dir() else []
    if len(steps) == 1 and stamp.is_file() and stamp.read_text().strip() == want:
        return steps[0]
    for old in steps:
        old.unlink()
    res = ag.Aguya().generate(**dict(P, part="wing")).export(out, formats=("step",), basename="wing")
    res.raise_for_status()
    stamp.write_text(want + "\n")
    return Path(res.artifacts["step"])


def lower_skins(step_file: Path) -> list[int]:
    """The two lower skin surfaces (the pressure of the lift goes on them)."""
    info = talos.inspect_step(step_file, units="mm-N-MPa")
    skins = [s for s in info.surfaces if s.kind == "BSpline surface" and s.area > 5e4]
    return [min((s for s in skins if (s.centroid[1] > 0) == right), key=lambda s: s.centroid[2]).tag for right in (False, True)]


def models(node: Assembly, step_file: Path) -> dict[str, talos.StructuralModel]:
    p, g = node.params, node.params["gust"]
    yc = p["geometry"]["fuselage_diameter"] / 2 + 5.0
    regions = [talos.SurfacesInBox("root", (-1.0, -yc - 0.1, -100.0, 400.0, yc + 0.1, 100.0)),
               talos.Surfaces("lift", lower_skins(step_file))]

    def model(n, name):
        return talos.StructuralModel(step_file, "mm-N-MPa", COMPOSITE, regions, [talos.FixedSupport("root")],
                                     [talos.Pressure("lift", n * g["weight_N"] / (g["wing_area_m2"] * 1e6))],
                                     talos.MeshSettings(element_size=p["element_mm"]), name=name)

    return {"limit_6g": model(g["limit_load_factor"], "limit_6g"), "gust_at_dash": model(g["load_factor"], "gust_at_dash")}


def solve(node: Assembly, out: Path, *, run: bool = True, threads: int = 1, progress=True) -> Assembly:
    """Both static cases (read back when solved with the same inputs) and the modes; recorded on ``node``."""
    out = Path(out)
    st = step(node.params["geometry"], out / "geometry")
    node.attach("wing.step", st, "geometry")
    ms = models(node, st)
    rs = talos.solve_models(list(ms.values()), [out / k for k in ms], threads=threads, run=run, progress=progress)
    rows = {}
    for (name, _), r in zip(ms.items(), rs):
        m = r.metrics if r.ok else {}
        rows[name] = {"ok": bool(r.ok), "tip_deflection_mm": m.get("max_displacement"), "max_von_mises_MPa": m.get("max_von_mises"),
                      "safety_factor": m.get("safety_factor_yield")}
        if (out / name).exists():
            node.attach(name, out / name, "mesh")
        if not r.ok:
            node.not_run(f"{name}: {r.messages[-1] if r.messages else r.status}")
    modes = None
    if run and rs[0].ok:
        mdir = out / "modes"
        if not ms["limit_6g"].mesh_is_current(mdir):
            ms["limit_6g"].mesh(mdir)
        mr = ms["limit_6g"].solve_modes(mdir, n_modes=node.params["n_modes"], threads=threads)
        if mr.ok:
            f = np.asarray(mr.metrics["frequencies_hz"], float)
            g = node.params["gust"]
            modes = {"frequency_hz": f, "reduced_frequency_at_dash": 2 * math.pi * f * g["mean_chord_m"] / (2 * g["v_dash"])}
            node.attach("modes", mdir, "mesh")
        else:
            node.not_run(f"modes: {mr.messages[-1] if mr.messages else mr.status}")
    if any(row["ok"] for row in rows.values()):
        node.record(stress=rows, modes=modes, complete=all(row["ok"] for row in rows.values()) and modes is not None)
    elif not run:
        node.meta["not_run"] = ["run_fea=False"]
    return node
