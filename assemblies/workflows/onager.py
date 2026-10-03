"""The Onager series: one wheel-leg chassis, four machines (notebooks 20-23).

The tree::

    onager (rover_family)
      sentinel (onager)            the reconnaissance unit
        body (onager_body)         mass budget, CAD numbers rebuilt and compared with the recorded ones
        scene (episode)            the patrol with two partial failures, both responses (scenarios/onager_patrol.py)
        leg_fea (plate_legs)       upper and lower leg under walking, braking, a 0.4 m drop and cornering (Talos)
      atlas (onager)               the forklift          body + scene: the pallet job (scenarios/onager_atlas_pallet.py)
      manus (onager)               the two-arm pincers   body + scene: the wire and the log (scenarios/onager_manus_tasks.py)
      sweeper (onager)             the street cleaner    body + scene: the street (scenarios/onager_sweeper_street.py)

The scenes run MuJoCo through Chiron (minutes each; ``--no-sim`` skips them). The leg cases of notebook 20 that need a
standing simulation or the quarter car over rocks (standing, rock strike) stay in the notebook. The suction CFD of
notebook 23 uses the recorded numbers the Sweeper's model carries (``onager_sweeper_cfd.SUCTION``).
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path

from .. import DATA, RUNS, vida
from .._cli import main, parser
from ..components import (actuators as act, leg, onager as cad_sentinel, onager_atlas as cad_atlas, onager_atlas_robot as oar,
                          onager_atlas_scenario as oas, onager_manus as cad_manus, onager_manus_robot as omr,
                          onager_manus_scenario as oms, onager_robot as orb, onager_scenario as osc, onager_sweeper as cad_sweeper,
                          onager_sweeper_robot as osr, onager_sweeper_scenario as oss)
from ..components._cad import export_kept
from ..vida import Assembly
from ._common import add_after, solve_fea
from ._scene import run_scene

NAME = "onager"
G = 9.81
VARIANTS = {"sentinel": (orb, cad_sentinel.OnagerSentinel), "atlas": (oar, cad_atlas.OnagerAtlas),
            "manus": (omr, cad_manus.OnagerManus), "sweeper": (osr, cad_sweeper.OnagerSweeper)}
DROP_M, S_LEG = 0.40, 0.15                                                                      # 20 cell 22
LEG_ELEMENT = {"smoke": (20.0, 6.0), "quick": (14.0, 4.0), "full": (10.0, 3.0)}                  # full: the notebook's


def _scene_sentinel(duration=None):
    lab = orb.onager_lab(osc.gravel_road())
    out = {}
    for response in osc.RESPONSES:
        ep = osc.run(lab, response, duration=duration or osc.DURATION)
        out[response] = {"events": ep.log["events"], "phases": osc.phase_table(ep).round(4)}
    return out


def _scene(module):
    def episode(duration=None):
        scene = module.Scene()
        lab = module.make_lab(scene)
        ep = module.run(lab, scene, **({"duration": duration} if duration else {}))
        return {"events": ep.log.get("events", []), "phases": module.phase_table(ep).round(4)}
    return episode


SCENES = {"sentinel": _scene_sentinel, "atlas": _scene(oas), "manus": _scene(oms), "sweeper": _scene(oss)}


def build(fidelity: str = "full", variants=tuple(VARIANTS)) -> Assembly:
    root = Assembly(NAME, "rover_family", params={"fidelity": fidelity})
    for v in variants:
        module, _ = VARIANTS[v]
        p = module.design_params()
        root.add(Assembly(v, "onager", params={"design": p})).add(Assembly("body", "onager_body", params={"design": p}))
    return root


def body(node: Assembly, variant: str) -> Assembly:
    """The mass budget on CAD numbers recomputed from the design (and how far they are from the recorded ones)."""
    module, _ = VARIANTS[variant]
    p = node.params["design"]
    cad = module.cad_numbers(p, recompute=True)
    drift = {k: abs(cad[k] / module.CAD[k] - 1) for k, ref in module.CAD.items()
             if k in cad and isinstance(ref, (int, float)) and ref}                    # areas and volumes, not the centres
    budget = module.mass_budget(p, cad)
    total = sum(budget.values())
    return node.record(mass_budget_kg=budget, mass_kg=total, cad_numbers=cad, cad_drift_max=max(drift.values()) if drift else None)


def run(*, fidelity: str = "full", variants=tuple(VARIANTS), run_fea: bool = True, run_sim: bool = True, threads: int = 1,
        duration: float | None = None, out: Path | None = None, vida_path: Path | None = None, include: str = "results",
        force: bool = False, redo=(), progress: bool = True, **_ignored) -> Assembly:
    out = Path(out or RUNS / NAME).resolve()
    vida_path = Path(vida_path or DATA / f"{NAME}.vida")
    if force and out.exists():
        shutil.rmtree(out)
    prior = vida.load(vida_path) if vida_path.is_file() and not force else None
    root = build(fidelity, variants)
    root.reuse(prior)
    from tqdm.auto import tqdm
    for v in tqdm(variants, desc="Onager variants (CAD, MuJoCo scene)", disable=not progress):
        node = root.child(v)
        b = node.child("body")
        if v in redo or "body" in redo:
            b.forget()
        if not b.results:
            body(b, v)
        sc = add_after(node, Assembly("scene", "episode", params={"variant": v, "design": node.params["design"],
                                                                  "duration": duration}), prior and _child(prior, v), redo)
        run_scene(sc, lambda v=v: SCENES[v](duration), run=run_sim)

    if "sentinel" in variants:
        _sentinel_legs(root, root.child("sentinel"), out, fidelity, prior, redo, run_fea, threads, progress)
    root.record(masses_kg={v: root.child(f"{v}/body").results["mass_kg"] for v in variants})
    root.meta["workflow"] = f"assemblies.workflows.{NAME}"
    root.meta["saved_to"] = str(root.save(vida_path, include=include))
    return root


def _child(tree: Assembly, name: str):
    try:
        return tree.child(name)
    except KeyError:
        return None


def _sentinel_legs(root, node, out, fidelity, prior, redo, run_fea, threads, progress):
    """Notebook 20 cells 19 and 22, the cases that need no simulation: walk peak, braking, drop landing, cornering."""
    p = cad_sentinel.OnagerSentinel().resolve()
    geo = orb.geometry()
    M = node.child("body").results["mass_kg"]
    W = M * G
    hub = act.get(orb.WHEEL_MOTOR)
    landing_mean = M * G * (DROP_M / S_LEG + 1) / 2
    cases = {"walk peak (beta 0.75)": dict(fx=0.3 * leg.foot_peak(W, 0.75), fy=0.0, fz=leg.foot_peak(W, 0.75)),
             "braking, full motor torque": dict(fx=-hub.stall_Nm / geo["r_wheel"], fy=0.0, fz=1.3 * W / 4),
             "drop landing 0.4 m on two wheels": dict(fx=0.2 * landing_mean * math.pi / 2, fy=0.0, fz=landing_mean * math.pi / 2),
             "cornering 0.4 g, outer wheel": dict(fx=0.0, fy=0.4 * W / 4 * 1.5, fz=1.5 * W / 4)}
    el, el_min = LEG_ELEMENT[fidelity]
    lf = add_after(node, Assembly("leg_fea", "plate_legs", params={"cad": p, "a1": geo["a1"], "a2": geo["a2"], "cases": cases,
                                                                   "material": leg.AL7075, "element_mm": [el, el_min]}),
                   prior and _child(prior, "sentinel"), redo)
    if lf.results.get("complete"):
        return
    d = cad_sentinel.OnagerSentinel()
    steps = {}
    for part in ("upper_leg", "lower_leg"):
        q = dict(p, part=part)
        files, _ = export_kept(lambda q=q: d.generate(**q), q, out / "cad" / part, part, formats=("step",))
        steps[part] = files["step"]
        lf.attach(f"{part}.step", files["step"], "geometry")
    models = {}
    for name, f in cases.items():
        key = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")
        models.update(leg.plate_leg_models(steps["lower_leg"], steps["upper_leg"], p, f, a1=geo["a1"], a2=geo["a2"], name=key,
                                           element_mm=el, min_element_mm=el_min))
    solve_fea(lf, models, out / "leg_fea", run=run_fea, threads=threads, progress=progress)


def _parser():
    ap = parser("The Onager series: mass budgets, the scenes in MuJoCo, the Sentinel's legs in FEA", cfd=False)
    ap.add_argument("--variant", dest="variants", action="append", choices=tuple(VARIANTS), help="only these (repeatable)")
    ap.add_argument("--no-sim", dest="run_sim", action="store_false", help="run no MuJoCo scene (simulated ones are kept)")
    ap.add_argument("--duration", type=float, default=None, help="scene length [s] (default: each scene's own)")
    return ap


def _run_cli(*, variants=None, **kw):
    return run(variants=tuple(variants or VARIANTS), **kw)


if __name__ == "__main__":
    raise SystemExit(main(_run_cli, _parser(), DATA / f"{NAME}.vida"))
