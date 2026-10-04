"""The Onager series: one wheel-leg chassis, four machines (notebooks 20-23).

The tree::

    onager (rover_family)
      sentinel (onager)            the reconnaissance unit
        body (onager_body)         mass budget, CAD numbers rebuilt and compared with the recorded ones
        scene (episode)            the patrol with two partial failures, both responses (scenarios/onager_patrol.py)
        leg_fea (plate_legs)       upper and lower leg under walking, braking, a 0.4 m drop and cornering (Talos)
        stand (episode)            standing 2 s on flat ground in ChironLab: CG, corner loads, rear share, sag (20 cell 8)
        drive (wheel_leg_drive)    wheel mode, walking mode, the stand-up, thermal duty (20 cells 10-15, 26, 27)
        terrains (quarter_car)     the active leg as suspension over three terrains, the rock strike (20 cells 17, 22)
        pins (hand_checks)         stub axle and knee pin by hand under the leg cases (20 cell 23)
      atlas (onager)               the forklift          body + scene: the pallet job (scenarios/onager_atlas_pallet.py)
        stand, sizing (forklift_sizing: the lift drive), load_chart (forklift_load_chart: load chart, stance, ride)
      manus (onager)               the two-arm pincers   body + scene: the wire and the log (scenarios/onager_manus_tasks.py)
        stand, sizing (pincer_sizing: arm torques holding the log, cutting envelope, shoulder servo mode)
      sweeper (onager)             the street cleaner    body + scene: the street (scenarios/onager_sweeper_street.py)
        stand, sizing (sweeper_sizing: power and endurance, arms, pick-up table, unbalance forcing)

The scenes and the stands run MuJoCo through Chiron (minutes each; ``--no-sim`` skips them). The scenes record each
mission's outcome too (finished, the phase log, what the notebooks' exports carry). The analytic nodes are the lifted
cells of ``components.onager_sizing``; what needs the standing CG or rear share is filled once the stand has been
simulated: the Sentinel's standing rows are left out before (``holding_power_W`` None), the Atlas's ``load_chart`` node
stays empty (``-``). The nodes added after the first version (stand, drive, terrains, pins, sizing, load_chart) reuse
only themselves from the saved tree (``_add_new``), so ``--redo`` keeps its old effect on the others. The rock strike
and the standing case are
recorded as loads (and checked by hand on the pins); their leg FEA, and the FEA of the Atlas, Manus and Sweeper parts,
stay in the notebooks. The suction CFD of notebook 23 uses the recorded numbers the Sweeper's model carries
(``onager_sweeper_cfd.SUCTION``).
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path

from .. import DATA, RUNS, vida, results
from .._cli import main, parser
from ..components import (actuators as act, leg, onager as cad_sentinel, onager_atlas as cad_atlas, onager_atlas_robot as oar,
                          onager_atlas_scenario as oas, onager_manus as cad_manus, onager_manus_robot as omr,
                          onager_manus_scenario as oms, onager_robot as orb, onager_scenario as osc, onager_sweeper as cad_sweeper,
                          onager_sweeper_robot as osr, onager_sweeper_scenario as oss)
from ..components import onager_controller as oc, onager_sizing as sizing
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


def _scene(module, outcome=None):
    """The mission's episode: its events and phase table, whether the mission finished, the phase log (with the
    'done' / 'timeout' notes the table lacks), the duration run, and what ``outcome(ep, scene, timeseries)`` adds."""
    def episode(duration=None):
        scene = module.Scene()
        lab = module.make_lab(scene)
        ep = module.run(lab, scene, **({"duration": duration} if duration else {}))
        out = {"events": ep.log.get("events", []), "phases": module.phase_table(ep).round(4)}
        out.update(finished=bool(ep.log.get("mission_finished")), mission=ep.log.get("mission", []),
                   duration_s=float(ep.log["t"][-1]))
        if outcome is not None:
            out.update(outcome(ep, scene, module.timeseries(ep)))
        return out
    return episode


def _atlas_outcome(ep, scene, ts) -> dict:
    """Where the pallet came down (21 cells 20, 24: ``pallet_job.pallet_final``)."""
    return {"pallet_final": {"x": float(ts.pallet_x.iloc[-1]), "z": float(ts.pallet_z.iloc[-1]), "tilt_deg": float(ts.pallet_tilt_deg.iloc[-1])}}


def _manus_outcome(ep, scene, ts) -> dict:
    """When the wire parted, how high the log was lifted and where it was put down (22 cells 16, 17, 20)."""
    return {"cut_at": ep.log.get("cut_at"), "log_max_height_m": float(ts.log_z.max()), "log_final_y_m": float(ts.log_y.iloc[-1])}


def _sweeper_outcome(ep, scene, ts) -> dict:
    """The litter vacuumed (name: time) of how many pieces, and the pick-ups in the basket (23 cells 23, 27)."""
    return {"vacuumed": dict(ep.log["collected"]), "n_litter": len(scene.litter),
            "in_basket": {name: oss.in_basket(ep, name) for name, _, _ in scene.pickups}}


SCENES = {"sentinel": _scene_sentinel, "atlas": _scene(oas, _atlas_outcome), "manus": _scene(oms, _manus_outcome),
          "sweeper": _scene(oss, _sweeper_outcome)}
#: The machine and its ChironLab builder per variant (20 cell 8, 21-23 cell 6).
STAND = {"sentinel": (orb.onager, orb.onager_lab), "atlas": (oar.atlas, oar.atlas_lab), "manus": (omr.manus, omr.manus_lab),
         "sweeper": (osr.sweeper, osr.sweeper_lab)}


def _stand(variant: str, cad: dict) -> dict:
    """The machine on its recorded CAD numbers standing 2 s on flat ground (20 cell 8; 21, 22, 23 cell 6):
    ``sizing.stand_summary`` of the episode."""
    from vegeta import chiron
    make_robot, make_lab = STAND[variant]
    robot = make_robot(cad=cad)
    lab_flat = make_lab(chiron.Flat(), robot=robot)
    ep_stand = lab_flat.run(oc.Stand(), duration=2.0, rules=None, settle=0.0)
    return sizing.stand_summary(ep_stand.log)


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


def sub_masses(variant: str, p: dict, cad: dict) -> dict:
    """The tables the mass budget's 'see ...' rows point to, from the recorded CAD numbers (no CAD rebuilt): the
    Atlas's forklift and its counterweight (21 cell 6), the Manus's one arm (22 cell 6), the Sweeper's sweeping gear and
    one arm (23 cells 6, 10); None where the variant has none (the Sentinel)."""
    if variant == "atlas":
        return {"sub_masses_kg": {"forklift_masses": oar.forklift_masses(p, cad)}, "counterweight": dict(oar.COUNTERWEIGHT)}
    if variant == "manus":
        return {"sub_masses_kg": {"arm_masses": omr.arm_masses(p, cad)}, "counterweight": None}
    if variant == "sweeper":
        return {"sub_masses_kg": {"sweeper_masses": osr.sweeper_masses(p, cad), "arm_masses": omr.arm_masses(p, cad)},
                "counterweight": None}
    return {"sub_masses_kg": None, "counterweight": None}


def run(*, fidelity: str = "full", variants=tuple(VARIANTS), run_fea: bool = True, run_sim: bool = True, threads: int = 1,
        duration: float | None = None, out: Path | None = None, vida_path: Path | None = None, include: str = "results",
        force: bool = False, redo=(), progress: bool = True, export: bool = True, **_ignored) -> Assembly:
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
        if "sub_masses_kg" not in b.results:                    # also on a body saved before the sub-mass tables
            b.record(**sub_masses(v, node.params["design"], b.results["cad_numbers"]))
        sc = add_after(node, Assembly("scene", "episode", params={"variant": v, "design": node.params["design"],
                                                                  "duration": duration}), prior and _child(prior, v), redo)
        run_scene(sc, lambda v=v: SCENES[v](duration), run=run_sim)
        if v != "sentinel":
            _stand_node(node, v, prior, redo, run_sim)
            _variant_sizing(node, v, prior, redo)

    if "sentinel" in variants:
        _sentinel_legs(root, root.child("sentinel"), out, fidelity, prior, redo, run_fea, threads, progress)
        _stand_node(root.child("sentinel"), "sentinel", prior, redo, run_sim)     # after leg_fea (see _stand_node)
        _sentinel_sizing(root.child("sentinel"), prior, redo)
    root.record(masses_kg={v: root.child(f"{v}/body").results["mass_kg"] for v in variants})
    root.meta["workflow"] = f"assemblies.workflows.{NAME}"
    root.meta["saved_to"] = str(root.save(vida_path, include=include))
    if export:                                                    # plain data for notebooks (assemblies.results)
        root.meta["results_to"] = str(results.write(root, vida_path))
    return root


def _child(tree: Assembly, name: str):
    try:
        return tree.child(name)
    except KeyError:
        return None


def _add_new(parent: Assembly, node: Assembly, prior_parent: Assembly | None, redo=()) -> Assembly:
    """Add a node this workflow did not have before (stand, drive, terrains, pins, sizing, load_chart) and take its
    results from the saved tree when its key is unchanged, as ``add_after`` does, but reuse only ``node`` (and below):
    ``add_after`` reuses over the whole of ``parent``, which would copy back a sibling that ``--redo`` emptied and could
    not solve again (a scene with ``--no-sim``, the legs with ``--no-fea``), where the old workflow left it NOT RUN."""
    parent.add(node)
    if node.name not in redo and prior_parent is not None:
        node.reuse(_child(prior_parent, node.name))
    return node


def _stand_node(node: Assembly, v: str, prior, redo, run_sim: bool) -> Assembly:
    """The ``stand`` node of variant ``v`` (``_stand`` through ``run_scene``: NOT RUN with ``run_sim=False``). The
    Sentinel's is added after ``leg_fea``, whose ``add_after`` reuses over the whole variant: a stand ``--redo stand``
    emptied stays NOT RUN when it cannot be simulated."""
    st = _add_new(node, Assembly("stand", "episode", params={"variant": v, "design": node.params["design"]}),
                  prior and _child(prior, v), redo)
    return run_scene(st, lambda v=v, cad=node.child("body").results["cad_numbers"]: _stand(v, cad), run=run_sim)


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


def _sentinel_sizing(node, prior, redo):
    """The Sentinel's analytic nodes (``components.onager_sizing``, seconds of numpy): ``drive`` (20 cells 10-15, 26,
    27), ``terrains`` (cells 17, 22) and ``pins`` (cell 23) under leg_fea's cases, the rock strike and, once the stand
    has been simulated, the standing case. The stand's rear share is a parameter of ``drive`` and the standing case
    one of ``pins``: both are computed again (cheaply) when the stand gets its results."""
    pv = prior and _child(prior, "sentinel")
    b, st, lf = node.child("body"), node.child("stand"), node.child("leg_fea")
    p, budget, M = node.params["design"], b.results["mass_budget_kg"], b.results["mass_kg"]
    geo = orb.geometry(p)
    rear_share = st.results.get("rear_share")
    dn = _add_new(node, Assembly("drive", "wheel_leg_drive", params={
        "mass_budget_kg": budget, "design": p, "inputs": sizing.SENTINEL_DRIVE, "rear_share": rear_share,
        "actuators": {"leg": act.get(orb.LEG_ACTUATOR).as_dict(), "hub": act.get(orb.WHEEL_MOTOR).as_dict()},
        "servo": {"kp": orb.LEG_KP, "kd": orb.LEG_KD, "armature": orb.LEG_ARMATURE, "wheel_armature": orb.WHEEL_ARMATURE}}), pv, redo)
    if not dn.results:
        dn.record(**sizing.sentinel_drive(M, budget, geo, p, sizing.SENTINEL_DRIVE, REAR_SHARE=rear_share))
    tn = _add_new(node, Assembly("terrains", "quarter_car", params={
        "mass_budget_kg": budget, "design": p, "kp_kd": [orb.LEG_KP, orb.LEG_KD], "k_tyre": sizing.K_TYRE_20,
        "terrains": sizing.TERRAINS}), pv, redo)
    if not tn.results:
        corner = sizing.sentinel_corner(M, budget, geo, sizing.K_TYRE_20)
        tn.record(suspension=corner, **sizing.sentinel_terrains(M, geo, corner, sizing.TERRAINS))
    cases = {**lf.params["cases"], sizing.ROCK_STRIKE: tn.results["rock_strike"]}
    if st.results.get("rear_wheel_max_N") is not None:                                     # 20 cell 22's standing case
        cases[sizing.STANDING] = dict(fx=0.0, fy=0.0, fz=st.results["rear_wheel_max_N"])
    geom = sizing.pin_geometry(lf.params["cad"])
    pn = _add_new(node, Assembly("pins", "hand_checks", params={"cases": cases, "pins": geom}), pv, redo)
    if not pn.results:
        pn.record(pins=sizing.sentinel_pins(cases, geom))


def _variant_sizing(node, v, prior, redo):
    """The Atlas's, the Manus's and the Sweeper's analytic node ``sizing`` (``components.onager_sizing``); the Atlas's
    ``load_chart`` on the stand's CG: the node stays empty until the stand has been simulated (its ``com_m`` is a
    parameter, so the chart is computed on the first run after that)."""
    pv = prior and _child(prior, v)
    b = node.child("body")
    p, cad, budget, M = node.params["design"], b.results["cad_numbers"], b.results["mass_budget_kg"], b.results["mass_kg"]
    if v == "atlas":
        sz = _add_new(node, Assembly("sizing", "forklift_sizing", params={"design": p, "cad_numbers": cad, "inputs": sizing.ATLAS}),
                      pv, redo)
        if not sz.results:
            sz.record(drives=sizing.atlas_drives(p, cad, sizing.ATLAS))
        com = node.child("stand").results.get("com_m")
        lc = _add_new(node, Assembly("load_chart", "forklift_load_chart", params={
            "mass_budget_kg": budget, "design": p, "com_m": com, "inputs": sizing.ATLAS}), pv, redo)
        if not lc.results and com is not None:              # empty ('-') until atlas/stand has its CG (MuJoCo)
            lc.record(**sizing.atlas_load_chart(M, com[0], com[2], budget, p, sizing.ATLAS))
    elif v == "manus":
        sz = _add_new(node, Assembly("sizing", "pincer_sizing", params={"design": p, "cad_numbers": cad, "inputs": sizing.MANUS}),
                      pv, redo)
        if not sz.results:
            sz.record(**sizing.manus_sizing(p, cad, sizing.MANUS))
    elif v == "sweeper":
        sz = _add_new(node, Assembly("sizing", "sweeper_sizing", params={"design": p, "cad_numbers": cad, "mass_kg": M,
                                                                         "inputs": sizing.SWEEPER}), pv, redo)
        if not sz.results:
            sz.record(**sizing.sweeper_sizing(p, cad, M, sizing.SWEEPER))


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
