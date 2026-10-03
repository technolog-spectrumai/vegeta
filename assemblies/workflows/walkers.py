"""The walkers: the robot dog (notebook 16), the Myropods Persephone (17) and Cleopatra (18), and Apheloria (19).

The tree::

    walkers (walker_family)
      dog (walker)
        body (walker_body)           mass budget from the CAD numbers, the Chiron robot (mass, joints) against it
        gaits (foot_peaks)           peak foot force of walk, trot and bound, alone and with the payload (16 cell 8)
      cleopatra (walker)
        body (walker_body)           notebook 18's budget and the Chiron robot
      persephone (walker)
        body (walker_cad)            the crawler's CAD (myropod.Myropod defaults): volume, area, size
      apheloria (walker)
        body (walker_body)           the Chiron robot: head + 8 segments, 96 leg servos, 8 body pitch joints
        pack (episode)               packs into its ball in MuJoCo   (scenarios/apheloria_pack.py, without the movie)
        unpack (episode)             and opens again

What stays in the notebooks: the leg torque tables of 18 and 19 (they need each notebook's module table), the
gait simulations of 16 and 17, Persephone's flue climb. The trials of the dog and Cleopatra are in ``benchmark/``.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

from .. import DATA, vida
from .._cli import main, parser
from ..components import apheloria_pack as ap_scene, apheloria_robot as ar, leg, myropod, myropod_robot as mr
from ..components import robot_dog_robot as rdr
from ..vida import Assembly
from ._common import add_after
from ._scene import run_scene

NAME = "walkers"
G = 9.81
MACHINES = ("dog", "cleopatra", "persephone", "apheloria")
DOG_GAITS = {"walk": 0.75, "trot": 0.50, "bound": 0.30}                                   # 16 cell 8
DOG_PAYLOAD_KG = 5.0                                                                       # 16 cell 6: the deck allowance


def _robot(robot) -> dict:
    s = robot.summary()
    return {"robot_mass_kg": robot.total_mass(), "servos": len(robot.actuated_joints()),
            "links": s.get("n_links"), "passive_joints": s.get("n_passive"), "link_mass_kg": s.get("link_mass_kg")}


def _design(machine: str) -> dict:
    if machine == "dog":
        return rdr.design_params()
    if machine == "cleopatra":
        return dataclasses.asdict(mr.CleopatraParams())
    if machine == "persephone":
        return myropod.Myropod().resolve()
    return {**dataclasses.asdict(ar.PARAMS), "n_segments": ar.N_SEGMENTS, "plates": True}


def build(machines=MACHINES) -> Assembly:
    root = Assembly(NAME, "walker_family")
    for m in machines:
        p = _design(m)
        node = root.add(Assembly(m, "walker", params={"design": p}))
        node.add(Assembly("body", "walker_cad" if m == "persephone" else "walker_body", params={"design": p}))
        if m == "dog":
            node.add(Assembly("gaits", "foot_peaks", params={"duty": DOG_GAITS, "payload_kg": DOG_PAYLOAD_KG}))
    return root


def body(node: Assembly, machine: str) -> Assembly:
    """The machine's mass: the budget the notebook writes down and the Chiron robot built from the same numbers."""
    if machine == "dog":
        budget = rdr.mass_budget()
        return node.record(mass_budget_kg=budget, mass_kg=sum(budget.values()), **_robot(rdr.dog_robot()))
    if machine == "cleopatra":
        budget = mr.mass_budget()
        return node.record(mass_budget_kg=budget, mass_kg=budget["total"], **_robot(mr.cleopatra()))
    if machine == "persephone":
        m = myropod.Myropod().generate().measure()
        return node.record(volume_mm3=m["volume"], surface_area_mm2=m["surface_area"], dimensions_mm=m["dimensions"],
                           center_of_mass_mm=m["center_of_mass"])
    r = ar.apheloria()
    return node.record(mass_kg=r.total_mass(), **_robot(r))


def dog_gaits(node: Assembly, mass_kg: float) -> Assembly:
    """16 cell 8: the half-sine peak foot force of each gait, the dog alone and with the payload allowance."""
    W, W_loaded = mass_kg * G, (mass_kg + node.params["payload_kg"]) * G
    rows = {g: {"duty": b, "feet_down_mean": 4 * b, "peak_foot_N": leg.foot_peak(W, b), "peak_foot_loaded_N": leg.foot_peak(W_loaded, b),
                "F_over_W_per_foot": leg.foot_peak(1.0, b)} for g, b in node.params["duty"].items()}
    return node.record(gaits=rows)


def run(*, machines=MACHINES, run_sim: bool = True, vida_path: Path | None = None, include: str = "results", force: bool = False,
        redo=(), progress: bool = True, **_ignored) -> Assembly:
    vida_path = Path(vida_path or DATA / f"{NAME}.vida")
    prior = vida.load(vida_path) if vida_path.is_file() and not force else None
    root = build(machines)
    root.reuse(prior)
    from tqdm.auto import tqdm
    for m in tqdm(machines, desc="walkers (CAD, robots, MuJoCo scenes)", disable=not progress):
        node = root.child(m)
        b = node.child("body")
        if m in redo or "body" in redo:
            b.forget()
        if not b.results:
            body(b, m)
        if m == "dog":
            g = node.child("gaits")
            if "gaits" in redo:
                g.forget()
            if not g.results:
                dog_gaits(g, b.results["mass_kg"])
        if m == "apheloria":
            for scene in ("pack", "unpack"):
                sc = add_after(node, Assembly(scene, "episode", params={"design": node.params["design"], "scene": scene}),
                               prior and _child(prior, m), redo)
                run_scene(sc, lambda scene=scene: ap_scene.scene(scene), run=run_sim)
    root.record(masses_kg={m: root.child(f"{m}/body").results["mass_kg"] for m in machines if m != "persephone"})
    root.meta["workflow"] = f"assemblies.workflows.{NAME}"
    root.meta["saved_to"] = str(root.save(vida_path, include=include))
    return root


def _child(tree: Assembly, name: str):
    try:
        return tree.child(name)
    except KeyError:
        return None


def _parser():
    ap = parser("The walkers: masses, the dog's foot loads, Persephone's CAD, Apheloria packing in MuJoCo", cfd=False, fea=False)
    ap.add_argument("--machine", dest="machines", action="append", choices=MACHINES, help="only these (repeatable)")
    ap.add_argument("--no-sim", dest="run_sim", action="store_false", help="run no MuJoCo scene (simulated ones are kept)")
    return ap


def _run_cli(*, machines=None, **kw):
    return run(machines=tuple(machines or MACHINES), **kw)


if __name__ == "__main__":
    raise SystemExit(main(_run_cli, _parser(), DATA / f"{NAME}.vida"))
