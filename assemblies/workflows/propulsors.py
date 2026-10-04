"""The propulsor libraries: propellers and ducted fans solved over airspeed x rpm, the maps MERLIN's race and mission
read (notebooks 25, 25b and 25c, their last section; scenarios/propulsor_maps.py).

The tree::

    propulsors (propulsor_libraries)
      propellers (propulsor_library)   two- and three-blade propellers, 9-12 in, pitch 0.6-1.2 D, with MERLIN's installation
      exotic (propulsor_library)       six and twelve blades
      edf (propulsor_library)          ducted fans 70-120 mm, one to three stages, and the lossless bound

Each node holds its whole library (``results["entries"]``, the maps as arrays). At ``full`` fidelity the library is
also written to ``assemblies/data/<kind>_maps.json`` (the format ``propulsor_maps.load`` reads); smaller fidelities
solve a 2 x 2 x ... corner of the space and write nothing shared. A committed map file made from the same space and
constants is loaded instead of solved again (the tree says ``source: file``). Each node also holds its notebook's
table at MERLIN's 1900 W electrical limit (``results["at_1900W"]``, one record per entry and layout: static thrust,
at 30 and 60 m/s; notebook 25 cell 48, 25b cell 30, 25c cell 10), interpolated from the maps.

    python -m assemblies.workflows.propulsors -j 4                # about a minute per propeller set, ten for the fans
    python -m assemblies.workflows.propulsors --fidelity smoke
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .. import DATA, RUNS, vida, results
from .._cli import main, parser
from ..components import propulsor_maps as pm
from ..vida import Assembly

NAME = "propulsors"
KINDS = ("propellers", "exotic", "edf")
SPACES = {"propellers": pm.PROP_SPACE, "exotic": pm.EXOTIC_SPACE, "edf": pm.EDF_SPACE}
COMPARED = ("kind", "space", "edf_bound", "section", "edf_tip_speed_m_s", "prop_tip_mach", "installation_speed_m_s", "rho")
MAP_ARRAYS = ("V", "rpm", "thrust", "power", "exit_velocity")      # an entry's tables (``propulsor_maps.load`` makes arrays)


def space_for(kind: str, fidelity: str) -> tuple[dict, dict | None]:
    space = SPACES[kind]
    bound = pm.EDF_BOUND if kind == "edf" else None
    if fidelity == "full":
        return dict(space), (dict(bound) if bound else None)
    corner = {k: tuple(v[:2]) for k, v in space.items()}
    return corner, ({k: tuple(v[:1]) for k, v in bound.items()} if bound else None)


def expected_meta(kind: str, space: dict, bound: dict | None) -> dict:
    """What ``build_library`` writes in ``meta`` for this space (the keys compared with a map file)."""
    return {"kind": kind, "space": space, "edf_bound": bound, "section": pm.SECTION_KW, "edf_tip_speed_m_s": pm.EDF_TIP_SPEED,
            "prop_tip_mach": pm.PROP_TIP_MACH, "installation_speed_m_s": pm.INSTALLATION_SPEED, "rho": 1.225}


def _same_meta(file_meta: dict, want: dict) -> bool:
    return all(vida.canonical(file_meta.get(k)) == vida.canonical(want.get(k)) for k in COMPARED)


def build(fidelity: str = "full", kinds=KINDS) -> Assembly:
    root = Assembly(NAME, "propulsor_libraries", params={"fidelity": fidelity, "v_table": pm.V_TABLE})
    for kind in kinds:
        space, bound = space_for(kind, fidelity)
        root.add(Assembly(kind, "propulsor_library", params=expected_meta(kind, space, bound) | {"v_table": pm.V_TABLE}))
    return root


def library(node: Assembly, out: Path, *, write_shared: bool, processes: int, progress) -> Assembly:
    kind = node.params["kind"]
    want = {k: node.params[k] for k in COMPARED}
    shared = Path(pm.LIBRARIES[kind])
    path = shared if write_shared else Path(out) / f"{kind}_maps.json"
    if path.is_file() and _same_meta(json.loads(path.read_text())["meta"], want):
        source = f"file {path.name}"
    else:
        # in the library's own key order: build_library takes the values positionally (node params are sorted)
        space = {k: tuple(node.params["space"][k]) for k in SPACES[kind]}
        bound = ({k: tuple(node.params["edf_bound"][k]) for k in pm.EDF_BOUND} if node.params["edf_bound"] else None)
        lib = pm.build_library(kind, space, processes=processes, progress=progress, edf_bound=bound)
        pm.save(lib, path)
        source = "solved"
    lib = pm.load(path)
    entries = [{k: v for k, v in e.items()} for e in lib["entries"]]
    node.attach(path.name, path, "geometry")
    node.record(entries=entries, meta=lib["meta"][kind], source=source, n=len(entries),
                failed=[e.get("id") for e in entries if "error" in e])
    return node


# ------------------------------------------------------------------------------------------------ the tables at 1900 W
def map_table(PMAPS: dict, pmaps=pm) -> dict:
    """Notebook 25 cell 48 (the library's load left out): every propeller as a tractor and as a pusher at MERLIN's
    1900 W electrical limit; the rows of its ``MAP_TAB``, keyed by (label, layout)."""
    rows = {}
    for e in PMAPS["entries"]:
        if "error" in e:
            continue
        for lay in ("tractor", "pusher"):
            u = pmaps.unit(e, 1900.0, layout=lay)
            rows[(e["label"], lay)] = {"static thrust at 1900 W [N]": u.full(0.0)[0], "at 30 m/s [N]": u.full(30.0)[0], "at 60 m/s [N]": u.full(60.0)[0],
                                       "w": e["installation"][lay]["w"], "t": e["installation"][lay]["t"]}
    return rows


def fan_table(FMAPS: dict, pmaps=pm) -> dict:
    """Notebook 25b cell 30 (the library's load and the scatter left out): every fan at 1900 W electrical, with its jet
    scrubbing MERLIN's fairing; the rows of its ``FAN_TAB``, keyed by label."""
    fans = [e for e in FMAPS["entries"] if "error" not in e]
    rows = {}
    for e in fans:
        u = pmaps.unit(e, 1900.0)
        rows[e["label"]] = {"stages": e["stages"], "quality": e["quality"], "static thrust at 1900 W [N]": u.full(0.0)[0],
                            "at 30 m/s [N]": u.full(30.0)[0], "at 60 m/s [N]": u.full(60.0)[0]}
    return rows


def exotic_table(XMAPS: dict, LAYOUTS=("tractor", "pusher"), pmaps=pm) -> dict:
    """Notebook 25c cell 10 (the library's load left out; ``LAYOUTS`` its cell 4's): the six- and twelve-blade propellers
    as tractor and pusher at 1900 W electrical; its table's rows, keyed by (label, layout)."""
    rows = {}
    for e in XMAPS["entries"]:
        if "error" in e:
            continue
        for lay in LAYOUTS:
            u = pmaps.unit(e, 1900.0, layout=lay)
            rows[(e["label"], lay)] = {"static thrust at 1900 W [N]": u.full(0.0)[0], "at 30 m/s [N]": u.full(30.0)[0], "at 60 m/s [N]": u.full(60.0)[0]}
    return rows


TABLES = {"propellers": map_table, "exotic": exotic_table, "edf": fan_table}       # the notebook's table per library


def at_1900W(node: Assembly) -> list[dict]:
    """A library node's table at 1900 W electrical (``TABLES``: its notebook's cell) as records: the entry's ``id``,
    ``label`` and (a propeller) ``layout``, then the cell's columns. Interpolation in the saved maps only."""
    entries = [dict(e, **{k: np.asarray(e[k], float) for k in MAP_ARRAYS if k in e}) for e in node.results["entries"]]
    rows = TABLES[node.params["kind"]]({"entries": entries})
    ids = {e["label"]: e["id"] for e in entries if "error" not in e}
    out = []
    for key, cols in rows.items():
        label, layout = key if isinstance(key, tuple) else (key, None)
        out.append({"id": ids[label], "label": label, **({"layout": layout} if layout else {}), **cols})
    return out


def run(*, fidelity: str = "full", kinds=KINDS, processors: int = 4, out: Path | None = None, vida_path: Path | None = None,
        include: str = "results", force: bool = False, redo=(), progress: bool = True, export: bool = True, **_ignored) -> Assembly:
    out = Path(out or RUNS / NAME).resolve()
    vida_path = Path(vida_path or DATA / f"{NAME}.vida")
    if force and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    root = build(fidelity, kinds)
    root.reuse(vida.load(vida_path) if vida_path.is_file() and not force else None)
    for path in redo:
        root.child(path).forget()
    for node in root.children:
        if not node.results:
            library(node, out, write_shared=fidelity == "full", processes=processors, progress=progress)
    for node in root.children:                                    # the notebooks' tables at 1900 W (interpolation, seconds)
        if "at_1900W" not in node.results:
            node.record(at_1900W=at_1900W(node))
    root.record(summary={n.name: {"maps": n.results["n"], "failed": len(n.results["failed"]), "source": n.results["source"]}
                         for n in root.children})
    root.meta["workflow"] = f"assemblies.workflows.{NAME}"
    root.meta["saved_to"] = str(root.save(vida_path, include=include))
    if export:                                                    # plain data for notebooks (assemblies.results)
        root.meta["results_to"] = str(results.write(root, vida_path))
    return root


def load_library(tree: Assembly) -> dict:
    """A saved propulsors tree as the library ``propulsor_maps.load`` returns (entries with arrays, by_id, meta, kinds)."""
    lib = {"entries": [], "meta": {}, "kinds": []}
    for n in tree.children:
        lib["kinds"].append(n.name)
        lib["meta"][n.name] = n.results["meta"]
        for e in n.results["entries"]:
            e = dict(e)
            for k in ("V", "rpm", "thrust", "power", "exit_velocity"):
                if k in e:
                    e[k] = np.asarray(e[k], float)
            lib["entries"].append(e)
    lib["by_id"] = {e["id"]: e for e in lib["entries"] if "error" not in e}
    return lib


def _parser():
    ap = parser("The propulsor libraries (propellers, exotic, ducted fans) over airspeed x rpm", cfd=False, fea=False)
    ap.add_argument("--kind", dest="kinds", action="append", choices=KINDS, help="only these libraries (repeatable)")
    return ap


def _run_cli(*, kinds=None, **kw):
    return run(kinds=tuple(kinds or KINDS), **kw)


if __name__ == "__main__":
    raise SystemExit(main(_run_cli, _parser(), DATA / f"{NAME}.vida"))
