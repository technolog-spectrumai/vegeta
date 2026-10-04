"""What a workflow computed, as plain data a notebook imports: ``assemblies/data/<name>_results.json``.

Every workflow writes it next to its ``.vida`` when it exports (``--no-export`` skips it); ``python -m assemblies.results
all`` writes it again from the saved ``.vida`` files without running anything. It is ordinary JSON::

    {"workflow": ..., "fidelity": ..., "written_at": ..., "git": {...}, "vida": "<name>.vida",
     "nodes": {"<path>": {"kind", "status" (computed / reused / NOT RUN / -), "not_run": [...], "computed_at",
                          "params": {...}, "results": {...}}}}

with numpy arrays as lists and NaN / infinities as null. A sub-tree grafted from another product's ``.vida``
(AGUYA's ``engine``, MERLIN's ``propulsors``) is left out: that product's own file has it. A single result larger
than ``INLINE_MAX_BYTES`` stays in the ``.vida`` and the JSON holds a reference to it
(``{"__in_vida__": "<file>", "node": ..., "key": ..., "bytes": ...}``); ``load`` resolves it.

In a notebook (the repository root on ``sys.path``, e.g. ``sys.path.insert(0, "..")`` from ``notebooks/``)::

    from assemblies import results
    q = results.load("quadcopter")
    q.status()                                    # one row per node: kind, status, why NOT RUN
    q["life/fatigue"]["hours_to_failure"]          # a node's results (a dict)
    q.table("life/fatigue", "life")                # a table-like result as a pandas DataFrame

or without ``assemblies``: ``json.load(open("../assemblies/data/quadcopter_results.json"))["nodes"]``.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from . import DATA, vida

INLINE_MAX_BYTES = 1_000_000
SUFFIX = "_results.json"
GRAFTED = {"aguya": ("engine",), "merlin": ("propulsors",)}        # sub-trees another product's file holds


def plain(x):
    """JSON-ready: numpy to lists and numbers, tuples to lists, keys to strings, NaN / inf to None."""
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, np.generic):
        return plain(x.item())
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if x is None or isinstance(x, (bool, int, str)):
        return x
    return str(x)


def path_for(vida_path: str | Path) -> Path:
    """``<dir>/<stem>_results.json`` next to a ``.vida``."""
    vida_path = Path(vida_path)
    return vida_path.with_name(f"{vida_path.stem}{SUFFIX}")


def to_dict(root: vida.Assembly, *, skip=(), vida_name: str | None = None) -> dict:
    """The tree as plain data (``skip``: names of top-level sub-trees to leave out)."""
    nodes = {}
    for p, n in root.walk():
        if p and p.split("/")[0] in skip:
            continue
        res = {}
        for k, v in n.results.items():
            pv = plain(v)
            size = len(json.dumps(pv))
            res[k] = pv if size <= INLINE_MAX_BYTES else {"__in_vida__": vida_name, "node": p, "key": k, "bytes": size}
        reused = n.meta.get("reused") or {}
        nodes[p] = {"kind": n.kind, "status": n.status(), "not_run": list(n.meta.get("not_run") or []),
                    "computed_at": n.meta.get("computed_at") or reused.get("computed_at"),
                    "params": plain(n.params), "results": res}
    return {"workflow": root.meta.get("workflow") or f"assemblies.workflows.{root.name}", "name": root.name,
            "fidelity": root.params.get("fidelity"), "skipped": list(skip), "nodes": nodes}


def write(root: vida.Assembly, vida_path: str | Path, *, skip=None, path: str | Path | None = None) -> Path:
    """Write ``<stem>_results.json`` next to the saved ``vida_path`` (or to ``path``); returns its path."""
    vida_path = Path(vida_path)
    skip = GRAFTED.get(root.name, ()) if skip is None else tuple(skip)
    out = Path(path) if path is not None else path_for(vida_path)
    d = to_dict(root, skip=skip, vida_name=vida_path.name)
    try:
        git = vida.manifest(vida_path).get("git")
    except (OSError, KeyError, ValueError):
        git = None
    d = {**{k: d[k] for k in ("workflow", "name", "fidelity")}, "written_at": vida.utc_now(), "git": git,
         "vida": vida_path.name, **{k: d[k] for k in ("skipped", "nodes")}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=1, allow_nan=False))
    return out


class Results:
    """A ``<name>_results.json`` read back: ``r[path]`` is that node's results (``r[""]`` the root's)."""

    def __init__(self, data: dict, path: Path):
        self.data, self.path = data, Path(path)
        self.nodes = data["nodes"]
        self._vida = None

    def __getitem__(self, node_path: str) -> dict:
        return {k: self._resolve(node_path, k, v) for k, v in self.nodes[node_path]["results"].items()}

    def __contains__(self, node_path: str) -> bool:
        return node_path in self.nodes

    def __iter__(self):
        return iter(self.nodes)

    def __repr__(self) -> str:
        return f"Results({self.data.get('name')!r}, {len(self.nodes)} nodes, fidelity={self.data.get('fidelity')!r}, {self.path})"

    @property
    def name(self) -> str:
        return self.data.get("name")

    def params(self, node_path: str = "") -> dict:
        return self.nodes[node_path]["params"]

    def status(self):
        """One row per node (pandas DataFrame): kind, status, why NOT RUN, when computed."""
        import pandas as pd
        return pd.DataFrame([{"node": p or self.name, "kind": n["kind"], "status": n["status"],
                              "not_run": "; ".join(n["not_run"]), "computed_at": n["computed_at"]}
                             for p, n in self.nodes.items()]).set_index("node")

    def table(self, node_path: str, key: str):
        """A table-like result as a pandas DataFrame: columns (``{column: [values]}``), rows (``{row: {column: v}}``)
        or a list of records (``[{column: v}, ...]``)."""
        import pandas as pd
        v = self[node_path][key]
        if isinstance(v, list):
            return pd.DataFrame(v)
        if isinstance(v, dict) and v and all(isinstance(x, dict) for x in v.values()):
            return pd.DataFrame.from_dict(v, orient="index")
        if isinstance(v, dict) and v and all(isinstance(x, list) for x in v.values()):
            return pd.DataFrame(v)
        return pd.DataFrame([v])

    def _resolve(self, node_path, key, v):
        if isinstance(v, dict) and "__in_vida__" in v:
            if self._vida is None:
                self._vida = vida.load(self.path.with_name(v["__in_vida__"]))
            return self._vida.child(v["node"]).results[v["key"]] if v["node"] else self._vida.results[v["key"]]
        return v


def load(name_or_path: str | Path, data_dir: str | Path | None = None) -> Results:
    """A workflow's results by name (``"quadcopter"`` -> ``assemblies/data/quadcopter_results.json``) or by path."""
    p = Path(name_or_path)
    if p.suffix != ".json":
        p = Path(data_dir or DATA) / f"{name_or_path}{SUFFIX}"
    return Results(json.loads(p.read_text()), p)


def main(argv=None) -> int:
    """``python -m assemblies.results all | NAME ...``: write ``<name>_results.json`` from the saved ``.vida``."""
    names = list(argv if argv is not None else sys.argv[1:])
    if not names or names == ["all"]:
        names = sorted(p.stem for p in DATA.glob("*.vida"))
    for n in names:
        vp = DATA / f"{n}.vida"
        if not vp.is_file():
            print(f"no {vp}: run python -m assemblies.workflows.{n} first", file=sys.stderr)
            return 2
        print("wrote", write(vida.load(vp), vp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
