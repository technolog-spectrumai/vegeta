"""A simulated episode (Chiron / MuJoCo) as a node: run it only when allowed, keep its tables as plain results."""
from __future__ import annotations

import json

from ..vida import Assembly


def plain(x):
    """Episode tables and events as plain JSON values (DataFrames to row dicts, numpy to numbers, keys to strings)."""
    if hasattr(x, "to_dict") and hasattr(x, "columns"):
        x = x.to_dict(orient="index")
    return json.loads(json.dumps(x, default=lambda o: o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o))))


def run_scene(node: Assembly, episode, *, run: bool) -> Assembly:
    """``episode()`` returns a dict of results; it is called only when ``run`` (MuJoCo, minutes) and the node has none."""
    if node.results:
        return node
    if not run:
        return node.not_run("run_sim=False")
    try:
        import mujoco  # noqa: F401
    except ImportError:
        return node.not_run("MuJoCo is not installed (pip install mujoco)")
    node.record(**{k: plain(v) for k, v in episode().items()})
    return node
