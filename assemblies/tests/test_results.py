"""``assemblies.results``: the plain data every workflow writes for notebooks (``<name>_results.json``). Quick: a small
tree written and read back; the committed files in ``assemblies/data`` match their ``.vida``; every workflow writes the
file when it exports; the JSON reads with the standard library alone."""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest

from assemblies import DATA, results, vida
from assemblies.vida import Assembly

WORKFLOWS = ("microjet", "aguya", "quadcopter", "fixed_wing", "propulsors", "merlin", "boat", "submarine", "rover", "onager",
             "walkers")


def _tree():
    root = Assembly("demo", "thing", params={"fidelity": "smoke", "size": (1, 2)})
    a = root.add(Assembly("a", "solved", params={"n": 3}))
    a.record(curve=np.linspace(0.0, 1.0, 5), table={"x": [1, 2], "y": [3.0, float("nan")]},
             rows={"r1": {"p": 1.0, "q": 2}, "r2": {"p": 3.0, "q": 4}}, records=[{"k": 1}, {"k": 2}], scalar=np.float64(2.5))
    root.add(Assembly("b", "solver", params={})).not_run("run_fea=False")
    g = root.add(Assembly("graft", "other"))
    g.add(Assembly("deep", "x")).record(v=1)
    root.record(total=7)
    return root


def test_round_trip_status_tables_and_nan(tmp_path):
    root = _tree()
    vp = root.save(tmp_path / "demo.vida")
    out = results.write(root, vp, skip=("graft",))
    assert out == tmp_path / "demo_results.json"
    raw = json.loads(out.read_text())                      # strict JSON: no NaN in the file
    assert raw["name"] == "demo" and raw["fidelity"] == "smoke" and raw["vida"] == "demo.vida" and raw["skipped"] == ["graft"]
    assert set(raw["nodes"]) == {"", "a", "b"}
    r = results.load(out)
    assert r["a"]["curve"] == [0.0, 0.25, 0.5, 0.75, 1.0] and r["a"]["scalar"] == 2.5 and r["a"]["table"]["y"][1] is None
    assert r[""]["total"] == 7 and r.params("a") == {"n": 3} and r.params()["size"] == [1, 2]
    st = r.status()
    assert st.loc["b", "status"] == "NOT RUN" and st.loc["b", "not_run"] == "run_fea=False" and st.loc["a", "status"] == "computed"
    assert r.table("a", "table").shape == (2, 2) and math.isnan(r.table("a", "table")["y"][1])
    assert list(r.table("a", "rows").index) == ["r1", "r2"] and list(r.table("a", "records")["k"]) == [1, 2]


def test_large_results_stay_in_the_vida(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "INLINE_MAX_BYTES", 300)
    root = _tree()
    root.child("a").record(long=np.arange(100.0))                      # ~ 600 bytes of JSON: over the limit
    vp = root.save(tmp_path / "demo.vida")
    raw = json.loads(results.write(root, vp).read_text())
    ref = raw["nodes"]["a"]["results"]["long"]
    assert ref["__in_vida__"] == "demo.vida" and ref["node"] == "a" and ref["key"] == "long"
    assert raw["nodes"]["a"]["results"]["curve"] == [0.0, 0.25, 0.5, 0.75, 1.0]             # small ones stay inline
    np.testing.assert_allclose(results.load(results.path_for(vp))["a"]["long"], np.arange(100.0))


def test_grafted_trees_are_left_to_their_own_file():
    assert results.GRAFTED == {"aguya": ("engine",), "merlin": ("propulsors",)}


@pytest.mark.parametrize("name", WORKFLOWS)
def test_committed_results_match_their_vida(name):
    vp = DATA / f"{name}.vida"
    rp = results.path_for(vp)
    assert rp.is_file(), f"{rp.name} missing: python -m assemblies.results {name}"
    tree = vida.load(vp)
    r = results.load(name)
    skip = results.GRAFTED.get(name, ())
    expected = {p for p, _ in tree.walk() if not (p and p.split("/")[0] in skip)}
    assert set(r.nodes) == expected
    for p, n in tree.walk():
        if p in r.nodes:
            assert r.nodes[p]["status"] == n.status() and r.nodes[p]["kind"] == n.kind
            assert set(r.nodes[p]["results"]) == set(n.results)
    plain = json.loads(rp.read_text())                     # the standard library alone reads it
    assert plain["nodes"][""]["kind"] == tree.kind


@pytest.mark.parametrize("name", WORKFLOWS)
def test_every_workflow_writes_its_results_when_it_exports(name):
    src = (Path(results.__file__).parent / "workflows" / f"{name}.py").read_text()
    run = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "run")
    assert "export" in [a.arg for a in run.args.kwonlyargs], f"{name}.run has no export=..."
    assert "results.write(root, vida_path)" in ast.get_source_segment(src, run)
