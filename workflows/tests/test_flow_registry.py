"""The chapter map lives once, in workflows/flow.py: library's dicts (what wf and the GUI read) are generated from it
and equal what they were before Yanus (a frozen copy), and kestrel_campaigns reads the same map."""
import json
from pathlib import Path

import library

FROZEN = json.loads((Path(__file__).parent / "data" / "library_frozen.json").read_text())


def _j(x):
    return json.loads(json.dumps(x))


def test_kinds_books_imports_unchanged():
    assert _j(library.KINDS) == FROZEN["KINDS"]
    assert _j({k: {kk: vv for kk, vv in v.items() if kk != "describe"} for k, v in library.BOOKS.items()}) == FROZEN["BOOKS"]
    assert _j(library.IMPORTS) == FROZEN["IMPORTS"]


def test_chapters_unchanged():
    got = {k: {kk: vv for kk, vv in v.items() if kk != "description"} for k, v in library.chapters("kestrel").items()}
    assert _j(got) == FROZEN["chapters"]
    assert library.by_code("k10")[1] == "joint"


def test_flow_validates_and_campaigns_share_it():
    import kestrel_campaigns as kcp
    library.FLOW.validate()
    assert kcp.CHAPTERS == {(k.workflow, ch.campaign): ch.name for ch in library.FLOW.books["kestrel"].chapters.values() for k in [ch.kind]}
    assert kcp.SCRIPTS["airframe_cfd"] == library.KINDS["cfd"]["script"]
    assert kcp.SCRIPTS["import_legacy"] == library.IMPORTS["import_legacy"]["script"]


def test_pin_commands_name_the_notebooks_that_read_a_run(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"schema": 1, "book": "kestrel", "chapter": "airframe-sweep", "config": {"variant": "4Q-2"}}))
    assert library.pin_commands(d) == [f"yn pin 43_kestrel_study K1 --variant 4Q-2 {d}"]
    (d / "manifest.json").write_text(json.dumps({"schema": 1, "book": "kestrel", "chapter": "airframe-stall", "config": {"variant": "4Q-2"}}))
    assert library.pin_commands(d) == []


def test_needs_status():
    ns = library.needs_status("kestrel", "frame")
    assert set(ns) >= {"ok", "missing", "line"} and "ccx" in ns["line"] and "gmsh" in ns["line"]


def test_wf_run_stops_when_a_need_is_missing():
    import shutil
    import subprocess
    import sys
    import pytest
    if shutil.which("ccx"):
        pytest.skip("CalculiX is installed here")
    cli = Path(library.ROOT) / "workflows" / "cli.py"
    out = subprocess.run([sys.executable, str(cli), "run", "K9", "--variant", "4Q-2"], capture_output=True, text=True, timeout=120, cwd=library.ROOT)
    assert out.returncode == 2 and "needs ccx" in out.stderr and "--prepare-only" in out.stderr


def test_notebook_43_contract_matches_its_runs_shape():
    import kestrel_study as kstu
    from vegeta import yanus
    g = yanus.enter("43_kestrel_study", flow=library.FLOW, deep=False, raise_closed=False, screening=False, auto=False)   # the manual shape (43 is yanus_auto)
    runs, required = kstu.runs_from_gate(g)
    assert set(runs["airframe_cfd"]) == {"4Q-1", "4Q-2", "4Q-3"} and "4Q-2" in runs["structure_fea"]
    assert required == {("airframe_cfd", "4Q-2", "sweep"), ("structure_fea", "4Q-2")}
    g = yanus.enter("43_kestrel_study", flow=library.FLOW, deep=False, raise_closed=False, screening=True, auto=False)
    assert kstu.runs_from_gate(g)[1] == set()


def test_notebook_43_is_yanus_auto_and_its_plan_runs_screening_quality():
    from vegeta import yanus
    nb = library.FLOW.notebooks["43_kestrel_study"]
    assert nb.auto and library.FLOW.notebooks["45_kestrel_x"].auto
    first = yanus.enter(nb, flow=library.FLOW, deep=False, raise_closed=False, auto=False)
    steps = yanus.auto.plan(library.FLOW, first)
    cfd = [s for s in steps if s.unit.kind and s.unit.kind.name == "cfd"]
    fea = [s for s in steps if s.unit.kind and s.unit.kind.name == "fea"]
    assert all(s.argv[-2:] == ["--quality", "screening"] for s in cfd)
    assert all("--element-size" in s.argv for s in fea)
    assert all(s.state in ("planned", "skipped") for s in steps)        # a plan starts nothing


def test_notebook_47_contract_is_all_optional_and_f4_is_steady_only():
    from vegeta import yanus
    nb = library.FLOW.notebooks["47_svarog_iii"]
    assert not nb.auto                                                         # nothing is rerun silently
    g = yanus.enter(nb, flow=library.FLOW, deep=False, raise_closed=False, auto=False)
    keys = {r.key for r in g.rows}
    assert any(k.startswith("fenix/shield-steady@") for k in keys) and any("46_svarog:" in k for k in keys)
    assert library.by_code("f4")[1] == "shield-steady"
    import sys
    sys.path.insert(0, str(Path(library.ROOT) / "workflows" / "fenix"))
    import plate_thermal
    cases = plate_thermal.cases_of("shield-steady", "WFL-3000", "screening")
    assert len(cases) == 45 and all(c["mode"] == "steady" for c in cases) and not any("t_max" in c for c in cases)
