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


def test_read_snippet_names_the_find_line_of_a_run(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"schema": 1, "book": "kestrel", "chapter": "airframe-sweep", "config": {"variant": "4Q-2"}}))
    assert library.read_snippet(d) == f'yanus.find("K1", variant="4Q-2", run="{d}")'
    (d / "manifest.json").write_text(json.dumps({"schema": 1, "book": "kestrel", "chapter": "airframe-sweep", "config": {"variant": "4Q-2"}, "mock": True}))
    assert library.read_snippet(d) is None                                         # a mock run is read by no notebook
    (d / "manifest.json").write_text(json.dumps({"schema": 1, "book": "other", "chapter": "x"}))
    assert library.read_snippet(d) is None


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


def test_notebook_43_reads_its_runs_with_yanus_and_stops_on_the_required_ones(tmp_path, monkeypatch):
    import kestrel_campaigns as kcp
    import kestrel_study as kstu
    import pytest
    monkeypatch.setenv("VEGETA_WORKFLOW_RUNS", str(tmp_path))                     # no run on disk
    fps = {v: kcp.fingerprint(v) for v in kstu.VARIANTS}
    sel = kstu.select_runs(fps, {("K1", "4Q-2"): tmp_path / "legacy_gone"})
    assert set(sel) == set(kstu.INPUTS) and not any(s.ok for s in sel.values())
    assert sel[("K1", "4Q-2")].explicit and sel[("K1", "4Q-2")].command.startswith("wf import_legacy")     # an explicit run is never replaced
    assert {k for k, req in kstu.INPUTS.items() if req} == {("K1", "4Q-2"), ("K11", "4Q-2")}
    with pytest.raises(kcp.PrerequisiteError) as exc:
        kstu.prerequisites(sel, {("K1", "4Q-2"), ("K11", "4Q-2")})
    msg = str(exc.value)
    assert "wf import_legacy" in msg and "wf K11 --variant 4Q-2" in msg and "KESTREL_SCREENING" in msg
    table, summ = kstu.prerequisites(sel, set())                                   # screening: everything optional, NOT RUN
    assert (table.loc[[s.label for s in sel.values()], "role"] == "optional").all() and summ == {"airframe_cfd": {}, "structure_fea": {}}
