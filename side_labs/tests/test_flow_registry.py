"""The chapter map lives once, in side_labs/flow.py: library's dicts (what wf and the GUI read) are generated from it
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
    cli = Path(library.ROOT) / "side_labs" / "cli.py"
    out = subprocess.run([sys.executable, str(cli), "run", "K9", "--variant", "4Q-2"], capture_output=True, text=True, timeout=120, cwd=library.ROOT)
    assert out.returncode == 2 and "needs ccx" in out.stderr and "--prepare-only" in out.stderr
