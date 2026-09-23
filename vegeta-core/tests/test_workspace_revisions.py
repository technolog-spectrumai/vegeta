import json

import pytest

from vegeta import core
from vegeta.core import ImmutableError, Workspace


def test_create_open_and_refuse_twice(tmp_path, ws):
    assert Workspace.open(ws.root).name == "test"
    with pytest.raises(FileExistsError):
        Workspace.create(ws.root)
    with pytest.raises(FileNotFoundError):
        Workspace.open(tmp_path / "nothing")


def test_add_design_module_and_file(ws, plate_file):
    d = ws.add_design("beam", "vegeta.dedalus.examples:CantileverBeam")
    assert [p["name"] for p in d.record["parameters"]] == ["length", "width", "height"]
    p = ws.add_design("plate", f"{plate_file}:Plate")
    assert not p.source.startswith("/")  # stored relative to the workspace
    assert [x.name for x in ws.designs()] == ["beam", "plate"]
    with pytest.raises(FileExistsError):
        ws.add_design("beam", "vegeta.dedalus.examples:CantileverBeam")
    with pytest.raises(ValueError):
        ws.add_design("bad name", "vegeta.dedalus.examples:Cube")


def test_revision_records_everything_and_runs_nothing(ws):
    d = ws.add_design("beam", "vegeta.dedalus.examples:CantileverBeam")
    r1 = d.new_revision(length=120.0, note="first")
    assert r1.id == "r1" and r1.params == {"length": 120.0, "width": 20.0, "height": 10.0}
    assert r1.changes == {"length": 120.0} and r1.parent is None
    rec = json.loads((r1.dir / "revision.json").read_text())
    assert len(rec["source"]["build_sha256"]) == 64 and rec["source"]["snapshot"] == "source/examples.py"
    assert rec["environment"]["packages"]["vegeta-cli"]
    assert not r1.is_generated and r1.evaluations() == []  # nothing ran
    with pytest.raises(ValueError, match="below minimum"):
        d.new_revision(length=-1.0)
    with pytest.raises(ValueError, match="unknown parameter"):
        d.new_revision(colour=3)


def test_revision_files_are_write_once(ws):
    r = ws.add_design("cube", "vegeta.dedalus.examples:Cube").new_revision()
    from vegeta.core._io import write_once

    with pytest.raises(ImmutableError):
        write_once(r.dir / "revision.json", {"tampered": True})


def test_branch_keeps_parent_and_changes(ws):
    r1 = ws.add_design("beam", "vegeta.dedalus.examples:CantileverBeam").new_revision(length=120.0)
    r2 = r1.branch(height=12.0, note="stiffer")
    assert r2.id == "r2" and r2.parent == "r1"
    assert r2.params["length"] == 120.0 and r2.changes == {"height": 12.0}
    assert ws.revision("r1").params["height"] == 10.0  # parent untouched


def test_labels_and_notes_keep_history(ws):
    r = ws.add_design("cube", "vegeta.dedalus.examples:Cube").new_revision()
    assert r.current_label == "unclassified"
    r.label("preferred", note="looks good")
    r.label("rejected", note="too heavy")
    r.note("check supplier")
    assert r.current_label == "rejected"
    assert [a["type"] for a in r.annotations()] == ["label", "label", "note"]
    with pytest.raises(ValueError):
        r.label("awesome")


def test_generate_once_and_detect_changed_source(ws, plate_file):
    d = ws.add_design("plate", f"{plate_file}:Plate")
    r1 = d.new_revision(width=40.0)
    res = r1.generate()
    assert res.ok and r1.is_generated and r1.step.name == "Plate.step"
    assert r1.geometry_summary()["metrics"]["volume"] == pytest.approx(40 * 10 * 2)
    again = r1.generate()
    assert not again.ok and "already generated" in again.messages[0]
    r2 = d.new_revision()
    plate_file.write_text(plate_file.read_text().replace("p[\"width\"], 10,", "p[\"width\"], 12,"))
    stale = r2.generate()
    assert not stale.ok and "source changed" in stale.messages[0] and not r2.is_generated
    r3 = r2.branch()  # picks up the new source
    assert r3.generate().ok and r3.geometry_summary()["metrics"]["volume"] == pytest.approx(30 * 12 * 2)


def test_failed_build_is_recorded_and_retry_keeps_history(ws):
    r = ws.add_design("bracket", "vegeta.dedalus.examples:Bracket").new_revision(hole_diameter=30.0)
    res = r.generate()
    assert not res.ok and not r.is_generated
    assert ws.status().rows[0]["CAD"] == core.FAILED
    assert any(a["type"] == "generate" and a["status"] == "failed" for a in r.annotations())


def test_no_automatic_chaining_and_not_run(ws):
    r1 = ws.add_design("cube", "vegeta.dedalus.examples:Cube").new_revision()
    refused = r1.run_fea("static", lambda rev: None)
    assert refused.status == "failed" and "generate() first" in refused.messages[0] and not refused.recorded
    assert r1.generate().ok
    assert r1.evaluations() == []  # generating ran no analysis
    row = ws.status().rows[0]
    assert row["CAD"].startswith("V=") and row["rev"] == "r1"


def test_status_table_columns_and_not_run(ws):
    d = ws.add_design("cube", "vegeta.dedalus.examples:Cube")
    d.new_revision()
    d.new_revision(size=10.0)
    t = ws.status()
    assert t.columns[:6] == ["rev", "design", "parent", "label", "changes", "CAD"]
    assert [r["CAD"] for r in t.rows] == [core.NOT_RUN, core.NOT_RUN]
    assert "NOT RUN" in str(t) and "<table>" in t._repr_html_()
    assert list(t.to_dataframe()["rev"]) == ["r1", "r2"]


def test_core_uses_only_public_tool_modules():
    import re
    from pathlib import Path

    src = Path(core.__file__).parent
    allowed = {"vegeta.dedalus", "vegeta.dedalus.loading", "vegeta.talos", "vegeta.aeromant", "vegeta.mellonia"}
    used = set()
    for p in src.glob("*.py"):
        used |= set(re.findall(r"from (vegeta\.[\w.]+) import", p.read_text()))
    assert used <= allowed, used - allowed
