import json
import os
import time

import pytest

from conftest import make_run
from vegeta import yanus
from vegeta.yanus import GateClosed, pins
from vegeta.yanus.stamps import read_stamp, write_stamp


def _maker(repo, schema=2):
    p = repo.root / "data" / "maker.json"
    p.write_text(json.dumps({"schema": schema, "calibration": {"k": 1}}))
    return p


def test_closed_lists_everything(repo):
    with pytest.raises(GateClosed) as e:
        yanus.enter("20_study")
    msg = str(e.value)
    assert "D1 sweep, A: not pinned" in msg and "wf D1 --variant A" in msg and "DEMO_SCREENING=1" in msg
    g = yanus.enter("20_study", raise_closed=False)
    roles = {r.input: (r.role, r.usable) for r in g.rows}
    assert roles["D1 sweep, B"] == ("optional", False)
    assert roles["maker.json (10_maker)"] == ("optional", False)


def test_pinned_run_opens_and_suggestion(repo):
    d = make_run(repo.root, "demo", "sweep", "r1")
    g = yanus.enter("20_study", raise_closed=False)
    row = next(r for r in g.rows if r.key == "demo/sweep@A")
    assert "1 usable run(s)" in row.status and row.fix.startswith("yn pin 20_study D1 --variant A workflows/runs/demo/sweep/r1")
    pins.set_pin(repo.root, "20_study", "demo/sweep@A", d)
    g = yanus.enter("20_study")
    assert g.ok and g.inputs["demo/sweep@A"]["counts"]["solved"] == 2
    assert json.loads(pins.pin_file(repo.root, "20_study").read_text()) == {"demo/sweep@A": "workflows/runs/demo/sweep/r1"}


@pytest.mark.parametrize("kw,expect", [
    (dict(state="running", pid=999999999), "its process is gone"),
    (dict(state="partial"), "run state 'partial'"),
    (dict(state="failed"), "failed — read run.log"),
    (dict(state="completed", summary=False), "no completed summary.json"),
    (dict(workflow="wfea"), "a wfea run, not wcfd"),
    (dict(variant="B"), "a run of B, not A"),
    (dict(design={"v": 2}), "incompatible: v differs"),
    (dict(schema=7), "schema 7"),
])
def test_run_statuses(repo, kw, expect):
    d = make_run(repo.root, "demo", "sweep", "r1", **kw)
    pins.set_pin(repo.root, "20_study", "demo/sweep@A", d)
    with pytest.raises(GateClosed, match=expect):
        yanus.enter("20_study")


def test_missing_legacy_pin_names_the_import(repo):
    pins.set_pin(repo.root, "20_study", "demo/sweep@A", repo.root / "workflows/runs/demo/sweep/legacy_2026_A")
    g = yanus.enter("20_study", raise_closed=False)
    row = next(r for r in g.rows if r.key == "demo/sweep@A")
    assert row.status.startswith("not found") and row.fix.startswith("wf import_old")


def test_screening_makes_takes_optional(repo, monkeypatch):
    monkeypatch.setenv("DEMO_SCREENING", "1")
    g = yanus.enter("20_study")
    assert g.ok and g.screening and all(r.role != "required" for r in g.rows)


def test_files_schema_stamp_and_staleness(repo):
    d = make_run(repo.root, "demo", "sweep", "r1")
    pins.set_pin(repo.root, "20_study", "demo/sweep@A", d)
    p = _maker(repo, schema=1)
    assert "schema 1 < 2" in next(r.status for r in yanus.enter("20_study").rows if r.kind == "file")
    p = _maker(repo, schema=2)
    row = next(r for r in yanus.enter("20_study").rows if r.kind == "file")
    assert row.usable and "unstamped" in row.status
    write_stamp(p, producer="10_maker", kind="notebook", schema=2, root=repo.root)
    row = next(r for r in yanus.enter("20_study").rows if r.kind == "file")
    assert row.usable and "unstamped" not in row.status
    p.write_text(json.dumps({"schema": 2, "calibration": {"k": 2}}))
    row = next(r for r in yanus.enter("20_study").rows if r.kind == "file")
    assert not row.usable and row.status.startswith("stale: modified since")
    write_stamp(p, producer="someone_else", kind="notebook", root=repo.root)
    row = next(r for r in yanus.enter("20_study").rows if r.kind == "file")
    assert "expected '10_maker'" in row.status and row.fix == "scripts/run_notebooks.sh 10_maker.ipynb"


def test_leave_checks_and_stamps(repo):
    old = repo.root / "data" / "maker.json"
    old.write_text("{}")
    past = time.time() - 3600
    os.utime(old, (past, past))
    g = yanus.enter("10_maker")
    with pytest.raises(GateClosed, match="not written by this execution"):
        g.leave()
    old.write_text(json.dumps({"schema": 2}))
    with pytest.raises(GateClosed, match="missing keys: calibration"):
        g.leave()
    old.write_text(json.dumps({"schema": 3, "calibration": 1}))
    with pytest.raises(GateClosed, match="schema 3 in the file, 2 declared"):
        g.leave()
    old.write_text(json.dumps({"schema": 2, "calibration": 1}))
    stamps = g.leave()
    st = read_stamp(old)
    assert st["producer"] == "10_maker" and st["schema"] == 2 and stamps[0]["sha256"] == st["sha256"]


def test_leave_records_inputs(repo):
    d = make_run(repo.root, "demo", "sweep", "r1")
    pins.set_pin(repo.root, "20_study", "demo/sweep@A", d)
    p = _maker(repo)
    write_stamp(p, producer="10_maker", kind="notebook", schema=2, root=repo.root)
    g = yanus.enter("20_study")
    (repo.root / "data" / "report.json").write_text("{}")
    g.leave()
    ins = read_stamp(repo.root / "data" / "report.json")["inputs"]
    assert {i["kind"] for i in ins} == {"run", "file"}
    # the maker is rewritten and re-stamped: the report is now stale for whoever takes it
    p.write_text(json.dumps({"schema": 2, "calibration": {"k": 9}}))
    write_stamp(p, producer="10_maker", kind="notebook", schema=2, root=repo.root)
    from vegeta.yanus.stamps import check_file
    c = check_file("data/report.json", root=repo.root)
    assert c["stale"] and "rewritten after it" in c["status"]


def test_chapter_gate_and_check_chapter(repo, capsys):
    with pytest.raises(SystemExit) as e:
        yanus.check_chapter("D2", variant="A", expected={"v": 1})
    assert e.value.code == 2
    assert "needs a completed compatible run" in capsys.readouterr().out
    res = yanus.check_chapter("D2", variant="A", expected={"v": 1}, ignore=True)
    assert len(res["missing"]) == 1
    make_run(repo.root, "demo", "sweep", "r1", design={"v": 2})          # incompatible: still missing
    assert yanus.check_chapter("D2", variant="A", expected={"v": 1}, ignore=True)["missing"]
    make_run(repo.root, "demo", "sweep", "r2", created="2026-10-17T11:00:00Z")
    assert yanus.check_chapter("D2", variant="A", expected={"v": 1})["missing"] == []
    out = yanus.check_chapter("D3", variant="A")
    assert out["needs"]["tool definitely-not-a-tool-xyz"] == "missing"
    assert "not found on this machine" in capsys.readouterr().out


def test_table_and_text(repo):
    g = yanus.enter("20_study", raise_closed=False)
    t = g.text()
    assert "✘ D1 sweep, A" in t and "→ wf D1 --variant A" in t
    pytest.importorskip("pandas")
    assert list(g.table().columns)[:3] == ["role", "selection", "status"]


def test_optional_give_and_debug(repo, monkeypatch):
    from vegeta.yanus import File
    repo.notebooks["10_maker"].gives.append(File("data/extra.csv", optional=True))
    p = repo.root / "data" / "maker.json"
    g = yanus.enter("10_maker")
    p.write_text(json.dumps({"schema": 2, "calibration": 1}))
    g.leave()                                              # extra.csv not written: optional, no error
    assert any("optional output" in r.status for r in g.left)
    p.unlink()
    monkeypatch.setenv("VEGETA_DEBUG", "1")
    g = yanus.enter("10_maker")
    assert g.debug and "DEBUG" in g.text()
    assert g.leave() == []                                 # DEBUG: a missing output is reported, not raised
    p.write_text(json.dumps({"schema": 2, "calibration": 1}))
    assert g.leave()[0]["debug"] is True


def test_debug_output_is_not_taken_by_a_real_run(repo, monkeypatch):
    d = make_run(repo.root, "demo", "sweep", "r1")
    pins.set_pin(repo.root, "20_study", "demo/sweep@A", d)
    p = _maker(repo)
    write_stamp(p, producer="10_maker", kind="notebook", schema=2, root=repo.root, debug=True)
    row = next(r for r in yanus.enter("20_study").rows if r.kind == "file")
    assert not row.usable and "in DEBUG" in row.status
    row = next(r for r in yanus.enter("20_study", debug=True).rows if r.kind == "file")
    assert row.usable and "made in DEBUG" in row.status
