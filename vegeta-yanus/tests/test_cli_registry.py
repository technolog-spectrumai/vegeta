import json

import pytest

from conftest import build_flow, make_run
from vegeta import yanus
from vegeta.yanus import FlowError, Run, cli, scaffold


def test_find_cli(repo, capsys):
    make_run(repo.root, "sweep", "r1")
    make_run(repo.root, "sweep", "r2", state="failed", summary=False, created="2026-10-18T00:00:00Z")
    assert cli.main(["find", "D1", "--variant", "A"]) == 0
    out = capsys.readouterr().out
    assert "latest run r1" in out and "r2: run state 'failed'" in out and "unverified" in out
    assert cli.main(["find", "D1", "--variant", "A", "--quality", "screening"]) == 2
    assert "produce one with:  wf D1 --variant A --quality screening" in capsys.readouterr().out
    assert cli.main(["find", "D1", "--variant", "A", "--run", "workflows/runs/demo/sweep/r2", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["request"]["mode"] == "explicit"


def test_doctor_validate(repo, capsys):
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "✘ tool definitely-not-a-tool-xyz" in out and "this machine can run: D1, D2" in out
    assert cli.main(["validate"]) == 0 and "3 chapter(s)" in capsys.readouterr().out


def test_registry_lookups_and_errors(tmp_path):
    f = build_flow(tmp_path)
    assert f.chapter("D2").prerequisites == ("sweep",) and f.chapter("demo 3").name == "frame" and f.chapter(f.chapter("D1")).code == "D1"
    with pytest.raises(KeyError, match="D1–D3"):
        f.chapter("D7")
    f.books["demo"].chapter("loop", kind="cfd", campaign="loop", takes=[Run("loop")])
    with pytest.raises(FlowError, match="cycle"):
        f.validate()
    with pytest.raises(FlowError, match="declared twice"):
        f.books["demo"].chapter("loop", kind="cfd", campaign="x")


def test_check_chapter_reports_needs_and_prerequisites(repo, capsys):
    with pytest.raises(SystemExit) as exc:
        yanus.check_chapter("D2", variant="A")
    assert exc.value.code == 2 and "D1 (sweep) has no usable run for A" in capsys.readouterr().out
    g = yanus.check_chapter("D2", variant="A", ignore=True)
    assert g["missing"] and g["needs"] == {"package json": g["needs"]["package json"]}
    make_run(repo.root, "sweep", "r1")
    assert yanus.check_chapter("D2", variant="A", expected={"v": 1})["missing"] == []
    assert yanus.check_chapter("D2", variant="A", expected={"v": 9}, ignore=True)["missing"]           # incompatible prerequisite


def test_new_chapter_scaffold(repo, tmp_path):
    out = scaffold.new_chapter(repo.root, "demo/thermal", kind="heat", new_kind=True, takes=("sweep",))
    assert (repo.root / "workflows/demo/heat.py").exists() and 'chapter("thermal", kind="heat"' in out[-1].read_text()
    with pytest.raises(FileExistsError):
        scaffold.new_chapter(repo.root, "demo/thermal", kind="heat")
