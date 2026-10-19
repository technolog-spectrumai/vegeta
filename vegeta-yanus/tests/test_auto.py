"""Auto mode: a flagged notebook produces what it is missing (chapters in screening quality, prerequisites first,
producer notebooks) and its gate never closes."""
import json
import os

import pytest

from conftest import make_run
from vegeta import yanus
from vegeta.yanus import File, Flow, GateClosed, Package, Run, Tool


def fake_runner(root, log):
    """Stands in for subprocess: a chapter script call writes a completed run; a notebook call writes its give."""
    def run(argv, env, cwd):
        log.append(argv)
        assert env["YANUS_AUTO"] == "1"
        if "nbconvert" in argv:
            (root / "data" / "maker.json").write_text(json.dumps({"schema": 2, "calibration": 1}))
            return 0
        campaign = argv[argv.index("--chapter") + 1]
        variant = argv[argv.index("--variant") + 1] if "--variant" in argv else None
        assert argv[-2:] == ["--quality", "screening"], argv
        make_run(root, "demo", {"sweep": "sweep", "blown": "blown"}[campaign], f"auto_{campaign}_{variant}", variant=variant)
        return 0
    return run


def auto_flow(root):
    flow = Flow("t", root=root)
    b = flow.book("demo", "D", variants=("A", "B"), fingerprint=lambda v: {"v": 1}, compatible=lambda e, f, ch=None: {"level": "compatible", "reasons": []})
    b.kind("cfd", title="CFD", script="side_labs/demo/cfd.py", workflow="wcfd", needs=[Package("json")], screening=("--quality", "screening"))
    b.kind("fea", title="FEA", script="side_labs/demo/fea.py", workflow="wfea", needs=[Tool("definitely-not-a-tool-xyz")])
    b.chapter("sweep", kind="cfd", campaign="sweep")
    b.chapter("blown", kind="cfd", campaign="blown", takes=[Run("sweep")])
    b.chapter("frame", kind="fea", campaign="frame")
    flow.notebook("10_maker", gives=[File("data/maker.json", schema=2, keys=("calibration",))])
    flow.notebook("30_auto", needs=[Package("json")], auto=True,
                  takes=[Run("demo/blown", variant="A", required=True), Run("demo/frame", variant="A", required=True), File("10_maker:data/maker.json", min_schema=2)],
                  gives=[File("data/auto_report.json")])
    flow.notebook("31_manual", takes=[Run("demo/blown", variant="A", required=True)])
    return flow.validate()


@pytest.fixture
def auto_repo(tmp_path, monkeypatch):
    monkeypatch.delenv("VEGETA_WORKFLOW_RUNS", raising=False)
    monkeypatch.delenv("YANUS_AUTO", raising=False)
    monkeypatch.delenv("VEGETA_DEBUG", raising=False)
    (tmp_path / "data").mkdir()
    flow = auto_flow(tmp_path)
    yanus.use(flow)
    yield flow
    yanus.use(None)


def test_auto_runs_what_is_missing_in_order_and_never_raises(auto_repo, tmp_path):
    log, lines = [], []
    g = yanus.enter("30_auto", runner=fake_runner(tmp_path, log), log=lines.append)
    assert g.auto_mode and g.ok and g.screening
    labels = [s.label for s in g.auto]
    assert labels == ["D1 sweep --variant A", "D2 blown --variant A", "D3 frame --variant A", "notebook 10_maker"]
    assert [s.state for s in g.auto] == ["done", "done", "skipped", "done"]         # frame: its tool is not on this machine
    assert "definitely-not-a-tool-xyz" in g.auto[2].reason
    assert len(log) == 3 and "nbconvert" in log[2]
    assert g.inputs["demo/blown@A"]["counts"]["solved"] == 2                        # the run it made is read back
    assert g.inputs["demo/frame@A"] is None
    row = next(r for r in g.rows if r.input.startswith("D3 frame"))
    assert row.role == "optional" and row.status.startswith("NOT RUN — auto skipped")
    assert g.inputs["10_maker:data/maker.json"] is not None
    assert any("yanus auto" in l for l in lines)
    assert "AUTO" in g.text()


def test_auto_skips_prerequisites_that_already_ran(auto_repo, tmp_path):
    make_run(tmp_path, "demo", "sweep", "old_sweep", variant="A")
    log = []
    g = yanus.enter("30_auto", runner=fake_runner(tmp_path, log), log=lambda *_: None)
    sweep = next(s for s in g.auto if s.label.startswith("D1"))
    assert sweep.state == "skipped" and "already there" in sweep.reason
    assert [a for a in log if "--chapter" in a and a[a.index("--chapter") + 1] == "sweep"] == []


def test_auto_failed_step_is_not_run_not_an_error(auto_repo, tmp_path):
    def failing(argv, env, cwd):
        return 1
    g = yanus.enter("30_auto", runner=failing, log=lambda *_: None)
    assert g.ok and all(s.state in ("failed", "skipped") for s in g.auto)
    assert g.inputs["demo/blown@A"] is None
    assert "auto failed" in next(r for r in g.rows if r.input.startswith("D2")).status


def test_auto_off_by_default_and_env_switches(auto_repo, tmp_path, monkeypatch):
    with pytest.raises(GateClosed):
        yanus.enter("31_manual")
    log = []
    monkeypatch.setenv("YANUS_AUTO", "1")
    g = yanus.enter("31_manual", runner=fake_runner(tmp_path, log), log=lambda *_: None)
    assert g.ok and len(log) == 2
    monkeypatch.setenv("YANUS_AUTO", "0")
    with pytest.raises(GateClosed):
        yanus.enter("30_auto")                                  # the flag can be switched off from the shell
    monkeypatch.delenv("YANUS_AUTO")
    g = yanus.enter("30_auto", auto=False, raise_closed=False)
    assert not g.ok and not g.auto_mode


def test_auto_nothing_missing_runs_nothing(auto_repo, tmp_path):
    make_run(tmp_path, "demo", "blown", "b", variant="A")
    (tmp_path / "data" / "maker.json").write_text(json.dumps({"schema": 2, "calibration": 1}))
    log = []
    g = yanus.enter("31_manual", auto=True, runner=fake_runner(tmp_path, log), log=lambda *_: None)
    assert g.auto == [] and log == [] and g.ok and not g.screening      # nothing was missing: the takes stay required


def test_debug_runs_nothing(auto_repo, tmp_path, monkeypatch):
    monkeypatch.setenv("VEGETA_DEBUG", "1")
    log = []
    g = yanus.enter("30_auto", runner=fake_runner(tmp_path, log), log=lambda *_: None)
    assert g.ok and log == [] and g.auto == []


def test_minutes_cap_and_plan_text(auto_repo, tmp_path, monkeypatch):
    monkeypatch.setenv("YANUS_AUTO_MINUTES", "30")
    first = yanus.enter("30_auto", auto=False, raise_closed=False)
    steps = yanus.auto.plan(auto_repo, first)
    assert steps[0].argv[-4:] == ["--quality", "screening", "--max-minutes", "30"]
    assert "D1 sweep" in yanus.auto.report(steps)
