import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).parent))
import json
import os
from pathlib import Path

import pytest

from vegeta import yanus
from vegeta.yanus import File, Flow, Package, Run, Tool


def make_run(root: Path, book: str, chapter: str, run_id: str, *, workflow="wcfd", variant="A", state="completed", summary=True,
             design=None, created="2026-10-17T10:00:00Z", pid=None, schema=1):
    d = root / "side_labs" / "runs" / book / chapter / run_id
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"schema": schema, "run_id": run_id, "workflow": workflow, "created_at": created, "host": "h",
                                                 "config": {"variant": variant}, "design": design or {"v": 1}}))
    (d / "status.json").write_text(json.dumps({"state": state, "pid": pid}))
    if summary:
        (d / "summary.json").write_text(json.dumps({"state": state, "counts": {"n": 2, "solved": 2}}))
    return d


def build_flow(root: Path) -> Flow:
    flow = Flow("t", root=root)
    b = flow.book("demo", "D", variants=("A", "B"), fingerprint=lambda v: {"v": 1},
                  compatible=lambda e, f, ch=None: {"level": "compatible" if (f or {}).get("v") == e.get("v") else "incompatible",
                                                     "reasons": [] if (f or {}).get("v") == e.get("v") else ["v differs"]})
    b.kind("cfd", title="CFD", script="side_labs/demo/cfd.py", workflow="wcfd", needs=[Package("json")])
    b.kind("fea", title="FEA", script="side_labs/demo/fea.py", workflow="wfea", needs=[Tool("definitely-not-a-tool-xyz")])
    b.chapter("sweep", kind="cfd", campaign="sweep")
    b.chapter("blown", kind="cfd", campaign="blown", takes=[Run("sweep")])
    b.chapter("frame", kind="fea", campaign="frame")
    flow.add_import("import_old", title="old results", script="side_labs/demo/import_old.py", marker="legacy_")
    flow.notebook("10_maker", gives=[File("data/maker.json", schema=2, keys=("calibration",))])
    flow.notebook("20_study", needs=[Package("json")],
                  takes=[Run("demo/sweep", variant="A", required=True), Run("demo/sweep", variant="B"), File("10_maker:data/maker.json", min_schema=2)],
                  gives=[File("data/report.json")], screening_env="DEMO_SCREENING")
    return flow.validate()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.delenv("VEGETA_WORKFLOW_RUNS", raising=False)
    monkeypatch.delenv("DEMO_SCREENING", raising=False)
    (tmp_path / "data").mkdir()
    (tmp_path / "side_labs").mkdir()
    (tmp_path / "side_labs" / "flow.py").write_text("from vegeta.yanus import Flow\nflow = Flow('file')\nflow.notebook('01_x')\n")
    flow = build_flow(tmp_path)
    yanus.use(flow)
    yield flow
    yanus.use(None)
