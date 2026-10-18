import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).parent))
import json
from pathlib import Path

import pytest

from vegeta import yanus
from vegeta.yanus import Flow, Package, Run, Tool


def make_run(root: Path, chapter: str, run_id: str, *, book="demo", workflow="wcfd", variant="A", quality="fine", state="completed",
             summary=True, design=None, created="2026-10-17T10:00:00Z", pid=None, schema=1, mock=None, base=None):
    """A run directory in runkit's file contract (manifest, status, summary), under ``base`` or ``<root>/workflows/runs``."""
    d = Path(base or root / "workflows" / "runs") / book / chapter / run_id
    d.mkdir(parents=True)
    cfg = {"variant": variant}
    if quality is not None:
        cfg["quality"] = quality
    man = {"schema": schema, "run_id": run_id, "workflow": workflow, "created_at": created, "host": "h", "config": cfg,
           "design": {"v": 1} if design is None else design, "tool_versions": {"git": "abc123"}}
    if mock is not None:
        man["mock"] = mock
    (d / "manifest.json").write_text(json.dumps(man))
    (d / "status.json").write_text(json.dumps({"state": state, "pid": pid}))
    if summary is True:
        (d / "summary.json").write_text(json.dumps({"state": state, "counts": {"n": 2, "solved": 2}}))
    elif isinstance(summary, str):
        (d / "summary.json").write_text(summary)
    return d


def build_flow(root: Path) -> Flow:
    flow = Flow("t", root=root)
    b = flow.book("demo", "D", variants=("A", "B"),
                  compatible=lambda e, f, ch=None: ({"level": "compatible", "reasons": []} if (f or {}).get("v") == e.get("v") else
                                                    {"level": "warnings", "reasons": ["sources changed"]} if (f or {}).get("v") == e.get("warn") else
                                                    {"level": "incompatible", "reasons": ["v differs"]}))
    b.kind("cfd", title="CFD", script="workflows/demo/cfd.py", workflow="wcfd", needs=[Package("json")],
           options=[("--variant", ("A", "B"), "A"), ("--quality", ("screening", "fine"), "screening")])
    b.kind("fea", title="FEA", script="workflows/demo/fea.py", workflow="wfea", needs=[Tool("definitely-not-a-tool-xyz")])
    b.chapter("sweep", kind="cfd", campaign="sweep")
    b.chapter("blown", kind="cfd", campaign="blown", takes=[Run("sweep")])
    b.chapter("frame", kind="fea", campaign="frame")
    flow.add_import("import_old", title="old results", script="workflows/demo/import_old.py", marker="legacy_")
    return flow.validate()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.delenv("VEGETA_WORKFLOW_RUNS", raising=False)
    (tmp_path / "workflows").mkdir()
    (tmp_path / "workflows" / "flow.py").write_text("from vegeta.yanus import Flow\nflow = Flow('file')\n")
    flow = build_flow(tmp_path)
    yanus.use(flow)
    yield flow
    yanus.use(None)
