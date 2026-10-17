"""batch.yaml: read, written, validated, run in order with a stand-in launcher, and the closing lines name the outputs."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

import batch
import library

ROOT = library.ROOT
PY = sys.executable


def test_example_round_trip_both_readers():
    b = batch.loads(batch.EXAMPLE)
    assert [s.chapter for s in b.steps] == ["K1", "K2", "K3", "K11"] and b.steps[0].options == {"quality": "fine", "processors": 4}
    text = batch.dumps(b)
    assert batch.from_dict(batch._mini_yaml(text)) == b
    yaml = pytest.importorskip("yaml")
    assert batch.from_dict(yaml.safe_load(text)) == b
    assert batch.validate(b) == []


def test_bad_files_are_named():
    with pytest.raises(ValueError):
        batch.loads("name: x\n")
    with pytest.raises(ValueError):
        batch.loads("steps:\n  - variant: 4Q-2\n")
    b = batch.loads("steps:\n  - chapter: K99\n  - chapter: K1\n    variant: 9Q\n  - chapter: K1\n    options: {nonsense: 1}\n")
    probs = batch.validate(b)
    assert len(probs) == 3 and "K99" in probs[0] and "9Q" in probs[1] and "--nonsense" in probs[2]


def test_argv_and_dry_run(capsys):
    b = batch.loads("name: t\nsteps:\n  - chapter: k2\n    variant: 4Q-1\n    options: {quality: fine}\n    mode: prepare\n    cache: false\n    limits: --max-cases 2\n    ignore_prerequisites: true\n")
    argv = batch.argv_for(b.steps[0], Path("/tmp/x"))
    assert argv == ["workflows/kestrel/airframe_cfd.py", "--chapter", "stall", "--variant", "4Q-1", "--quality", "fine", "--prepare-only", "--no-cache", "--ignore-prerequisites",
                    "--max-cases", "2", "--out", "/tmp/x"]
    rep = batch.run(b, dry_run=True, log=lambda *_: None)
    assert rep["exit_code"] == 0 and rep["steps"][0]["chapter"] == "K2"


def fake_launch(outcomes: dict):
    """Writes a run directory the way a script would, then returns the exit code of the step."""
    def launch(cmd, cwd, f):
        out = Path(cmd[cmd.index("--out") + 1])
        campaign = cmd[cmd.index("--chapter") + 1]
        rc = outcomes.get(campaign, 0)
        out.mkdir(parents=True)
        (out / "manifest.json").write_text(json.dumps({"schema": 1, "run_id": out.name, "workflow": "airframe_cfd", "created_at": "2026-10-17T10:00:00Z", "config": {"variant": "4Q-2", "campaign": campaign}}))
        (out / "status.json").write_text(json.dumps({"state": {0: "completed", 3: "partial", 1: "failed"}[rc]}))
        if rc == 0:
            (out / "summary.json").write_text(json.dumps({"state": "completed", "counts": {"n": 1, "solved": 1}}))
        elif rc == 3:
            (out / "summary.partial.json").write_text("{}")
        f.write(b"fake solver output\n")
        return rc
    return launch


@pytest.fixture
def runs_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(library, "needs_status", lambda book, chapter: {"ok": True, "missing": [], "fixes": []})
    return tmp_path / "runs"


def test_runs_in_order_stops_on_failure_and_names_outputs(runs_tmp, tmp_path):
    b = batch.loads("name: night\nsteps:\n  - chapter: K1\n    variant: 4Q-2\n  - chapter: K2\n    variant: 4Q-2\n  - chapter: K3\n    variant: 4Q-2\n  - chapter: K6\n    variant: 4Q-2\n")
    lines = []
    rep = batch.run(b, launch=fake_launch({"stall": 1}), log=lines.append, folder=tmp_path / "b")
    states = [r["state"] for r in rep["steps"]]
    assert states == ["completed", "failed", "not started (an earlier step failed)", "not started (an earlier step failed)"]
    assert rep["exit_code"] == 3
    assert rep["steps"][0]["summary"].endswith("summary.json") and "b_s1_4Q-2" in rep["steps"][0]["run_dir"]
    assert (tmp_path / "b" / "batch.json").exists() and json.loads((tmp_path / "b" / "status.json").read_text())["state"] == "partial"
    assert (tmp_path / "b" / "step_01_K1.out").read_text() == "fake solver output\n"
    text = "\n".join(lines)
    assert "results: " in text and "batch.json" in text and "✔  1. K1" in text and "✘  2. K2" in text


def test_prerequisite_is_checked_per_step_and_go_on_mode(runs_tmp, tmp_path):
    b = batch.loads("name: go\nstop_on_failure: false\nsteps:\n  - chapter: K3\n    variant: 4Q-2\n  - chapter: K1\n    variant: 4Q-2\n  - chapter: K3\n    variant: 4Q-2\n")
    rep = batch.run(b, launch=fake_launch({}), log=lambda *_: None, folder=tmp_path / "b")
    states = [r["state"] for r in rep["steps"]]
    assert states[0].startswith("did not start") and states[1] == "completed" and states[2] == "completed"     # K3 before K1 fails, after K1 runs
    assert rep["exit_code"] == 3
    rep = batch.run(b, launch=fake_launch({}), log=lambda *_: None, folder=tmp_path / "c", start=2)
    assert [r["state"] for r in rep["steps"]] == ["skipped", "completed", "completed"] and rep["exit_code"] == 0


def test_cli_batch_example_and_dry_run(tmp_path):
    out = subprocess.run([PY, str(ROOT / "workflows" / "cli.py"), "batch", "--example"], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "chapter: K1" in out.stdout
    f = tmp_path / "batch.yaml"
    f.write_text(out.stdout)
    out = subprocess.run([PY, str(ROOT / "workflows" / "cli.py"), str(f), "--dry-run"], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "airframe_cfd.py --chapter sweep" in out.stdout and "4 step(s)" in out.stdout
