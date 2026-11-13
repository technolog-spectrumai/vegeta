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
    assert argv == ["labs/aero/campaigns/side_labs/kestrel/airframe_cfd.py", "--chapter", "stall", "--variant", "4Q-1", "--quality", "fine", "--prepare-only", "--no-cache", "--ignore-prerequisites",
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
    out = subprocess.run([PY, str(ROOT / "labs" / "aero" / "campaigns" / "side_labs" / "cli.py"), "batch", "--example"], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "chapter: K1" in out.stdout
    f = tmp_path / "batch.yaml"
    f.write_text(out.stdout)
    out = subprocess.run([PY, str(ROOT / "labs" / "aero" / "campaigns" / "side_labs" / "cli.py"), str(f), "--dry-run"], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "airframe_cfd.py --chapter sweep" in out.stdout and "4 step(s)" in out.stdout


def test_estimate_from_history_rule_of_thumb_and_expected_minutes(runs_tmp, tmp_path):
    """The time per case comes from finished runs of the chapter (same quality first), else the kind's rule of thumb;
    expected_minutes in the file wins; the dry run prints the total and the finishing time."""
    import json as _json
    d = runs_tmp / "kestrel" / "airframe-sweep" / "old_run"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(_json.dumps({"schema": 1, "run_id": "old_run", "workflow": "airframe_cfd", "created_at": "2026-10-17T09:00:00Z",
                                                  "config": {"variant": "4Q-2", "quality": "screening"}, "planned_cases": ["a", "b", "c"]}))
    (d / "status.json").write_text(_json.dumps({"state": "completed"}))
    (d / "summary.json").write_text(_json.dumps({"cases": {"a": {"status": "solved", "duration_s": 100.0}, "b": {"status": "solved", "duration_s": 300.0}, "c": {"status": "FAILED"}}}))
    b = batch.loads("name: e\nsteps:\n  - chapter: K1\n    variant: 4Q-2\n  - chapter: K2\n    variant: 4Q-2\n    options: {quality: fine}\n  - chapter: K9\n    variant: 4Q-2\n    expected_minutes: 12\n")
    est = batch.estimate(b, count=False)
    assert est[0]["cases"] == 3 and abs(est[0]["case_s"] - 200.0) < 1e-9 and "run(s) of the chapter" in est[0]["source"]
    assert est[1]["case_s"] == batch.ESTIMATE_S[("cfd", "fine")] and est[1]["cases"] == 1          # no run of K2: the rule of thumb, 1 case assumed without --list
    assert est[2]["seconds"] == 12 * 60 and "expected_minutes" in est[2]["source"]
    lines = []
    rep = batch.run(b, dry_run=True, count=False, log=lines.append)
    assert rep["estimate"]["cases"] == 5 and any(l.startswith("estimated total:") and "would finish about" in l for l in lines)
    assert batch.from_dict(batch._mini_yaml(batch.dumps(b))).steps[2].expected_minutes == 12


def test_progress_and_eta_written_to_status(runs_tmp, tmp_path):
    """While a step runs, the batch's status.json carries the cases done over every case of the batch, the fraction and
    the remaining time; a finished batch reports the time against the estimate."""
    import json as _json
    seen = []

    def launch(cmd, cwd, f):
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True)
        (out / "manifest.json").write_text(_json.dumps({"schema": 1, "run_id": out.name, "workflow": "airframe_cfd", "created_at": "2026-10-17T10:00:00Z", "config": {"variant": "4Q-2"}, "planned_cases": ["x", "y"]}))
        (out / "status.json").write_text(_json.dumps({"state": "completed", "cases": {"planned": 2, "done": 2}}))
        (out / "summary.json").write_text(_json.dumps({"state": "completed", "counts": {"n": 2, "solved": 2}}))
        seen.append(_json.loads((tmp_path / "b" / "status.json").read_text()))
        return 0
    b = batch.loads("name: p\nsteps:\n  - chapter: K1\n    variant: 4Q-2\n    expected_minutes: 10\n  - chapter: K2\n    variant: 4Q-2\n    expected_minutes: 10\n")
    lines = []
    rep = batch.run(b, launch=launch, log=lines.append, folder=tmp_path / "b", count=False)
    assert rep["exit_code"] == 0 and rep["estimate"]["seconds"] == 1200 and "actual_seconds" in rep["estimate"]
    st0, st1 = seen
    assert st0["cases"]["planned"] == 2 and st0["cases"]["done"] == 0 and st0["eta_s"] == 1200 and st0["eta_at"]
    assert st1["cases"]["done"] == 2 and st1["cases"]["planned"] == 3 and abs(st1["progress"] - 2 / 3) < 1e-9 and 0 < st1["eta_s"] <= 600   # step 1 turned out to be 2 cases: the total followed
    final = _json.loads((tmp_path / "b" / "status.json").read_text())
    assert final["state"] == "completed" and final["cases"]["done"] == 4 and final["eta_at"] is None
    assert any(l.startswith("estimate:") for l in lines) and any(l.startswith("time:") for l in lines)
