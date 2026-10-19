"""The run-directory contract on a fake solver: states, atomic summaries, cancellation, resume of finished cases."""
import json
import threading
import time

import pytest

import runkit


def fake_solver(d, cancel, n=10, status="solved"):
    log = d / "log.fake"
    for i in range(n):
        if cancel.is_set():
            return {"status": "cancelled", "process": "cancelled"}
        with log.open("a") as f:
            f.write(f"Time = {i}\n")
        time.sleep(0.005)
    return {"status": status, "process": "ok", "converged": status == "solved", "metrics": {"x": 1.0}}


def test_create_run_completed_summary_is_atomic_and_only_on_completion(tmp_path):
    run = runkit.Run.create(tmp_path / "r", workflow="demo", campaign="c", config={"q": 1}, planned_cases=["a", "b"], heartbeat_s=0.01, echo=False)
    assert json.loads((tmp_path / "r" / "status.json").read_text())["state"] == "prepared"
    m = json.loads((tmp_path / "r" / "manifest.json").read_text())
    assert m["schema"] == runkit.SCHEMA and m["config_hash"] == runkit.sha256_json({"q": 1}) and m["planned_cases"] == ["a", "b"]
    run.status("running")
    run.run_case("a", "k1", fake_solver)
    assert not (tmp_path / "r" / "summary.json").exists()                     # nothing published before the run finishes
    run.run_case("b", "k1", lambda d, c: fake_solver(d, c, status="FAILED"))
    code = run.finish({"note": "done"}, state="completed")
    run.close()
    assert code == 0
    s = json.loads((tmp_path / "r" / "summary.json").read_text())
    assert s["state"] == "completed" and s["cases"]["a"]["status"] == "solved" and s["cases"]["b"]["status"] == "FAILED"
    assert s["cases"]["a"]["converged"] is True and s["cases"]["a"]["process"] == "ok"
    assert not list((tmp_path / "r").glob("*.tmp*"))
    log = (tmp_path / "r" / "run.log").read_text()
    assert "case a: start" in log and "case a: solved" in log and "heartbeat: a alive" in log and "advancing" in log


def test_exception_in_a_case_is_a_failed_result_not_the_end(tmp_path):
    run = runkit.Run.create(tmp_path / "r", workflow="demo", campaign="c", config={}, planned_cases=["a"], heartbeat_s=10, echo=False)
    rec = run.run_case("a", "k", lambda d, c: 1 / 0)
    assert rec.status == "FAILED" and "ZeroDivisionError" in rec.note
    assert run.finish({}, state="completed") == runkit.EXIT["nothing solved"]
    run.close()


def test_cancel_leaves_partial_results_and_nonzero_exit(tmp_path):
    run = runkit.Run.create(tmp_path / "r", workflow="demo", campaign="c", config={}, planned_cases=["a", "b"], heartbeat_s=10, echo=False)
    run.status("running")
    run.run_case("a", "k", fake_solver)
    threading.Timer(0.02, run.cancel.set).start()
    rec = run.run_case("b", "k", lambda d, c: fake_solver(d, c, n=400))
    assert rec.status == "cancelled"
    code = run.finish({}, state="cancelled")
    run.close()
    assert code == 130
    assert (tmp_path / "r" / "summary.partial.json").exists() and not (tmp_path / "r" / "summary.json").exists()
    st = json.loads((tmp_path / "r" / "status.json").read_text())
    assert st["state"] == "cancelled" and st["cases"]["done"] == 2
    assert (tmp_path / "r" / "cases" / "a" / "case.json").exists()            # the finished case is kept


def test_resume_reuses_finished_cases_with_the_same_key_and_refuses_another_config(tmp_path):
    run = runkit.Run.create(tmp_path / "r", workflow="demo", campaign="c", config={"q": 1}, planned_cases=["a", "b", "c"], heartbeat_s=10, echo=False)
    run.run_case("a", "k1", fake_solver)
    run.run_case("b", "k1", lambda d, c: fake_solver(d, c, status="FAILED"))
    run.finish({}, state="cancelled")
    run.close()
    with pytest.raises(ValueError, match="different configuration"):
        runkit.Run.open(tmp_path / "r", config={"q": 2})
    run2 = runkit.Run.open(tmp_path / "r", config={"q": 1}, heartbeat_s=10, echo=False)
    calls = []

    def counting(d, c):
        calls.append(d.name)
        return fake_solver(d, c)
    assert run2.run_case("a", "k1", counting).reused                         # same key: reused
    assert run2.run_case("b", "k1", counting).reused                         # FAILED is a result: kept unless asked
    assert not run2.run_case("b", "k1", counting, rerun_failed=True).reused   # asked: run again
    assert not run2.run_case("a", "k2", counting).reused                     # another key (geometry or values changed): run again
    run2.run_case("c", "k1", counting)
    assert calls == ["b", "a", "c"]
    assert run2.finish({}, state="completed") == 0
    run2.close()
    assert (tmp_path / "r" / "summary.json").exists() and not (tmp_path / "r" / "summary.partial.json").exists()
    assert len(json.loads((tmp_path / "r" / "manifest.json").read_text())["resumed"]) == 1


def test_prepared_and_partial_states_never_publish_a_summary(tmp_path):
    run = runkit.Run.create(tmp_path / "r", workflow="demo", campaign="c", config={}, planned_cases=["a"], heartbeat_s=10, echo=False)
    run.run_case("a", "k", lambda d, c: {"status": "NOT RUN", "process": None, "note": "prepared only"})
    assert run.finish({}, state="prepared") == 0
    run.close()
    assert not (tmp_path / "r" / "summary.json").exists() and (tmp_path / "r" / "summary.partial.json").exists()
    d = runkit.load_run(tmp_path / "r")
    assert d["summary"] is None and d["partial"]["state"] == "prepared"
    assert runkit.EXIT["partial"] == 3


def test_refuses_a_non_empty_directory_and_lists_runs(tmp_path):
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "x").write_text("x")
    with pytest.raises(FileExistsError):
        runkit.Run.create(tmp_path / "r", workflow="demo", campaign="c", config={})
    run = runkit.Run.create(tmp_path / "s", workflow="demo", campaign="c", config={}, heartbeat_s=10, echo=False)
    run.close()
    runs = runkit.list_runs(tmp_path)
    assert [r["run_id"] for r in runs] == ["s"] and runs[0]["state"] == "prepared"


def test_heartbeat_distinguishes_alive_from_advancing(tmp_path):
    run = runkit.Run.create(tmp_path / "r", workflow="demo", campaign="c", config={}, heartbeat_s=10, echo=False)
    d = tmp_path / "r" / "cases" / "x"
    d.mkdir(parents=True)
    hb = runkit.Heartbeat(run, 10)
    hb._case, hb._dir, hb._t0, hb._quiet_since = "x", d, time.monotonic(), time.monotonic()
    (d / "log.simpleFoam").write_text("Time = 1\nSolving for Ux\nTime = 2\n")
    b1 = hb.beat()
    assert b1["advancing"] and b1["last_progress"] == "Time = 2" and b1["pid"] > 0
    b2 = hb.beat()
    assert not b2["advancing"]                                               # the log did not grow: alive, not advancing
    run.close()


def test_slug_is_a_safe_directory_name():
    assert runkit.slug("forward 3 s: α 54°, 1.5 m/s, T 4.36/4.89 N") == "forward_3_s_54_1.5_m_s_T_4.36_4.89_N"
    assert "/" not in runkit.slug("a/b") and runkit.slug("") == "case"


def test_progress_bars_are_silent_without_a_terminal_and_count_cases(tmp_path):
    import _kestrel as wk
    run = runkit.Run.create(tmp_path / "r", workflow="demo", campaign="c", config={}, planned_cases=["a", "b"], heartbeat_s=10, echo=False)
    bars = wk.Progress(run, 2)                                   # pytest's stderr is not a tty: disabled, every call a no-op
    assert not bars.enabled and bars.outer is None
    bars.solver("solver", 0.5, "iteration 200"); bars.start_case("a"); bars.case_done("solved")
    state = wk.drive(run, ["a", "b"], {"a": "k", "b": "k2"}, lambda n, d, c: {"status": "solved", "process": "ok"}, budget=runkit.Budget(), progress=bars)
    bars.close()
    assert state == "completed" and run.echo_fn is print
    run.finish({}, state="completed"); run.close()


def test_a_run_made_under_the_mocks_is_labelled_mock(tmp_path, monkeypatch):
    import mimesis
    monkeypatch.delenv("VEGETA_DEBUG", raising=False)
    real = runkit.Run.create(tmp_path / "real", workflow="w", campaign="c", config={}, echo=False)
    real.close()
    assert not mimesis.is_mock(runkit.read_json(tmp_path / "real" / "manifest.json"))
    monkeypatch.setenv("VEGETA_DEBUG", "1")
    fake = runkit.Run.create(tmp_path / "fake", workflow="w", campaign="c", config={}, echo=False)
    fake.close()
    assert mimesis.is_mock(runkit.read_json(tmp_path / "fake" / "manifest.json"))
    runkit.Run.open(tmp_path / "real", echo=False).close()                 # resumed under the mocks: mock from then on
    assert mimesis.is_mock(runkit.read_json(tmp_path / "real" / "manifest.json"))
