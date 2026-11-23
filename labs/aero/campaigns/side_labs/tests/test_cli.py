"""The workflows CLI: it lists, explains and inspects without running a solver."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
CLI = ROOT / "labs" / "aero" / "campaigns" / "side_labs" / "cli.py"


def run(*args):
    return subprocess.run([sys.executable, str(CLI), *args], capture_output=True, text=True, timeout=300, cwd=ROOT)


def test_list_and_info():
    out = run("list")
    assert out.returncode == 0 and "kestrel  (K)" in out.stdout and "K1   airframe-sweep" in out.stdout and "K9   frame" in out.stdout and "[after K1]" in out.stdout and "15–30 min" in out.stdout
    out = run("info", "kestrel")
    assert out.returncode == 0 and "K10  joint" in out.stdout and "FEA" in out.stdout
    out = run("info", "k10")
    assert out.returncode == 0 and "K10  joint" in out.stdout and "prerequisites (same book): K9" in out.stdout and "CalculiX" in out.stdout
    out = run("info", "K2", "--variant", "4Q-1")
    assert out.returncode == 0 and "5 case(s)" in out.stdout and "--list" in out.stdout


def test_runs_status_and_stop_on_a_fake_run(tmp_path, monkeypatch):
    import runkit
    r = runkit.Run.create(tmp_path / "kestrel" / "airframe-stall" / "demo_run", workflow="airframe_cfd", campaign="stall", config={"variant": "4Q-2", "chapter": "airframe-stall", "quality": "fine"}, planned_cases=["a"], heartbeat_s=10, echo=False)
    r.manifest.update(book="kestrel", chapter="airframe-stall", kind="cfd"); runkit.write_json_atomic(r.root / "manifest.json", r.manifest)
    r.run_case("a", "k", lambda d, c: {"status": "solved", "process": "ok", "converged": True})
    r.finish({"counts": {"solved": 1}}, state="completed")
    r.close()
    env = {"VEGETA_WORKFLOW_RUNS": str(tmp_path)}
    import os
    out = subprocess.run([sys.executable, str(CLI), "runs"], capture_output=True, text=True, cwd=ROOT, env={**os.environ, **env}, timeout=120)
    assert out.returncode == 0 and "demo_run" in out.stdout and "completed" in out.stdout and "K2    kestrel / airframe-stall" in out.stdout
    out = subprocess.run([sys.executable, str(CLI), "status", "demo_run"], capture_output=True, text=True, cwd=ROOT, env={**os.environ, **env}, timeout=120)
    assert out.returncode == 0 and "state completed" in out.stdout and "its results are in the run directory" in out.stdout
    out = subprocess.run([sys.executable, str(CLI), "stop", "demo_run"], capture_output=True, text=True, cwd=ROOT, env={**os.environ, **env}, timeout=120)
    assert out.returncode == 1 and "no running process" in out.stdout


def test_cache_commands(tmp_path):
    import os
    import runkit
    env = {**os.environ, "VEGETA_WORKFLOW_CACHE": str(tmp_path / "cache"), "VEGETA_WORKFLOW_RUNS": str(tmp_path / "runs")}
    out = subprocess.run([sys.executable, str(CLI), "cache"], capture_output=True, text=True, cwd=ROOT, env=env, timeout=120)
    assert out.returncode == 0 and "empty" in out.stdout
    runkit.Cache("kestrel", base=tmp_path / "cache").put("k", {"status": "solved", "metrics": {}})
    out = subprocess.run([sys.executable, str(CLI), "cache"], capture_output=True, text=True, cwd=ROOT, env=env, timeout=120)
    assert "kestrel" in out.stdout and "1 case(s)" in out.stdout
    out = subprocess.run([sys.executable, str(CLI), "clean-cache", "-y"], capture_output=True, text=True, cwd=ROOT, env=env, timeout=120)
    assert out.returncode == 0 and "removed 1" in out.stdout and not list((tmp_path / "cache").glob("*/*.json"))


def test_run_by_code_shortcut_lists_cases():
    out = run("k2", "--variant", "4Q-1", "--list")
    assert out.returncode == 0 and "5 case(s)" in out.stdout and "--chapter stall" in out.stdout
    out = run("K99")
    assert out.returncode != 0 and "K1–K12" in (out.stdout + out.stderr)
