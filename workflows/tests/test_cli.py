"""The workflows CLI: it lists, explains and inspects without running a solver."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "workflows" / "cli.py"


def run(*args):
    return subprocess.run([sys.executable, str(CLI), *args], capture_output=True, text=True, timeout=300, cwd=ROOT)


def test_list_and_info():
    out = run("list")
    assert out.returncode == 0 and "kestrel_airframe" in out.stdout and "kestrel_frame" in out.stdout and "stall" in out.stdout and "[after: sweep]" in out.stdout and "15–30 min" in out.stdout
    out = run("info", "kestrel_frame")
    assert out.returncode == 0 and "joint" in out.stdout and "CalculiX" in out.stdout
    out = run("info", "kestrel_airframe", "stall", "--variant", "4Q-1")
    assert out.returncode == 0 and "5 case(s)" in out.stdout and "--list" in out.stdout


def test_runs_status_and_notebook_line_on_a_fake_run(tmp_path, monkeypatch):
    import runkit
    r = runkit.Run.create(tmp_path / "kestrel_airframe" / "stall" / "demo_run", workflow="airframe_cfd", campaign="stall", config={"variant": "4Q-2", "chapter": "stall", "quality": "fine"}, planned_cases=["a"], heartbeat_s=10, echo=False)
    r.manifest.update(book="kestrel_airframe", chapter="stall"); runkit.write_json_atomic(r.root / "manifest.json", r.manifest)
    r.run_case("a", "k", lambda d, c: {"status": "solved", "process": "ok", "converged": True})
    r.finish({"counts": {"solved": 1}}, state="completed")
    r.close()
    env = {"VEGETA_WORKFLOW_RUNS": str(tmp_path)}
    import os
    out = subprocess.run([sys.executable, str(CLI), "runs"], capture_output=True, text=True, cwd=ROOT, env={**os.environ, **env}, timeout=120)
    assert out.returncode == 0 and "demo_run" in out.stdout and "completed" in out.stdout and "kestrel_airframe / stall" in out.stdout
    out = subprocess.run([sys.executable, str(CLI), "status", "demo_run"], capture_output=True, text=True, cwd=ROOT, env={**os.environ, **env}, timeout=120)
    assert out.returncode == 0 and "state completed" in out.stdout and "use it in the notebook" in out.stdout
    out = subprocess.run([sys.executable, str(CLI), "notebook", "demo_run"], capture_output=True, text=True, cwd=ROOT, env={**os.environ, **env}, timeout=120)
    assert out.returncode == 0 and 'RUNS["airframe_cfd"]["4Q-2"]["stall"]' in out.stdout
    out = subprocess.run([sys.executable, str(CLI), "stop", "demo_run"], capture_output=True, text=True, cwd=ROOT, env={**os.environ, **env}, timeout=120)
    assert out.returncode == 1 and "no running process" in out.stdout


def test_cache_commands(tmp_path):
    import os
    import runkit
    env = {**os.environ, "VEGETA_WORKFLOW_CACHE": str(tmp_path / "cache"), "VEGETA_WORKFLOW_RUNS": str(tmp_path / "runs")}
    out = subprocess.run([sys.executable, str(CLI), "cache"], capture_output=True, text=True, cwd=ROOT, env=env, timeout=120)
    assert out.returncode == 0 and "empty" in out.stdout
    runkit.Cache("kestrel_airframe", base=tmp_path / "cache").put("k", {"status": "solved", "metrics": {}})
    out = subprocess.run([sys.executable, str(CLI), "cache"], capture_output=True, text=True, cwd=ROOT, env=env, timeout=120)
    assert "kestrel_airframe" in out.stdout and "1 case(s)" in out.stdout
    out = subprocess.run([sys.executable, str(CLI), "clean-cache", "-y"], capture_output=True, text=True, cwd=ROOT, env=env, timeout=120)
    assert out.returncode == 0 and "removed 1" in out.stdout and not list((tmp_path / "cache").glob("*/*.json"))
