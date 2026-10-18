"""The fenix book: its chapters list without the interpreter (every case would be NOT RUN), run without one as a
failed run with NOT RUN cases, and — when the fenicsx interpreter exists — a screening steady run that completes and
agrees with the numpy solver."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "workflows" / "fenix" / "plate_thermal.py"
PY = sys.executable
ENV = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "vegeta-cli" / "src"), str(ROOT / "vegeta-yanus" / "src"), os.environ.get("PYTHONPATH", "")])}


def _run(args, env=ENV, timeout=600):
    return subprocess.run([PY, str(SCRIPT), *args], capture_output=True, text=True, env=env, cwd=ROOT, timeout=timeout)


def test_list_and_fingerprint():
    r = _run(["--chapter", "guard-ablation", "--variant", "1kW", "--list"])
    assert r.returncode == 0, r.stderr
    assert "5 case(s)" in r.stdout and "black_polycarbonate_10_mm" in r.stdout and "fenicsx interpreter" in r.stdout
    r = _run(["--chapter", "guard-steady", "--variant", "2kW", "--quality", "screening", "--list"])
    assert "11 case(s)" in r.stdout and "24×8" in r.stdout
    sys.path.insert(0, str(SCRIPT.parent)); sys.path.insert(0, str(ROOT / "workflows")); sys.path.insert(0, str(ROOT / "notebooks" / "designs"))
    import plate_thermal as pt
    fp = pt.fingerprint("1kW")
    assert fp["variant"] == "1kW" and 800 < fp["power_W"] < 1100 and len(fp["barriers"]) == 11
    assert pt.compatible(fp, fp)["level"] == "compatible"
    assert pt.compatible(fp, {**fp, "power_W": fp["power_W"] * 2})["level"] == "incompatible"


def test_without_interpreter_every_case_is_not_run(tmp_path):
    env = {**ENV, "VEGETA_FENICSX_PYTHON": str(tmp_path / "nowhere")}
    if (ROOT / ".fenicsx" / "env" / "bin" / "python").exists():
        pytest.skip("a .fenicsx env exists on this machine: the no-interpreter path is not reachable")
    r = _run(["--chapter", "guard-steady", "--variant", "1kW", "--quality", "screening", "--out", str(tmp_path / "run"), "--quiet"], env=env)
    assert r.returncode == 1
    summ = json.loads((tmp_path / "run" / "summary.partial.json").read_text())
    assert summ["state"] == "failed" and summ["counts"]["not_run"] == 11 and summ["counts"]["solved"] == 0


def _interpreter():
    sys.path.insert(0, str(SCRIPT.parent)); sys.path.insert(0, str(ROOT / "workflows")); sys.path.insert(0, str(ROOT / "notebooks" / "designs"))
    import plate_thermal as pt
    return pt.probe()


def test_screening_steady_run_agrees_with_numpy(tmp_path):
    if not _interpreter():
        pytest.skip("no fenicsx interpreter (VEGETA_FENICSX_PYTHON / .fenicsx/env): NOT RUN")
    r = _run(["--chapter", "guard-steady", "--variant", "1kW", "--quality", "screening", "--cases", "mild_steel_1p5,aluminium_3", "--out", str(tmp_path / "run"), "--quiet"], timeout=900)
    assert r.returncode == 0, r.stderr[-800:]
    summ = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summ["state"] == "completed" and summ["counts"]["solved"] == 2
    steel = summ["cases"]["mild_steel_1p5_mm"]
    assert steel["status"] == "solved" and 1150 < steel["T_max"] < 1320 and not steel["reaches_damage"]      # numpy: 1222 K
    assert (tmp_path / "run" / "cases" / "mild_steel_1p5_mm" / "result.json").exists()
