import json
import sys
import threading
import time

import numpy as np
import pytest
from _helpers import DESIGN, TWO_PART

from vegeta.fidia import sandbox as sandbox_mod
from vegeta.fidia._process import clean_environment, run_walled
from vegeta.fidia.mesh import load_parts
from vegeta.fidia.sandbox import Sandbox, execute

pytestmark = pytest.mark.slow

LOOP = TWO_PART.replace('        base = ', '        while True:\n            pass\n        base = ')


def test_two_part_assembly(tmp_path):
    r = execute(TWO_PART, tmp_path)
    assert r.ok, r.error
    assert [p["name"] for p in r.parts] == ["base", "post"]
    assert r.parts[0]["color"][:3] == pytest.approx([0.8, 0.1, 0.1], abs=1e-3)
    assert r.parts[1]["color_given"] is True
    post = r.parts[1]["brep"]
    assert post["valid"] and post["n_solids"] == 1
    assert post["bbox_min"][0] == pytest.approx(20, abs=0.2) and post["bbox_min"][2] == pytest.approx(10, abs=0.2)
    assert r.bbox_max[2] == pytest.approx(60, abs=0.01)
    assert r.contacts["pairs"] and r.contacts["pairs"][0][:2] == [0, 1]
    parts = load_parts(tmp_path)
    assert parts[1].vertices[:, 0].min() == pytest.approx(20, abs=1e-6)
    for f in ("design.py", "execution.json", "runner.log", "result.json", "parts.npz", "model.step"):
        assert (tmp_path / f).is_file(), f
    assert json.loads((tmp_path / "execution.json").read_text())["status"] == "ok"


def test_parameters_and_gap(tmp_path):
    r = execute(TWO_PART, tmp_path, parameters={"gap": 15.0})
    assert r.ok and r.parameters["gap"] == 15.0
    assert r.contacts["pairs"] == []


def test_plain_workplane_is_one_part(tmp_path):
    r = execute(DESIGN, tmp_path)
    assert r.ok and len(r.parts) == 1
    assert r.parts[0]["name"] == "body" and r.parts[0]["color_given"] is False


def test_rejected_code_never_runs(tmp_path):
    r = execute("import os\n" + DESIGN, tmp_path)
    assert r.status == "rejected" and "os" in r.error
    assert not (tmp_path / "runner.log").exists()


def test_build_error_has_traceback(tmp_path):
    r = execute(TWO_PART.replace("        base = ", '        raise ValueError("boom")\n        base = '), tmp_path)
    assert r.status == "build_error"
    assert "boom" in r.error and "design.py" in r.traceback


def test_timeout(tmp_path):
    t0 = time.monotonic()
    r = execute(LOOP, tmp_path, sandbox=Sandbox(timeout_s=4))
    assert r.status == "timeout"
    assert time.monotonic() - t0 < 12


def test_cancel(tmp_path):
    ev = threading.Event()
    threading.Timer(3.0, ev.set).start()
    t0 = time.monotonic()
    r = execute(LOOP, tmp_path, cancel=ev, sandbox=Sandbox(timeout_s=60))
    assert r.status == "cancelled"
    assert time.monotonic() - t0 < 3.0 + 2.0


def test_memory_limit(tmp_path):
    src = TWO_PART.replace("import cadquery as cq", "import cadquery as cq\nimport numpy as np").replace(
        "        base = ", "        big = np.ones(3_000_000_000)\n        base = ")
    r = execute(src, tmp_path, sandbox=Sandbox(memory_mb=3000, timeout_s=60))
    assert r.status == "resource_limit", r.error


def test_clean_environment_has_no_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    env = clean_environment(tmp_path, {"MY_PASSWORD": "x", "FOO": "1"})
    assert "ANTHROPIC_API_KEY" not in env and "GITHUB_TOKEN" not in env and "MY_PASSWORD" not in env
    assert env["FOO"] == "1" and env["HOME"] == str(tmp_path)
    rec = run_walled([sys.executable, "-I", "-c", "import os; print(os.environ.get('ANTHROPIC_API_KEY'))"], tmp_path, env=env)
    assert rec.ok and rec.stdout.strip() == "None"


def test_execute_uses_the_clean_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    seen = {}
    real = sandbox_mod.run_walled

    def spy(command, cwd, *, env, **kw):
        seen["env"] = dict(env)
        return real(command, cwd, env=env, **kw)
    monkeypatch.setattr(sandbox_mod, "run_walled", spy)
    assert execute(DESIGN, tmp_path).ok
    assert "ANTHROPIC_API_KEY" not in seen["env"]
    assert all(not any(w in k for w in ("KEY", "TOKEN", "SECRET")) for k in seen["env"])


def test_runner_output_is_not_pickled(tmp_path):
    assert execute(DESIGN, tmp_path).ok
    with np.load(tmp_path / "parts.npz", allow_pickle=False) as z:
        assert set(z.files) == {"v0", "t0"}
