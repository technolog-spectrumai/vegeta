"""vegeta.chiron must not import the other tools, other vegeta subpackages or its parent namespace."""
import re
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "vegeta" / "chiron"
FORBIDDEN = re.compile(
    r"^\s*(from|import)\s+(vegeta\b|dedalus|talos|aeromant\b|mellonia|boreas|chronos|\.\.)", re.M
)
ROBOT_SPECIFIC = re.compile(r"cleopatra|myropod|persephone|robot.dog|quadruped|hexapod", re.I)


def test_no_forbidden_imports():
    offenders = [str(p.relative_to(SRC)) for p in SRC.rglob("*.py") if FORBIDDEN.search(p.read_text())]
    assert offenders == []


def test_core_is_robot_agnostic():
    core = ["__init__.py", "robot.py", "terrain.py", "servo.py", "lab.py", "gaits.py", "viz.py"]
    offenders = [n for n in core if (SRC / n).exists() and ROBOT_SPECIFIC.search((SRC / n).read_text())]
    assert offenders == []


def test_result_shape():
    from vegeta.chiron import Result

    d = Result(kind="chiron.test").to_dict()
    assert set(d) == {"kind", "status", "metrics", "artifacts", "messages", "duration_s", "execution", "metadata"}


def test_import_does_not_load_mujoco_or_pyvista():
    code = "import sys, vegeta.chiron; print('mujoco' in sys.modules, 'pyvista' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False"]
