"""vegeta.zephyr must not import the other tools, other vegeta subpackages or its parent namespace."""
import re
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "vegeta" / "zephyr"
FORBIDDEN = re.compile(
    r"^\s*(from|import)\s+(vegeta\b|dedalus|talos|aeromant\b|mellonia|boreas|chronos|chiron|atropos|\.\.)", re.M
)


def test_no_forbidden_imports():
    offenders = [str(p.relative_to(SRC)) for p in SRC.rglob("*.py") if FORBIDDEN.search(p.read_text())]
    assert offenders == []


def test_result_shape():
    from vegeta.zephyr import Result

    d = Result(kind="zephyr.test").to_dict()
    assert set(d) == {"kind", "status", "metrics", "artifacts", "messages", "duration_s", "execution", "metadata"}


def test_import_loads_neither_matplotlib_nor_neuralfoil():
    code = "import sys, vegeta.zephyr; print('matplotlib' in sys.modules, 'neuralfoil' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False"]
