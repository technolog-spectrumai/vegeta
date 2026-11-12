"""vegeta.mellonia must not import the other tools, other vegeta subpackages or its parent namespace."""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "vegeta" / "mellonia"
FORBIDDEN = re.compile(
    r"^\s*(from|import)\s+(vegeta\b|dedalus|talos|aeromant\b|\.\.)", re.M
)


def test_no_forbidden_imports():
    offenders = [str(p.relative_to(SRC)) for p in SRC.rglob("*.py") if FORBIDDEN.search(p.read_text())]
    assert offenders == []


def test_result_shape():
    from vegeta.mellonia import Result

    d = Result(kind="mellonia.test").to_dict()
    assert set(d) == {"kind", "status", "metrics", "artifacts", "messages", "duration_s", "execution", "metadata"}
