"""Dedalus must not import the other engineering packages or Vegeta; results follow the common shape."""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "talos"
FORBIDDEN = ("dedalus", "aeromant", "mellonia", "vegeta")


def test_no_forbidden_imports():
    pattern = re.compile(r"^\s*(from|import)\s+(" + "|".join(FORBIDDEN) + r")\b", re.M)
    offenders = [p.name for p in SRC.rglob("*.py") if pattern.search(p.read_text())]
    assert offenders == []


def test_result_shape():
    from talos import Result

    d = Result(kind="talos.test").to_dict()
    assert set(d) == {"kind", "status", "metrics", "artifacts", "messages", "duration_s", "execution", "metadata"}
