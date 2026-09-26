"""vegeta.ai is the provider connection only: it imports nothing from vegeta (no geometry, no tools)."""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "vegeta" / "ai"
FORBIDDEN = re.compile(r"^\s*(from|import)\s+(vegeta\b|\.\.)", re.M)


def test_no_vegeta_imports():
    assert [p.name for p in SRC.rglob("*.py") if FORBIDDEN.search(p.read_text())] == []
