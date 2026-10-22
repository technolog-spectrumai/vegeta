"""Side labs know no notebook: no code here names a notebook, the notebooks/ folder or a notebook registry. Their
results reach readers only through the cache (todo S.6–S.11)."""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"notebook|notebooks/|\.ipynb", re.I)


def test_no_side_lab_code_names_a_notebook():
    hits = [f"{p.relative_to(HERE)}:{n}" for p in HERE.rglob("*.py") if "tests" not in p.parts
            for n, line in enumerate(p.read_text().splitlines(), 1) if PATTERN.search(line)]
    assert hits == []
