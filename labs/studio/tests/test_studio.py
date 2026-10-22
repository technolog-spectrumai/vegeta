"""Studio: every notebook starts with the lab's bootstrap, and its bootstrap and imports run (mocks on)."""
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

STUDIES = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


def _code(nb: Path) -> str:
    cells = ["".join(c["source"]) for c in json.loads(nb.read_text())["cells"] if c["cell_type"] == "code"]
    return "\n".join("\n".join("" if re.match(r"\s*[%!]", ln) else ln for ln in s.splitlines()) for s in cells)


@pytest.mark.parametrize("nb", STUDIES, ids=[p.stem for p in STUDIES])
def test_notebook_bootstraps_and_imports(nb):
    src = _code(nb)
    assert 'bootstrap.lab("studio"' in src
    tree = ast.parse(src)
    keep = [ast.get_source_segment(src, n) for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom)) or "bootstrap" in (ast.get_source_segment(src, n) or "")
            or (isinstance(n, ast.Assign) and ast.get_source_segment(src, n).startswith("DEBUG"))]
    env = dict(os.environ, VEGETA_DEBUG="1", VEGETA_NO_SOLVERS="1", MPLBACKEND="Agg")
    env.pop("ANTHROPIC_API_KEY", None)
    r = subprocess.run([sys.executable, "-c", "\n".join(keep)], cwd=nb.parent, capture_output=True, text=True,
                       timeout=300, env=env)
    assert r.returncode == 0, r.stderr[-1500:]
