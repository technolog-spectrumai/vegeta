"""og.gears, og.soil, og.tracks, og.cutting, og.actuators return exactly what Vegeta's designs/ modules returned
before they moved (golden/parts.json, recorded at Vegeta 787b988 with the same inputs: parts_cases.py)."""
import json
import math
from pathlib import Path

import pytest

from ogma import actuators, cutting, gears, soil, tracks

HERE = Path(__file__).resolve().parent
import importlib.util
_spec = importlib.util.spec_from_file_location("parts_cases", HERE / "parts_cases.py")
parts_cases = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parts_cases)

GOLDEN = json.loads((HERE / "golden" / "parts.json").read_text())
NOW = parts_cases.cases(gears, tracks, soil, cutting, actuators)


def same(a, b, path=""):
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and isinstance(b, float):
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15) or (math.isnan(a) and math.isnan(b)), (path, a, b)
    else:
        assert a == b, (path, a, b)


@pytest.mark.parametrize("module", sorted(GOLDEN))
def test_module_reproduces_vegeta(module):
    same(NOW[module], GOLDEN[module], module)


def test_tables_are_lazy_on_pandas():
    import subprocess
    import sys
    code = "import sys, ogma.cutting, ogma.actuators, ogma.soil, ogma.tracks, ogma.gears; print('pandas' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={"PYTHONPATH": str(HERE.parent / "src")}).stdout.strip()
    assert out == "False"
