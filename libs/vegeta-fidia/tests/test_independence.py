"""vegeta.fidia may use Dedalus (geometry), vegeta.ai (the provider) and, lazily, vegeta.core; nothing else."""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "vegeta" / "fidia"
IMPORT = re.compile(r"^\s*(?:from|import)\s+(vegeta(?:\.\w+)*)", re.M)
ALLOWED = ("vegeta.dedalus", "vegeta.ai", "vegeta.core", "vegeta.fidia")


def test_imports_only_allowed_vegeta_packages():
    bad = {p.name: m for p in SRC.rglob("*.py") for m in IMPORT.findall(p.read_text()) if not m.startswith(ALLOWED)}
    assert bad == {}


def test_import_is_light():
    import subprocess
    import sys
    code = "import sys, vegeta.fidia; heavy = [m for m in ('pyvista', 'vegeta.core', 'anthropic') if m in sys.modules]; print(heavy)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]"
