"""vegeta.yanus stands alone: it imports no other vegeta package, nothing of side_labs/ or the design modules, and
it starts no process except a tool's version probe (needs.py) — it guards, it never drives."""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "vegeta" / "yanus"
PROJECT = ("runkit", "library", "flow", "kestrel", "side_labs")


def _imports(p: Path):
    for node in ast.walk(ast.parse(p.read_text())):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield ("." * node.level) + (node.module or "")


def test_no_other_vegeta_imports():
    bad = []
    for p in SRC.rglob("*.py"):
        for name in _imports(p):
            if name.startswith("..") or (name.startswith("vegeta") and not name.startswith("vegeta.yanus")) or name.split(".")[0] in PROJECT:
                bad.append(f"{p.name}: {name}")
    assert bad == []


def test_no_process_launch_outside_the_version_probe_and_auto_mode():
    """Yanus guards: only ``needs.py`` (the version probe) and ``auto.py`` (auto mode, the one face that drives) may
    start a process."""
    bad = [f"{p.name}: {n}" for p in SRC.rglob("*.py") if p.name not in ("needs.py", "auto.py") for n in _imports(p) if n in ("subprocess", "multiprocessing", "pty")]
    for p in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "os" and \
                    (node.attr in ("system", "popen", "fork") or node.attr.startswith(("exec", "spawn"))):
                bad.append(f"{p.name}: os.{node.attr}")
    assert bad == []
