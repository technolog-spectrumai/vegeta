"""Ozaru is a common layer: its libraries import no host (Vegeta, Alabastron), none of a host's workflow modules,
and nothing outside the standard library and themselves."""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIBS = ("porunga", "mimezia", "yanus")


def test_libraries_import_only_the_standard_library_and_themselves():
    allowed = set(sys.stdlib_module_names) | set(LIBS) | {"IPython"}      # mimezia.banner: optional, inside try/except ImportError
    bad = []
    for lib in LIBS:
        for p in (ROOT / lib / "src").rglob("*.py"):
            for node in ast.walk(ast.parse(p.read_text())):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                        [node.module or ""] if isinstance(node, ast.ImportFrom) and node.level == 0 else []
                bad += [f"{p.relative_to(ROOT)}: {n}" for n in names if n.split(".")[0] not in allowed]
    assert bad == []


def test_libraries_do_not_import_each_other():
    for lib in LIBS:
        for p in (ROOT / lib / "src").rglob("*.py"):
            for node in ast.walk(ast.parse(p.read_text())):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if node.level == 0 else []
                    assert not [n for n in names if n.split(".")[0] in set(LIBS) - {lib}], p
