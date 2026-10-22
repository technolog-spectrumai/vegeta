import ast
import sys
from pathlib import Path

import yanus
from yanus import AnyOf, Env, Package, Tool, check_need, check_needs, table

HERE = Path(__file__).resolve().parents[1]


def test_tool_package_env_and_anyof(monkeypatch):
    yanus.clear_cache()
    monkeypatch.setenv("WHIZ_TEST_VAR", "x")
    monkeypatch.delenv("WHIZ_TEST_UNSET", raising=False)
    rows = check_needs([Tool("python3", version_cmd=(sys.executable, "--version")), Tool("no-such-tool-xyz"), Package("json"),
                        Package("no_such_package_xyz"), Env("WHIZ_TEST_VAR"), Env("WHIZ_TEST_UNSET"),
                        AnyOf("either", Package("no_such_package_xyz"), Package("json"))])
    assert [r.ok for r in rows] == [True, False, True, False, True, False, True]
    assert rows[0].version.startswith("Python")
    assert "PATH" in rows[1].fix and rows[5].fix.startswith("export ")
    assert "json" in rows[6].version


def test_probe_decides_and_failures_are_not_found():
    yanus.clear_cache()
    assert check_need(Tool("probed", probe=lambda: "1.2")).version == "1.2"
    assert not check_need(Tool("broken", probe=lambda: 1 / 0)).ok
    assert not check_need(Tool("none", probe=lambda: None)).ok


def test_table_is_plain_text():
    yanus.clear_cache()
    t = table(check_needs([Package("json"), Package("no_such_package_xyz", fix="pip install it")]))
    assert t.splitlines()[0].startswith("ok") and "MISSING" in t and "→ pip install it" in t


def test_standard_library_only():
    allowed = set(sys.stdlib_module_names) | {"yanus"}
    for p in (HERE / "src" / "yanus").glob("*.py"):
        for node in ast.walk(ast.parse(p.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                    [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 else []
            assert all(n.split(".")[0] in allowed for n in names), (p.name, names)
