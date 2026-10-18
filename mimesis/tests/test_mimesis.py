import json
import sys
from pathlib import Path

import pytest

import mimesis as mm


def test_mode_env_overrides_and_exports(monkeypatch):
    m = mm.Mode("MIMESIS_TEST_MODE")
    monkeypatch.delenv("MIMESIS_TEST_MODE", raising=False)
    assert m.resolve() == "mock" and m.resolve("real") == "mock"           # exported: the first resolve wins in this process
    monkeypatch.setenv("MIMESIS_TEST_MODE", "off")
    assert m.resolve("real") == "off" and not m.is_mock() and not mm.Mode.launches("off") and mm.Mode.launches("mock")
    monkeypatch.setenv("MIMESIS_TEST_MODE", "nonsense")
    with pytest.raises(ValueError):
        m.resolve()


def test_registry_dispatch_and_label():
    @mm.mimic("test.square")
    def stub(x):
        return {"y": 0.0}
    assert "test.square" in mm.available()
    out = mm.call("test.square", 3, mock=True, real=lambda x: {"y": x * x})
    assert mm.is_mock(out) and out["mimic"] == "test.square" and out["y"] == 0.0
    assert mm.call("test.square", 3, mock=False, real=lambda x: {"y": x * x}) == {"y": 9}
    with pytest.raises(mm.MimicError):
        mm.call("test.nothing", mock=True)
    with pytest.raises(mm.MimicError):
        mm.call("test.square", 3, mock=False)


def test_isolation(monkeypatch, tmp_path):
    assert mm.isolated("a/runs") == "a/runs-mock" == mm.isolated("a/runs-mock") and mm.isolated(Path("r")) == Path("r-mock")
    monkeypatch.setenv("MIMESIS_RUNS", str(tmp_path / "runs"))
    monkeypatch.delenv("MIMESIS_CACHE", raising=False)
    got = mm.isolate_env({"MIMESIS_RUNS": "x", "MIMESIS_CACHE": tmp_path / "cache"})
    assert got == {"MIMESIS_RUNS": str(tmp_path / "runs-mock"), "MIMESIS_CACHE": str(tmp_path / "cache-mock")}


def test_schema_record_and_check(tmp_path):
    real = {"T_max": 1232.5, "converged": True, "t": [0.0, 1.0], "nested": {"a": 1}, "name": "steel", "nan": None}
    mm.record_schema(real, tmp_path / "s.json")
    good = mm.label({"T_max": 300.0, "converged": True, "t": [0.0], "nested": {"a": 2.5}, "name": "x", "nan": 1.0, "extra": 1}, "t")
    mm.check(good, tmp_path / "s.json")
    with pytest.raises(mm.SchemaError):
        mm.check({**good, "converged": 1.0}, tmp_path / "s.json")
    with pytest.raises(mm.SchemaError):
        mm.check({k: v for k, v in good.items() if k != "t"}, tmp_path / "s.json")
    with pytest.raises(mm.SchemaError):
        mm.check(good, tmp_path / "s.json", extra_ok=False)


def test_interpreter_and_banner(monkeypatch, capsys, tmp_path):
    assert mm.interpreter(mock=True) == Path(sys.executable)
    monkeypatch.setenv("MIMESIS_PY", str(tmp_path / "nope"))
    assert mm.interpreter(mock=False, env="MIMESIS_PY") is None
    assert mm.interpreter(mock=False, env="MIMESIS_PY", candidates=[sys.executable]) == Path(sys.executable)
    assert mm.banner("mock", "hello") == "hello" and "hello" in capsys.readouterr().out


def test_standard_library_only():
    src = Path(mm.__file__).parent
    import ast
    stdlib = set(sys.stdlib_module_names) | {"mimesis", "IPython"}
    for p in src.glob("*.py"):
        for n in ast.walk(ast.parse(p.read_text())):
            if isinstance(n, ast.Import):
                assert all(a.name.split(".")[0] in stdlib for a in n.names), p
            elif isinstance(n, ast.ImportFrom) and n.level == 0:
                assert n.module.split(".")[0] in stdlib, p
