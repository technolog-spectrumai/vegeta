"""Every vegeta module imports; every tool's command answers --help; the engineering tools import without MuJoCo."""
import importlib
import pkgutil
import subprocess
import sys

import pytest

import vegeta
from vegeta.cli import TOOLS

SKIP = ()                           # module prefixes that may not be imported (none today)


def all_modules():
    return sorted(m.name for m in pkgutil.walk_packages(vegeta.__path__, "vegeta.") if not m.name.startswith(SKIP) and not m.name.endswith(".__main__"))


def test_every_module_imports():
    failed = {}
    for name in all_modules():
        try:
            importlib.import_module(name)
        except Exception as e:                                   # noqa: BLE001 - report every broken module at once
            failed[name] = f"{type(e).__name__}: {e}"
    assert failed == {}


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_tool_help(tool, capsys):
    cli = importlib.import_module(f"vegeta.{tool}.cli")
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"], prog=tool)
    assert e.value.code == 0 and "usage" in capsys.readouterr().out.lower()


def test_vegeta_command_lists_the_tools(capsys):
    from vegeta.cli import main
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    assert all(t in out for t in TOOLS)


ENGINEERING = ["vegeta.dedalus", "vegeta.talos", "vegeta.aeromant", "vegeta.mellonia", "vegeta.boreas", "vegeta.chronos",
               "vegeta.atropos", "vegeta.zephyr", "vegeta.cache", "vegeta.core", "vegeta.fidia", "vegeta.ai", "vegeta.mock"]


def test_engineering_works_without_mujoco():
    code = ("import sys, importlib; sys.modules['mujoco'] = None\n"
            f"for m in {ENGINEERING!r}: importlib.import_module(m)\n"
            "from vegeta.dedalus import Geometry; import cadquery as cq\n"
            "assert abs(Geometry.from_cadquery(cq.Workplane().box(1, 2, 3)).volume - 6) < 1e-9\n"
            "assert 'mujoco' not in [k for k, v in sys.modules.items() if v is not None]\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
