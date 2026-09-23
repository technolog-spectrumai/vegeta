import subprocess
import sys

import vegeta
from vegeta.cli import TOOLS, main


def test_vegeta_is_a_namespace_package():
    assert getattr(vegeta, "__file__", None) is None  # no vegeta/__init__.py
    from vegeta import aeromant, dedalus, mellonia, talos  # noqa: F401


def test_help_version_and_unknown_tool(capsys):
    assert main([]) == 0 and "dedalus" in capsys.readouterr().out
    assert main(["--version"]) == 0 and capsys.readouterr().out.startswith("vegeta-cli ")
    assert main(["nope"]) == 2 and "unknown tool" in capsys.readouterr().err
    assert set(TOOLS) == {"dedalus", "talos", "aeromant", "mellonia"}


def test_delegates_to_tool_with_prog_name(capsys):
    assert main(["dedalus", "params", "vegeta.dedalus.examples:Cube"]) == 0
    assert "size" in capsys.readouterr().out
    assert main(["dedalus", "generate", "vegeta.dedalus.examples:Cube", "-p", "nope=1"]) == 2
    assert capsys.readouterr().err.startswith("vegeta dedalus: error")


def test_python_dash_m_entry_points():
    for args in (["-m", "vegeta.cli", "--help"], ["-m", "vegeta.mellonia", "--help"]):
        r = subprocess.run([sys.executable, *args], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
