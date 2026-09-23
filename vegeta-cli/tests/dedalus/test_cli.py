import pytest
import json
import subprocess
import sys
from pathlib import Path

from dedalus.cli import main

DESIGN_FILE = '''
import cadquery as cq
from dedalus import Design, Parameter

class Plate(Design):
    parameters = [Parameter("width", 30.0, "mm", min=1), Parameter("thickness", 2.0, "mm", min=0.5)]
    def build(self, p):
        return cq.Workplane().box(p["width"], 10, p["thickness"])
'''


def test_cli_generate_from_file(tmp_path, capsys):
    f = tmp_path / "plate.py"
    f.write_text(DESIGN_FILE)
    out = tmp_path / "out"
    rc = main(["generate", str(f), "-p", "width=40", "-o", str(out), "--json"])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["metrics"]["volume"] == pytest.approx(40 * 10 * 2)
    assert (out / "Plate.step").is_file() and (out / "Plate.stl").is_file()


def test_cli_params_and_measure(tmp_path, capsys):
    assert main(["params", "dedalus.examples:Cube", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["parameters"][0]["name"] == "size"
    assert main(["generate", "dedalus.examples:Cube", "-o", str(tmp_path), "--formats", "step"]) == 0
    capsys.readouterr()
    assert main(["measure", str(tmp_path / "Cube.step"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["measurements"]["volume"] == pytest.approx(8000.0)


def test_cli_errors_are_exit_codes(tmp_path, capsys):
    # invalid input is a usage error (2); a design that fails to build is a failed run (1)
    assert main(["generate", "dedalus.examples:Cube", "-p", "size=-3", "-o", str(tmp_path)]) == 2
    assert main(["generate", "dedalus.examples:Bracket", "-p", "hole_diameter=30", "-o", str(tmp_path)]) == 1
    assert main(["generate", "dedalus.examples:Cube", "-p", "nope=3", "-o", str(tmp_path)]) == 2
    assert "unknown parameter" in capsys.readouterr().err


def test_console_script_runs(tmp_path):
    r = subprocess.run([sys.executable, "-m", "dedalus", "params", "dedalus.examples:Bracket"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "hole_diameter" in r.stdout
