import json
import shutil

import pytest

from aeromant.cli import main
from conftest import DATA

CASE = """
import math, aeromant
case = aeromant.CFDCase(
    "laminar_external_simplefoam", r"{stl}",
    dict(velocity=1.0, kinematic_viscosity=0.01, density=1.0, reference_area=math.pi / 4,
         reference_length=1.0, center_of_rotation=(0, 0, 0)),
    workdir=r"{work}", geometry_units="m",
    environment=aeromant.OpenFOAMEnvironment(env={{"PATH": r"{nopath}"}}))
"""


def test_cli_templates(capsys):
    assert main(["templates"]) == 0
    assert "laminar_external_simplefoam" in capsys.readouterr().out
    assert main(["templates", "rans_ksst_external_simplefoam", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert any(p["name"] == "velocity" and p["required"] for p in data[0]["parameters"])


def test_cli_prepare_run_results(tmp_path, sphere_stl, capsys):
    f = tmp_path / "case.py"
    f.write_text(CASE.format(stl=sphere_stl, work=tmp_path / "c", nopath=tmp_path / "empty"))
    assert main(["run", str(f), "-q"]) == 1                 # not prepared: explicit failure
    capsys.readouterr()
    assert main(["prepare", str(f), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["metrics"]["reynolds_number"] == pytest.approx(100)
    assert main(["prepare", str(f)]) == 1                   # refuses to overwrite silently
    capsys.readouterr()
    assert main(["run", str(f), "--steps", "blockMesh", "--json"]) == 1   # OpenFOAM absent on the given PATH
    assert "not found" in json.loads(capsys.readouterr().out)["messages"][0]
    assert main(["results", str(tmp_path / "c")]) == 1      # NOT RUN
    # fake a finished solver run and read it back
    post = tmp_path / "c/postProcessing/forceCoeffs/0"
    post.mkdir(parents=True)
    shutil.copy(DATA / "coefficient_v1912.dat", post / "coefficient.dat")
    shutil.copy(DATA / "log.simpleFoam", tmp_path / "c/log.simpleFoam")
    capsys.readouterr()
    assert main(["results", str(tmp_path / "c"), "--json", "--png", "--window", "2"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["metrics"]["Cd"] == pytest.approx(1.09) and "coefficients_png" in out["artifacts"]


def test_cli_bad_case_file(tmp_path, capsys):
    f = tmp_path / "c.py"
    f.write_text("x = 1\n")
    assert main(["prepare", str(f)]) == 2
    assert "0 CFDCase" in capsys.readouterr().err
