import json

import pytest

from vegeta.talos.cli import main

MODEL = """
from vegeta import talos
model = talos.StructuralModel(
    geometry=r"{step}", units="mm-N-MPa",
    material=talos.Material("steel", 210000, 0.3, yield_strength=235),
    regions=[talos.SurfacesOnPlane("fixed", "x", 0), talos.SurfacesOnPlane("tip", "x", 200)],
    supports=[talos.FixedSupport("fixed")], loads=[talos.Force("tip", fz=-100)],
    mesh_settings=talos.MeshSettings(element_size=10))
"""


@pytest.mark.requires_gmsh
def test_cli_inspect(beam_step, capsys):
    assert main(["inspect", str(beam_step), "--units", "mm-N-MPa", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["surfaces"]) == 6


def test_cli_inspect_requires_units(capsys):
    with pytest.raises(SystemExit):
        main(["inspect", "x.step"])


@pytest.mark.requires_gmsh
@pytest.mark.requires_ccx
def test_cli_mesh_solve_results(beam_step, tmp_path, capsys):
    f = tmp_path / "model.py"
    f.write_text(MODEL.format(step=beam_step))
    w = tmp_path / "w"
    assert main(["solve", str(f), "-w", str(w), "-q"]) == 1          # no mesh yet: explicit failure
    capsys.readouterr()
    assert main(["mesh", str(f), "-w", str(w), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == "talos.mesh"
    assert main(["solve", str(f), "-w", str(w), "--json", "--png"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["metrics"]["max_displacement"] > 0 and "deformed_png" in out["artifacts"]
    assert main(["results", str(w)]) == 0


def test_cli_results_not_run(tmp_path, capsys):
    assert main(["results", str(tmp_path)]) == 1
    assert "NOT RUN" in capsys.readouterr().err


def test_cli_bad_model_file(tmp_path, capsys):
    f = tmp_path / "m.py"
    f.write_text("x = 1\n")
    assert main(["mesh", str(f), "-w", str(tmp_path)]) == 2
    assert "0 StructuralModel" in capsys.readouterr().err
