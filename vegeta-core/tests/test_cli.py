import json
from pathlib import Path

import pytest

from vegeta.cli import main as vegeta_main
from vegeta.core.cli import rev_main, ws_main

MODEL = '''
from vegeta import talos

def static(rev):
    return talos.StructuralModel(
        rev.step, "mm-N-MPa", talos.Material("steel", 210000, 0.3),
        regions=[talos.SurfacesOnPlane("fixed", "x", 0.0), talos.SurfacesOnPlane("tip", "x", rev.params["length"])],
        supports=[talos.FixedSupport("fixed")], loads=[talos.Force("tip", fz=-100.0)],
        mesh_settings=talos.MeshSettings(10.0))
'''


def test_plugin_is_discovered(capsys):
    assert vegeta_main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "ws" in out and "rev" in out


def test_cli_workflow(tmp_path, capsys):
    w = str(tmp_path / "ws")
    assert vegeta_main(["ws", "-w", w, "init", "--name", "demo"]) == 0
    assert ws_main(["-w", w, "add-design", "beam", "vegeta.dedalus.examples:CantileverBeam"]) == 0
    assert rev_main(["-w", w, "new", "beam", "-p", "length=100", "--note", "base"]) == 0
    assert rev_main(["-w", w, "fea", "r1", "--model", "x.py:static", "--name", "s"]) == 2  # no such file
    assert rev_main(["-w", w, "generate", "r1"]) == 0
    assert rev_main(["-w", w, "branch", "r1", "-p", "height=12"]) == 0
    assert rev_main(["-w", w, "label", "r2", "preferred", "--note", "stiffer"]) == 0
    capsys.readouterr()
    assert ws_main(["-w", w, "status", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[1]["CAD"] == "NOT RUN" and rows[1]["label"] == "preferred" and rows[1]["parent"] == "r1"
    assert rev_main(["-w", w, "show", "r2", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["revision"]["changes"] == {"height": 12.0}
    assert rev_main(["-w", w, "new", "beam", "-p", "nope=1"]) == 2


@pytest.mark.requires_gmsh
@pytest.mark.requires_ccx
def test_cli_fea(tmp_path, capsys):
    w = str(tmp_path / "ws")
    ws_main(["-w", w, "init"])
    ws_main(["-w", w, "add-design", "beam", "vegeta.dedalus.examples:CantileverBeam"])
    rev_main(["-w", w, "new", "beam", "-p", "length=100"])
    rev_main(["-w", w, "generate", "r1"])
    f = tmp_path / "analyses.py"
    f.write_text(MODEL)
    capsys.readouterr()
    assert rev_main(["-w", w, "fea", "r1", "--model", f"{f}:static", "--name", "static", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "success"
    ws_main(["-w", w, "status"])
    assert "fea:static" in capsys.readouterr().out
