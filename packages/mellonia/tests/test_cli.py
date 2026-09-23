import json

import pytest

from mellonia.cli import main
from conftest import DATA

SETTINGS = """
from mellonia import PrintSettings
from mellonia.examples import GENERIC_PLA_0_2MM
FINE = GENERIC_PLA_0_2MM.replace(print={"layer_height": 0.1, "first_layer_height": 0.2})
"""


def test_cli_parse(capsys):
    assert main(["parse", str(DATA / "tiny.gcode"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["metrics"]["layer_count"] == 3


@pytest.mark.requires_prusaslicer
def test_cli_slice_with_settings_file(tmp_path, cube_stl, capsys):
    f = tmp_path / "settings.py"
    f.write_text(SETTINGS)
    assert main(["slice", str(cube_stl), "-s", f"{f}:FINE", "-o", str(tmp_path / "o"), "--json", "--png"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["metrics"]["layer_count"] == 1 + round((20 - 0.2) / 0.1) and "layers_png" in out["artifacts"]


def test_cli_errors(tmp_path, cube_stl, capsys):
    f = tmp_path / "s.py"
    f.write_text("x = 1\n")
    assert main(["slice", str(cube_stl), "-s", str(f), "-o", str(tmp_path)]) == 2
    assert "0 PrintSettings" in capsys.readouterr().err
    assert main(["slice", str(cube_stl), "-s", "mellonia.examples:GENERIC_PLA_0_2MM", "-o", str(tmp_path / "x"),
                 "--prusa-slicer", "no-such-slicer"]) == 1
