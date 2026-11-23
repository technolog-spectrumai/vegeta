import json

from vegeta.atropos.cli import main

SETTINGS = """
from vegeta.atropos.examples import D1_PRO_BALSA_2MM

FINE = D1_PRO_BALSA_2MM.replace(spacing_mm=2.0)
"""


def test_rib_nest_info_round_trip(tmp_path, capsys):
    parts = tmp_path / "parts.json"
    assert main(["rib", "--chord", "250", "--naca", "3412", "--tube", "0.30:20:0.3", "--tube", "0.55:18:0.3",
                 "--sheeting", "0:0.32:1.5", "--split", "0.70", "--gap", "1.5", "--bevel", "12", "--name", "R1",
                 "--out", str(parts), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert [p["part"] for p in out["parts"]] == ["R1-fwd", "R1-aft"] and out["parts"][0]["holes"] == 2
    svg = tmp_path / "rib.svg"
    assert main(["rib", "--chord", "120", "--out", str(svg)]) == 0
    assert svg.is_file() and "rib" in capsys.readouterr().out
    assert main(["info", str(svg), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["paths"]["cut"] == 1

    f = tmp_path / "settings.py"
    f.write_text(SETTINGS)
    job = tmp_path / "job"
    assert main(["nest", str(parts), "-s", f"{f}:FINE", "-o", str(job), "--json"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["status"] == "success" and res["metrics"]["sheets"] == 1
    assert main(["info", str(job / "sheet_01.gcode")]) == 0
    assert "g1_length_mm" in capsys.readouterr().out
    assert main(["settings", "vegeta.atropos.examples:D1_PRO_BALSA_2MM"]) == 0
    assert json.loads(capsys.readouterr().out)["sheet"]["thickness_mm"] == 2.0


def test_cli_errors(tmp_path, capsys):
    f = tmp_path / "s.py"
    f.write_text("x = 1\n")
    (tmp_path / "p.json").write_text("[]")
    assert main(["nest", str(tmp_path / "p.json"), "-s", str(f), "-o", str(tmp_path / "o")]) == 2
    assert "0 CutSettings" in capsys.readouterr().err
    assert main(["rib", "--chord", "100", "--naca", "12"]) == 2
    assert "four digits" in capsys.readouterr().err
    assert main(["rib", "--chord", "100", "--tube", "0.3:80"]) == 2
    assert "does not fit" in capsys.readouterr().err
    assert main(["info", str(tmp_path / "nope.gcode")]) == 2
