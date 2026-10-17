import json
import math

from vegeta.hyperion.cli import main


def test_materials_and_material(capsys):
    assert main(["materials"]) == 0 and "Nd:YAG" in capsys.readouterr().out
    assert main(["material", "--name", "bk7", "--wavelength", "587.6e-9", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert abs(d["n"] - 1.5168) < 2e-4
    assert main(["material", "--name", "nope"]) == 2 and "unknown material" in capsys.readouterr().err


def test_lens_beam_cavity_laser(capsys):
    assert main(["lens", "--r1", "0.1", "--r2", "-0.1", "--thickness", "0.02", "--material", "1.5", "--json"]) == 0
    assert abs(json.loads(capsys.readouterr().out)["f"] - 0.103448) < 1e-5
    assert main(["beam", "--wavelength", "1064e-9", "--w0", "200e-6", "--lens", "0.1", "--at", "0.5", "--after", "0.3", "--z", "0", "0.2", "0.1", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert abs(d["rayleigh_range"] - math.pi * 200e-6 ** 2 / 1064e-9) < 1e-9 and len(d["after_lens"]["waists"]) == 1 and len(d["profile"]) == 3
    assert main(["cavity", "--r1", "0.5", "--r2", "inf", "--length", "0.2", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["stable_t"] and abs(d["g1g2"] - 0.6) < 1e-12 and d["spots_t"]["M2"] < d["spots_t"]["M1"]
    assert main(["cavity", "--r1", "0.5", "--r2", "inf", "--length", "0.2", "--lens", "0.05", "--at", "0.1"]) == 1   # a strong lens: unstable
    capsys.readouterr()
    assert main(["laser", "--length", "0.01", "--pump", "10", "--wp", "200e-6", "--wl", "200e-6", "--t", "0.05", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert 0.1 < d["threshold_W"] < 0.3 and 4 < d["output_W"] < 7 and 0.03 < d["thermal_lens_f"] < 0.3
