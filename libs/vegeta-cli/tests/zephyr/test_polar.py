"""The backends (each skipped when missing) against thin-airfoil theory and each other, the summary, the flap, the
JSON round trip, the CLI."""
import json
import math

import numpy as np
import pytest

from vegeta import zephyr
from vegeta.zephyr import Polar, analyze, naca

HAVE = zephyr.backends()
needs_nf = pytest.mark.skipif(not HAVE["neuralfoil"], reason="NeuralFoil not installed")
needs_xfoil = pytest.mark.skipif(not HAVE["xfoil"], reason="xfoil not on the PATH")


def fake_polar():
    a = np.arange(-6.0, 19.0, 1.0)
    cl = np.minimum(0.105 * (a + 2.0), 1.2)
    cl[a > 12] = 1.2 - 0.08 * (a[a > 12] - 12)
    cd = 0.012 + 0.01 * cl ** 2
    cm = -0.05 - 0.02 * cl
    return Polar("fake", 1e5, a, cl, cd, cm, np.ones_like(a), "test")


def test_summary_on_a_known_polar():
    s = fake_polar().summary(cl_design=0.4)
    assert s["cl_max"] == pytest.approx(1.2) and s["alpha_cl_max_deg"] in (10.0, 11.0, 12.0)
    assert s["alpha0_deg"] == pytest.approx(-2.0, abs=1e-9)
    assert s["cl_alpha_per_rad"] == pytest.approx(0.105 * 180 / math.pi, rel=1e-6)
    assert s["cm0"] == pytest.approx(-0.05) and s["x_ac"] == pytest.approx(0.27, abs=1e-6)       # cm = cm0 − 0.02 cl: the a.c. 2 % aft
    assert s["alpha_design_deg"] == pytest.approx(0.4 / 0.105 - 2.0) and s["LD_design"] == pytest.approx(0.4 / (0.012 + 0.0016), rel=0.01)


def test_json_round_trip(tmp_path):
    p = fake_polar()
    path = zephyr.save_polars(tmp_path / "p.json", [p], notes="t")
    q = zephyr.load_polars(path)[0]
    assert q.section == "fake" and np.allclose(q.cl, p.cl) and q.source == "test"
    with pytest.raises(ValueError):
        (tmp_path / "x.json").write_text(json.dumps({"kind": "other"}))
        zephyr.load_polars(tmp_path / "x.json")


def test_analyze_without_backends_says_so(monkeypatch):
    monkeypatch.setattr(zephyr.polar, "backends", lambda: {"neuralfoil": None, "xfoil": None})
    with pytest.raises(RuntimeError, match="no section backend"):
        analyze(naca("0012"), [0.0], 1e5)
    r = zephyr.analyze_result(naca("0012"), [0.0], 1e5, method="neuralfoil")
    assert not r.ok and "NeuralFoil" in r.messages[0]


@needs_nf
def test_neuralfoil_against_thin_airfoil_and_symmetry():
    s = naca("2412")
    p = analyze(s, np.arange(-6.0, 16.0, 1.0), 5e5, method="neuralfoil")
    chk = zephyr.check_against_thin_airfoil(p, s)
    assert abs(chk["d_alpha0_deg"]) < 0.6 and abs(chk["d_cm0"]) < 0.012
    assert 0.85 < chk["cl_alpha_ratio_to_2pi"] < 1.15
    assert 1.1 < p.summary()["cl_max"] < 1.5 and 10 <= p.summary()["alpha_cl_max_deg"] <= 15
    assert p.summary()["confidence_min_attached"] > 0.8
    q = analyze(naca("0012"), [-6.0, -3.0, 3.0, 6.0], 2e5, method="neuralfoil")
    assert np.allclose(q.cl[:2], -q.cl[2:][::-1], atol=0.02) and np.allclose(q.cd[:2], q.cd[2:][::-1], atol=0.001)


@needs_nf
def test_neuralfoil_flap_sweep():
    r = zephyr.flap_sweep(naca("0012"), 0.25, 1.5e5, alpha_deg=2.0, method="neuralfoil")
    assert 0.35 < r["tau"] / r["tau_thin"] < 1.0                                 # viscous, low Re: well below thin-airfoil theory
    assert 0.3 < r["x_cp_flap"] < 0.55                                          # the flap's lift centre behind the quarter chord
    assert r["cd_increment"][0] > 0.01 and r["cd_increment"][-1] > 0.005          # a deflection costs drag both ways
    assert r["cl"][-1] > r["cl"][0]


@needs_nf
def test_result_and_cli(tmp_path, capsys):
    res = zephyr.analyze_result(naca("23112"), np.arange(-4, 15, 1.0), 1e5, cl_design=0.4, out=tmp_path / "p.json")
    assert res.ok and res.metrics["cl_design"] == 0.4 and (tmp_path / "p.json").exists()
    from vegeta.zephyr.cli import main
    assert main(["polar", "--naca", "23112", "--re", "1e5", "--alpha", "-4", "14", "2", "--cl", "0.4", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["status"] == "success" and doc["metrics"]["section"] == "NACA 23112"
    assert main(["section", "--naca", "2412"]) == 0 and "alpha0_deg" in capsys.readouterr().out
    assert main(["backends"]) == 0
    assert main(["section", "--naca", "99"]) == 2


@needs_xfoil
def test_xfoil_runs_and_agrees_with_theory(tmp_path):
    s = naca("2412")
    p = analyze(s, np.arange(-4.0, 11.0, 2.0), 3e5, method="xfoil", workdir=tmp_path)
    assert p.source == "xfoil" and len(p) >= 5
    chk = zephyr.check_against_thin_airfoil(p, s)
    assert abs(chk["d_alpha0_deg"]) < 0.6 and abs(chk["d_cm0"]) < 0.012


@needs_xfoil
@needs_nf
def test_the_two_backends_agree_in_the_attached_range(tmp_path):
    s = naca("0012")
    a = np.arange(-4.0, 9.0, 2.0)
    x = analyze(s, a, 3e5, method="xfoil", workdir=tmp_path)
    n = analyze(s, x.alpha_deg, 3e5, method="neuralfoil")
    assert np.allclose(x.cl, n.cl, atol=0.05) and np.allclose(x.cd, n.cd, atol=0.003)
