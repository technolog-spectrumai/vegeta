"""Halka's jet: the Aeromant case prepares, Fulgur's sources are written into it (zones, fvOptions, run.sh), the
jet estimate follows; the trends run (the 1-D numbers under the mocks)."""
import math

import pytest

pytest.importorskip("cadquery")
pytest.importorskip("fipy")

from vegeta import mock

import halka_cfd as hc
import halka_trends as ht

G = 9.81


def test_case_prepares_with_the_sources(tmp_path):
    pr = hc.prepare("mark4c", tmp_path / "mark4c", V=22e3, lc=3.0)
    w = pr["sources"]["written"]
    assert w["zones"] >= 1 and w["force_z"] > 0
    assert (tmp_path / "mark4c" / "system" / "topoSetDict").exists()
    fvo = (tmp_path / "mark4c" / "system" / "fvOptions").read_text()
    assert "vectorSemiImplicitSource" in fvo and "cylinderAnnulusToCell" in (tmp_path / "mark4c" / "system" / "topoSetDict").read_text()
    sh = (tmp_path / "mark4c" / "run.sh").read_text()
    assert sh.index("topoSet") < sh.index("simpleFoam")
    row = hc.table(pr)
    assert row["cfd"] == "NOT RUN" and 0.5 < row["v_mean [m/s]"] < 8.0
    # the zones' force is the solve's thrust (binned, nothing lost)
    assert w["force_z"] == pytest.approx(pr["sources"]["solve"]["thrust"], rel=0.15)


def test_jet_estimate_scales():
    from vegeta.fulgur.jet import jet_estimate
    a, b = jet_estimate(0.1, 0.03), jet_estimate(0.4, 0.03)
    assert b.v_mean == pytest.approx(2 * a.v_mean, rel=1e-6)
    assert a.v_focal > a.v_mean and a.focal_distance > 0


@pytest.mark.skipif(not mock.installed(), reason="the trends solve the corona several times: mocks only in the lab's tests")
def test_trends_run_under_the_mocks():
    t = ht.teeth(lc=4.0)
    assert "velocity gain" in t
    o = ht.offset_loss(lc=4.0)
    assert len(o["velocity steps"]) == 2
