"""The tail-frame study (notebook 34): the variants, the hand checks, what the tail puts into the wing, the drag and the
propeller maps."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import frame_study as fst
import nisus_plus as npl


@pytest.fixture(scope="module")
def F():
    return fst.frames()


def test_variants_follow_the_propeller(F):
    p = npl.resolve()
    assert fst.boom_y_for_prop(p, 15.0) == pytest.approx(p["boom_y"], abs=0.5)
    twins = [f for f in F if f.kind == "twin"]
    assert [f.boom_y for f in twins] == sorted(f.boom_y for f in twins) and twins[-1].boom_y > twins[0].boom_y + 80
    for f in twins:                                                      # the tail volumes are kept
        assert f.tail_span * f.tail_chord == pytest.approx(p["tail_span"] * p["tail_chord"], rel=1e-6)
        assert f.flap_y0 > f.boom_y and f.rear_spar_half_length > f.boom_y
    assert [f.feasible for f in F] == [True, True, False, False, True]  # 20" and 22" fail the wing's spar
    assert all("main spar" in f.note for f in F if not f.feasible)


def test_hand_margins_and_stiffness(F):
    h = fst.hand_table(F)
    assert (h["margin (500 MPa x 0.8)"].astype(float) > 0).all()
    s, t = h.loc['single 20"'], h.loc['twin 15"']
    assert s["frame mass [g]"] < t["frame mass [g]"] - 150                  # two root fittings fewer
    assert s["first bending mode, hand [Hz]"] > t["first bending mode, hand [Hz]"]
    assert s["tail roll, asymmetric case [deg]"] > t["tail roll, asymmetric case [deg]"]   # torsion against differential bending
    assert h["fin twist under its side load [deg]"].astype(float).max() < 5.0


def test_wing_loading_grows_with_the_spacing(F):
    w = fst.wing_loading(F)
    spar = w["main spar: bending from the boom, tail case [MPa]"].astype(float)
    assert spar['single 20"'] == 0.0 and spar['twin 22"'] > spar['twin 20"'] > spar['twin 15"']
    assert w.loc['twin 22"', "crow drag kept [%]"] < 70 < w.loc['twin 18"', "crow drag kept [%]"] < 100


def test_frame_drag_and_mass(F):
    d = {f.name: fst.frame_drag(f) for f in F}
    assert d['single 20"']["cd_area_m2"] < d['twin 15"']["cd_area_m2"] < d['twin 22"']["cd_area_m2"]
    m = {f.name: fst.frame_mass(f)["frame [g]"] for f in F}
    assert 250 < m['single 20"'] < m['twin 15"'] < 600


def test_frame_cad_is_one_solid(F):
    for f in (F[0], F[4]):
        s = fst._frame_cad(f).val()
        assert s.isValid() and len(s.Solids()) == 1


def test_motor_for_prop_meets_the_rated_current():
    prop = fst.propeller(20.0, 13.0)
    motor, fit = fst.motor_for_prop(prop)
    assert 250 < fit["kv"] < 540 and fit["static_thrust_n"] > 40.0
    assert fit["shaft_power_w"] == pytest.approx(fst.fs.motor_model()[1]["shaft_power_w"], rel=0.05)


@pytest.mark.skipif(not (fst.DATA / "frame_study_drive_20x13.json").exists(), reason="the 20x13 map is built by the notebook")
def test_big_propeller_map():
    dr = fst.drive_for_prop(20.0)
    base = fst.fs.drive()
    assert dr.max_thrust(0.0) > base.max_thrust(0.0)                  # more disc: more static thrust at the same power
    assert math.isfinite(dr.electrical_for_thrust(20.0, 15.0))
