"""PEREGRINE (notebook 28): the folding wing's geometry, the vortex lattice against theory, the flight models' physics.

Run: .venv/bin/python -m pytest -q notebooks/designs/tests/test_peregrine.py   (about a minute; the CAD tests need
CadQuery, the movie test OpenCV)
"""
import math

import numpy as np
import pytest

import peregrine as pg
import peregrine_flight as pf


def _rect(AR, sweep_deg=0.0):
    b = AR
    dx = math.tan(math.radians(sweep_deg)) * b / 2
    q = np.array([[0, 0, 0], [dx, -b / 2, 0], [dx + 1, -b / 2, 0], [1, 0, 0]], float)
    return [q, q * [1, -1, 1]]


@pytest.mark.parametrize("AR", [6.0, 10.0])
def test_lattice_rectangular_wing_against_helmbold(AR):
    r = pf.vlm(_rect(AR), alpha_deg=5.0, S_ref=AR, c_ref=1.0, n_span=30, n_chord=4)
    cla = r["CL"] / math.radians(5.0)
    helmbold = 2 * math.pi * AR / (2 + math.sqrt(AR ** 2 + 4))
    assert cla == pytest.approx(helmbold, rel=0.08)
    e = r["CL"] ** 2 / (math.pi * AR * r["CDi"])
    assert 0.93 < e < 1.03                                     # a rectangular wing: a little below elliptic


def test_lattice_neutral_point_near_quarter_chord_and_sweep_lowers_lift_slope():
    r0 = pf.vlm(_rect(6.0), alpha_deg=0.0, S_ref=6.0, c_ref=1.0, n_span=20, n_chord=4)
    r1 = pf.vlm(_rect(6.0), alpha_deg=5.0, S_ref=6.0, c_ref=1.0, n_span=20, n_chord=4)
    x_np = -(r1["Cm"] - r0["Cm"]) / (r1["CL"] - r0["CL"])
    assert 0.22 < x_np < 0.27
    swept = pf.vlm(_rect(6.0, 45.0), alpha_deg=5.0, S_ref=6.0, c_ref=1.0, n_span=20, n_chord=4)
    assert swept["CL"] < 0.85 * r1["CL"]


def test_lattice_streamwise_strips_survive_a_rotated_panel():
    """A panel rotated about a hinge (its chords no longer streamwise) loses lift smoothly with the angle."""
    base = np.array([[0, -1, 0], [0, -3, 0], [1, -3, 0], [1, -1, 0]], float)
    cen = np.array([[0, 0, 0], [0, -1, 0], [1, -1, 0], [1, 0, 0]], float)
    cl = []
    for ang in (0.0, 15.0, 30.0, 45.0, 60.0):
        q = pg._rotate_xy(base, 0.2, -1.0, ang)
        r = pf.vlm([cen, cen * [1, -1, 1], q, q * [1, -1, 1]], alpha_deg=5.0, S_ref=6.0, c_ref=1.0, n_span=16, n_chord=4,
                   hidden=[0, 0, 1.0, 1.0])
        cl.append(r["CL"])
    assert all(np.diff(cl) < 0) and cl[-1] > 0.3 * cl[0]


def test_planform_and_fold_limit():
    pl0, pl60 = pg.planform({"variant": "C"}, 0.0), pg.planform({"variant": "C"}, 60.0)
    assert pl0["b"] == pytest.approx(1.1)
    assert pl60["b"] < 0.8 * pl0["b"]
    assert pl0["S_ref"] == pytest.approx(pl60["S_ref"])
    assert pg.fold_limit({"variant": "C"})["max_fold_deg"] == 90.0                     # the hinge at 30 % of the half span
    assert pg.fold_limit({"variant": "C", "hinge_y_frac": 0.0})["max_fold_deg"] < 15.0  # a true root hinge collides early
    ex0, ex60 = pg.exposed_wing_area({"variant": "C"}, 0.0), pg.exposed_wing_area({"variant": "C"}, 60.0)
    assert ex60["exposed"] < ex0["exposed"]
    with pytest.raises(ValueError):
        pg.planform({"variant": "A"}, 30.0)


def test_drag_buildup_and_fold_polar():
    a = pg.drag_buildup({"variant": "A"}, 20.0)
    c = pg.drag_buildup({"variant": "C"}, 20.0)
    assert 0.01 < a["cd0"] < 0.04
    assert c["cd_area_m2"] > a["cd_area_m2"]                   # the hinges cost a little drag
    fp = pf.fold_polar({"variant": "C"}, (0.0, 30.0, 60.0))
    assert fp["CLα [1/rad]"].is_monotonic_decreasing
    assert fp["k = CDi/CL²"].is_monotonic_increasing
    assert fp["x_np [m]"].iloc[-1] > fp["x_np [m]"].iloc[0]   # the tuck moves the neutral point aft
    assert fp["CL_max"].is_monotonic_decreasing


def test_mass_table_sets_the_margin_and_bom_sums():
    for v in "ABC":
        t = pf.mass_table({"variant": v})
        st = pf.stability({"variant": v}, 0.0, t)
        assert st["static margin [%MAC]"] == pytest.approx(10.0, abs=0.1)
        assert abs(st["elevator trim [deg]"]) < 10
        b = pf.bom({"variant": v})
        assert b["mass [g]"].sum() == pytest.approx(t["mass [kg]"].sum() * 1000)
    assert pf.bom({"variant": "C"})["EUR"].sum() > pf.bom({"variant": "A"})["EUR"].sum()


@pytest.fixture(scope="module")
def fleet():
    unit = pf.drive()
    acs = {}
    for v in "ABC":
        m = pf.mass_table({"variant": v})["mass [kg]"].sum()
        acs[v] = pf.make_aircraft(v, {"variant": v}, m, folds=(0.0, 30.0, 60.0) if v == "C" else (0.0,))
    return unit, acs


def test_turn_radius_and_terminal_speed(fleet):
    unit, acs = fleet
    af = acs["A"].af(0.0)
    tr = pf.turn(af, unit, np.array([15.0]), n_struct=5.0)
    n = tr["n_inst"][0]
    assert tr["radius_inst"][0] == pytest.approx(15.0 ** 2 / (pf.G * math.sqrt(n ** 2 - 1)))
    vt = pf.terminal_speed(af)
    assert vt == pytest.approx(math.sqrt(2 * af.mass_kg * pf.G / (pf.RHO * af.cd0 * af.wing_area_m2)), rel=1e-6)
    s = pf.stoop(acs["A"], unit, h0=80.0, x_target=40.0)
    assert s["V_max"] < vt and s["h_bottom"] == pytest.approx(3.0, abs=0.3)


def test_roof_start_and_perch(fleet):
    unit, acs = fleet
    for ac in acs.values():
        ph = pf.prop_hang(ac, unit)
        assert ph["T/W"] > 1.2 and ph["possible"]
        assert pf.hand_throw(ac, unit, V0=9.0)["flies"]
    w0 = pf.perched_wind({"variant": "C"}, 0.0, acs["C"].mass_kg)
    w85 = pf.perched_wind({"variant": "C"}, 85.0, acs["C"].mass_kg)
    assert w85["holds to [m/s]"] > w0["holds to [m/s]"]


def test_flock_flees_the_drone():
    fl = pf.Flock(n=20, seed=1)
    pos, vel = fl.start(np.array([0.0, 0.0]))
    drone = pos.mean(axis=0) + [-20.0, 0.0, 0.0]
    d0 = np.linalg.norm(pos.mean(axis=0) - drone)
    for _ in range(40):
        pos, vel, afraid = fl.step(pos, vel, drone, np.array([0.0, 0.0]), 0.05)
    assert np.linalg.norm(pos.mean(axis=0) - drone) > d0


def test_hinge_loads(fleet):
    _, acs = fleet
    h = pf.hinge_loads({"variant": "C"}, 0.0, acs["C"].mass_kg, n=4.0, V=25.0)
    assert 0 < h["panel lift [N]"] < 4.0 * acs["C"].mass_kg * pf.G / 2
    assert h["bearing force [N]"] > h["panel lift [N]"]
    assert h["servo torque [N m]"] > 0


def test_cad_all_variants_and_folds():
    pytest.importorskip("cadquery")
    P = pg.Peregrine()
    for v, folds in (("A", (0.0,)), ("B", (0.0,)), ("C", (0.0, 60.0, 85.0))):
        for f in folds:
            g = P.generate(**pg.VARIANTS[v], fold_deg=f)
            assert g.shape.isValid() and len(g.shape.Solids()) == 1
            assert g.dimensions[1] == pytest.approx(pg.planform({"variant": v}, f)["b"] * 1000, rel=0.02)
    lug = P.generate(part="hinge")
    assert lug.shape.isValid() and len(lug.shape.Solids()) == 1


def test_mission_and_movie(fleet, tmp_path):
    unit, acs = fleet
    ep = pf.mission(acs["C"], unit, pf.Farm(), pf.Flock.of("starling", seed=0), tuck=60.0, fold_fast=30.0)
    assert ep.phase[-1].startswith("prop-hang landing")
    assert np.linalg.norm(ep.pos[-1, :2] - np.array(pf.Farm().perch[:2])) < 1.0
    assert 0 < ep.energy_wh[-1] < 24.4 * 0.8
    pytest.importorskip("cv2")
    path, n = pf.render_movie(ep, pf.Farm(), {"variant": "C"}, tmp_path / "m.mp4", seconds=0.2, fps=10, size=(640, 360))
    assert path.exists() and n == 2
