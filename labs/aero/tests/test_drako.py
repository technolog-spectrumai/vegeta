"""Drako: sizing against the spec, the cost model, and the FEA and CFD cases prepared and pushed through the mocks
(``vegeta.mock``: nothing is meshed or solved, the numbers mean nothing). The CAD is skipped without CadQuery; the
cases then take a stand-in STEP / STL.

Run: VEGETA_DEBUG=1 python -m pytest -q -p vegeta.mock.pytest_plugin labs/aero/tests/test_drako.py
"""
import math

import numpy as np
import pytest

import drako
import drako_cfd as dcfd
import drako_cost as dcost
import drako_fea as dfea
from vegeta import mock

P = {q.name: q.default for q in drako.Drako.parameters}


@pytest.fixture(autouse=True, scope="module")
def mocks():
    was = mock.installed()
    if not was:
        mock.install()
    yield
    if not was:
        mock.uninstall()


def test_sizing_meets_the_spec():
    s = drako.Drako.sizing(P)
    assert s["endurance_empty_min"] >= 40 and s["endurance_design_min"] >= 25
    assert s["thrust_to_weight_max"] >= 1.8 and s["motor_out_T_W_max"] > 1.2
    assert s["tilt_80kmh_empty_deg"] < 30


def test_cost_radius_and_comparison():
    r = dcost.max_radius()
    assert 8.0 < r < 12.0
    assert dcost.trip_energy_Wh(r) == pytest.approx(0.8 * dcost.flight_power()["usable_Wh"], rel=1e-9)
    df = dcost.comparison()
    d, car = df.iloc[0], df.loc["car, petrol (Pb95) (dedicated round trip)"]
    assert d["energy_kWh"] < car["energy_kWh"] / 5 and d["CO2_kg"] < car["CO2_kg"]


def test_arm_reference_is_the_cantilever():
    a = dfea.arm_reference(P)
    ro, ri = P["arm_diameter"] / 2, P["arm_diameter"] / 2 - P["arm_wall"]
    I = math.pi / 4 * (ro ** 4 - ri ** 4)
    assert a["root bending stress [MPa]"] == pytest.approx(a["tip force, ultimate [N]"] * a["length [mm]"] * ro / I)
    assert a["margin"] > 0
    rpm = dfea.rotor_rpm(P)
    assert 1500 < rpm["design"]["rpm"] < rpm["max"]["rpm"] < 4000


@pytest.fixture(scope="module")
def step(tmp_path_factory):
    d = tmp_path_factory.mktemp("drako_fea")
    try:
        import cadquery  # noqa: F401
    except ImportError:
        (d / "stand_in.step").write_text("ISO-10303-21; stand-in for the mocks\n")
        return d, d / "stand_in.step", d / "stand_in.step"
    return d, None, None


def test_fea_cases_prepare_and_solve_under_the_mocks(step):
    wd, arm, frame = step
    cs = dfea.cases(wd, P, arm_step=arm, frame_step=frame)
    assert [c.name for c in cs] == ["arm_full_thrust", "arm_modes", "frame_manoeuvre", "frame_motor_out", "frame_modes"]
    mo = cs[3].model
    assert {l.region: l.fz for l in mo.loads}["M0"] < 0          # the dead motor: only its weight
    res = dfea.solve(cs, wd)
    tab = dfea.table(cs, res, P)
    assert (tab["status"] == "solved").all(), tab
    assert np.isfinite(tab.loc["frame_modes", "closest rotor harmonic [Hz]"])
    assert all(mock.is_mock(r.metadata) or r.metadata.get("mock") for r in res.values())


def _box_stl(path):
    from vegeta import aeromant
    v = np.array([[x, y, z] for x in (0, 600) for y in (0, 400) for z in (0, 350)], float)
    f = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    return aeromant.write_stl_ascii(aeromant.Surface(v[np.array(f)], "box"), path)


def test_cfd_cases_prepare_and_run_under_the_mocks(tmp_path):
    stls = {n: _box_stl(tmp_path / f"{n}.stl") for n in dcfd.CASES}
    dcfd.tilt_stl(stls["empty_80"], 19.0)
    cs = dcfd.cases(tmp_path / "cfd", P, stls=stls, iterations=50)
    assert cs["empty_80"].user_parameters["velocity"] == pytest.approx(80 / 3.6)
    res = dcfd.run(cs)
    tab = dcfd.table(cs, res, P)
    assert (tab["status"] == "solved").all(), tab
    assert (tmp_path / "cfd" / "loaded_40" / "system").is_dir()   # prepare is real: the OpenFOAM case is written
    assert dcfd.frontal_area_m2(P, True) > dcfd.frontal_area_m2(P, False)
