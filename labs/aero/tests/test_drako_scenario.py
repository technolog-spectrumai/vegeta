"""Drako's cargo scenario (drako_scenario, drako_controller): the map, the wind, and a short MuJoCo run that latches
the first package and lifts it. The whole route is drako_missions.py (minutes).

Run: python -m pytest -q labs/aero/tests/test_drako_scenario.py
"""
import math

import numpy as np
import pytest

pytest.importorskip("mujoco")

import drako_controller as dcn
import drako_scenario as ds


def test_pads_are_level_and_the_route_clears_the_terrain():
    sc = ds.Scene()
    for n in sc.pads:
        x, y, z = sc.pad(n)
        ring = sc.height(x + 20 * np.cos(np.linspace(0, 6.28, 12)), y + 20 * np.sin(np.linspace(0, 6.28, 12)))
        assert np.allclose(ring, z, atol=1e-6)
    hut = sc.pad("hut")[2]
    assert 450 < hut < 650 and 40 < sc.pad("village")[2] < 120
    alt = dcn.cruise_altitude(sc, sc.pad("helipad")[:2], sc.pad("hut")[:2], dcn.Plan())
    assert alt >= hut + dcn.Plan().clearance_m - 1e-6


def test_wind_profile_direction_and_gusts():
    w = ds.Wind(10.0, from_deg=270.0, gust=0.0)
    v10, v100 = w.at((0, 0, 10), 0.0, 0.0), w.at((0, 0, 100), 0.1, 0.0)
    assert v10[0] == pytest.approx(10.0) and abs(v10[1]) < 1e-9                    # from the west: blows to +x
    assert np.linalg.norm(v100) == pytest.approx(10.0 * 10 ** 0.14, rel=1e-6)
    g = ds.Wind(10.0, gust=0.3, seed=4)
    u = np.array([g.at((0, 0, 10), 0.1 * k, 0.0)[0] for k in range(50000)])          # ~1700 correlation times
    assert u.mean() == pytest.approx(10.0, abs=0.4) and u.std() == pytest.approx(3.0, rel=0.15)
    assert not ds.Wind().at((0, 0, 50), 0.0, 0.0).any()


def test_rho_falls_with_altitude():
    assert ds.rho(0.0) == pytest.approx(1.225) and ds.rho(1000.0) == pytest.approx(1.112, abs=0.002)


def test_drako_latches_the_tote_and_lifts_it():
    sc = ds.Scene()
    lab = ds.make_lab(sc)
    ep = ds.run(lab, sc, duration=15.0)
    log = ep.log["mission"]
    assert any("medical latched" in note for _, _, note in log)
    i = ep.log["props"].index("medical")
    z_tote = np.asarray(ep.log["prop_pos"])[-1][i][2]
    ts = ds.timeseries(ep)
    assert ts.z.iloc[-1] > 20.0 and z_tote > 18.0                                  # climbing at 4 m/s with it
    assert 3000 < ts.P_W.iloc[-1] < 5000                                           # ~33 kg climbing: the sizing's 4 kW
