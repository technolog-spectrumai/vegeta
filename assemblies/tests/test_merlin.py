"""(Copied from notebooks/designs/tests with only the imports changed: the promoted copy must pass the same tests.)
MERLIN (notebook 26): the three variants build, the EDF nozzle closes around the fuselage, the race behaves (time
rises with distance, the reserve is kept), the plume carries its emission and the passes recover the fire's size, the
mission comes home, and the movie renders.

Run: cd /home/user/vegeta && .venv/bin/python -m pytest -q notebooks/designs/tests/test_merlin.py   (~40 s)
"""
import math

import numpy as np
import pytest

from assemblies.components import merlin_flight as mf


@pytest.fixture(scope="module")
def merlins():
    return mf.build_merlins(quick=True)


@pytest.mark.parametrize("kind", ["edf", "tractor", "pusher"])
def test_every_variant_builds_as_one_solid(kind):
    pytest.importorskip("cadquery")
    from assemblies.components import merlin as m
    g = m.Merlin().generate(propulsion=kind)
    assert g.shape.isValid() and len(g.shape.Solids()) == 1
    bb = g.shape.BoundingBox()
    assert bb.ymax == pytest.approx(700.0, abs=0.5)                       # half the span
    L = m.Merlin.layout(m.Merlin().resolve(propulsion=kind))
    if kind == "pusher":
        assert L["prop_x"] > L["x_tail"]                                  # the propeller behind the tail
    else:
        assert L["prop_x"] < 0                                            # on the nose, ahead of the wing


def test_edf_nozzle_closes_around_the_fuselage():
    from assemblies.components import ducted_fan as df
    from assemblies.components import merlin as m
    hp = m.edf_housing_params(m.Merlin().resolve())
    g = df.housing_geometry(hp)
    assert g["exit_area_ratio"] == pytest.approx(hp["exit_area_ratio"], rel=0.01)   # the annular jet's area as designed
    assert g["tail_radius_at_exit"] == pytest.approx(hp["hub_diameter"] / 2, rel=1e-3)   # a cylinder, not a cone


def test_drag_buildup_is_a_clean_small_airframe():
    from assemblies.components import merlin as m
    b = m.drag_buildup({}, 25.0)
    assert 0.012 < b["cd0"] < 0.04 and 6 < b["aspect_ratio"] < 7


def test_race_slows_with_distance_and_keeps_the_reserve(merlins):
    for kind, v in merlins.items():
        rr = [mf.race(v["airframe"], v["unit"], d * 1000) for d in mf.DISTANCES_KM]
        assert all(r["reachable"] for r in rr), kind
        t = [r["time_to_fire_s"] for r in rr]
        assert all(a < b for a, b in zip(t, t[1:]))
        for r in rr:
            assert r["energy_out_wh"] + r["energy_sampling_wh"] + r["energy_home_wh"] <= r["budget_wh"] * (1 + 1e-6)
            assert r["dash_speed"] <= r["top_speed"] + 1e-9
    far = mf.race(merlins["edf"]["airframe"], merlins["edf"]["unit"], 120e3)
    assert not far["reachable"]


def test_reach_return_and_landing_add_up(merlins):
    for v in merlins.values():
        r = mf.race(v["airframe"], v["unit"], 8000.0)
        assert r["landing_time_s"] == pytest.approx(r["time_to_fire_s"] + r["sampling_s"] + r["return_time_s"])
        assert r["range_speed"] - 1e-9 <= r["return_speed"] <= r["top_speed"] + 1e-9


def test_a_fire_up_a_mountain_costs_reach_time_and_keeps_the_reserve(merlins):
    for v in merlins.values():
        for d in (5.0, 10.0):
            flat = mf.race(v["airframe"], v["unit"], d * 1000)
            hill = mf.race(v["airframe"], v["unit"], d * 1000, fire_elevation_m=600.0)
            assert hill["time_to_fire_s"] > flat["time_to_fire_s"] and hill["top_speed"] < flat["top_speed"]
            assert 0 < hill["dash_climb_deg"] < 10
            assert hill["energy_out_wh"] + hill["energy_sampling_wh"] + hill["energy_home_wh"] <= hill["budget_wh"] * (1 + 1e-6)


def test_full_throttle_respects_the_power_limit(merlins):
    for v in merlins.values():
        u = v["unit"]
        for V in (0.0, 20.0, 45.0):
            T, P, n = u.full(V)
            assert P <= u.max_electrical_w + 1e-6 and n <= u.rpm[-1] + 1e-6
            assert u.electrical_power(V, 0.5 * T) < P


def test_plume_carries_its_emission():
    pl = mf.Plume(heat_mw=40.0, wind_speed=5.0, wind_from_deg=225.0)
    d, n = pl.downwind, np.array([-pl.downwind[1], pl.downwind[0]])
    c, z = np.linspace(-2000, 2000, 2001), np.linspace(0, 2000, 1001)
    for s in (300.0, 1500.0):
        C, Z = np.meshgrid(c, z)
        X = pl.source_xy[0] + s * d[0] + C * n[0]
        Y = pl.source_xy[1] + s * d[1] + C * n[1]
        kg = pl.concentration_ppm(X, Y, Z) * 1e-6 * mf.RHO * 28.01 / 28.97
        flux = np.trapezoid(np.trapezoid(kg, c, axis=1), z) * pl.wind_speed
        assert flux == pytest.approx(pl.co_kg_s, rel=0.01)
    assert pl.concentration_ppm(pl.source_xy[0] - 500 * d[0], pl.source_xy[1] - 500 * d[1], 100.0) == 0.0   # nothing upwind


def test_source_estimate_recovers_a_synthetic_fire():
    pl = mf.Plume(heat_mw=25.0)
    d, n = pl.downwind, np.array([-pl.downwind[1], pl.downwind[0]])
    c = np.linspace(-500, 500, 401)
    passes = []
    for h in (80.0, 120.0, 160.0, 200.0, 240.0):
        x = pl.source_xy[0] + 600 * d[0] + c * n[0]; y = pl.source_xy[1] + 600 * d[1] + c * n[1]
        passes.append({"x": x, "y": y, "z_agl": np.full_like(c, h), "ppm": pl.concentration_ppm(x, y, h)})
    est = mf.source_estimate(pl, passes)
    assert est["heat_mw"] == pytest.approx(25.0, rel=0.05)


def test_mission_samples_the_smoke_and_comes_home(merlins):
    v = merlins["pusher"]
    pl = mf.Plume(source_xy=(10000.0, 0.0), wind_from_deg=225.0)
    r = mf.race(v["airframe"], v["unit"], 10000.0)
    ep = mf.fly(v["airframe"], v["unit"], mf.Forest(), pl, race_result=r)
    s = ep.summary()
    assert len(s["events"]) == 1 and "touchdown" in s["events"][0] and s["passes"] >= 2 and s["peak_ppm"] > 0.5
    assert np.linalg.norm(ep.pos[-1, :2]) < 400 and ep.agl[-1] < 0.3          # on the ground near the launch site
    assert ep.energy_wh[-1] < mf.Mission().battery_wh * mf.Mission().usable_fraction
    assert ep.airspeed.max() <= r["dash_speed"] + 0.5


def test_movie_renders(merlins):
    pytest.importorskip("cv2")
    v = merlins["tractor"]
    pl = mf.Plume(source_xy=(8000.0, 0.0))
    ep = mf.fly(v["airframe"], v["unit"], mf.Forest(), pl, race_result=mf.race(v["airframe"], v["unit"], 8000.0))
    path, frames = mf.render_movie(ep, mf.Forest(), pl, "/tmp/merlin_test_movie.mp4", seconds=0.2, fps=10, size=(640, 360))
    assert len(frames) == 2 and frames[0].shape == (360, 640, 3) and path.exists()
