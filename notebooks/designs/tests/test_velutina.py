"""Velutina (notebook 24): the Dedalus design builds in every part, the flight model flies both deliveries, the
terrain, the ISA, the parachute estimate and the movie renderer.

Run: cd /home/user/vegeta && python3 -m pytest -q notebooks/designs/tests/test_velutina.py
"""
import math

import numpy as np
import pytest

import velutina_flight as vf  # noqa: E402


def _aircraft(**kw):
    base = dict(mass_kg=2.6, capsule_kg=0.54, cda_axial_m2=0.0031, cda_cross_m2=0.0575, hover_power_w_sl=506.0,
                hover_thrust_n_sl=25.5, max_thrust_n_sl=100.0, battery_wh=151.0)
    base.update(kw)
    return vf.Aircraft(**base)


def test_isa_density():
    assert vf.isa_density(0.0) == pytest.approx(1.225, rel=1e-3)
    assert vf.isa_density(2400.0) == pytest.approx(0.967, rel=0.01)


def test_terrain_depot_and_site():
    t = vf.Terrain()
    assert t.depot[2] == pytest.approx(t.base, abs=10)
    assert t.site[2] == pytest.approx(t.site_altitude, abs=1)
    assert np.linalg.norm(t.wall_normal) == pytest.approx(1.0)
    # the route climbs: every waypoint is above the terrain by the clearance
    for w in vf.route(t, t.depot, t.site, 6, 120.0):
        assert w[2] - float(t.height(w[0], w[1])) == pytest.approx(120.0)


def test_power_model_scales_like_momentum_theory():
    ac = _aircraft()
    p1 = ac.power(ac.hover_thrust_n_sl, 0.0, vf.RHO0)
    assert p1 == pytest.approx(ac.hover_power_w_sl, rel=1e-6)
    assert ac.power(2 * ac.hover_thrust_n_sl, 0.0, vf.RHO0) > 2 * p1              # more than linear in thrust
    assert ac.power(ac.hover_thrust_n_sl, 0.0, 0.9) > p1                          # thinner air costs power
    assert ac.power(ac.hover_thrust_n_sl, 30.0, vf.RHO0) < p1                     # forward flight relieves the induced power


@pytest.mark.parametrize("mode", ["pad", "hand"])
def test_mission_flies_and_comes_home(mode):
    ac = _aircraft()
    ep = vf.simulate(ac, vf.Terrain(), vf.Wind(), vf.Plan(mode=mode))
    s = ep.summary()
    names = [e[1] for e in ep.events]
    assert any("capsule" in n for n in names) and any("landed at the depot" in n for n in names)
    assert not any("battery" in n for n in names), s
    assert s["energy_wh"] < ac.battery_wh
    assert s["max_speed_m_s"] <= ac.max_speed + 0.1
    assert ep.pos[-1][2] == pytest.approx(vf.Terrain().depot[2], abs=0.5)
    assert np.linalg.norm(ep.pos[-1][:2]) < 1.0
    table = ep.phase_table()
    assert {"transit out", "transit home", "landing"} <= set(table.index)
    if mode == "hand":
        assert "hand-over" in table.index and ep.hold_error_m is not None and len(ep.hold_error_m) > 0
        assert ep.min_wall_distance_m > 0.5                                     # never touched the rock


def test_wall_relative_sensing_holds_tighter():
    terrain, wind, plan = vf.Terrain(), vf.Wind(), vf.Plan(mode="hand")
    gnss = vf.touchdown_statistics(_aircraft(), terrain, wind, plan, n=3)
    rel = vf.touchdown_statistics(_aircraft(position_noise_m=0.05, response_s=0.6, hold_gain=2.0), terrain, wind, plan, n=3)
    assert rel["hold_error_max_mean_m"] < 0.5 * gnss["hold_error_max_mean_m"]
    assert rel["hold_error_max_mean_m"] < 0.3


def test_parachute_estimate():
    r = vf.parachute_descent(2.06, 2520.0, 2100.0, 1.1, vf.Wind(), 600.0)
    v_t = math.sqrt(2 * 2.06 * vf.G / (float(vf.isa_density(2520.0)) * 1.1))
    assert r["terminal_speed_m_s"] == pytest.approx(v_t)
    assert r["descent_time_s"] > 420.0 / v_t
    assert r["opening_shock_N"] > 2.06 * vf.G * 5


def test_movie_renders(tmp_path):
    pytest.importorskip("cv2")
    ep = vf.simulate(_aircraft(), vf.Terrain(), vf.Wind(), vf.Plan(mode="hand"))
    path, frames = vf.render_movie(ep, vf.Terrain(), tmp_path / "v.mp4", seconds=0.4, fps=5)
    assert path.exists() and len(frames) == 2 and frames[0].shape == (540, 960, 3)


def test_design_builds_every_part():
    pytest.importorskip("cadquery")
    from vegeta import dedalus
    import pathlib
    d = dedalus.load_design(f"{pathlib.Path(vf.__file__).with_name('velutina.py')}:Velutina")
    for part in ("aircraft", "body", "arm", "capsule", "fin"):
        g = d.generate(part=part)
        m = g.measure()
        assert m["valid"] and m["n_solids"] == 1, part
    p = d.resolve()
    assert d.generate(part="aircraft").measure()["dimensions"][0] == pytest.approx(p["body_length"] + p["handle_diameter"] / 2, abs=1)
    # nose-up rotation puts the nose higher than the tail (mean z of the nose region vs the tail region; the fins are symmetric)
    g = d.generate(angle_of_attack_deg=10.0)
    pts = np.asarray(g.tessellate(0.5)[0])
    nose_z, tail_z = pts[pts[:, 0] < 100.0, 2].mean(), pts[pts[:, 0] > 480.0, 2].mean()
    assert nose_z > tail_z + 20.0
    with pytest.raises(dedalus.BuildError):            # the design's own check (arm on the cylinder), wrapped by Dedalus
        d.generate(arm_x=100.0)


# ---------------------------------------------------------------- Velutina v2: the wind-turbine inspection
import velutina_inspection as vi  # noqa: E402


def _inspector():
    return _aircraft(mass_kg=2.3, capsule_kg=0.0, position_noise_m=0.05, response_s=0.6, hold_gain=2.0, max_speed=8.0)


def test_turbine_geometry():
    tb = vi.Turbine()
    assert tb.chord(tb.chord_max_r) == pytest.approx(tb.chord_max)
    assert tb.chord(tb.rotor_radius) == pytest.approx(tb.chord_tip)
    tip_down = tb.blade_point(0, tb.rotor_radius)
    assert tip_down[2] < tb.hub_height - 40.0 and abs(tip_down[1]) < 1.0            # blade 1 points down along the tower
    for k in range(3):
        span, chordwise, normal = tb.blade_frame(k)
        assert abs(np.dot(span, chordwise)) < 1e-9
        assert abs(np.dot(span, normal)) == pytest.approx(math.sin(math.radians(tb.cone_deg)), abs=1e-9)   # the cone leans the blade upwind
    assert tb.blade_point(1, 30.0)[2] > tb.hub_height and tb.blade_point(2, 30.0)[2] > tb.hub_height


def test_tower_shadow_and_shear():
    tb, w = vi.Turbine(), vi.SiteWind(speed=7.0, hub_height=80.0)
    free = w.mean((-40.0, 0.0, 40.0), tb)[0]
    assert free < 7.0 and free > 5.0                                              # shear: slower below hub height
    assert w.mean((-40.0, 0.0, 80.0), tb)[0] == pytest.approx(7.0)
    a = tb.tower_diameter(40.0) / 2
    assert w.mean((a + 3.0, 0.0, 40.0), tb)[0] < 0.7 * free                        # the wake deficit right behind the tower
    assert w.turbulence_factor((a + 3.0, 0.0, 40.0), tb) > 1.5 > w.turbulence_factor((-40.0, 0.0, 40.0), tb) == 1.0
    assert w.mean((0.0, a + 1.0, 40.0), tb)[0] > free                              # the flow speeds up past the side
    assert w.mean((a + 30.0, 0.0, 40.0), tb)[0] > w.mean((a + 3.0, 0.0, 40.0), tb)[0]   # the wake fills in downstream


def test_camera_numbers():
    cam = vi.Camera()
    assert cam.gsd_mm(cam.max_standoff_m()) * cam.pixels_per_crack == pytest.approx(cam.crack_width_mm)
    v, lim = cam.max_scan_speed(4.0)
    assert v == pytest.approx(min(lim.values()))
    assert cam.dwell_time_s(np.zeros(100), 0.05)[0] == pytest.approx(cam.sharp_frames_needed / cam.fps)
    assert cam.dwell_time_s(np.full(100, 1.0), 0.05)[0] == math.inf
    dwell, inside = cam.dwell_time_s(np.array([0.05] * 50 + [0.5] * 50), 0.05)
    assert inside == pytest.approx(0.5) and dwell == pytest.approx(2 * cam.sharp_frames_needed / cam.fps)


def test_inspection_path_covers_every_pass():
    tb, cam, plan = vi.Turbine(), vi.Camera(), vi.InspectionPlan(blades=(0,))
    wps, v = vi.inspection_path(tb, plan, cam, dwell_s=3.0)
    labels = {lb for _, _, _, lb in wps}
    for name, *_ in plan.passes:
        assert f"blade 1: {name}" in labels
    assert sum(1 for _, _, d, _ in wps if d > 0) == 2                              # blade 1 has two suspect points
    for p, _, _, lb in wps:
        if lb.startswith("blade 1: ") and "to " not in lb:
            r = np.linalg.norm(p - tb.hub)
            assert tb.blade_root_r < r < tb.rotor_radius + plan.standoff_m + 1.0
    assert v == pytest.approx(cam.max_scan_speed(plan.standoff_m)[0])


def test_inspection_flight_one_blade():
    tb, wind, cam = vi.Turbine(), vi.SiteWind(speed=7.0, hub_height=80.0, gust_sigma=1.5), vi.Camera()
    wps, _ = vi.inspection_path(tb, vi.InspectionPlan(blades=(0,)), cam, dwell_s=2.0)
    ep = vi.follow_path(_inspector(), tb, wind, wps, seed=0)
    s = ep.summary()
    assert not any("time limit" in e[1] for e in ep.events)
    assert sum("steady video" in e[1] for e in ep.events) == 2
    assert s["track_error_p95_m"] < 2.0 and s["energy_wh"] < 60.0
    assert ep.pos[-1][2] < 0.8                                                     # back at the van (the last waypoint's arrival radius)
    f = vi.flights_needed(ep, 151.0, n_blades=1)
    assert f["flights"] == 1 and f["energy_per_blade_wh"][0] > 0


def test_station_keeping_gives_a_finite_dwell():
    tb, wind, cam = vi.Turbine(), vi.SiteWind(speed=7.0, hub_height=80.0, gust_sigma=1.5), vi.Camera()
    point = tb.blade_point(0, 20.0, 0.5, 1.0) + vi.pass_offset(tb, 0, 0.5, 1.0, 4.0)
    err, _ = vi.station_keeping_at_blade(_inspector(), tb, wind, point, seconds=20.0, seed=1)
    dwell, inside = cam.dwell_time_s(err, 0.05)
    assert len(err) > 300 and inside > 0.5 and dwell < 10.0


def test_inspection_movie_renders(tmp_path):
    pytest.importorskip("cv2")
    tb, wind, cam = vi.Turbine(), vi.SiteWind(speed=7.0, hub_height=80.0), vi.Camera()
    wps, _ = vi.inspection_path(tb, vi.InspectionPlan(blades=(0,), passes=vi.InspectionPlan().passes[:1]), cam, dwell_s=1.0)
    ep = vi.follow_path(_inspector(), tb, wind, wps)
    path, frames = vi.render_movie(ep, tb, tmp_path / "i.mp4", seconds=0.4, fps=5)
    assert path.exists() and len(frames) == 2
