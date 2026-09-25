"""Particle movies without OpenFOAM: an analytic stub field stands in for the solved case."""
import json
import math

import numpy as np
import pytest

from vegeta.aeromant import movie

cv2 = pytest.importorskip("cv2")

D, U0, UJ, OMEGA_SWIRL = 0.2, 2.0, 3.0, 40.0          # diameter [m], inflow, slipstream extra speed [m/s], swirl rate [1/s]


def stub_sampler(center=(0.0, 0.0, 0.0), rotation=1.0):
    """Uniform inflow along +x; behind the disc, inside the slipstream radius, faster and swirling in the
    sense of rotation; a solid hub (invalid) near the centre; invalid outside a box."""
    c = np.asarray(center, float)

    def sample(points):
        p = np.atleast_2d(points) - c
        r = np.hypot(p[:, 1], p[:, 2])
        jet = (p[:, 0] > 0) & (r < D / 2)
        u = np.zeros_like(p)
        u[:, 0] = U0 + UJ * jet
        u[:, 1] = -rotation * OMEGA_SWIRL * p[:, 2] * jet
        u[:, 2] = rotation * OMEGA_SWIRL * p[:, 1] * jet
        valid = (np.abs(p).max(axis=1) < 5 * D) & ~((np.abs(p[:, 0]) < 0.02) & (r < 0.01))
        return u, valid

    return sample


def rotor(**kw):
    return movie.RotorView(**{"center": (0.0, 0.0, 0.0), "diameter": D, "rpm": 6000.0, "blades": 3, **kw})


def test_rotor_view_and_time_step():
    r = rotor()
    assert r.omega == pytest.approx(6000 * 2 * math.pi / 60) and rotor(rotation=-1).omega < 0
    assert movie.time_step(r, 10.0) == pytest.approx(10.0 / (360 * 100))
    with pytest.raises(ValueError):
        movie.time_step(r, 0)
    with pytest.raises(ValueError):
        rotor(rotation=2)


def test_rotor_view_from_case_record(tmp_path):
    cfg = {"template": "rotor_mrf", "parameters": {"rpm": 8000.0, "diameter": 0.127, "rotation": -1, "airspeed": 14.0,
                                                   "center": [0.1, 0, 0]}}
    (tmp_path / "aeromant_case.json").write_text(json.dumps({"config": cfg}))
    r = movie.RotorView.from_case(tmp_path, blades=2)
    assert (r.rpm, r.diameter, r.rotation, r.inflow, r.center, r.blades) == (8000.0, 0.127, -1.0, 14.0, (0.1, 0.0, 0.0), 2)
    cfg = {"template": "rotor_mrf_static", "parameters": {"rpm": 6000.0, "diameter": 0.1, "inflow_fraction": 0.02}}
    (tmp_path / "aeromant_case.json").write_text(json.dumps({"config": cfg}))
    r = movie.RotorView.from_case(tmp_path, blades=3)
    assert r.inflow == pytest.approx(0.02 * 6000 * 2 * math.pi / 60 * 0.05) and r.rotation == 1.0 and r.center == (0.0, 0.0, 0.0)
    (tmp_path / "aeromant_case.json").write_text(json.dumps({"config": {"template": "rans_ksst_external", "parameters": {}}}))
    with pytest.raises(ValueError, match="rotor cases"):
        movie.RotorView.from_case(tmp_path, blades=2)
    with pytest.raises(FileNotFoundError):
        movie.RotorView.from_case(tmp_path / "nope", blades=2)


def test_field_sampler_interpolates_and_flags_points_outside():
    g = np.linspace(0, 1, 11)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    centres = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    u = np.column_stack([centres[:, 0], np.zeros(len(centres)), np.ones(len(centres))])   # u = (x, 0, 1)
    s = movie.field_sampler(centres, u, np.full(len(centres), 0.1))
    vel, ok = s(np.array([[0.5, 0.5, 0.5], [0.52, 0.31, 0.77], [3.0, 3.0, 3.0]]))
    assert ok.tolist() == [True, True, False]
    assert vel[0] == pytest.approx([0.5, 0.0, 1.0]) and vel[1, 0] == pytest.approx(0.52, abs=0.03) and vel[1, 2] == pytest.approx(1.0)


def test_tracer_advects_swirls_and_reseeds():
    r = rotor()
    tr = movie.Tracer(stub_sampler(), r, n=30, seed=1, trail=5, fill=False)
    x_start = tr.pos[:, 0].copy()
    assert (x_start <= -0.2 * D + 1e-12).all() and (x_start >= -1.2 * D - 1e-12).all()          # fed upstream of the disc
    dt = 1e-3
    tr.step(dt)
    assert tr.pos[:, 0] - x_start == pytest.approx(np.full(30, U0 * dt))                           # uniform inflow upstream
    assert tr.trails.shape == (5, 30, 3) and np.allclose(tr.trails[-1], tr.pos)
    # run until particles cross the disc; those in the slipstream turn in the sense of rotation
    for _ in range(200):
        prev = tr.pos.copy()
        tr.step(dt)
        rel, relp = tr.pos, prev
        inside = (relp[:, 0] > 0.01) & (np.hypot(relp[:, 1], relp[:, 2]) < 0.4 * D) & (np.abs(tr.pos - prev).max(axis=1) < 0.1)
        if inside.any():
            a0 = np.arctan2(relp[inside, 2], relp[inside, 1])
            a1 = np.arctan2(rel[inside, 2], rel[inside, 1])
            assert (np.angle(np.exp(1j * (a1 - a0))) > 0).all()                                   # counter-clockwise about +x
            break
    else:
        pytest.fail("no particle reached the slipstream")
    # particles leaving the view downstream are re-seeded upstream
    for _ in range(400):
        tr.step(2e-3)
    assert tr.reseeded > 0 and (tr.pos[:, 0] <= 3.0 * D + 1e-9).all()


def test_stalled_particles_are_reseeded():
    def dead_water(points):
        p = np.atleast_2d(points)
        return np.zeros_like(p), np.ones(len(p), bool)

    tr = movie.Tracer(dead_water, rotor(), n=6, seed=3, stall_speed=0.1, stall_steps=3)
    for _ in range(3):
        tr.step(1e-3)
    assert tr.reseeded == 0
    tr.step(1e-3)
    assert tr.reseeded == 6 and (tr.pos[:, 0] <= -0.2 * D + 1e-12).all()


def test_render_frame_draws_particles_and_turning_blades():
    tr = movie.Tracer(stub_sampler(), rotor(), n=10, seed=2)
    a = movie.render_frame(tr, 0.0, size=(640, 270), speed_max=5.0, time_s=0.0)
    b = movie.render_frame(tr, 0.7, size=(640, 270), speed_max=5.0, time_s=0.0)
    assert a.shape == (270, 640, 3) and a.dtype == np.uint8
    assert (a != 250).any(axis=2).sum() > 500                              # something is drawn
    assert (a != b).any()                                                   # the blades moved
    blank = movie.render_frame(movie.Tracer(stub_sampler(), rotor(), n=1, seed=2), 0.0, size=(640, 270), speed_max=5.0)
    assert (a != 250).any(axis=2).sum() > (blank != 250).any(axis=2).sum()   # more particles, more ink


def test_make_movie_writes_video_with_stubs(tmp_path):
    out = movie.make_movie(None, tmp_path / "m" / "prop.mp4", rotor=rotor(), sampler=stub_sampler(), n=12, seconds=1.0, fps=8,
                           size=(480, 200), title="stub rotor")
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 8
    ok, img = cap.read()
    assert ok and img.shape == (200, 480, 3)


def test_make_movie_reads_the_case_record_when_no_rotor_is_given(tmp_path):
    cfg = {"template": "rotor_mrf", "parameters": {"rpm": 6000.0, "diameter": D, "airspeed": U0}}
    (tmp_path / "aeromant_case.json").write_text(json.dumps({"config": cfg}))
    out = movie.make_movie(tmp_path, tmp_path / "prop.mp4", blades=2, sampler=stub_sampler(), n=5, seconds=0.5, fps=4, size=(320, 160))
    assert int(cv2.VideoCapture(str(out)).get(cv2.CAP_PROP_FRAME_COUNT)) == 2


def test_openfoam_sampler_needs_a_solved_case(tmp_path):
    pytest.importorskip("pyvista")
    with pytest.raises(FileNotFoundError, match="no time directories"):
        movie.openfoam_sampler(tmp_path)


def test_concat_videos_joins_in_order(tmp_path):
    a = movie.make_movie(None, tmp_path / "a.mp4", rotor=rotor(), sampler=stub_sampler(), n=4, seconds=0.5, fps=8, size=(320, 160))
    b = movie.make_movie(None, tmp_path / "b.mp4", rotor=rotor(rpm=3000.0), sampler=stub_sampler(), n=4, seconds=0.25, fps=8, size=(480, 200))
    out = movie.concat_videos([a, b], tmp_path / "ab.mp4", captions=["slow", "fast"])
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 4 + 2
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (320, 160)
    with pytest.raises(ValueError):
        movie.concat_videos([], tmp_path / "none.mp4")
