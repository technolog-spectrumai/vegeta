"""Kerchak's quad (``designs/gorilla_quad``): sized round the gorilla, stable with him on board, he fits it, the CAD
builds, the MuJoCo model compiles and stands on its springs."""
import math

import mujoco
import numpy as np
import pytest

import gorilla_quad as gq
import motion_kit as mk
import space_unit_robots as sr


@pytest.fixture(scope="module")
def kin():
    return mk.Kin(sr.bare(casing="plastic"))


def test_rider_measures_are_kerchaks(kin):
    m, d = kin.model, kin.data
    mujoco.mj_forward(m, d)
    assert sum(m.body_mass) == pytest.approx(gq.RIDER["mass"], abs=0.5)
    assert 2 * d.xpos[m.body("L_upper_arm").id][1] == pytest.approx(gq.RIDER["shoulder_width"], abs=0.01)
    assert 2 * m.geom_size[m.geom("pelvis_shell").id][1] == pytest.approx(gq.RIDER["pelvis_width"], abs=0.01)
    assert 2 * m.geom_size[m.geom("L").id][0] == pytest.approx(gq.RIDER["foot_length"], abs=0.01)


def test_stable_with_the_gorilla_on_board():
    s = gq.sizing()
    assert s["ssf"] >= gq.SSF_TARGET
    assert gq.GEOM["track"] >= s["track_for_target_m"]


def test_suspension_and_drive_numbers():
    s = gq.sizing()
    for c in gq.CORNERS:
        k, ms = s["corners"][c]["k_N_m"], s["corners"][c]["sprung_kg"]
        assert math.sqrt(k / ms) / (2 * math.pi) == pytest.approx(gq.RIDE_HZ, rel=1e-3)
        assert s["corners"][c]["sag_m"] < gq.GEOM["droop"] + gq.GEOM["bump"]
    assert s["seat"]["sag_m"] > 0 and s["seat"]["travel_m"] > 0
    theta = math.atan(gq.GRADE)
    assert 4 * s["wheel_torque_Nm"] / gq.GEOM["wheel_r"] >= s["mass_total_kg"] * gq.G * math.sin(theta)
    assert s["range_30kmh_km"] > 40


def test_kerchak_fits_across_the_steering_range(kin):
    q, err = gq.seated_pose(kin, 0.0)
    assert err < 0.01
    for sgn in (1, -1):
        qi = q
        for st in np.linspace(0.15, gq.GEOM["steer_max"], 4):
            qi, e = gq.seated_pose(kin, sgn * st, q_init=qi)
            assert e < 0.01


def test_seat_spring_carries_the_rider():
    """The seat's preload pushes up (springref above zero): a seat pulled down onto its stop is a rigid seat."""
    prop = gq.quad_prop()
    seat = next(lk for lk in prop.links() if lk.name == "quad_seat")
    j = seat.joints[0]
    assert j.springref > 0
    assert j.stiffness * j.springref == pytest.approx(gq.RIDER["mass"] * gq.G, rel=1e-2)
    for lk in prop.links():
        for jj in lk.joints:
            if jj.name.startswith("susp_"):
                assert jj.springref < 0                                    # the corner springs push the wheel down


def test_cad_parts_build():
    d = gq.GorillaQuad()
    for part in ("wheel", "handlebar", "trunk", "wishbone"):
        g = d.generate(part=part, detail=0)
        assert g.measure()["volume"] > 0
    lo, hi = d.generate(part="trunk").bounding_box
    assert hi[2] - lo[2] == pytest.approx(gq.GEOM["trunk"]["h"] * 1000 + 12, abs=15)   # an open tray: the walls and the top rails, no lid


def test_quad_model_stands_on_its_springs():
    from vegeta.chiron import SimOptions

    robot = sr.bare()
    robot.root.pos = (0.0, 0.0, 3.0)
    xml = robot.to_mjcf(options=SimOptions(), props=[gq.quad_prop(x0=5.0)])
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    for _ in range(1500):
        d.qpos[0:3] = (0.0, 0.0, 3.0)
        d.qvel[0:6] = 0.0
        mujoco.mj_step(m, d)
    z = d.qpos[m.jnt_qposadr[m.joint("quad_chassis:free").id] + 2]
    assert -0.01 < z <= gq.GEOM["droop"] + 0.01                            # without the rider the springs lift it, at most to full droop


def test_render_clips_below_the_ground():
    import kerchak_render as br

    tris = np.array([[[0, 0, -1], [1, 0, 1], [0, 1, 1]], [[0, 0, 1], [1, 0, -1], [0, 1, -1]], [[0, 0, -1], [1, 0, -1], [0, 1, -1]]], float)
    out, _ = br.clip_below_ground(tris, np.ones((3, 3)))
    assert len(out) == 3 and out[:, :, 2].min() >= 0.0                   # the crossing ones cut, the buried one gone


@pytest.mark.slow
def test_obstacles_are_solid():
    """Driven straight at the cones the quad knocks them over; on the slalom's line it leaves them standing."""
    import gorilla_quad_ride as gr

    ride = gr.Ride()
    t = 0.0
    while ride.pose()[0] < 60.0 and t < 25.0:
        ride.step(t, 0.05)
        t += 0.05
    assert all(c["moved_m"] < 0.01 and c["tilt_deg"] < 1.0 for c in ride.cones())
    path = gr.path_y
    gr.path_y = lambda x: 0.0
    try:
        ride = gr.Ride()
        t = 0.0
        while ride.pose()[0] < 31.0 and t < 15.0:
            ride.step(t, 0.05)
            t += 0.05
        assert ride.cones()[0]["tilt_deg"] > 45.0
    finally:
        gr.path_y = path
