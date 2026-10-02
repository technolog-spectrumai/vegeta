"""Water in ChironLab: the medium's density and viscosity reach MuJoCo, ellipsoid fluid shapes, no buoyancy by
MuJoCo (a scene adds it), and a falling body reaches the terminal speed its drag and net weight give."""
import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from vegeta.chiron import ChironLab, Flat, Geom, Link, Robot, SimOptions  # noqa: E402

RHO, MU = 1000.0, 1.0e-3


def ball(mass=2.0, r=0.05, fluidshape="ellipsoid"):
    root = Link("ball", log=True, geoms=[Geom("ball_g", "sphere", (r,), mass=mass, role="body", fluidshape=fluidshape)])
    return Robot("ball", root, nominal_base_height=r)


def test_defaults_are_vacuum_and_options_reach_mujoco():
    lab = ChironLab(ball(), Flat())
    assert lab.model.opt.density == 0.0 and lab.model.opt.viscosity == 0.0
    wet = ChironLab(ball(), Flat(), density=RHO, viscosity=MU, wind=(0.0, 0.5, 0.0))
    assert wet.model.opt.density == RHO and wet.model.opt.viscosity == MU and tuple(wet.model.opt.wind) == (0.0, 0.5, 0.0)
    assert wet.model.geom_fluid[wet._geom_id("ball_g")][0] > 0           # the ellipsoid model is on for that geom
    assert 'fluidshape="ellipsoid"' in ball().to_mjcf(Flat(), SimOptions())
    with pytest.raises(ValueError):
        Geom("g", "sphere", (0.1,), fluidshape="blob")
    with pytest.raises(ValueError):
        SimOptions(density=-1.0)


def test_mujoco_adds_no_buoyancy():
    lab = ChironLab(ball(), Flat(), density=RHO, viscosity=MU)
    lab.reset(seed=0)
    mujoco.mj_forward(lab.model, lab.data)
    assert np.allclose(lab.data.qfrc_fluid, 0.0)                        # at rest: no fluid force at all


def test_sinking_ball_reaches_its_terminal_speed():
    """A 2 kg Ø100 mm ball, buoyancy from a hook: at the terminal speed MuJoCo's fluid force carries the net weight
    (m − ρV) g, and the speed lies between the drag estimates ½ ρ C_d A v² for C_d 0.4 and 1.2 (MuJoCo's ellipsoid
    model on a sphere behaves like C_d ≈ 1: blunt and slender terms together)."""
    m, r = 2.0, 0.05
    V, A = 4 / 3 * math.pi * r ** 3, math.pi * r * r
    lab = ChironLab(ball(m, r), Flat(), density=RHO, viscosity=MU, course_extent=(-1, 1, -1, 1))

    def buoyancy(lab):
        lab.body_force("ball", (0.0, 0.0, RHO * V * 9.81))
    lab.add_hook(buoyancy)
    ep = lab.run(None, duration=3.0, rules=None, settle=0.0, base_pos=(0.0, 0.0, 20.0))   # well above the floor
    vz = np.asarray(ep.log["com_vel"])[:, 2]
    assert abs(vz[-1] - vz[-20]) < 1e-3                                  # steady
    net = (m - RHO * V) * 9.81
    assert lab.data.qfrc_fluid[2] == pytest.approx(net, rel=0.02)
    v_lo, v_hi = (math.sqrt(2 * net / (RHO * cd * A)) for cd in (1.2, 0.4))
    assert v_lo < -vz[-1] < v_hi
