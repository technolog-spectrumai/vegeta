"""The Pinna launch clips' MuJoCo models (labs/marine/sims/pinna_launch): each builds and steps; the locked Gannet
rides its pad and climbs once released; the catapult gives the card's launch speed; the davit lifts the Walrus;
the Fulmar flies its approach. No rendering (that needs OpenGL): ``python -m pinna_launch`` makes the videos."""
import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from pinna_launch import world  # noqa: E402
from pinna_launch.fulmar_sim import FulmarLaunchAgent, FulmarNetAgent  # noqa: E402
from pinna_launch.gannet_sim import GannetAgent  # noqa: E402
from pinna_launch.walrus_sim import WalrusAgent  # noqa: E402


def _finite(sim):
    assert np.all(np.isfinite(sim.data.qpos)) and np.all(np.isfinite(sim.data.qvel))


def test_gannet_locked_then_climbs():
    g = GannetAgent()
    sim = world.Sim([g], use_meshes=False)
    sim.run(1.2)                                               # locked: it rides the heaving, rolling pad
    p, _, _, _ = g.state()
    pad, _ = sim.carrier_point(g.pad_local)
    assert np.linalg.norm(p - pad) < 0.03 and abs(sim.sea.state(1.2)[0][0]) > 0.2
    sim.run(6.0)
    p, v, R, _ = g.state()
    assert not g.locked and p[2] > pad[2] + 5.0 and v[2] > 1.0 and R[2, 2] > math.cos(math.radians(15))
    _finite(sim)


def test_catapult_launch_speed():
    f = FulmarLaunchAgent(t_fire=0.5)
    sim = world.Sim([f], use_meshes=False)
    sim.run(f.t_fire + f.t_stroke + 1.0)
    assert f.released and abs(f.v_release - f.c["v_launch"]) < 0.5           # 18 m/s at the end of the 2.76 m stroke
    assert abs(f.a_cat / world.G - f.c["p"]["catapult_g"]) < 0.2
    p, v, _, _ = f.state()
    assert p[2] > world.geometry()["rail_top"] and f.info["V"] > f.c["v_stall"]
    _finite(sim)


def test_davit_lifts_walrus():
    w = WalrusAgent()
    sim = world.Sim([w], use_meshes=False)
    z0 = w.state()[0][2]
    sim.run(3.0)
    com = w.state()[0]
    assert com[2] > z0 + 0.25 and w.released is None
    weight = w.c["mass"] * world.G
    assert 0.5 * weight < w.tension < 2.0 * weight                           # it hangs on the cable
    _finite(sim)


def test_fulmar_approach_flies():
    f = FulmarNetAgent()
    sim = world.Sim([f], use_meshes=False)
    sim.run(3.0)
    p, v, _, _ = f.state()
    assert f.caught is None and p[2] > 5.0 and f.info["V"] > f.c["v_stall"] and v[0] > 5.0
    _finite(sim)


def test_all_with_cad_meshes():
    pytest.importorskip("cadquery")
    sim = world.Sim([GannetAgent(), FulmarLaunchAgent(), WalrusAgent()], use_meshes=True)
    assert sim.model.nmesh >= 20
    sim.run(0.2)
    _finite(sim)


def _hook_residual(compensated):
    w = WalrusAgent(compensated=compensated)
    sim = world.Sim([w], sea=world.Sea.heavy(), use_meshes=False)
    sim.run(14.0)
    _finite(sim)
    return w.residual(t0=9.0, t1=13.5)


def test_compensated_davit_holds_the_hook():
    """In a heavy sea (heave +/-1 m, roll +/-5 deg) the compensated davit's hook stays within centimetres in the
    earth frame while the deck at the davit moves tens of centimetres; without compensation the hook follows it."""
    on, off = _hook_residual(True), _hook_residual(False)
    assert on["deck_at_davit"]["vertical_rms_m"] > 0.3
    assert on["hook"]["vertical_rms_m"] < 0.08 and on["sheave"]["horizontal_rms_m"] < 0.08
    assert off["hook"]["vertical_rms_m"] > 4 * on["hook"]["vertical_rms_m"]
    assert off["hook"]["horizontal_rms_m"] > 2 * on["hook"]["horizontal_rms_m"]


def test_oil_platform_scene_builds():
    from pinna_launch import clips
    sim, ag = clips.build("oil_platform_deploy", use_meshes=False)
    _, g, sq = ag
    sim.run(2.0)
    _finite(sim)
    assert g.locked and not g.crate and sq.name == "squilla" and sim.data.qpos[0] > 0.5          # Pinna under way, Gannet locked
    assert abs(g.c["mass"] - g.c["s"]["empty_kg"]) < 1e-9 and sq.c["mass"] > 100
