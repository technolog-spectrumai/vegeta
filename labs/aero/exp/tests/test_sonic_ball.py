"""The ball in the blowdown pipe (notebook 51): the mesh cut at the ball, the moving zones and their bounding planes,
the 0-D gun model's trends (no OpenFOAM)."""
import pytest

import sonic_ball as sb
import sonic_flow as sf


def test_mesh_leaves_a_gap_for_the_ball(tmp_path):
    rig, ball = sf.Rig(), sb.Ball()
    zones = {b[-1] for b in sb.blocks(rig, sf.Mesh(), ball)}
    assert {"pipeBehindBall", "pipeAheadOfBall"} <= zones
    behind = [b for b in sb.blocks(rig, sf.Mesh(), ball) if b[-1] == "pipeBehindBall"]
    ahead = [b for b in sb.blocks(rig, sf.Mesh(), ball) if b[-1] == "pipeAheadOfBall"]
    assert max(b[1] for b in behind) == ball.rear_x and ahead[0][0] == ball.rear_x + ball.diameter
    c = sb.write_case(tmp_path / "c", sf.AIR)
    dyn = (c / "constant/dynamicMeshDict").read_text()
    assert "displacementLayeredMotion" in dyn and "pipeInlet" in dyn and "exitPlane" in dyn
    pd = (c / "0/pointDisplacement").read_text()
    assert "sixDoFRigidBodyDisplacement" in pd and f"{ball.mass * 5 / 360:.9g}" in pd
    assert "ballRear" in (c / "system/topoSetDict").read_text()


def test_gun_model_trends():
    small = sb.gun(sf.AIR, *sb.bore_setup(5.0)[:1], ball=sb.bore_setup(5.0)[1], friction=0.0)
    big = sb.gun(sf.AIR, *sb.bore_setup(20.0)[:1], ball=sb.bore_setup(20.0)[1], friction=0.0)
    assert small["u"][-1] > big["u"][-1]                                  # a ~ A p / m ~ 1/d
    base = sb.gun(sf.AIR, friction=0.0)
    assert sb.gun(sf.AIR, friction=50.0)["u"][-1] < base["u"][-1]
    assert base["force"].max() == pytest.approx((50e5 - 101325) * sf.Rig().exit_area_m2, rel=0.03)
