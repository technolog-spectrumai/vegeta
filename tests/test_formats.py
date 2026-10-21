"""Yax 2's formats, checked by the engines that use them: MuJoCo compiles what Yax writes and gets the same bodies and
joints; JSBSim loads a converted aircraft and gives the same mass, inertia and aerodynamics; round trips through USD
are exact; every conversion reports what the target cannot hold."""
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

import yax
from yax import hub
from helpers import assert_same

HERE = Path(__file__).parent
ROVER = HERE / "fixtures" / "rover.xml"
mujoco = pytest.importorskip("mujoco")


def _compiled(path):
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    return m, d


def _same_mujoco(a_path, b_path, *, actuators=True, free_name=True):
    a, da = _compiled(a_path)
    b, db = _compiled(b_path)
    for i in range(1, a.nbody):
        n = mujoco.mj_id2name(a, mujoco.mjtObj.mjOBJ_BODY, i)
        j = mujoco.mj_name2id(b, mujoco.mjtObj.mjOBJ_BODY, n)
        assert a.body_mass[i] == pytest.approx(b.body_mass[j], rel=1e-12), n
        np.testing.assert_allclose(np.sort(b.body_inertia[j]), np.sort(a.body_inertia[i]), rtol=1e-9, err_msg=n)
        np.testing.assert_allclose(db.xpos[j], da.xpos[i], atol=1e-12, err_msg=n)
        np.testing.assert_allclose(db.xipos[j], da.xipos[i], atol=1e-12, err_msg=n)
    for i in range(a.njnt):
        n = mujoco.mj_id2name(a, mujoco.mjtObj.mjOBJ_JOINT, i)
        j = mujoco.mj_name2id(b, mujoco.mjtObj.mjOBJ_JOINT, n)
        if j < 0 and a.jnt_type[i] == 0 and not free_name:
            continue
        assert a.jnt_type[i] == b.jnt_type[j], n
        np.testing.assert_allclose(db.xanchor[j], da.xanchor[i], atol=1e-12, err_msg=n)
        np.testing.assert_allclose(db.xaxis[j], da.xaxis[i], atol=1e-12, err_msg=n)
        np.testing.assert_allclose(b.jnt_range[j], a.jnt_range[i], atol=1e-12, err_msg=n)
    if actuators:
        assert a.nu == b.nu and np.allclose(a.actuator_gainprm[:, 0], b.actuator_gainprm[:, 0])
    for g in range(a.ngeom):
        if a.geom_bodyid[g] == 0:
            continue
        n = mujoco.mj_id2name(a, mujoco.mjtObj.mjOBJ_GEOM, g)
        h = mujoco.mj_name2id(b, mujoco.mjtObj.mjOBJ_GEOM, n)
        assert a.geom_type[g] == b.geom_type[h], n
        np.testing.assert_allclose(db.geom_xpos[h], da.geom_xpos[g], atol=1e-9, err_msg=n)
        if a.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            np.testing.assert_allclose(b.geom_size[h], a.geom_size[g], atol=1e-12, err_msg=n)


def test_mjcf_through_usd_is_exact(tmp_path):
    rep = yax.convert(ROVER, tmp_path / "rover.xml")
    assert rep.dropped == ["reading: geom floor: a world geom (terrain/scenery)"]
    _same_mujoco(ROVER, tmp_path / "rover.xml")


def test_usd_file_round_trip(tmp_path):
    m, _ = yax.read(ROVER)
    yax.save(m, tmp_path / "rover.usda")
    back, _ = yax.read(tmp_path / "rover.usda")
    assert_same(m, back, tol=1e-12)
    from pxr import Usd, UsdGeom, UsdPhysics
    st = Usd.Stage.Open(str(tmp_path / "rover.usda"))
    assert UsdGeom.GetStageUpAxis(st) == "Z" and UsdGeom.GetStageMetersPerUnit(st) == 1.0
    assert sum(1 for p in st.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)) == 5
    assert sum(1 for p in st.Traverse() if p.IsA(UsdPhysics.Joint)) == 4


def test_urdf_round_trip_and_report(tmp_path):
    rep = yax.convert(ROVER, tmp_path / "rover.urdf")
    text = "\n".join(rep.dropped)
    assert "spherical" in text and "ellipsoid" in text and "capsule" in text and "drive" in text
    m, _ = yax.read(ROVER)
    back, _ = yax.read(tmp_path / "rover.urdf")
    assert_same(m, back, kinds={"spherical": "fixed"})
    assert (tmp_path / "rover_meshes").is_dir()
    yax.convert(tmp_path / "rover.urdf", tmp_path / "from_urdf.xml")
    _compiled(tmp_path / "from_urdf.xml")


def test_sdf_round_trip_keeps_poses(tmp_path):
    rep = yax.convert(ROVER, tmp_path / "rover.sdf")
    assert any("drive" in d for d in rep.dropped)
    m, _ = yax.read(ROVER)
    back, _ = yax.read(tmp_path / "rover.sdf")
    assert_same(m, back)
    for a, b in zip(m.links, back.links):
        np.testing.assert_allclose(b.pos, a.pos, atol=1e-12)
    yax.convert(tmp_path / "rover.sdf", tmp_path / "from_sdf.xml")
    _same_mujoco(ROVER, tmp_path / "from_sdf.xml", actuators=False, free_name=False)


def test_urdf_to_sdf_to_urdf(tmp_path):
    yax.convert(ROVER, tmp_path / "a.urdf")
    yax.convert(tmp_path / "a.urdf", tmp_path / "b.sdf")
    yax.convert(tmp_path / "b.sdf", tmp_path / "c.urdf")
    a, _ = yax.read(tmp_path / "a.urdf")
    c, _ = yax.read(tmp_path / "c.urdf")
    assert_same(a, c)


def test_step_volumes_and_placement(tmp_path):
    pytest.importorskip("cadquery")
    rep = yax.convert(ROVER, tmp_path / "rover.step")
    assert any("mass" in d for d in rep.dropped) and any("joints" in d for d in rep.dropped)
    m, _ = yax.read(tmp_path / "rover.step", density_kg_m3=1000.0)
    src, _ = yax.read(ROVER)
    V = 0.0
    for lk in src.links:
        for s in lk.shapes:
            if s.kind == "box":
                V += np.prod(s.size)
            elif s.kind == "sphere":
                V += 4 / 3 * math.pi * s.size[0] ** 3
            elif s.kind == "ellipsoid":
                V += 4 / 3 * math.pi * np.prod(s.size)
            elif s.kind == "cylinder":
                V += math.pi * s.size[0] ** 2 * s.size[1]
            elif s.kind == "capsule":
                V += math.pi * s.size[0] ** 2 * s.size[1] + 4 / 3 * math.pi * s.size[0] ** 3
            elif s.kind == "mesh":
                p = hub.tessellate(s)[0][np.asarray(s.faces)]
                V += abs(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum()) / 6
    assert m.links[0].mass == pytest.approx(1000.0 * V, rel=1e-6)
    lo = np.min([np.min(s.points, axis=0) for s in m.links[0].shapes], axis=0)
    hi = np.max([np.max(s.points, axis=0) for s in m.links[0].shapes], axis=0)
    allp = np.vstack([(hub.R(lk.quat) @ hub.tessellate(s, 64)[0].T).T + lk.pos for lk in src.links for s in lk.shapes])
    np.testing.assert_allclose(lo, allp.min(0), atol=2e-3)
    np.testing.assert_allclose(hi, allp.max(0), atol=2e-3)


def test_stl_and_yax1_package(tmp_path):
    yax.convert(ROVER, tmp_path / "rover.stl")
    m, rep = yax.read(tmp_path / "rover.stl")
    assert len(m.links) == 1 and any("mass" in d for d in rep.dropped)
    cube, _ = yax.read(HERE.parent / "examples" / "cube.yax")
    d = yax.load(HERE.parent / "examples" / "cube.yax")
    assert cube.links[0].mass == d.mass["kg"]
    np.testing.assert_allclose(cube.links[0].inertia, d.mass["inertia"])
    yax.save(cube, tmp_path / "cube.usda")
    back, _ = yax.read(tmp_path / "cube.usda")
    assert back.links[0].mass == d.mass["kg"]


def test_jsbsim_aircraft_flies_the_same(tmp_path):
    jsbsim = pytest.importorskip("jsbsim")
    R = Path(jsbsim.get_default_root_dir())
    root = tmp_path / "root"
    for d in ("engine", "systems"):
        (root / d).mkdir(parents=True)
    rep = yax.convert(R / "aircraft" / "c172x" / "c172x.xml", root / "aircraft" / "c172y" / "c172y.xml", dst_fmt="jsbsim")
    assert rep.dropped == []

    def run(r, name):
        f = jsbsim.FGFDMExec(str(r))
        f.set_debug_level(0)
        f.load_model(name)
        for i in range(2):
            f.set_property_value(f"propulsion/tank[{i}]/contents-lbs", 0.0)
        for k, v in {"ic/h-sl-ft": 3000, "ic/vc-kts": 100, "ic/alpha-deg": 4, "ic/beta-deg": 2, "ic/p-rad_sec": 0.1}.items():
            f.set_property_value(k, v)
        f.run_ic()
        keys = [f"inertia/{k}" for k in ("mass-slugs", "cg-x-in", "cg-y-in", "cg-z-in", "ixx-slugs_ft2", "iyy-slugs_ft2", "izz-slugs_ft2",
                                         "ixz-slugs_ft2", "ixy-slugs_ft2", "iyz-slugs_ft2")] + \
               [f"forces/fb{a}-aero-lbs" for a in "xyz"] + [f"moments/{a}-aero-lbsft" for a in "lmn"]
        return {k: f.get_property_value(k) for k in keys}

    a, b = run(R, "c172x"), run(root, "c172y")
    for k in a:
        assert b[k] == pytest.approx(a[k], rel=1e-12, abs=1e-9), k


def test_jsbsim_mass_balance_from_a_robot(tmp_path):
    pytest.importorskip("jsbsim")
    rep = yax.convert(ROVER, tmp_path / "rover_fdm.xml", dst_fmt="jsbsim")
    assert any("one rigid body" in d for d in rep.dropped)
    m, _ = yax.read(tmp_path / "rover_fdm.xml")
    src, _ = yax.read(ROVER)
    assert m.links[0].mass == pytest.approx(src.total_mass(), rel=1e-12)


def test_detect_and_cli(tmp_path, capsys):
    from yax import cli
    assert yax.detect(ROVER) == "mjcf" and yax.detect(tmp_path / "x.urdf", writing=True) == "urdf"
    assert cli.main(["convert", str(ROVER), str(tmp_path / "r.sdf")]) == 0
    assert cli.main(["show", str(tmp_path / "r.sdf")]) == 0
    out = capsys.readouterr().out
    assert "5 links, 4 joints" in out


def test_repeated_shape_names(tmp_path):
    """Gazebo models name a link's collision and visual alike, and every rotor's shape alike (Piccolo's drako_drone):
    USD prims and MJCF geoms stay unique, the names are kept."""
    m, _ = yax.read(HERE / "fixtures" / "twins.sdf")
    yax.save(m, tmp_path / "twins.usda")
    back, _ = yax.read(tmp_path / "twins.usda")
    assert [s.name for s in back.link("rotor_0").shapes] == ["disk", "disk"]
    yax.convert(HERE / "fixtures" / "twins.sdf", tmp_path / "twins.xml")
    model, _ = _compiled(tmp_path / "twins.xml")
    assert model.body_subtreemass[1] == pytest.approx(1.1) and model.ngeom == 5


def test_gazebo_plugins_and_sensors_survive_sdf_and_are_reported_elsewhere(tmp_path):
    """A Gazebo model's plugins (motor models, controllers) and sensors (cameras) go through USD verbatim and come back
    in SDF; formats that cannot hold them (MJCF, URDF) report them."""
    import xml.etree.ElementTree as ET
    src = HERE / "fixtures" / "twins.sdf"
    assert yax.convert(src, tmp_path / "twins.usda").dropped == []
    rep = yax.convert(tmp_path / "twins.usda", tmp_path / "back.sdf")
    assert rep.dropped == []
    model = ET.parse(tmp_path / "back.sdf").getroot().find("model")
    plug = model.find("plugin")
    assert plug.get("name") == "gz::sim::systems::MulticopterMotorModel" and plug.find("motorConstant").text == "8.54858e-06"
    cam = model.find("link[@name='base']/sensor")
    assert cam.get("type") == "camera" and cam.find("topic").text == "camera/image_raw"
    rep = yax.convert(src, tmp_path / "twins.xml")
    assert any("machine data 'sdf'" in d for d in rep.dropped) and any("link base: sdf data (1 sensor)" in d for d in rep.dropped)
