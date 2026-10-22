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


def test_sync_takes_only_the_mapped_physics(tmp_path):
    """yax.sync: a design (USD with part data) updates a Gazebo model's masses, poses and plugin constants; sensors,
    plugins, comments and every unmapped value stay; check mode reports drift before and nothing after."""
    import xml.etree.ElementTree as ET
    from yax.hub import Joint, Link, Machine
    target = tmp_path / "twins.sdf"
    target.write_text("<?xml version=\"1.0\"?>\n<!-- the team's header comment -->\n" + (HERE / "fixtures" / "twins.sdf").read_text().split("?>", 1)[1])
    d = Machine("quad", links=[Link("frame", pos=(1.0, 2.0, 0.5), mass=1.4, com=(0.01, 0, 0), inertia=np.diag([0.02, 0.03, 0.04])),
                                Link("prop_a", pos=(1.12, 2.12, 0.53), mass=0.008, inertia=np.diag([1e-6, 1e-6, 2e-6]))],
                joints=[Joint("spin_a", "continuous", "frame", "prop_a", pos=(1.12, 2.12, 0.53), axis=(0, 0, 1))],
                data={"ogma": {"rotors": [{"kF": 1.2e-5, "pos": [1.12, 2.12, 0.53]}]}})
    yax.save(d, tmp_path / "quad.usda")
    mp = {"design_root": "frame", "target_root": "base",
          "links": {"frame": "base", "prop_a": "rotor_0"}, "joints": {"spin_a": "r0"}, "take": ["mass", "poses", "axes"],
          "values": [{"target": "plugin[jointName='r0']/motorConstant", "from": "data.ogma.rotors.0.kF"}]}
    before = yax.sync(tmp_path / "quad.usda", target, mapping=mp, check=True)
    assert any("link base mass: 1.0 → 1.4" in c for c in before.changes) and any("motorConstant" in c for c in before.changes)
    assert target.read_text().count("1.2e-05") == 0                         # check mode writes nothing
    rep = yax.sync(tmp_path / "quad.usda", target, mapping=mp)
    assert rep.changes == before.changes and rep.dropped == []
    assert yax.sync(tmp_path / "quad.usda", target, mapping=mp, check=True).changes == []
    text = target.read_text()
    assert "<!-- the team's header comment -->" in text
    model = ET.fromstring(text).find("model")
    assert model.find("plugin/motorConstant").text == "1.2e-05"
    assert model.find("link[@name='base']/sensor/topic").text == "camera/image_raw"            # behaviour untouched
    assert model.find("link[@name='rotor_1']/inertial/mass").text.strip() == "0.05"           # unmapped untouched
    pose = [float(x) for x in model.find("link[@name='rotor_0']/pose").text.split()]
    np.testing.assert_allclose(pose[:3], [0.12, 0.12, 0.03], atol=1e-12)                       # relative to the root
    m, _ = yax.read(target)
    assert m.link("base").mass == 1.4 and m.link("rotor_0").mass == 0.008


def test_lump_keeps_the_rigid_body():
    """hub.lump: the rover as one rigid link has the compiled whole body's mass, centre and inertia at q = 0."""
    m, _ = yax.read(ROVER)
    one = hub.lump(m)
    assert len(one.links) == 1 and not one.joints and len(one.links[0].shapes) == sum(len(lk.shapes) for lk in m.links)
    model, data = _compiled(ROVER)
    mujoco.mj_forward(model, data)
    assert one.links[0].mass == pytest.approx(model.body_subtreemass[1], rel=1e-12)
    com = np.asarray(one.links[0].pos) + hub.R(one.links[0].quat) @ np.asarray(one.links[0].com)
    np.testing.assert_allclose(com, data.subtree_com[1], atol=1e-12)


# ------------------------------------------------------------------------------------------ propulsion part data
def _propulsion(axis=(1.0, 0.0, 0.0), spin=1, *, grid=False, power=True):
    """A rotor at x = 0.3 m with a synthetic propeller: CT = 0.10 − 0.08 J, CP = 0.05 − 0.02 J, D = 0.254 m."""
    J = np.linspace(0.0, 1.2, 13)
    prop = {"blades": 2, "diameter_m": 0.254, "inertia_kgm2": 2e-5}
    if grid:                                                                   # the same law on an rpm × airspeed grid
        rpm, V, D, rho = np.array([6000.0, 9000.0, 12000.0]), np.array([0.0, 5.08, 10.16, 15.24]), 0.254, 1.225
        n = (rpm / 60)[:, None]
        Jg = V[None, :] / (n * D)
        prop["grid"] = {"rpm": rpm.tolist(), "airspeed_m_s": V.tolist(), "rho": rho,
                        "thrust_n": ((0.10 - 0.08 * Jg) * rho * n ** 2 * D ** 4).tolist(),
                        "torque_nm": ((0.05 - 0.02 * Jg) * rho * n ** 3 * D ** 5 / (2 * np.pi * n)).tolist()}
    else:
        prop["coefficients"] = {"J": J.tolist(), "CT": (0.10 - 0.08 * J).tolist(), "CP": (0.05 - 0.02 * J).tolist()}
    return {"rotors": [{"name": "r0", "pos": [0.3, 0.0, 0.0], "axis": list(axis), "spin": spin, "propeller": "p", "motor": "m"}],
            "propellers": {"p": prop}, "motors": {"m": {"power_w": 500.0} if power else {"kv": 2400}}}


def _plane(name, blk):
    return hub.Machine(name, links=[hub.Link("airframe", mass=2.0, inertia=np.diag([0.05, 0.08, 0.1]))], data={"propulsion": blk})


def _fly(root, name, *, vt_kts, steps=12000):
    import jsbsim
    f = jsbsim.FGFDMExec(str(root))
    f.set_debug_level(0)
    assert f.load_model(name)
    f.set_dt(1e-4)                                                             # a small propeller spins up fast
    f.set_property_value("ic/h-sl-ft", 0.0)
    f.set_property_value("ic/vt-kts", vt_kts)
    f.set_property_value("fcs/throttle-cmd-norm", 1.0)
    f.run_ic()
    for _ in range(steps):
        f.run()
    g = f.get_property_value
    return {"rpm": g("propulsion/engine/propeller-rpm"), "J": g("propulsion/engine/advance-ratio"),
            "thrust_n": g("propulsion/engine/thrust-lbs") * 4.4482216152605,
            "power_w": g("propulsion/engine/propeller-power-ftlbps") * 1.3558179483314004,
            "rho": g("atmosphere/rho-slugs_ft3") * 515.378818,
            "moments_nm": [g(f"moments/{a}-prop-lbsft") * 1.3558179483314004 for a in "lmn"]}


def test_jsbsim_propeller_from_part_data(tmp_path):
    """A propeller map in the propulsion part data becomes a JSBSim propeller file (C_THRUST, C_POWER against J) on
    an electric engine: JSBSim's thrust and power at the settled rpm are the table's CT ρ n² D⁴ and CP ρ n³ D⁵, and
    the reaction torque on the airframe is along +axis for spin +1 (sense = −spin)."""
    pytest.importorskip("jsbsim")
    root = tmp_path / "root"
    for d in ("engine", "systems"):
        (root / d).mkdir(parents=True)
    yax.save(_plane("tractor", _propulsion()), tmp_path / "tractor.usda")   # the block rides through USD untouched
    m, _ = yax.read(tmp_path / "tractor.usda")
    assert m.data["propulsion"] == _propulsion()
    rep = yax.save(m, root / "aircraft" / "tractor" / "tractor.xml", fmt="jsbsim")
    assert not any("propulsion" in d for d in rep.dropped) and any("aerodynamics, ground reactions" in d for d in rep.dropped)
    assert (root / "aircraft" / "tractor" / "Engines" / "prop_p.xml").exists()
    r = _fly(root, "tractor", vt_kts=30)
    n, D = r["rpm"] / 60, 0.254
    assert 0.3 < r["J"] < 1.0
    assert r["thrust_n"] == pytest.approx((0.10 - 0.08 * r["J"]) * r["rho"] * n ** 2 * D ** 4, rel=1e-4)
    assert r["power_w"] == pytest.approx((0.05 - 0.02 * r["J"]) * r["rho"] * n ** 3 * D ** 5, rel=1e-4)
    assert r["power_w"] == pytest.approx(500.0, rel=1e-2)                     # the electric motor's power, settled
    assert r["moments_nm"][0] > 0                                              # drag torque on the frame along +x
    for spin in (1, -1):                                                       # a rotor pointing up: yaw moment
        name = f"rotor_{spin:+d}"
        yax.save(_plane(name, _propulsion(axis=(0.0, 0.0, 1.0), spin=spin)), root / "aircraft" / name / f"{name}.xml", fmt="jsbsim")
        r = _fly(root, name, vt_kts=0, steps=6000)
        assert r["thrust_n"] > 10 and r["moments_nm"][2] * spin < 0           # JSBSim's n is about z down: +axis = −n
        assert r["moments_nm"][1] == pytest.approx(0.3 * r["thrust_n"], rel=2e-2)   # the thrust at x = 0.3 m pitches


def test_jsbsim_propeller_grid_collapses_to_coefficients(tmp_path):
    """Without coefficients, the rpm × airspeed grid is collapsed onto the advance ratio (reported), and a motor
    without a power takes the map's maximum shaft power (reported)."""
    from yax import propulsion
    blk = _propulsion(grid=True, power=False)
    co, note = propulsion.coefficients(blk["propellers"]["p"])
    J = np.asarray(co["J"])
    np.testing.assert_allclose(co["CT"], 0.10 - 0.08 * J, atol=1e-12)
    np.testing.assert_allclose(co["CP"], 0.05 - 0.02 * J, atol=1e-12)
    assert "collapsed" in note and len(J) == 8                                 # 3 × 4 points, 8 distinct J (J = 0 thrice, 0.2 and 0.4 twice)
    rep = yax.save(_plane("g", blk), tmp_path / "g" / "g.xml", fmt="jsbsim")
    assert any("collapsed to CT(J), CP(J)" in d for d in rep.dropped) and any("motor power assumed" in d for d in rep.dropped)
    text = (tmp_path / "g" / "Engines" / "prop_p.xml").read_text()
    assert 'name="C_THRUST"' in text and 'name="C_POWER"' in text and "<numblades>2</numblades>" in text
    with pytest.raises(ValueError):
        propulsion.coefficients({"name": "x"})


def test_sdf_reports_the_propulsion_reduction(tmp_path):
    """SDF has no propeller element: the report names, per rotor, the reduced constants a motor plugin carries (and
    where they were taken), or that none were given; the block itself survives USD and is dropped by MJCF generically."""
    blk = _propulsion()
    rep = yax.save(_plane("q", blk), tmp_path / "q.sdf")
    assert any("rotor r0: its propeller map (SDF has no propeller element and no reduced constants" in d for d in rep.dropped)
    blk["reduced"] = {"r0": {"kF": 8.5e-6, "kM": 0.016, "w_max": 1100.0, "tau_up": 0.0125, "tau_down": 0.025,
                             "taken_at": "hover", "residual": 0.31}}
    rep = yax.save(_plane("q", blk), tmp_path / "q2.sdf")
    line = next(d for d in rep.dropped if d.startswith("propulsion: rotor r0"))
    assert "reduced to kF = 8.5e-06, kM = 0.016 at hover, residual 0.31" in line and "yax sync" in line
    assert not any("machine data 'propulsion'" in d for d in rep.dropped) and any("data:propulsion: the propellers' maps" in d for d in rep.dropped)
    rep = yax.save(_plane("q", blk), tmp_path / "q.xml")
    assert any("data:propulsion: the propellers' maps (1 propellers, 1 rotors)" in d for d in rep.dropped)
