"""V6.3: a machine leaves Vegeta only as standard files written through Yax (``tools/export.py``), from what the labs
register (``vegeta.labkit.machines``)."""
import sys
from pathlib import Path

import numpy as np
import pytest

yax = pytest.importorskip("yax")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export  # noqa: E402


def test_drongo_exports_with_its_rotors(tmp_path):
    reps = export.export("drongo", tmp_path / "drongo.usda", also=("sdf",))
    assert (tmp_path / "drongo.usda").exists() and (tmp_path / "drongo.sdf").exists()
    assert any("ogma" in d for d in reps[-1].dropped)                     # SDF cannot hold part data: reported
    m, _ = yax.read(tmp_path / "drongo.usda")
    assert m.producer["machine"] == "drongo"
    rot = m.data["ogma"]["rotors"]
    assert len(rot) == 4 and sorted(r["spin"] for r in rot) == [-1, -1, 1, 1]
    for r in rot:
        assert r["kF"] * r["w_max"] ** 2 == pytest.approx(r["T_max"])
        assert r["tau_down"] > r["tau_up"] > 0


def test_lumped_drongo_is_the_body_minus_the_rotors():
    m, _ = export.machine("drongo")
    ml, _ = export.machine("drongo", lump=True)
    rot = ml.data["ogma"]["rotors"]
    assert len(ml.links) == 1 and not ml.joints
    total = sum(lk.mass for lk in m.links)
    assert ml.links[0].mass == pytest.approx(total - sum(r["kg"] for r in rot), rel=1e-12)
    assert np.all(np.linalg.eigvalsh(ml.links[0].inertia) > 0)


def test_unknown_machine_is_refused():
    with pytest.raises(KeyError):
        export.machine("no_such_machine")


@pytest.mark.parametrize("name", sorted(set(export.registry().names()) - {"drongo"}))
def test_every_machine_goes_through_yax(name, tmp_path):
    m, _ = export.machine(name)
    assert m.links and all(lk.mass is None or lk.mass >= 0 for lk in m.links)
    yax.save(m, tmp_path / f"{name}.usda")
    m2, _ = yax.read(tmp_path / f"{name}.usda")
    assert [lk.name for lk in m2.links] == [lk.name for lk in m.links]


# ------------------------------------------------------------------------------------- propellers as part data
WITH_PROPELLERS = ("drongo", "nisus", "nisus_plus", "falco")


def test_drongo_propulsion_block_matches_its_rotors(tmp_path):
    """Drongo's propulsion block: the 5x4.3's BEMT map (coefficients against J, the grid), the 2306 on 4S, and the
    reduced constants that ``ogma.rotors`` carries — the same numbers, checked at hover; the block survives USD and
    SDF reports it as the reduction."""
    reps = export.export("drongo", tmp_path / "drongo.usda", also=("sdf",))
    m, _ = yax.read(tmp_path / "drongo.usda")
    blk, rot = m.data["propulsion"], m.data["ogma"]["rotors"]
    assert [r["name"] for r in blk["rotors"]] == ["rotor0", "rotor1", "rotor2", "rotor3"]
    for r, o in zip(blk["rotors"], rot):
        assert r["pos"] == o["pos"] and r["spin"] == o["spin"] and r["axis"] == [0.0, 0.0, 1.0]
        red = blk["reduced"][r["name"]]
        assert red["kF"] == o["kF"] and red["kM"] == o["kM"] and red["w_max"] == o["w_max"] and red["tau_up"] == o["tau_up"]
        assert 0 <= red["residual"] < 1 and "hover" not in red["residual_at"] and red["residual_at"].endswith("0 m/s")
    (prop,) = blk["propellers"].values()
    co, grid = prop["coefficients"], prop["grid"]
    assert prop["blades"] == 3 and prop["diameter_m"] == pytest.approx(5 * 0.0254) and len(prop["stations"]) >= 3
    assert np.all(np.diff(co["J"]) > 0) and co["J"][0] == 0 and 0.05 < co["CT"][0] < 0.3 and 0 < co["CP"][0] < 0.2
    assert np.asarray(grid["thrust_n"]).shape == (len(grid["rpm"]), len(grid["airspeed_m_s"])) and grid["rho"] == 1.225
    (mot,) = blk["motors"].values()
    assert mot["kv"] == 2400 and 0 < mot["power_w"] < mot["battery_v"] * mot["I_max_A"]
    assert blk["battery"]["cells"] == 4
    assert any("propulsion: rotor rotor0: its propeller map reduced to kF" in d for d in reps[-1].dropped)
    assert any("ogma" in d for d in reps[-1].dropped)


@pytest.mark.parametrize("name", WITH_PROPELLERS)
def test_propulsion_blocks_are_consistent(name):
    """Every rotor names a propeller and a motor the block holds; the propeller's coefficients come from its grid
    (CT at J = 0 is the grid's static thrust coefficient); the motor's power is a point of the grid within the pack."""
    from vegeta.labkit import propulsion as lp
    m, _ = export.machine(name)
    blk = m.data["propulsion"]
    for r in blk["rotors"]:
        assert r["propeller"] in blk["propellers"] and r["motor"] in blk["motors"] and r["spin"] in (-1, 1)
        assert np.linalg.norm(r["axis"]) == pytest.approx(1.0) and r["name"] in blk["reduced"]
    for p in blk["propellers"].values():
        g, D = p["grid"], p["diameter_m"]
        n0 = g["rpm"][-1] / 60
        ct0 = lp.at(g, "thrust_n", g["rpm"][-1], 0.0) / (g["rho"] * n0 ** 2 * D ** 4)
        assert p["coefficients"]["CT"][0] == pytest.approx(ct0, rel=0.05)      # Boreas: no Reynolds effect, CT(0) ~ constant
        assert p["inertia_kgm2"] > 0 and p["mass_kg"] > 0
    for mo in blk["motors"].values():
        assert mo["power_w"] > 0 and (("I_max_A" not in mo) or mo["power_w"] < mo["battery_v"] * mo["I_max_A"])
    for red in blk["reduced"].values():
        assert red["kF"] > 0 and red["kM"] > 0 and red["w_max"] > 0 and red["residual"] >= 0


def test_nisus_flies_in_jsbsim_on_its_own_map(tmp_path):
    """``--also jsbsim`` writes NISUS as a JSBSim aircraft with its 9x6 on the fitted X2216: JSBSim's thrust at its
    settled rpm and advance ratio is the exported grid's (the propeller file holds the map), and the reaction torque
    on the airframe follows the assumed spin."""
    jsbsim = pytest.importorskip("jsbsim")
    from vegeta.labkit import propulsion as lp
    reps = export.export("nisus", tmp_path / "nisus.usda", also=("jsbsim",))
    assert not any("propulsion" in d for d in reps[-1].dropped)
    root = tmp_path / "jsbsim"
    assert (root / "aircraft" / "nisus" / "nisus.xml").exists() and list((root / "aircraft" / "nisus" / "Engines").glob("prop_*.xml"))
    for d in ("engine", "systems"):
        (root / d).mkdir()
    m, _ = yax.read(tmp_path / "nisus.usda")
    blk = m.data["propulsion"]
    (prop,) = blk["propellers"].values()
    (rot,) = blk["rotors"]
    f = jsbsim.FGFDMExec(str(root))
    f.set_debug_level(0)
    assert f.load_model("nisus")
    f.set_dt(1e-4)
    f.set_property_value("ic/h-sl-ft", 0.0)
    f.set_property_value("ic/vt-kts", 0.0)
    f.set_property_value("fcs/throttle-cmd-norm", 1.0)
    f.run_ic()
    for _ in range(5000):
        f.run()
    g = f.get_property_value
    rpm, J = g("propulsion/engine/propeller-rpm"), g("propulsion/engine/advance-ratio")
    V = max(J, 0.0) * rpm / 60 * prop["diameter_m"]
    T = g("propulsion/engine/thrust-lbs") * 4.4482216152605
    assert prop["grid"]["rpm"][0] < rpm < prop["grid"]["rpm"][-1]
    assert T == pytest.approx(lp.at(prop["grid"], "thrust_n", rpm, V), rel=0.03)
    assert g("moments/l-prop-lbsft") * rot["spin"] > 0                       # drag torque on the frame along spin · axis (+x forward)
