"""V6.3: a machine leaves Vegeta only as standard files written through Yax (``designs/export.py``)."""
import numpy as np
import pytest

yax = pytest.importorskip("yax")
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


@pytest.mark.parametrize("name", sorted(set(export.MACHINES) - {"drongo"}))
def test_every_machine_goes_through_yax(name, tmp_path):
    m, _ = export.machine(name)
    assert m.links and all(lk.mass is None or lk.mass >= 0 for lk in m.links)
    yax.save(m, tmp_path / f"{name}.usda")
    m2, _ = yax.read(tmp_path / f"{name}.usda")
    assert [lk.name for lk in m2.links] == [lk.name for lk in m.links]
