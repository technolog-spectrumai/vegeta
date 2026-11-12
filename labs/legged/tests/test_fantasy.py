"""The fantasy objects: the CAD builds, its volumes and centres of mass are the stored ``fantasy_robots.CAD``, the
bodies carry exactly the CAD masses (volume × density per material), the materials leave in Yax's shape, and the
chest's lid hinges."""
import pytest

import fantasy_robots as fr
import fantasy_specs as fs

#: object -> (mass [kg], links)
MASSES = {"chest": (32.784, 2), "crate": (11.009, 1), "sword": (1.857, 1), "shield": (6.479, 1),
          "coin": (0.0198, 1), "amulet": (0.0926, 1), "elixir_flask": (0.798, 1)}


@pytest.mark.parametrize("obj", sorted(MASSES))
def test_body_mass_is_the_cads(obj):
    r = fr.BUILDERS[obj]()
    m, links = MASSES[obj]
    assert r.total_mass() == pytest.approx(m, rel=2e-3)
    assert len(r.links()) == links
    whole = sum(fr.mass_of(k) for k in fr.CAD if k.split("/")[0] == obj and "/lid_" not in k)
    assert r.total_mass() == pytest.approx(whole, rel=1e-6)


@pytest.mark.parametrize("obj", sorted(MASSES))
def test_materials_are_yax_shaped(obj):
    mats = fr.materials_block(obj)
    assert mats and set(mats) <= set(fs.MATERIALS)
    for v in mats.values():
        assert {"density_kg_m3", "E_pa", "nu", "yield_pa", "ultimate_pa", "source"} <= set(v)


def test_chest_lid_hinges_open():
    import fantasy_motions as fm
    import numpy as np

    kin, ms = fm.motions_chest()
    opened = {m.name: m for m in ms}["open"]
    assert opened.Q[0, 0] == pytest.approx(0.0) and opened.Q[-1, 0] == pytest.approx(np.radians(fm.OPEN_DEG))
    q = kin.q0.copy()
    q[kin.index["lid_hinge"]] = opened.Q[-1, 0]
    kin.set(q)
    lid = kin.data.xpos[kin.body("lid")]
    front = kin.world(kin.body("lid"), kin.geom_point("lid_shell")[1])
    assert front[2] > lid[2] + 0.1                                       # the open lid stands up behind the hinge


@pytest.mark.parametrize("obj", sorted(fs.OBJECTS))
def test_cad_builds_and_matches(obj):
    pytest.importorskip("cadquery")
    import numpy as np

    import fantasy_objects as fo

    cls, spec = fs.OBJECTS[obj]
    C = getattr(fo, cls)
    assert C().generate().measure()["valid"]
    for part, mat in spec["_parts"].items():
        if mat is None:
            continue
        g = C().generate(part=part).shape
        v, c = fr.CAD[f"{obj}/{part}"]
        assert g.Volume() == pytest.approx(v, rel=1e-4), part
        assert np.allclose(g.Center().toTuple(), c, atol=0.05), part


def test_heater_shield_builds():
    pytest.importorskip("cadquery")
    import fantasy_objects as fo

    m = fo.Shield().generate(shape_heater=1.0).measure()
    assert m["valid"] and m["n_solids"] == 1
