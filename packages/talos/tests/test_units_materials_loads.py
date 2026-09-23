import pytest

import talos
from talos import (Acceleration, Displacement, FixedSupport, Force, Material, MeshSettings, Pressure,
                   StructuralModel, SurfacesOnPlane)

STEEL = Material("steel", 210000, 0.3)


@pytest.mark.parametrize("kwargs, msg", [
    (dict(youngs_modulus=-1), "youngs_modulus"),
    (dict(poissons_ratio=0.5), "poissons_ratio"),
    (dict(density=0), "density"),
    (dict(yield_strength=-3), "yield_strength"),
])
def test_material_validation(kwargs, msg):
    base = dict(name="m", youngs_modulus=1.0, poissons_ratio=0.3)
    base.update(kwargs)
    with pytest.raises(ValueError, match=msg):
        Material(**base)


def test_loads_reject_empty_values():
    with pytest.raises(ValueError):
        Force("a")
    with pytest.raises(ValueError):
        Pressure("a", 0.0)
    with pytest.raises(ValueError):
        Acceleration()
    with pytest.raises(ValueError):
        Displacement("a")
    assert Displacement("a", uy=0.0).dofs() == [(2, 0.0)]


def _model(**over):
    kw = dict(geometry="x.step", units="mm-N-MPa", material=STEEL,
              regions=[SurfacesOnPlane("fix", "x", 0), SurfacesOnPlane("tip", "x", 1)],
              supports=[FixedSupport("fix")], loads=[Force("tip", fz=-1)], mesh_settings=MeshSettings(1.0))
    kw.update(over)
    return StructuralModel(**kw)


def test_model_validation_never_guesses():
    assert _model().units == "mm-N-MPa"
    with pytest.raises(ValueError, match="unit system"):
        _model(units="inch-lbf")
    with pytest.raises(ValueError, match="supports"):
        _model(supports=[])
    with pytest.raises(ValueError, match="loads"):
        _model(loads=[])
    with pytest.raises(ValueError, match="unknown region"):
        _model(loads=[Force("nowhere", fx=1)])
    with pytest.raises(ValueError, match="density"):
        _model(loads=[Acceleration(az=-9810)])
    with pytest.raises(ValueError, match="unique"):
        _model(regions=[SurfacesOnPlane("a", "x", 0), SurfacesOnPlane("A", "x", 1)], supports=[FixedSupport("a")],
               loads=[Force("a", fx=1)])
    with pytest.raises(ValueError, match="tagged"):
        _model(material=Material("s", 2.1e11, 0.3, units="m-N-Pa"))


def test_region_and_mesh_validation():
    with pytest.raises(ValueError):
        SurfacesOnPlane("bad name", "x", 0)
    with pytest.raises(ValueError):
        SurfacesOnPlane("ok", "w", 0)
    with pytest.raises(ValueError):
        talos.SurfacesInBox("b", (0, 0, 0))
    with pytest.raises(ValueError):
        MeshSettings(0)
    with pytest.raises(ValueError):
        MeshSettings(1.0, order=3)


def test_solve_without_mesh_is_failed_result(tmp_path):
    res = _model().solve(tmp_path)
    assert res.status == "failed" and "run model.mesh" in res.messages[0]
    assert (tmp_path / "summary.json").is_file()
