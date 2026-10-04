"""The "rotating" components (air_propeller, ducted_fan, turbojet) build the same shapes as the originals in
``notebooks/designs`` with default parameters: volume, surface area, bounding box and face count of every design and of
each ``part`` / ``layout`` it can make."""
from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("cadquery")

DESIGNS = str(Path(__file__).resolve().parents[2] / "notebooks" / "designs")
if DESIGNS not in sys.path:
    sys.path.append(DESIGNS)          # the originals use bare imports

REL = 1e-9
slow = pytest.mark.slow

# (module, class, overrides); the heavy ones are slow (each still compared once by hand)
CASES = [
    ("air_propeller", "PropPod", {"layout": "tractor"}),
    ("air_propeller", "PropPod", {"layout": "pusher"}),
    ("ducted_fan", "EDFHousing", {}),
    pytest.param("ducted_fan", "EDFRotor", {}, marks=slow),
    ("turbojet", "Turbojet", {"part": "impeller"}),
    ("turbojet", "Turbojet", {"part": "engine"}),
    pytest.param("turbojet", "Turbojet", {"part": "turbine"}, marks=slow),
]
DESIGN_CLASSES = [("air_propeller", "PropPod"), ("ducted_fan", "EDFRotor"), ("ducted_fan", "EDFHousing"),
                  ("turbojet", "Turbojet")]


def _close(a, b):
    return math.isclose(a, b, rel_tol=REL, abs_tol=1e-9)


def _measure(module, cls, overrides):
    return getattr(importlib.import_module(module), cls)().generate(**overrides).measure()


@pytest.mark.parametrize("module,cls,overrides", CASES)
def test_same_shape_as_original(module, cls, overrides):
    new = _measure(f"components.{module}", cls, overrides)
    old = _measure(module, cls, overrides)
    assert new["valid"] == old["valid"]
    assert new["n_solids"] == old["n_solids"] and new["n_faces"] == old["n_faces"]
    assert _close(new["volume"], old["volume"]), (new["volume"], old["volume"])
    assert _close(new["surface_area"], old["surface_area"]), (new["surface_area"], old["surface_area"])
    for k in ("bbox_min", "bbox_max"):
        assert all(_close(x, y) for x, y in zip(new[k], old[k])), (k, new[k], old[k])


def test_parameters_are_the_originals():
    """Same parameter names, defaults, units, limits and choices for every design."""
    for module, cls in DESIGN_CLASSES:
        new = getattr(importlib.import_module(f"components.{module}"), cls).parameters
        old = getattr(importlib.import_module(module), cls).parameters
        key = lambda q: (q.name, q.default, q.units, q.min, q.max, q.choices, q.description)  # noqa: E731
        assert [key(q) for q in new] == [key(q) for q in old], (module, cls)


def test_housing_layout_is_the_originals():
    """MERLIN's nose uses EDFHousing's layout, duct profile and vane section: same numbers as the original."""
    import ducted_fan as old
    from components import ducted_fan as new
    p = new.EDFHousing().resolve()
    a, b = new.EDFHousing.layout(p), old.EDFHousing.layout(p)
    for k in a:
        if k != "te":
            assert _close(a[k], b[k]), k
    assert (new.EDFHousing.duct_profile(p)["closed"] == old.EDFHousing.duct_profile(p)["closed"]).all()
    assert (new.EDFHousing.vane_section(p) == old.EDFHousing.vane_section(p)).all()
