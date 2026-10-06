"""The "air" components (quad_frame, fixed_wing, aguya, merlin, velutina) build the same shapes as the originals in
``notebooks/designs`` with default parameters: volume, surface area, bounding box and face count of every design and
of each ``part`` it can make (and, for MERLIN, each propulsion layout)."""
from __future__ import annotations

import importlib
import importlib.util
import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("cadquery")

DESIGNS = str(Path(__file__).resolve().parents[2] / "notebooks" / "designs")
if DESIGNS not in sys.path:
    sys.path.append(DESIGNS)          # the originals use bare imports (``from fixed_wing import ...``)

REL = 1e-9
slow = pytest.mark.slow

# (module, class, overrides); the heavy ones are slow (each still compared once by hand before commit)
CASES = [
    ("quad_frame", "QuadFrame", {}),
    ("fixed_wing", "FixedWing", {"part": "aircraft"}),
    ("fixed_wing", "FixedWing", {"part": "wing"}),
    ("fixed_wing", "FixedWing", {"part": "nacelle"}),
    ("fixed_wing", "FixedWing", {"part": "fuselage"}),
    ("aguya", "Aguya", {"part": "aircraft"}),
    ("aguya", "Aguya", {"part": "wing"}),
    ("aguya", "Aguya", {"part": "nacelle"}),
    ("merlin", "Merlin", {"part": "nose", "propulsion": "edf"}),
    pytest.param("merlin", "Merlin", {"part": "aircraft", "propulsion": "edf"}, marks=slow),
    pytest.param("merlin", "Merlin", {"part": "wing", "propulsion": "edf"}, marks=slow),
    pytest.param("merlin", "Merlin", {"part": "aircraft", "propulsion": "tractor"}, marks=slow),
    ("merlin", "Merlin", {"part": "nose", "propulsion": "tractor"}),
    pytest.param("merlin", "Merlin", {"part": "aircraft", "propulsion": "pusher"}, marks=slow),
    ("merlin", "Merlin", {"part": "nose", "propulsion": "pusher"}),
    ("velutina", "Velutina", {"part": "aircraft"}),
    ("velutina", "Velutina", {"part": "body"}),
    ("velutina", "Velutina", {"part": "arm"}),
    ("velutina", "Velutina", {"part": "capsule"}),
    ("velutina", "Velutina", {"part": "fin"}),
]
DESIGN_CLASSES = [("quad_frame", "QuadFrame"), ("fixed_wing", "FixedWing"), ("aguya", "Aguya"), ("merlin", "Merlin"),
                  ("velutina", "Velutina")]


def _close(a, b):
    return math.isclose(a, b, rel_tol=REL, abs_tol=1e-9)


def _measure(module, cls, overrides):
    return getattr(importlib.import_module(module), cls)().generate(**overrides).measure()


@pytest.mark.parametrize("module,cls,overrides", CASES)
def test_same_shape_as_original(module, cls, overrides):
    if module == "merlin":
        pytest.importorskip("components.ducted_fan", reason="MERLIN's nose is components.ducted_fan's EDFHousing")
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
        if module == "merlin" and importlib.util.find_spec("components.ducted_fan") is None:
            continue
        new = getattr(importlib.import_module(f"components.{module}"), cls).parameters
        old = getattr(importlib.import_module(module), cls).parameters
        key = lambda q: (q.name, q.default, q.units, q.min, q.max, q.choices, q.description)  # noqa: E731
        assert [key(q) for q in new] == [key(q) for q in old], module


def test_aguya_uses_the_fixed_wing_component():
    from components.aguya import Aguya
    from components.fixed_wing import FixedWing
    assert issubclass(Aguya, FixedWing)
