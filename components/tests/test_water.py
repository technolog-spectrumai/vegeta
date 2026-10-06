"""The "water" components (survey_boat, submarine) build the same shapes as the originals in ``notebooks/designs``
with default parameters: volume, surface area, bounding box and face count of each ``part`` every design can make."""
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

CASES = [
    ("survey_boat", "SurveyBoat", {"part": "boat"}),
    ("survey_boat", "SurveyBoat", {"part": "hull_solid"}),
    ("survey_boat", "SurveyBoat", {"part": "hull_shell"}),
    ("survey_boat", "SurveyBoat", {"part": "bracket"}),
    ("submarine", "Submarine", {"part": "vehicle"}),
    ("submarine", "Submarine", {"part": "body"}),
    ("submarine", "Submarine", {"part": "pressure_hull"}),
]
DESIGN_CLASSES = [("survey_boat", "SurveyBoat"), ("submarine", "Submarine")]


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
        assert [key(q) for q in new] == [key(q) for q in old], module
