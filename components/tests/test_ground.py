"""The "ground" components (rover, robot_dog, myropod, apheloria and the Onager series) build the same shapes as the
originals in ``notebooks/designs`` with default parameters: validity, solid and face counts, volume, surface area and
bounding box of each ``part`` every design can make (plus the build variants: Myropod versions 2 and 3, Apheloria
curled into the ball). The whole multi-segment crawlers take ~20 s a side, so they are marked slow; their parts run
in the default tier."""
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
#: The whole crawlers (hundreds of booleans, 1300+ faces) are not repeatable to 1e-9 in OCCT itself: building the
#: unchanged original Myropod (version 3) twice in one process gave volumes 760339.8513 and 760339.8573 (7.8e-9).
#: Their files are verbatim copies; these cases compare to 1e-7 (counts and validity stay exact).
REL_CHAIN = 1e-7


def slow(module, cls, overrides):
    return pytest.param(module, cls, overrides, REL_CHAIN, marks=pytest.mark.slow)


def _parts(*parts, module, cls):
    return [(module, cls, {"part": q}, REL) for q in parts]


CASES = (
    _parts("rover", "chassis", "arm", module="rover", cls="Rover")
    + _parts("dog", "body", "upper_leg", "lower_leg", module="robot_dog", cls="RobotDog")
    + _parts("segment", "leg", "head", "pincer", "case", "guide", module="myropod", cls="Myropod")
    + [slow("myropod", "Myropod", {"part": "crawler"}),
       slow("myropod", "Myropod", {"part": "crawler", "version": 2}),
       slow("myropod", "Myropod", {"part": "crawler", "version": 3})]
    + _parts("segment", "plate", "leg", "head", module="apheloria", cls="Apheloria")
    + [slow("apheloria", "Apheloria", {"part": "apheloria"}),
       slow("apheloria", "Apheloria", {"part": "apheloria", "mode": "ball"})]
    + _parts("robot", "hull", "upper_leg", "lower_leg", "wheel", module="onager", cls="OnagerSentinel")
    + _parts("robot", "hull", "upper_leg", "lower_leg", "wheel", "fork", "mast", "mast_fea",
             module="onager_atlas", cls="OnagerAtlas")
    + _parts("robot", "hull", "upper_leg", "lower_leg", "wheel", "upper_arm", "forearm", "jaw",
             module="onager_manus", cls="OnagerManus")
    + _parts("robot", "hull", "upper_leg", "lower_leg", "wheel", "upper_arm", "forearm", "jaw", "broom_disc", "hood",
             "hood_shell", "basket", module="onager_sweeper", cls="OnagerSweeper")
    + _parts("rover", "hull", "track_module", "sprocket", "idler", "road_wheel", "track_link", "pinion", "basket",
             module="pekari_rover", cls="PekariRover")
)
DESIGN_CLASSES = [("rover", "Rover"), ("robot_dog", "RobotDog"), ("myropod", "Myropod"), ("apheloria", "Apheloria"),
                  ("onager", "OnagerSentinel"), ("onager_atlas", "OnagerAtlas"), ("onager_manus", "OnagerManus"),
                  ("onager_sweeper", "OnagerSweeper"), ("pekari_rover", "PekariRover")]


def _close(a, b, rel=REL):
    return math.isclose(a, b, rel_tol=rel, abs_tol=1e-9)


def _measure(module, cls, overrides):
    return getattr(importlib.import_module(module), cls)().generate(**overrides).measure()


@pytest.mark.parametrize("module,cls,overrides,rel", CASES)
def test_same_shape_as_original(module, cls, overrides, rel):
    new = _measure(f"components.{module}", cls, overrides)
    old = _measure(module, cls, overrides)
    assert new["valid"] == old["valid"]
    assert new["n_solids"] == old["n_solids"] and new["n_faces"] == old["n_faces"]
    assert _close(new["volume"], old["volume"], rel), (new["volume"], old["volume"])
    assert _close(new["surface_area"], old["surface_area"], rel), (new["surface_area"], old["surface_area"])
    for k in ("bbox_min", "bbox_max"):
        assert all(_close(x, y, rel) for x, y in zip(new[k], old[k])), (k, new[k], old[k])


def test_every_part_is_covered():
    """Each choice of every design's ``part`` parameter has a case above."""
    covered = {(c.values[0], c.values[2]["part"]) if hasattr(c, "values") else (c[0], c[2]["part"]) for c in CASES}
    for module, cls in DESIGN_CLASSES:
        part = next(q for q in getattr(importlib.import_module(f"components.{module}"), cls).parameters if q.name == "part")
        assert {(module, q) for q in part.choices} <= covered, module


def test_parameters_are_the_originals():
    """Same parameter names, defaults, units, limits and choices for every design."""
    for module, cls in DESIGN_CLASSES:
        new = getattr(importlib.import_module(f"components.{module}"), cls).parameters
        old = getattr(importlib.import_module(module), cls).parameters
        key = lambda q: (q.name, q.default, q.units, q.min, q.max, q.choices, q.description)  # noqa: E731
        assert [key(q) for q in new] == [key(q) for q in old], module


def test_onager_variants_derive_from_the_component_sentinel():
    """The Atlas, Manus and Sweeper build on the component Sentinel (package-relative), not on the notebook copy."""
    from components.onager import OnagerSentinel
    from components.onager_atlas import OnagerAtlas
    from components.onager_manus import OnagerManus
    from components.onager_sweeper import OnagerSweeper
    assert issubclass(OnagerAtlas, OnagerSentinel) and issubclass(OnagerManus, OnagerSentinel)
    assert issubclass(OnagerSweeper, OnagerManus)
