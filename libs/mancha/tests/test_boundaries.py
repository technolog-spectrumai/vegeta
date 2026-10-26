"""Boundary configuration: parsing, validation and the edge/corner precedence rule."""
import numpy as np
import pytest

from mancha import D3Q19, InvalidParameterError
from mancha.boundaries import BoundaryConditions, Kind, link_groups, sphere, validate_mask

C = D3Q19().c


def test_defaults_are_periodic():
    bc = BoundaryConditions()
    bc.validate()
    assert all(f.kind == Kind.PERIODIC for f in bc.faces)


@pytest.mark.parametrize("face,kind,params", [
    ("x-", "wall", {"velocity": (0.1, 0, 0)}),     # normal wall motion
    ("w+", "wall", {}),                            # no such face
    ("x-", "slip", {}),                            # no such kind
    ("x-", "velocity", {}),                        # missing velocity
    ("x+", "pressure", {"density": -1.0}),
    ("x+", "pressure", {"velocity": (0, 0, 0)}),
    ("y-", "periodic", {"density": 1.0}),
])
def test_bad_conditions(face, kind, params):
    with pytest.raises(InvalidParameterError):
        BoundaryConditions().set(face, kind, **params)


def test_unpaired_periodic_is_refused():
    bc = BoundaryConditions()
    bc.set("x-", "wall")
    with pytest.raises(InvalidParameterError):
        bc.validate()


def test_round_trip_dict():
    bc = BoundaryConditions()
    bc.set("x-", "velocity", velocity=(0.05, 0, 0))
    bc.set("x+", "pressure", density=1.01)
    bc.set("y+", "wall", velocity=(0.1, 0, 0))
    bc.set("y-", "wall")
    again = BoundaryConditions.from_dict(bc.to_dict())
    assert again.faces == bc.faces
    assert again["y+"].kind == Kind.MOVING_WALL


def _kind_at(groups, i, node, shape):
    flat = np.ravel_multi_index(node, shape)
    hits = [g for g in groups[i] if flat in g.nodes]
    assert len(hits) <= 1
    return (hits[0].kind, hits[0].face) if hits else (Kind.PERIODIC, None)


def test_precedence_at_an_edge():
    shape = (5, 5, 1)
    bc = BoundaryConditions()
    bc.set("x-", "velocity", velocity=(0.05, 0, 0))
    bc.set("x+", "pressure")
    bc.set("y-", "wall")
    bc.set("y+", "wall")
    codes = bc.table()[0]
    groups = link_groups(shape, C, codes)
    i = int(np.flatnonzero((C == (1, 1, 0)).all(axis=1))[0])     # pulls from (x-1, y-1)
    assert _kind_at(groups, i, (0, 0, 0), shape) == (Kind.WALL, 2)       # leaves x- and y-: the wall wins
    assert _kind_at(groups, i, (0, 2, 0), shape) == (Kind.VELOCITY, 0)   # leaves x- only
    assert _kind_at(groups, i, (2, 0, 0), shape) == (Kind.WALL, 2)
    assert _kind_at(groups, i, (2, 2, 0), shape) == (Kind.PERIODIC, None)
    j = int(np.flatnonzero((C == (0, 0, 1)).all(axis=1))[0])     # z is periodic: no replacement anywhere
    assert groups[j] == []


def test_masks():
    m = sphere((10, 10, 10), (4.5, 4.5, 4.5), 2.0)
    assert m.dtype == bool and m[4, 4, 4] and not m[0, 0, 0]
    with pytest.raises(InvalidParameterError):
        validate_mask(np.ones((10, 10, 10), bool), (10, 10, 10))
    with pytest.raises(InvalidParameterError):
        validate_mask(m, (10, 10, 9))
