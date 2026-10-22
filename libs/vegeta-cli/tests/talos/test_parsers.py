from pathlib import Path
import numpy as np
import pytest

from vegeta.talos import read_dat_reactions, read_frd, von_mises
from vegeta.talos.mesh import consistent_nodal_forces, element_faces, MeshData

DATA = Path(__file__).parent / "data"



def test_read_frd_fixture():
    fr = read_frd(DATA / "tiny.frd")
    assert list(fr.node_ids) == [1, 2]
    assert fr.coords[1] == pytest.approx([10, -5, 2.5])
    assert fr.displacement[1] == pytest.approx([1e-3, -2e-3, -3.5e-2])
    assert fr.von_mises[0] == pytest.approx(100.0)


def test_von_mises_known_values():
    assert von_mises(np.array([[100, 0, 0, 0, 0, 0]]))[0] == pytest.approx(100)
    assert von_mises(np.array([[0, 0, 0, 50, 0, 0]]))[0] == pytest.approx(50 * np.sqrt(3))
    assert von_mises(np.array([[-70, -70, -70, 0, 0, 0]]))[0] == pytest.approx(0)  # hydrostatic


def test_read_dat_reactions():
    r = read_dat_reactions(DATA / "tiny.dat")
    assert r["N_FIXED"] == pytest.approx([2.072596e-09, -7.389618e-10, 100.0])
    assert r["N_OTHER"] == pytest.approx([-15, 0, 0])


def test_consistent_forces_sum_and_quadratic_rule():
    cmap = {1: np.array([0., 0, 0]), 2: np.array([2., 0, 0]), 3: np.array([0., 2, 0]),
            4: np.array([1., 0, 0]), 5: np.array([1., 1, 0]), 6: np.array([0., 1, 0])}
    lin = consistent_nodal_forces(np.array([[1, 2, 3]]), cmap, np.array([0, 0, -9.0]))
    assert set(lin) == {1, 2, 3} and all(f[2] == pytest.approx(-3) for f in lin.values())
    quad = consistent_nodal_forces(np.array([[1, 2, 3, 4, 5, 6]]), cmap, np.array([0, 0, -9.0]))
    assert set(quad) == {4, 5, 6} and sum(f[2] for f in quad.values()) == pytest.approx(-9)


def test_element_face_lookup():
    mesh = MeshData(node_ids=np.arange(1, 5), coords=np.eye(4, 3), element_type="C3D4",
                    element_ids=np.array([7]), connectivity=np.array([[1, 2, 3, 4]]),
                    region_nodes={}, region_triangles={})
    assert element_faces(mesh, np.array([[3, 2, 1], [1, 4, 2], [2, 3, 4], [4, 1, 3]])) == [(7, 1), (7, 2), (7, 3), (7, 4)]
    with pytest.raises(ValueError):
        element_faces(mesh, np.array([[1, 2, 9]]))
