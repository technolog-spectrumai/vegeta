import math

from legged_kit import pin_stress


def test_pin_stress_is_von_mises_of_double_shear():
    assert pin_stress(100.0, 5.0) == math.sqrt(3) * 100.0 / (2 * math.pi * 25 / 4)


def test_posed_moves_a_copy():
    import numpy as np

    from legged_kit import posed

    class Mesh:
        def __init__(self, pts):
            self.points = pts

        def copy(self):
            return Mesh(self.points.copy())

    m = Mesh(np.array([[0.0, 0.0, 1.0]]))
    out = posed(m, np.eye(3), np.array([1.0, 2.0, 3.0]), zc=1.0)
    assert np.allclose(out.points, [[1.0, 2.0, 3.0]]) and np.allclose(m.points, [[0.0, 0.0, 1.0]])
