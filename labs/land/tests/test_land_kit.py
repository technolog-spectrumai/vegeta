import math

import numpy as np

from land_kit import add_drop, add_rocks, pin_bending, pin_shear


def test_terrain_profiles():
    x = np.linspace(0, 10, 1001)
    z = np.zeros_like(x)
    assert add_drop(x, z, 5.0, 0.2)[-1] == -0.2 and z.max() == 0.0             # a copy, the input untouched
    r = add_rocks(x, z, 0.1, 0.4, 2.0, seed=1)
    assert 0.09 < r.max() <= 0.1 and r.min() >= 0.0


def test_pin_stresses():
    assert pin_bending(100.0, 10.0, 5.0) == 100.0 * 10.0 / (math.pi * 125 / 32)
    assert pin_shear(100.0, 5.0) == 100.0 / (2 * math.pi * 25 / 4)
