import math

from legged_kit import pin_stress


def test_pin_stress_is_von_mises_of_double_shear():
    assert pin_stress(100.0, 5.0) == math.sqrt(3) * 100.0 / (2 * math.pi * 25 / 4)
