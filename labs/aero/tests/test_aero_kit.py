import numpy as np

from aero_kit import plain


def test_plain_makes_json_values():
    assert plain(np.float64(1.5)) == 1.5 and plain(np.nan) is None and plain(np.int32(3)) == 3
    assert plain(np.bool_(True)) is True and plain("x") == "x"
