import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from vegeta.talos._watermark import LOGO, watermark  # noqa: E402


@pytest.mark.parametrize("w,h", [(1280, 540), (960, 720), (400, 1000)])
def test_logo_bottom_right_at_most_ten_percent(w, h):
    assert LOGO.is_file()
    frame = np.zeros((h, w, 3), np.uint8)
    watermark(frame)
    ys, xs = np.nonzero(frame.max(axis=2))
    bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    assert bw <= 0.1 * w + 1 and bh <= 0.1 * h + 1           # small: at most 10 % of each side
    assert abs(bw - bh) <= 1                                  # the square logo keeps its aspect ratio
    assert xs.max() > 0.95 * w and ys.max() > 0.95 * h        # bottom-right corner
    assert xs.min() > 0.85 * w and ys.min() > 0.85 * h
