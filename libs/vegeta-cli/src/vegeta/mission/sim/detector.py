"""The simulated YOLOX: detections from the simulator's truth (YOLOX is not run).

For each bird in the frame (and in the crop, for a crop pass): its box from the projection, the wingspan and a
random wing attitude (0.55-1.0 of the span visible; height 0.4 of the width); its size in the network's input
pixels (letterbox of the full frame, or the crop resized); the probability of detection, a logistic in those pixels
(``px50`` for one half, per model in ``compute.YOLOX``: 9 px for YOLOX-Nano, 7 for Tiny, 6 for S, after the
small-object recall of COCO-trained detectors: ASSUMED, to be fitted on flight video) reduced by the motion blur; a score around that probability (so marginal birds give the low-score boxes
ByteTrack uses); box noise; overlapping birds merge into one box; false positives (clutter, other objects) with low
scores, Poisson per frame. The latency and the rates come from ``vegeta.mission.compute`` through ``DetectorNode``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..camera import PinholeCamera
from ..compute import DetectorConfig
from ..detector import DetectorNode
from ..messages import Detection

__all__ = ["SimDetectorConfig", "SimDetector"]


@dataclass(frozen=True)
class SimDetectorConfig:
    px50: float | None = None      # bird width in network pixels for 50 % detection; None: the model's (compute.YOLOX, ASSUMED)
    px_spread: float = 2.0
    p_max: float = 0.97
    fp_search: float = 0.25        # false positives per full frame
    fp_roi: float = 0.04
    fp_high_frac: float = 0.08     # of the false positives that score above 0.5
    centre_sigma_net: float = 0.6  # px in the network input
    width_sigma: float = 0.12      # log-normal
    merge_iou: float = 0.3


def _iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


class SimDetector(DetectorNode):
    name = "detector"

    def __init__(self, cam: PinholeCamera, cfg: DetectorConfig = DetectorConfig(), sim: SimDetectorConfig = SimDetectorConfig(), seed: int = 0):
        super().__init__(cam, cfg)
        self.sim = sim
        self.rng = np.random.default_rng(seed)
        self.last_truth_boxes = []

    def infer(self, frame, roi, bus):
        truth = bus.latest("sim/truth")
        cam, sim, rng = self.cam, self.sim, self.rng
        out = []
        if roi is None:
            scale, x0r, y0r, x1r, y1r = self.cfg.search_tiles * self.cfg.search_input / cam.width, 0.0, 0.0, cam.width, cam.height
        else:
            x0r, y0r, x1r, y1r = roi
            scale = self.cfg.roi_input / (x1r - x0r)
        cand = []
        px50 = sim.px50 if sim.px50 is not None else self.cfg.yolox().px50
        if truth is not None and len(truth.ids):
            uv, z = cam.project(truth.pos, frame.R_wc, frame.p_wc)
            for i in range(len(truth.ids)):
                if z[i] < 1.0:
                    continue
                w = cam.fx * truth.size[i] / z[i] * rng.uniform(0.55, 1.0)
                h = 0.4 * w
                cx, cy = uv[i]
                if not (x0r <= cx < x1r and y0r <= cy < y1r):
                    continue
                blur = cam.blur_px(truth.vel[i] - bus.latest("nav").vel, truth.pos[i] - frame.p_wc, frame.omega_b, frame.R_wc) * scale
                px_net = w * scale
                p = sim.p_max / (1 + np.exp(-(px_net - px50) / sim.px_spread)) * np.exp(-max(blur - 1.0, 0.0) / 4.0)
                cand.append((w, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), p, int(truth.ids[i])))
        cand.sort(key=lambda c: -c[0])
        kept = []
        for c in cand:                                    # occlusion: the nearer (bigger) bird hides the other
            if all(_iou(c[1], k[1]) < sim.merge_iou for k in kept):
                kept.append(c)
        self.last_truth_boxes = [(k[3], k[1]) for k in kept]
        for w, box, p, tid in kept:
            if rng.random() > min(1.0, 1.25 * p):
                continue
            s = float(np.clip(p + rng.normal(0, 0.12), 0.05, 0.99))
            sc = sim.centre_sigma_net / scale + 0.04 * w
            cx, cy = 0.5 * (box[0] + box[2]) + rng.normal(0, sc), 0.5 * (box[1] + box[3]) + rng.normal(0, sc)
            ww = w * float(np.exp(rng.normal(0, sim.width_sigma)))
            hh = 0.4 * ww
            out.append(Detection((cx - ww / 2, cy - hh / 2, cx + ww / 2, cy + hh / 2), s))
        n_fp = rng.poisson(sim.fp_search if roi is None else sim.fp_roi)
        for _ in range(n_fp):
            ww = rng.uniform(6, 30) / scale
            cx, cy = rng.uniform(x0r, x1r), rng.uniform(y0r + 0.3 * (y1r - y0r), y1r)
            s = rng.uniform(0.5, 0.75) if rng.random() < sim.fp_high_frac else rng.uniform(0.1, 0.45)
            out.append(Detection((cx - ww / 2, cy - 0.2 * ww, cx + ww / 2, cy + 0.2 * ww), float(s), "clutter"))
        return out
