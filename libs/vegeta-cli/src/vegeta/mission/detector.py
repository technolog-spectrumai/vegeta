"""The detector node: when to grab a frame, which crop to look at, and the latency of the answer.

``DetectorNode`` is the part that stays on the hardware: a full-frame search pass at ``search_hz`` and, while a
target is selected and predicted inside the frame, a crop of ``roi_px`` pixels around its predicted position at
``roi_hz`` (the crop sees the bird at full resolution: a far bird spans many more of the network's pixels). Each
result is published ``latency`` after the exposure (the budget in ``compute``), stamped with the exposure time and
the camera pose so the tracker can compensate the aircraft's motion. Subclasses implement ``infer(frame, roi, bus)``:
``vegeta.mission.sim.SimDetector`` from the simulator's truth, ``YoloxTrtDetector`` (later) from the pixels.
"""
from __future__ import annotations

import numpy as np

from .bus import Node
from .camera import PinholeCamera
from .compute import DetectorConfig, budget
from .messages import Detections

__all__ = ["DetectorNode", "YoloxTrtDetector"]


class DetectorNode(Node):
    name = "detector"

    def __init__(self, cam: PinholeCamera, cfg: DetectorConfig = DetectorConfig()):
        self.cam, self.cfg = cam, cfg
        b = budget(cfg)
        self.search_period = 1.0 / max(b["search [Hz] achieved"], 1e-3)
        self.roi_period = 1.0 / max(b["roi [Hz] achieved"], 1e-3) if cfg.roi_hz > 0 else float("inf")
        self.lat_search = b["latency search [ms]"] / 1000.0
        self.lat_roi = b["latency roi [ms]"] / 1000.0
        self.next_search = 0.0
        self.next_roi = 0.0
        self.pending: list = []
        self.enabled = True
        self.frames = 0

    def roi_for(self, frame, bus):
        sel, tracks = bus.latest("target"), bus.latest("tracks")
        if sel is None or sel.track_id is None or tracks is None:
            return None
        k = tracks.get(sel.track_id)
        if k is None:
            return None
        pos = k.pos + k.vel * max(frame.t - tracks.t, 0.0)
        uv, z = self.cam.project(pos[None], frame.R_wc, frame.p_wc)
        if not self.cam.in_frame(uv, z, margin=0.05)[0]:
            return None
        half = self.cfg.roi_px / 2
        cx = float(np.clip(uv[0, 0], half, self.cam.width - half))
        cy = float(np.clip(uv[0, 1], half, self.cam.height - half))
        return (cx - half, cy - half, cx + half, cy + half)

    def infer(self, frame, roi, bus):                      # pragma: no cover - interface
        raise NotImplementedError

    def step(self, t, bus):
        out = [p for p in self.pending if p.t <= t + 1e-9]
        self.pending = [p for p in self.pending if p.t > t + 1e-9]
        for d in out:
            bus.publish("detections", d)
        if not self.enabled:
            return
        frame = bus.latest("camera")
        if frame is None:
            return
        if t >= self.next_search:
            self.next_search = t + self.search_period
            self._run(frame, None, self.lat_search, bus)
        if t >= self.next_roi:
            roi = self.roi_for(frame, bus)
            if roi is not None:
                self.next_roi = t + self.roi_period
                self._run(frame, roi, self.lat_roi, bus)

    def _run(self, frame, roi, latency, bus):
        items = tuple(self.infer(frame, roi, bus))
        self.frames += 1
        self.pending.append(Detections(frame.t + latency, frame.t, frame.frame_id, frame.R_wc, frame.p_wc, items,
                                       "roi" if roi is not None else "search", roi))


class YoloxTrtDetector(DetectorNode):
    """The hardware detector (not implemented yet): YOLOX exported to ONNX and built into a TensorRT engine
    (FP16/INT8). Contract: ``frame.image`` is the full-resolution RGB frame; letterbox the frame (search) or the crop
    (``roi``) to ``cfg.search_input``/``cfg.roi_input``, run the engine, decode, NMS, keep the bird class, map the
    boxes back to full-resolution pixels and return ``Detection`` objects with the network's score."""

    def infer(self, frame, roi, bus):
        raise NotImplementedError("YOLOX is not run in this prototype: use vegeta.mission.sim.SimDetector "
                                  "(simulated detections with the assumed latency) until the TensorRT engine exists")
