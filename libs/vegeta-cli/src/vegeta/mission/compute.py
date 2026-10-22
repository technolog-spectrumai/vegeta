"""The detector's load on the Jetson: an ASSUMED budget until YOLOX is benchmarked with TensorRT on the hardware.

Model:  latency = overhead (resize, copy, decode, NMS) + work / effective throughput,
        work = GFLOPs at the network input (the YOLOX paper's figures, scaled with the input area),
        GPU load = Σ rate x (latency − CPU overhead).

What is published and what is assumed:
- YOLOX GFLOPs and parameters: Ge et al. 2021, "YOLOX: Exceeding YOLO Series in 2021" (Nano and Tiny at 416, S/M at 640).
- Jetson peaks: Jetson Nano 472 GFLOPS FP16 (NVIDIA); Jetson Orin Nano Super 67 TOPS INT8 sparse (NVIDIA, JetPack 6.2),
  so ~33 TOPS dense INT8 and ~17 TFLOPS dense FP16 (derived: half each step).
- ASSUMED: the effective throughputs (what a small detector really gets out of the GPU: low utilisation, memory bound)
  and the overheads. They put YOLOX-Tiny@416 at ~12 fps on the Nano and ~110 fps (FP16) on the Orin Nano Super:
  the order of the community TensorRT reports for similar-size detectors, not a measurement of this pipeline.
Replace ``effective_tflops`` and ``overhead_ms`` with the benchmark (todo 5n) and every table follows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .bus import Node
from .messages import Health

__all__ = ["YOLOX", "JETSONS", "DetectorConfig", "Jetson", "YoloxModel", "budget", "budget_table", "ComputeMonitor"]


@dataclass(frozen=True)
class YoloxModel:
    name: str
    params_m: float
    gflops: float          # at ``ref_input``
    ref_input: int
    ap_coco: float         # COCO val AP (paper), for the trade-off only
    px50: float = 7.0      # bird width in the network's pixels for 50 % detection: ASSUMED (smaller models need more)

    def gflops_at(self, size: int) -> float:
        return self.gflops * (size / self.ref_input) ** 2


YOLOX = {
    "yolox-nano": YoloxModel("YOLOX-Nano", 0.91, 1.08, 416, 25.8, 9.0),
    "yolox-tiny": YoloxModel("YOLOX-Tiny", 5.06, 6.45, 416, 32.8, 7.0),
    "yolox-s": YoloxModel("YOLOX-S", 9.0, 26.8, 640, 40.5, 6.0),
    "yolox-m": YoloxModel("YOLOX-M", 25.3, 73.8, 640, 47.2, 5.5),
}


@dataclass(frozen=True)
class Jetson:
    key: str
    name: str
    peak: dict                    # precision -> TFLOPS/TOPS (published or derived)
    effective_tflops: dict        # precision -> what a small detector achieves (ASSUMED)
    overhead_ms: float            # per frame: capture copy, resize, decode, NMS on the CPU (ASSUMED)
    power_w: tuple                # (idle, full) module power at the mode used (nisus_systems: 5/10 W and 7-25 W modes)
    memory_gb: float
    notes: str = ""


JETSONS = {
    "nano_b01": Jetson("nano_b01", "Jetson Nano 4 GB (B01, end of life)", {"fp16": 0.472}, {"fp16": 0.085}, 9.0, (2.5, 10.0), 4.0,
                       "Maxwell GPU: no INT8 tensor cores; JetPack 4.6 (TensorRT 8.2)"),
    "orin_nano_super": Jetson("orin_nano_super", "Jetson Orin Nano Super 8 GB", {"int8": 33.5, "fp16": 16.7}, {"int8": 2.2, "fp16": 1.25},
                              3.0, (5.0, 25.0), 8.0, "Ampere GPU with tensor cores; JetPack 6.2 (TensorRT 10)"),
}


@dataclass(frozen=True)
class DetectorConfig:
    """The detection pipeline: a full-frame search pass (``search_tiles``² tiles of ``search_input``) at ``search_hz`` and, while a
    target is selected, a crop of ``roi_px`` full-resolution pixels around it at ``roi_input`` and ``roi_hz``."""
    device: str = "orin_nano_super"
    model: str = "yolox-tiny"
    precision: str = "fp16"
    search_input: int = 640
    search_tiles: int = 2         # the full frame cut into tiles x tiles crops of search_input (small birds need the pixels)
    search_hz: float = 5.0
    roi_input: int = 416
    roi_px: int = 832
    roi_hz: float = 15.0
    max_gpu_load: float = 0.8     # the rest is left to the encoder, the tracker and the OS

    def jetson(self) -> Jetson:
        return JETSONS[self.device]

    def yolox(self) -> YoloxModel:
        return YOLOX[self.model]


def _latency(cfg: DetectorConfig, size: int) -> tuple:
    j, m = cfg.jetson(), cfg.yolox()
    prec = cfg.precision if cfg.precision in j.effective_tflops else "fp16"
    gpu = m.gflops_at(size) / (j.effective_tflops[prec] * 1000.0)       # s
    return gpu + j.overhead_ms / 1000.0, gpu


def budget(cfg: DetectorConfig) -> dict:
    """The planned load: latency and GPU time per pass, the GPU load and the achievable rates."""
    j, m = cfg.jetson(), cfg.yolox()
    lat_s, gpu_s = _latency(cfg, cfg.search_input)
    n = cfg.search_tiles ** 2                                            # tiles run as one batch
    lat_s, gpu_s = lat_s + (n - 1) * gpu_s, n * gpu_s
    lat_r, gpu_r = _latency(cfg, cfg.roi_input)
    load = cfg.search_hz * gpu_s + cfg.roi_hz * gpu_r
    feasible = load <= cfg.max_gpu_load
    scale = min(1.0, cfg.max_gpu_load / load) if load > 0 else 1.0
    max_search_alone = cfg.max_gpu_load / gpu_s
    p_idle, p_full = j.power_w
    return {"device": j.name, "model": m.name, "precision": cfg.precision if cfg.precision in j.effective_tflops else "fp16",
            "search input": f"{cfg.search_tiles}x{cfg.search_tiles} x {cfg.search_input}", "GFLOPs search": cfg.search_tiles ** 2 * m.gflops_at(cfg.search_input), "latency search [ms]": 1000 * lat_s,
            "roi input": cfg.roi_input, "GFLOPs roi": m.gflops_at(cfg.roi_input), "latency roi [ms]": 1000 * lat_r,
            "search [Hz] asked": cfg.search_hz, "roi [Hz] asked": cfg.roi_hz, "GPU load": load, "feasible": feasible,
            "search [Hz] achieved": cfg.search_hz * scale, "roi [Hz] achieved": cfg.roi_hz * scale,
            "max search-only [Hz]": max_search_alone, "module power [W]": p_idle + min(load, 1.0) * (p_full - p_idle),
            "status": "ASSUMED (no benchmark)"}


def budget_table(configs: dict | None = None) -> pd.DataFrame:
    """The budget for the candidate configurations (``{label: DetectorConfig}``; default: the models on both Jetsons)."""
    if configs is None:
        configs = {}
        for dev in JETSONS:
            for mk in ("yolox-nano", "yolox-tiny", "yolox-s"):
                for tiles in (1, 2):
                    configs[f"{dev} / {mk} / {tiles}x{tiles}"] = DetectorConfig(device=dev, model=mk, search_tiles=tiles,
                                                                                 precision="int8" if dev == "orin_nano_super" and mk == "yolox-s" else "fp16")
    return pd.DataFrame({k: budget(c) for k, c in configs.items()}).T


class ComputeMonitor(Node):
    """Publishes the planned load and the detection rate actually seen on ``detections`` (``health``)."""
    name = "compute"
    subscriptions = ("detections",)

    def __init__(self, cfg: DetectorConfig, window_s: float = 5.0):
        self.cfg, self.window = cfg, window_s
        self.b = budget(cfg)
        self.times: list = []

    def step(self, t, bus):
        for d in bus.take("detections", self.name):
            self.times.append(d.t_capture)
        self.times = [x for x in self.times if t - x <= self.window]
        b = self.b
        bus.publish("health", Health(t, b["device"], b["model"], b["GPU load"], b["latency search [ms]"] / 1000,
                                     b["search [Hz] achieved"], b["roi [Hz] achieved"], len(self.times) / self.window,
                                     b["module power [W]"], b["status"]))
