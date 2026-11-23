"""The standard node graph of the bird-photography mission, wired on one bus.

Topics (inputs from the autopilot/camera side: ``nav``, ``camera``; outputs to it: ``guidance``):

    camera ──► detector ──► detections ──► tracker ──► tracks ──► scheduler ──► target ──► guidance ──► guidance cmd
                  ▲                                       │            ▲                       ▲
                  └──────────── target, tracks (ROI) ◄────┘            └── photos ◄── photo ◄──┘  (+ nav everywhere)
    compute monitor: detections ──► health

``MissionStack.tick(t, nav, frame)`` publishes the inputs and runs every node that is due; ``command()`` is the
latest ``GuidanceCmd``. The detector is passed in (simulated or real), everything else is the flight software.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .bus import Bus, Executor
from .camera import IMX219, PinholeCamera
from .compute import ComputeMonitor, DetectorConfig
from .guidance import GuidanceConfig, PhotoPassGuidance
from .photo import PhotoConfig, PhotoTrigger
from .targeting import SchedulerConfig, TargetScheduler
from .tracking import BirdTracker, TrackerConfig

__all__ = ["MissionConfig", "MissionStack"]


@dataclass
class MissionConfig:
    camera: PinholeCamera = IMX219
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    photo: PhotoConfig = field(default_factory=PhotoConfig)
    rates: dict = field(default_factory=lambda: {"detector": 100.0, "tracker": 20.0, "scheduler": 2.0, "guidance": 10.0,
                                                 "photo": 20.0, "compute": 1.0})


class MissionStack:
    def __init__(self, cfg: MissionConfig, detector, *, record: bool = True, t0: float = 0.0):
        self.cfg = cfg
        self.bus = Bus(record=record)
        self.ex = Executor(self.bus)
        r = cfg.rates
        self.detector = self.ex.add(detector, r["detector"], t0)
        self.tracker = self.ex.add(BirdTracker(cfg.camera, cfg.tracker), r["tracker"], t0)
        self.scheduler = self.ex.add(TargetScheduler(cfg.scheduler), r["scheduler"], t0)
        self.guidance = self.ex.add(PhotoPassGuidance(cfg.guidance), r["guidance"], t0)
        self.photo = self.ex.add(PhotoTrigger(cfg.camera, cfg.photo), r["photo"], t0)
        self.compute = self.ex.add(ComputeMonitor(cfg.detector), r["compute"], t0)
        self.frame_id = 0

    def tick(self, t, nav, frame=None, **extra):
        """Publish the autopilot's state (and a camera frame, and simulator topics in ``extra``) and run the due nodes."""
        self.bus.publish("nav", nav)
        if frame is not None:
            self.bus.publish("camera", frame)
        for topic, msg in extra.items():
            self.bus.publish(topic, msg)
        self.ex.spin_until(t)

    def command(self):
        return self.bus.latest("guidance")
