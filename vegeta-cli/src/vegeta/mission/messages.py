"""The messages the mission software exchanges: plain, frozen dataclasses with a timestamp ``t`` [s].

Frames: the world is ENU (x east, y north, z up, metres from home). The body is x forward, y left, z up. The camera
uses the OpenCV convention (x right, y down, z along the optical axis); pixel (0, 0) is the top left corner.
Nothing here depends on a simulator or on ROS 2: the same types can later be mapped one to one onto ROS 2 messages.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["NavState", "CameraFrame", "Detection", "Detections", "TrackEstimate", "Tracks", "TargetSelection", "GuidanceCmd",
           "CameraCmd", "PhotoEvent", "Health", "MissionStatus"]


@dataclass(frozen=True)
class NavState:
    """The aircraft's navigation solution (from the autopilot)."""
    t: float
    pos: np.ndarray                 # (3,) world [m]
    vel: np.ndarray                 # (3,) world ground velocity [m/s]
    R_wb: np.ndarray                # (3, 3) body → world rotation
    omega_b: np.ndarray             # (3,) body angular rate [rad/s]
    airspeed: float
    energy_margin_wh: float = float("inf")   # energy above what the return needs (the autopilot's energy manager)
    phase: str = "mission"

    @property
    def course(self) -> float:
        return float(np.arctan2(self.vel[1], self.vel[0]))


@dataclass(frozen=True)
class CameraFrame:
    """One exposure: when, from where, and (on hardware) the pixels."""
    t: float
    frame_id: int
    R_wc: np.ndarray                # (3, 3) camera → world
    p_wc: np.ndarray                # (3,) camera centre, world
    omega_b: np.ndarray             # body rate during the exposure (motion blur)
    image: np.ndarray | None = None


@dataclass(frozen=True)
class Detection:
    box: tuple                      # (x0, y0, x1, y1) full-resolution pixels
    score: float
    label: str = "bird"

    @property
    def centre(self):
        return 0.5 * (self.box[0] + self.box[2]), 0.5 * (self.box[1] + self.box[3])

    @property
    def width(self):
        return self.box[2] - self.box[0]


@dataclass(frozen=True)
class Detections:
    """The detector's output for one frame, published ``latency`` after the exposure (``t`` is the publish time)."""
    t: float
    t_capture: float
    frame_id: int
    R_wc: np.ndarray
    p_wc: np.ndarray
    items: tuple                    # of Detection
    source: str = "search"          # 'search' (full frame) or 'roi' (crop around the target)
    roi: tuple | None = None        # the crop (x0, y0, x1, y1) in full-resolution pixels


@dataclass(frozen=True)
class TrackEstimate:
    id: int
    status: str                     # 'tentative', 'confirmed', 'lost'
    pos: np.ndarray                 # (3,) world
    vel: np.ndarray                 # (3,)
    cov: np.ndarray                 # (6, 6) position-velocity covariance
    turn_rate: float                # rad/s (coordinated-turn model's estimate)
    p_turn: float                   # IMM probability of the turning model
    hits: int
    age: float                      # s since birth
    since_update: float             # s since the last detection
    box: tuple | None = None        # the last associated detection's box
    score: float = 0.0


@dataclass(frozen=True)
class Tracks:
    t: float
    items: tuple                    # of TrackEstimate

    def get(self, track_id):
        return next((k for k in self.items if k.id == track_id), None)


@dataclass(frozen=True)
class TargetSelection:
    t: float
    track_id: int | None
    reason: str
    cost: float = float("nan")
    done: tuple = ()                # ids photographed well enough (released)
    blacklist: tuple = ()           # ids given up on for now


@dataclass(frozen=True)
class GuidanceCmd:
    """What the autopilot should fly: a course over the ground, a height, an airspeed. ``valid`` False: no command (the
    autopilot holds or returns by its own rules)."""
    t: float
    mode: str                       # 'search', 'intercept', 'pass', 'breakoff', 'reposition', 'done'
    course: float = 0.0             # rad, world (atan2 of the ground track)
    height: float = 80.0            # m
    airspeed: float = 15.0          # m/s
    valid: bool = True
    note: str = ""


@dataclass(frozen=True)
class CameraCmd:
    t: float
    roi: tuple | None               # crop for the high-rate target detection
    target_id: int | None = None


@dataclass(frozen=True)
class PhotoEvent:
    """A full-resolution still the trigger fired, with the trigger's own (estimated) quality."""
    t: float
    track_id: int
    range_m: float
    pixels_across: float
    blur_px: float
    centring: float                 # 0 at the image centre, 1 at the edge
    score: float
    good: bool


@dataclass(frozen=True)
class Health:
    t: float
    device: str
    model: str
    gpu_load: float                 # share of the GPU the detector is planned to use
    latency_s: float                # detector latency (search frame)
    search_hz: float
    roi_hz: float
    measured_hz: float = float("nan")
    power_w: float = float("nan")
    notes: str = ""


@dataclass(frozen=True)
class MissionStatus:
    t: float
    state: str                      # 'running', 'complete', 'aborted'
    photographed: tuple = ()
    reason: str = ""
    extra: dict = field(default_factory=dict)
