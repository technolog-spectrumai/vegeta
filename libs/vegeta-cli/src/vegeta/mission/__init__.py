"""vegeta.mission: onboard mission software, independent of any simulator and of ROS 2.

Typed messages (``messages``) on an in-process bus with a deterministic executor (``bus``); nodes for detection
(``detector``: scheduling, crops, latency; the YOLOX engine plugs in later), tracking (``tracking``: IMM filter per
bird, ByteTrack-style association), target selection (``targeting``), guidance for photo passes (``guidance``), the
shutter (``photo``), the Jetson compute budget (``compute``), the camera model (``camera``), the scores
(``metrics``) and the standard graph (``pipeline``). ``vegeta.mission.sim`` holds what only a simulation needs
(birds, the simulated detector, a kinematic aircraft); the flight software never imports it.
"""
from .bus import Bus, Executor, Node
from .camera import IMX219, IMX219_6MM, PinholeCamera, camera_pose
from .compute import DetectorConfig, budget, budget_table
from .messages import *  # noqa: F401,F403
from .pipeline import MissionConfig, MissionStack
