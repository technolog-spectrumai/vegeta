"""vegeta.bionic — shared primitives for human-robot limbs (``docs/bionic.md``).

Stage 1: a kinematic hand (``hand``), sensor models (``sensors``), scripted motions (``motions``), a simulated data
glove (``glove.SimGlove``), calibration and orientation estimation (``estimate``), the glove-to-host link with a
latency/loss model (``link``) and safety primitives (``safety``). Next: retargeting to robot pincers, haptic feedback
and the teleop loop (stage 2), the glove's CAD and electronics, force feedback, the puppet arm, exoskeletons.

    from vegeta.bionic import SimGlove, GloveSpec, Calibration, motions, estimate
    glove = SimGlove(GloveSpec.default("R"), seed=1)
    cal_samples, _ = glove.run(motions.calibration_routine())
    cal = Calibration.fit(cal_samples, motions.calibration_routine().windows, glove.spec)
    cal.with_magnetometer(glove.run(motions.wrist_tour(20.0))[0])          # wave the hand: the hard-iron offset
    samples, truth = glove.run(motions.grasp_cycles())
    poses = estimate.estimate_hand(samples, cal)
"""
from . import estimate, hand, link, motions, safety, sensors
from .estimate import Calibration, Mahony
from .glove import GloveSample, GloveSpec, SimGlove
from .hand import FINGERS, JOINTS, RANGES, HandPose, HandSize, fingertips, key_pose, pinch_aperture
from .link import InProcessLink, decode, encode
from .safety import Limits, Watchdog

__version__ = "0.1.0"

__all__ = ["Calibration", "FINGERS", "GloveSample", "GloveSpec", "HandPose", "HandSize", "InProcessLink", "JOINTS",
           "Limits", "Mahony", "RANGES", "SimGlove", "Watchdog", "decode", "encode", "estimate", "fingertips", "hand",
           "key_pose", "link", "motions", "pinch_aperture", "safety", "sensors"]
