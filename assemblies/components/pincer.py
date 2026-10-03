"""The pincer and the arm that carries it: an index of what the machines already share (copied unchanged).

- Onager Manus (notebook 22), reused by the Sweeper (23): the CAD in ``onager_manus.OnagerManus`` (parts ``upper_arm``,
  ``forearm``, ``jaw``: the V cutter notch and the hooked tip), the Chiron arm ``onager_manus_robot._arm`` (yaw, shoulder,
  elbow, wrist, two jaw hinges), its IK ``arm_ik`` / ``arm_fk``, the actuators ``ARM_ACTUATORS``, the stowed pose
  ``STOW``, the grip modes of ``onager_manus_controller.Mission``.
- Myropod v3 (notebook 17): the pincer on the head, ``myropod.Myropod`` part ``pincer``.
"""
from __future__ import annotations

from . import onager_manus as _manus_cad
from . import onager_manus_robot as _manus

arm = _manus._arm
arm_ik = _manus.arm_ik
arm_fk = _manus.arm_fk
ARM_ACTUATORS = _manus.ARM_ACTUATORS
STOW = _manus.STOW
OnagerManus = _manus_cad.OnagerManus
ARM_PARTS = ("upper_arm", "forearm", "jaw")


def arm_parts(**overrides) -> dict:
    """The Manus arm's parts as Dedalus geometries (``OnagerManus`` parts), with parameters other than the defaults."""
    d = OnagerManus()
    return {part: d.generate(**dict(overrides, part=part)) for part in ARM_PARTS}
