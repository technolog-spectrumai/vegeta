"""The machines this lab exports (``tools/export.py``): name → the Chiron robot builder, and the part data a simulator
needs that MJCF cannot hold (``vegeta.labkit.machines``)."""
from __future__ import annotations

from vegeta.labkit import machines

# The propellers leave as Yax part data (propulsion_parts: Drongo's rotors for Gazebo and the full propulsion block,
# the fixed wings' maps from their propulsion files); the export's --also jsbsim writes them as JSBSim propellers.
import propulsion_parts as pp  # noqa: E402

def _drongo_parts(m) -> dict:
    import drongo_robot as dr
    return {**pp.drongo(m), "materials": dr.materials_block()}


machines.register("drongo", "drongo_robot:drongo", parts=_drongo_parts)
machines.register("falco", "falco_robot:falco_robot", parts=pp.falco)
machines.register("nisus", "nisus_robot:nisus_robot", parts=pp.nisus)
machines.register("nisus_plus", "nisus_plus_robot:nisus_plus_robot", parts=pp.nisus_plus)
