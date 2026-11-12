"""The machines this lab exports (``tools/export.py``): name → the Chiron robot builder, and the part data a simulator
needs that MJCF cannot hold (``vegeta.labkit.machines``)."""
from __future__ import annotations

from vegeta.labkit import machines

# The propellers leave as Yax part data (propulsion_parts: Drongo's rotors for Gazebo and the full propulsion block,
# the fixed wings' maps from their propulsion files); the export's --also jsbsim writes them as JSBSim propellers.
import propulsion_parts as pp  # noqa: E402

def _drongo_parts(m) -> dict:
    """Drongo's part data and the design's own tags on it (vendor/yax docs/interchange.md): ``boreas`` on what the BEMT
    computed, ``nb08`` on what notebook 08 fixed (its materials, its motor and battery), ``assumed`` on the rotor
    constants that are assumptions (ogma.rotors: T_max, spin-up; see ASSUMED_PROPULSION)."""
    import drongo_robot as dr
    m.link("drongo").data["material"] = "PETG-CF"                               # the frame (QuadFrame, PETG-CF, notebook 08)
    from vegeta.labkit import detach as ld
    asm = ld.from_joint_tree(m, name="Drongo", kinds={"drongo": "body"},
                             source="drongo_robot: the frame and its jaw carriages (the propellers and motors are geoms of the frame, not links; "
                                    "no component is declared detachable — notebook 08b fixes no separation)")
    return {**pp.drongo(m), "materials": dr.materials_block(), "assembly": asm,
            "tags": {"data:propulsion/propellers*": "boreas", "data:materials*": "nb08", "data:propulsion/motors*": "nb08",
                     "data:propulsion/battery*": "nb08", "data:ogma*": "assumed", "data:propulsion/reduced*": "assumed"}}


machines.register("drongo", "drongo_robot:drongo", parts=_drongo_parts, visual="drongo:Drongo")
machines.register("quad", "quad_robot:quad", visual="quadcopter_assembly:Quadcopter")
machines.register("falco", "falco_robot:falco_robot", parts=pp.falco)
machines.register("nisus", "nisus_robot:nisus_robot", parts=pp.nisus)
machines.register("nisus_plus", "nisus_plus_robot:nisus_plus_robot", parts=pp.nisus_plus)
def _drako_parts(m) -> dict:
    """Drako's rotors and drive (propulsion_parts.drako), the frame's material and the tags: ``boreas`` on the BEMT map,
    ``assumed`` on the motor, the battery and the reduced constants (catalogue-class values, drako_robot)."""
    m.link("drako").data["material"] = "carbon"
    return {**pp.drako(m), "tags": {"data:propulsion/propellers*": "boreas", "data:propulsion/motors*": "assumed",
                                    "data:propulsion/battery*": "assumed", "data:ogma*": "assumed",
                                    "data:propulsion/reduced*": "assumed"}}


machines.register("drako", "drako_robot:drako", parts=_drako_parts, visual="drako:Drako")
