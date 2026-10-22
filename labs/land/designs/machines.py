"""The machines this lab exports (``tools/export.py``): name → the Chiron robot builder (``vegeta.labkit.machines``)."""
from vegeta.labkit import machines

machines.register("onager", "onager_robot:onager")
machines.register("onager_atlas", "onager_atlas_robot:atlas")
machines.register("onager_manus", "onager_manus_robot:manus")
machines.register("onager_sweeper", "onager_sweeper_robot:sweeper")
def _pekari_parts(m) -> dict:
    """Pekari's part data: its materials (densities notebook 30 fixed), the rollers' axles as pinned connections, and the
    link materials the module states (the hull shell, the wheels' moulding); tagged ``nb30`` as the notebook's."""
    import pekari_rover_robot as prr
    links = {"hull": "Al 6061-T6 hull shell"}
    for lk in m.links:
        if lk.name.endswith("_link") and "bogie" not in lk.name:
            links[lk.name] = "PA6-GF30 links, sprockets, wheels"
    for lk in m.links:
        if lk.name in links:
            lk.data["material"] = links[lk.name]
    from vegeta.labkit import detach as ld
    kinds = {"hull": "body", **{lk.name: ("bogie" if "bogie" in lk.name else "roller") for lk in m.links if lk.name != "hull"}}
    asm = ld.from_joint_tree(m, name="Pekari Rover", kinds=kinds,
                             source="pekari_rover_robot: the hull, its bogies and rollers on their axles; no component is declared detachable "
                                    "(notebook 30 fixes no separation)")
    return {"materials": prr.materials_block(), "connections": prr.connections_block(m), "assembly": asm,
            "tags": {"data:materials*": "nb30", "data:connections*": "nb30"}}


machines.register("pekari", "pekari_rover_robot:pekari", parts=_pekari_parts)
machines.register("perseus", "perseus_robot:perseus")
machines.register("perseus_mini", "perseus_mini_robot:mini")
