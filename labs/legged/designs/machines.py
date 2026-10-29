"""The machines this lab exports (``tools/export.py``): name → the Chiron robot builder (``vegeta.labkit.machines``)."""
from vegeta.labkit import machines

machines.register("apheloria", "apheloria_robot:apheloria")
machines.register("myropod", "myropod_robot:cleopatra")
machines.register("robot_dog", "robot_dog_robot:dog_robot")


def _motions(module):
    """The machine's motions (``<module>.parts``: ``data:motions``, tag ``motion``), imported when exported."""
    def parts(m):
        import importlib
        return importlib.import_module(module).parts(m)
    return parts


machines.register("humanoid", "humanoid_robot:talia", parts=_motions("humanoid_motions"), visual="humanoid:Humanoid")
