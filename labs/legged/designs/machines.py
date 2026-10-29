"""The machines this lab exports (``tools/export.py``): name → the Chiron robot builder (``vegeta.labkit.machines``)."""
from vegeta.labkit import machines

machines.register("robot_dog", "robot_dog_robot:dog_robot")


def _motions(module):
    """The machine's motions (``module[:function]``, default ``parts``: ``data:motions``, tag ``motion``), imported
    when exported."""
    name, _, fn = module.partition(":")

    def parts(m):
        import importlib
        return getattr(importlib.import_module(name), fn or "parts")(m)
    return parts


# the Myriapods, with their motions and their CAD looks (Persephone is CAD-only: no robot model)
machines.register("apheloria", "apheloria_robot:apheloria", parts=_motions("myriapod_motions:parts_apheloria"),
                  visual="apheloria:Apheloria")
machines.register("myropod", "myropod_robot:cleopatra", parts=_motions("myriapod_motions:parts_cleopatra"),
                  visual="myropod_robot:cleopatra_cad")

machines.register("humanoid", "humanoid_robot:talia", parts=_motions("humanoid_motions"), visual="humanoid:Humanoid")

# zoomorphic designs (design stage: CAD + robot model, no simulation yet)
machines.register("zoo_dog", "zoo_robots:dog", parts=_motions("zoo_motions:parts_dog"), visual="zoo_quadruped:Dog")
machines.register("zoo_bear", "zoo_robots:bear", parts=_motions("zoo_motions:parts_bear"), visual="zoo_quadruped:Bear")
machines.register("zoo_deer", "zoo_robots:deer", parts=_motions("zoo_motions:parts_deer"), visual="zoo_quadruped:Deer")
machines.register("zoo_elephant", "zoo_robots:elephant", parts=_motions("zoo_motions:parts_elephant"),
                  visual="zoo_quadruped:Elephant")
machines.register("zoo_snake", "zoo_robots:snake", parts=_motions("zoo_motions:parts_snake"), visual="zoo_others:Snake")
machines.register("zoo_bat", "zoo_robots:bat", parts=_motions("zoo_motions:parts_bat"), visual="zoo_others:Bat")
machines.register("zoo_octopus", "zoo_robots:octopus", parts=_motions("zoo_motions:parts_octopus"),
                  visual="zoo_others:Octopus")

# monsters (design stage: CAD + robot model, no simulation yet)
machines.register("monster_wolf", "monster_robots:wolf", visual="monsters:Wolf")
machines.register("monster_wyvern", "monster_robots:wyvern", visual="monsters:Wyvern")
machines.register("monster_centaur", "monster_robots:centaur", visual="monsters:Centaur")
machines.register("monster_leech", "monster_robots:leech", visual="monsters:Leech")
machines.register("monster_leshy", "monster_robots:leshy", visual="monsters:Leshy")
machines.register("monster_wraith", "monster_robots:wraith", visual="monsters:Wraith")
machines.register("monster_ghoul", "monster_robots:ghoul", visual="monsters:Ghoul")
