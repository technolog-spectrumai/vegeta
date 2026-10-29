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
machines.register("monster_wolf", "monster_robots:wolf", parts=_motions("monster_motions:parts_wolf"),
                  visual="monsters:Wolf")
machines.register("monster_wyvern", "monster_robots:wyvern", parts=_motions("monster_motions:parts_wyvern"),
                  visual="monsters:Wyvern")
machines.register("monster_centaur", "monster_robots:centaur", parts=_motions("monster_motions:parts_centaur"),
                  visual="monsters:Centaur")
machines.register("monster_leech", "monster_robots:leech", parts=_motions("monster_motions:parts_leech"),
                  visual="monsters:Leech")
machines.register("monster_leshy", "monster_robots:leshy", parts=_motions("monster_motions:parts_leshy"),
                  visual="monsters:Leshy")
machines.register("monster_wraith", "monster_robots:wraith", parts=_motions("monster_motions:parts_wraith"),
                  visual="monsters:Wraith")
machines.register("monster_ghoul", "monster_robots:ghoul", parts=_motions("monster_motions:parts_ghoul"),
                  visual="monsters:Ghoul")

# fantasy objects (design stage: CAD, materials, a rigid body each; the chest's lid hinges)
for _obj, _cls in (("chest", "Chest"), ("crate", "Crate"), ("sword", "Sword"), ("shield", "Shield"), ("coin", "Coin"),
                   ("amulet", "Amulet"), ("elixir_flask", "ElixirFlask")):
    machines.register(f"fantasy_{_obj}", f"fantasy_robots:{_obj}", parts=_motions(f"fantasy_robots:parts_{_obj}"),
                      visual=f"fantasy_objects:{_cls}")
