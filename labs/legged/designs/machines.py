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


def _persephone():
    """Persephone, the 12-segment chimney Myropod (notebook 17): the benchmark campaign's Chiron model, built from
    myropod_robot's blocks with notebook 17's mass budget (``campaigns/benchmark/persephone/persephone_robot.py``)."""
    import importlib.util
    from pathlib import Path
    f = Path(__file__).resolve().parents[1] / "campaigns" / "benchmark" / "persephone" / "persephone_robot.py"
    spec = importlib.util.spec_from_file_location("persephone_robot", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.persephone()


# the Myriapods, with their CAD looks (Cleopatra and Apheloria with their motions; Persephone without: no gait yet)
machines.register("apheloria", "apheloria_robot:apheloria", parts=_motions("myriapod_motions:parts_apheloria"),
                  visual="apheloria:Apheloria")
machines.register("myropod", "myropod_robot:cleopatra", parts=_motions("myriapod_motions:parts_cleopatra"),
                  visual="myropod_robot:cleopatra_cad")
machines.register("persephone", _persephone, visual="myropod:Myropod")    # the design's defaults are Persephone

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
