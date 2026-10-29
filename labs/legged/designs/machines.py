"""The machines this lab exports (``tools/export.py``): name → the Chiron robot builder (``vegeta.labkit.machines``)."""
from vegeta.labkit import machines

machines.register("apheloria", "apheloria_robot:apheloria")
machines.register("myropod", "myropod_robot:cleopatra")
machines.register("robot_dog", "robot_dog_robot:dog_robot")
machines.register("humanoid", "humanoid_robot:talia", visual="humanoid:Humanoid")

# zoomorphic designs (design stage: CAD + robot model, no simulation yet)
machines.register("zoo_dog", "zoo_robots:dog", visual="zoo_quadruped:Dog")
machines.register("zoo_bear", "zoo_robots:bear", visual="zoo_quadruped:Bear")
machines.register("zoo_deer", "zoo_robots:deer", visual="zoo_quadruped:Deer")
machines.register("zoo_elephant", "zoo_robots:elephant", visual="zoo_quadruped:Elephant")
machines.register("zoo_snake", "zoo_robots:snake", visual="zoo_others:Snake")
machines.register("zoo_bat", "zoo_robots:bat", visual="zoo_others:Bat")
machines.register("zoo_octopus", "zoo_robots:octopus", visual="zoo_others:Octopus")

# monsters (design stage: CAD + robot model, no simulation yet)
machines.register("monster_wolf", "monster_robots:wolf", visual="monsters:Wolf")
machines.register("monster_wyvern", "monster_robots:wyvern", visual="monsters:Wyvern")
machines.register("monster_centaur", "monster_robots:centaur", visual="monsters:Centaur")
machines.register("monster_leech", "monster_robots:leech", visual="monsters:Leech")
machines.register("monster_leshy", "monster_robots:leshy", visual="monsters:Leshy")
machines.register("monster_wraith", "monster_robots:wraith", visual="monsters:Wraith")
machines.register("monster_ghoul", "monster_robots:ghoul", visual="monsters:Ghoul")
