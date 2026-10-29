"""The machines this lab exports (``tools/export.py``): name → the Chiron robot builder (``vegeta.labkit.machines``)."""
from vegeta.labkit import machines

machines.register("apheloria", "apheloria_robot:apheloria")
machines.register("myropod", "myropod_robot:cleopatra")
machines.register("robot_dog", "robot_dog_robot:dog_robot")
machines.register("humanoid", "humanoid_robot:talia", visual="humanoid:Humanoid")
