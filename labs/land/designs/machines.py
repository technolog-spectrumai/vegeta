"""The machines this lab exports (``tools/export.py``): name → the Chiron robot builder (``vegeta.labkit.machines``)."""
from vegeta.labkit import machines

machines.register("onager", "onager_robot:onager")
machines.register("onager_atlas", "onager_atlas_robot:atlas")
machines.register("onager_manus", "onager_manus_robot:manus")
machines.register("onager_sweeper", "onager_sweeper_robot:sweeper")
machines.register("pekari", "pekari_rover_robot:pekari")
machines.register("perseus", "perseus_robot:perseus")
machines.register("perseus_mini", "perseus_mini_robot:mini")
