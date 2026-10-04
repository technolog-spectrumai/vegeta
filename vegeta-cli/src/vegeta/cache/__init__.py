"""Simulation results cached next to the notebook: the user-facing switch for Talos (FEA) and Aeromant (CFD).

    from vegeta import cache
    cache.notebook("08_quadcopter")                          # entries in ./08_quadcopter.cache/
    r = model.solve(RUNS / "thrust", cache="frame_thrust")  # solved once, then loaded
    c = case.run(cache="canopy_15ms")
    cache.entries(); cache.clear("frame_thrust"); cache.clear()

The entry exists -> it is loaded and nothing runs; it does not -> the simulation runs and, when it succeeds, is
saved. Nothing checks whether the inputs or the code changed: delete the entry when they do. The implementation is
``vegeta.talos.cache`` (Aeromant has an identical copy; the settings are shared through the process environment).
"""
from vegeta.talos.cache import (clear, directory, disable, enable, enabled, entries, exists, files_dir, notebook,  # noqa: F401
                                path)
