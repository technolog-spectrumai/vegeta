"""The machines this lab exports (``tools/export.py``, ``tools/publish.py``): the sea game's vessels as rigid bodies
(``sea_robots``) with their CAD as their looks."""
from vegeta.labkit import machines

machines.register("zarco", "sea_robots:zarco", visual="zarco:Zarco")
machines.register("esmeralda", "sea_robots:esmeralda", visual="esmeralda:Esmeralda")
machines.register("barracuda", "sea_robots:barracuda", visual="barracuda:Barracuda")
