"""The machines this lab exports (``tools/export.py``, ``tools/publish.py``): the sea game's vessels and its site as
rigid bodies (``sea_robots``) with their CAD as their looks and their published Ketos card as part data (``ketos``)."""
from vegeta.labkit import machines

from sea_robots import alien, alien_visual, craft, craft_visual, ketos_card

machines.register("zarco", "sea_robots:zarco", parts=ketos_card("zarco"), visual="zarco:Zarco")
machines.register("esmeralda", "sea_robots:esmeralda", parts=ketos_card("esmeralda"), visual="esmeralda:Esmeralda")
machines.register("barracuda", "sea_robots:barracuda", parts=ketos_card("barracuda"), visual="barracuda:Barracuda")
machines.register("squilla", "sea_robots:squilla", parts=ketos_card("squilla"), visual="squilla:Squilla")
machines.register("resident_dock", "sea_robots:resident_dock", parts=ketos_card("resident_dock"), visual="oil_platform:ResidentDock")
machines.register("jacket", "sea_robots:jacket", parts=ketos_card("jacket"), visual="oil_platform:Jacket")

for _n in ("fishing_boat", "pilot_boat", "yacht", "skiff"):
    machines.register(_n, (lambda n: lambda: craft(n))(_n), parts=ketos_card(_n), visual=craft_visual(_n))

for _n in ("alien_ship", "alien_submarine"):
    machines.register(_n, (lambda n: lambda: alien(n))(_n), parts=ketos_card(_n), visual=alien_visual(_n))
