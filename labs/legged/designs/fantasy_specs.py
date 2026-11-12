"""The fantasy objects' numbers, without CadQuery: dimensions [mm, deg], the material of each ``part`` and the
materials catalogue (Yax ``materials``: density, stiffness, strength; handbook values) the masses come from."""
from __future__ import annotations

__all__ = ["MATERIALS", "CHEST", "CRATE", "SWORD", "SHIELD", "COIN", "AMULET", "FLASK", "OBJECTS"]

#: Yax's materials shape: {name: {density_kg_m3, E_pa, nu, yield_pa, ultimate_pa, source}}
MATERIALS = {
    "oak": {"density_kg_m3": 750.0, "E_pa": 11.0e9, "nu": 0.35, "yield_pa": None, "ultimate_pa": 90e6,
            "source": "European oak, 12 % moisture, along the grain (Wood Handbook, FPL-GTR-282)"},
    "pine": {"density_kg_m3": 510.0, "E_pa": 9.0e9, "nu": 0.35, "yield_pa": None, "ultimate_pa": 80e6,
             "source": "Scots pine, 12 % moisture, along the grain (Wood Handbook)"},
    "wrought iron": {"density_kg_m3": 7700.0, "E_pa": 190e9, "nu": 0.28, "yield_pa": 210e6, "ultimate_pa": 340e6,
                     "source": "wrought iron, typical (ASM Handbook vol. 1)"},
    "spring steel": {"density_kg_m3": 7850.0, "E_pa": 205e9, "nu": 0.29, "yield_pa": 1170e6, "ultimate_pa": 1400e6,
                     "source": "AISI 5160 quenched and tempered, the blade steel of modern swords"},
    "leather": {"density_kg_m3": 860.0, "E_pa": 0.2e9, "nu": 0.4, "yield_pa": None, "ultimate_pa": 20e6,
                "source": "vegetable-tanned cowhide, typical"},
    "gold 22k": {"density_kg_m3": 17800.0, "E_pa": 79e9, "nu": 0.42, "yield_pa": 205e6, "ultimate_pa": 220e6,
                 "source": "22 carat gold (91.7 % Au), typical"},
    "bronze": {"density_kg_m3": 8800.0, "E_pa": 110e9, "nu": 0.34, "yield_pa": 150e6, "ultimate_pa": 350e6,
               "source": "tin bronze CuSn10, cast"},
    "garnet": {"density_kg_m3": 4000.0, "E_pa": 240e9, "nu": 0.27, "yield_pa": None, "ultimate_pa": None,
               "source": "almandine garnet (the amulet's red gem)"},
    "soda-lime glass": {"density_kg_m3": 2500.0, "E_pa": 70e9, "nu": 0.22, "yield_pa": None, "ultimate_pa": 50e6,
                        "source": "soda-lime glass, annealed"},
    "cork": {"density_kg_m3": 240.0, "E_pa": 20e6, "nu": 0.0, "yield_pa": None, "ultimate_pa": 1.5e6,
             "source": "natural cork stopper"},
    "elixir": {"density_kg_m3": 1100.0, "E_pa": None, "nu": None, "yield_pa": None, "ultimate_pa": None,
               "source": "assumed: a syrupy potion, a little denser than water"},
}

CHEST = {"length": 700.0, "width": 420.0, "height": 380.0, "wall": 22.0, "planks": 4.0, "band_width": 40.0,
         "open_deg": 0.0, "_parts": {"chest": None, "oak": "oak", "iron": "wrought iron"}}
CRATE = {"size": 500.0, "board": 15.0, "slats": 4.0, "gap": 12.0, "frame": 30.0,
         "_parts": {"crate": "pine"}}
SWORD = {"blade_length": 900.0, "blade_width": 48.0, "blade_thickness": 6.5, "fuller_width": 10.0,
         "guard_width": 220.0, "guard_thickness": 18.0, "grip_length": 200.0, "grip_diameter": 30.0,
         "grip_rings": 10.0, "pommel_diameter": 42.0,
         "_parts": {"sword": None, "steel": "spring steel", "leather": "leather"}}
SHIELD = {"diameter": 800.0, "thickness": 12.0, "curvature_radius": 1400.0, "planks": 6.0, "rim_width": 18.0,
          "boss_diameter": 160.0, "shape_heater": 0.0,
          "_parts": {"shield": None, "oak": "oak", "iron": "wrought iron"}}
COIN = {"diameter": 28.0, "thickness": 2.4, "rim_width": 1.6, "relief": 0.35, "reeds": 90.0,
        "_parts": {"coin": "gold 22k"}}
AMULET = {"setting_diameter": 42.0, "setting_width": 4.0, "setting_thickness": 5.0, "gem_length": 30.0,
          "gem_width": 22.0, "gem_height": 10.0, "chain_links": 60.0, "wire_diameter": 1.6,
          "chain_radius": 110.0, "_parts": {"amulet": None, "bronze": "bronze", "gem": "garnet"}}
FLASK = {"bulb_diameter": 110.0, "glass_thickness": 3.0, "neck_diameter": 26.0, "neck_length": 70.0,
         "base_height": 6.0, "cork_length": 30.0, "fill": 0.6,
         "_parts": {"flask": None, "glass": "soda-lime glass", "cork": "cork", "elixir": "elixir"}}

#: machine name -> (design class, spec)
OBJECTS = {"chest": ("Chest", CHEST), "crate": ("Crate", CRATE), "sword": ("Sword", SWORD),
           "shield": ("Shield", SHIELD), "coin": ("Coin", COIN), "amulet": ("Amulet", AMULET),
           "elixir_flask": ("ElixirFlask", FLASK)}
