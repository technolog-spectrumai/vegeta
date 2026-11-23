"""Example settings. Generic illustration values — verify them for your printer and material."""
from .settings import PrintSettings

GENERIC_PLA_0_2MM = PrintSettings(
    name="generic PLA 0.2 mm (example)",
    source="illustrative generic values, not a validated printer profile",
    printer={
        "bed_shape": "0x0,250x0,250x210,0x210",
        "max_print_height": 210,
        "nozzle_diameter": 0.4,
        "gcode_flavor": "marlin2",
    },
    filament={
        "filament_diameter": 1.75,
        "filament_density": 1.24,
        "filament_cost": 25,
        "temperature": 210,
        "first_layer_temperature": 215,
        "bed_temperature": 60,
        "first_layer_bed_temperature": 60,
    },
    print={
        "layer_height": 0.2,
        "first_layer_height": 0.2,
        "perimeters": 2,
        "top_solid_layers": 5,
        "bottom_solid_layers": 4,
        "fill_density": "20%",
        "fill_pattern": "gyroid",
        "skirts": 1,
        "support_material": 0,
    },
)
