# Mellonia — 3D-print manufacturability (PrusaSlicer)

Mellonia slices any STL with the PrusaSlicer command line, using explicit printer, filament and print
settings and an orientation chosen by the engineer. It does not optimise orientation. The G-code is
kept, and so is the full configuration PrusaSlicer used.

## Settings in Python
```python
from mellonia import PrintSettings, Orientation, slice_stl

settings = PrintSettings(
    name="my printer PLA 0.2",
    printer={"bed_shape": "0x0,250x0,250x210,0x210", "max_print_height": 210, "nozzle_diameter": 0.4},
    filament={"filament_diameter": 1.75, "filament_density": 1.24, "filament_cost": 25,
              "temperature": 210, "bed_temperature": 60},
    print={"layer_height": 0.2, "first_layer_height": 0.2, "perimeters": 2, "fill_density": "20%"},
)
finer = settings.replace(print={"layer_height": 0.1})
```
Keys are PrusaSlicer's own option names. The keys in `mellonia.REQUIRED_KEYS` must be given;
everything else falls back to PrusaSlicer defaults, and every run says how many settings did and
writes them all to `effective_config.ini`. PrusaSlicer silently ignores unknown keys, so Mellonia
compares your keys with the effective configuration and **fails** on any it does not know (typos).
Existing PrusaSlicer exports: `PrintSettings.from_ini(printer_ini, filament_ini, print_ini)`.
`mellonia.examples.GENERIC_PLA_0_2MM` is an illustration, not a validated profile.

## Slicing
```python
res = slice_stl("part.stl", settings, Orientation(rotate_x=90), "runs/part_x90")
res.metrics["estimated_time_s"], res.metrics["filament_used_g"], res.metrics["layer_count"]
mellonia.plot_layers(res)                       # filament per layer vs Z
```
`Orientation(rotate_x, rotate_y, rotate_z)` in degrees is passed to PrusaSlicer as `--rotate-x`,
`--rotate-y`, `--rotate`; PrusaSlicer places the rotated part on the bed.

| metric | source |
|--------|--------|
| `estimated_time_s`, `estimated_time_silent_s`, `first_layer_time_s` | G-code summary comments |
| `filament_used_mm`, `filament_used_cm3` | G-code summary comments |
| `filament_used_g`, `filament_cost` | only when `filament_density` / `filament_cost` are set, else `None` |
| `layer_count`, `max_z` | counted from `;LAYER_CHANGE` / `;Z:` markers |
| `layer_height`, `first_layer_height` | effective configuration |

Artifacts: `inputs/<stl>`, `profiles/{printer,filament,print}.ini`, `<stem>.gcode`,
`effective_config.ini`, `prusaslicer.log`, `summary.json`. Existing G-code:
`mellonia.read_gcode(path)`.

## CLI
```
mellonia slice part.stl -s settings.py:FINE --rotate-x 90 -o runs/part_x90 [--png] [--json]
mellonia slice part.stl -s mellonia.examples:GENERIC_PLA_0_2MM -o runs/part --prusa-slicer "xvfb-run -a prusa-slicer"
mellonia parse runs/part/part.gcode [--json]
```

## Validation (tests/test_integration.py)
- 20 mm cube: layer count = 1 + ⌈(20 − h₁)/h⌉ and top Z = h₁ + (n − 1)·h (PrusaSlicer quantises layers).
- Solid (100 % rectilinear) 20 mm cube: filament volume within 3 % of 8 cm³ (observed +0.9 %).
- Per-layer extrusion parsed from the moves sums to PrusaSlicer's reported filament length (±0.5 %).
- 10×10×40 mm part: 200 layers upright, 50 layers after `rotate_x=90`.
- A misspelled key and an invalid combination (gyroid at 100 %) give failed results with the reason.
