# Atropos — flat parts for a diode laser (ribs, kerf, nesting, SVG, G-code)

Atropos draws the flat parts the engineer specifies — wing ribs from a section, with the skin let in, slots, spar
holes, lightening holes and the hinge split — compensates the kerf, nests them on stock sheets and writes one SVG
and one GRBL G-code file per sheet, with the cut length, the time and the balsa they take. It picks no laser
setting: the sheet, the machine and every operation's power, feed, passes and kerf are stated by the engineer
(`vegeta.atropos.examples` is an illustration, not a validated profile — cut a test strip and measure the kerf
first). SVG (mm, one layer per operation) loads in xTool Creative Space and LightBurn; the G-code streams to a GRBL
machine such as the xTool D1 Pro from LightBurn or any sender.

## Settings in Python
```python
from vegeta import atropos
from vegeta.atropos import Sheet, Machine, Operation, CutSettings

sheet = Sheet("balsa 400 x 100 x 2", width_mm=400, height_mm=100, thickness_mm=2.0, density_g_cm3=0.16, grain="x", margin_mm=4)
d1 = Machine("xTool D1 Pro 10 W", bed_width_mm=432, bed_height_mm=406, travel_feed_mm_min=6000, s_max=1000)
ops = {"cut": Operation("cut", power_pct=100, feed_mm_min=600, passes=2, kerf_mm=0.15, order=2),
       "score": Operation("score", power_pct=25, feed_mm_min=1500, passes=1, kerf_mm=0.0, order=1),
       "engrave": Operation("engrave", power_pct=12, feed_mm_min=3000, passes=1, kerf_mm=0.0, order=0)}
settings = CutSettings(sheet, d1, ops, spacing_mm=3.0, rotate=True, labels=False)
thinner = settings.replace(sheet=atropos.examples.BALSA_400x100_1_5MM)
```
`cut` is required; `score` and `engrave` only when parts carry such lines. The sheet must fit the bed; its
front-left corner is the machine origin. Power is `power_pct / 100 × s_max` in the `S` word.

## Ribs
```python
up, lo = atropos.naca4(0.03, 0.4, 0.12)                        # or any section at chord 1, TE -> LE, as the designs use
spec = atropos.RibSpec("R07", chord_mm=250, upper=up, lower=lo,
    sheeting=[atropos.Sheeting(0.0, 0.32, 1.5), atropos.Sheeting(0.88, 1.0, 1.5)],   # the D-box and the trailing edge
    tubes=[atropos.TubeHole(0.30, 20.0, clearance_mm=0.3), atropos.TubeHole(0.55, 18.0, 0.3, name="rear")],
    notches=[atropos.Notch(0.42, "upper", 6.0, 2.0), atropos.Notch(0.42, "lower", 6.0, 2.0)],       # cap strips
    lightening=[atropos.Lightening(0.44, 30.0, wall_mm=4.0)],
    le_cut_mm=6.0, te_cut_mm=12.0,                              # leading- and trailing-edge stock take the nose and tail
    split_at_frac=0.70, split_gap_mm=1.5, bevel_deg=12.0)       # hinge line: R07-fwd and R07-aft
fwd, aft = atropos.make_rib(spec)
fwd.area(), fwd.cut_length(), fwd.holes
```
A tube hole that does not fit inside the rib (or leaves less than `min_wall_mm`) raises `ValueError`; lightening
holes shrink until the wall holds and are dropped below `min_diameter_mm`. A `Part` is an outer contour, holes,
score and engrave lines and metadata (`chord_mm`, `role`, `grain` ...), in mm, serialisable (`to_dict`,
`parts_to_json`) for the CLI.

## Nesting, SVG, G-code
```python
res = atropos.cut_job(parts, settings, "runs/wing_cut", name="wing")
res.metrics["sheets"], res.metrics["time_s"], res.metrics["cut_length_mm"], res.metrics["parts_mass_g"]
res.nesting.table()                                             # part, sheet, x, y, angle
atropos.plot_parts(parts); atropos.plot_nesting(res.nesting)
```
`cut_job` kerf-compensates (outer contours out by half the kerf, holes in), nests with a deterministic shelf
packer (tallest first, long side along X, a 90° turn when allowed and the grain permits), and writes
`parts.json`, `sheet_NN.svg`, `sheet_NN.gcode`, `preview.png` and `summary.json`. The G-code is
`G21 G90`, `M4 S<power>` per operation in `order`, `G0` travels, `G1 ... F<feed>` cuts repeated per pass, holes
before the contour that holds them, parts nearest-first, `M5` and home at the end. Time is path length over feed
plus travel over the rapid feed (no acceleration model: the machine is slower). A part that fits no sheet gives a
failed result. Existing files: `atropos.read_svg_paths(svg)`, `atropos.read_gcode(gcode)`.

| metric | meaning |
|--------|---------|
| `parts`, `sheets`, `utilisation` | parts placed, sheets used, mean part area / sheet area |
| `cut_length_mm` (`score_`, `engrave_`) | toolpath length per operation, passes included |
| `travel_length_mm`, `time_s` | rapid moves; the sum of all operations and travel |
| `parts_area_mm2`, `parts_mass_g`, `sheets_mass_g` | material in the parts (area × thickness × density) and bought |

## CLI
```
atropos rib --chord 250 --naca 3412 --tube 0.30:20:0.3 --tube 0.55:18:0.3 --sheeting 0:0.32:1.5 --split 0.70 --gap 1.5 --bevel 12 -o rib.svg
atropos rib --chord 250 --naca 3412 --tube 0.30:20:0.3 -o parts.json
atropos nest parts.json -s vegeta.atropos.examples:D1_PRO_BALSA_2MM -o runs/ribs [--no-gcode] [--no-png] [--json]
atropos info runs/ribs/sheet_01.gcode        # bbox, lengths, feeds, powers, time
atropos info rib.svg                         # page, paths per layer, lengths
atropos settings my_settings.py:FINE
```

## Validation (libs/vegeta-cli/tests/atropos)
- A rectangle offset by k has area A + P·k + 4k² (1e-9); a circle's offset is a radius change (1e-3); a NACA
  outline offset by 0.1 mm changes its area by P·k (0.5 %).
- A NACA 00xx section's area is 0.6851·t·c² (closed form of the thickness polynomial, 0.2 %).
- A plain rib is the section (1e-9); letting the skin in removes t × arc length (5 %); tube holes remove πd²/4 (0.2 %);
  the split pieces stop and start at the gap; a hole that does not fit raises.
- 40 ribs nest with no overlap, inside the margins, on at least ⌈Σarea / usable area⌉ sheets, reproducibly.
- The SVG parses, with one group per operation in the engineer's order; its paths re-read to the parts' lengths (1e-4).
- The G-code's G1 length is passes × cut length (1e-3), `S ≤ s_max`, the bounding box is inside the bed, the time
  is length / feed + travel / rapid; holes come before their contour.
