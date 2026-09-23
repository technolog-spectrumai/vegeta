# Chronos — missions, cyclic loads and long-term life

Chronos answers "how long does it last?" for a design whose static strength is already known. It
takes a **mission** (segments with explicit load levels and vibratory excitations), the structure's
**natural frequencies** (from `talos.StructuralModel.solve_modes`) and turns them into a **load
spectrum**: blocks of (pattern, mean, amplitude, cycles). Talos evaluates that spectrum on the FEA
stress fields (`talos.assess_fatigue`) to give damage per mission and a hotspot map, and Chronos
accumulates the damage over a **fleet usage** of several mission types into a life in flights and hours.

Pure numpy, no external program. Every number is an input you wrote: load levels, excitation
forces, damping, S-N curve, usage mix.

## Vocabulary
- **Pattern** — a named load shape whose level varies, e.g. `thrust` (all motors), `lateral` (an
  unbalance force at one motor mount), `landing` (one arm), `lift`. Each pattern has a *unit* FEA
  case in Talos; the spectrum's levels multiply it (linear superposition).
- **Segment** — `duration_s` at steady `loads` (pattern → level) with `excitations` superimposed;
  `repeat` stacks identical segments (twenty punch-outs).
- **Excitation** — a vibratory force amplitude at a frequency on a pattern, e.g. the rotor unbalance
  at the shaft frequency (`boreas.excitations` gives both).
- **Mission** — the ordered segments; `.profile()` draws it.
- **Structure** — natural frequencies + one damping ratio; `amplification(f)` is the single-degree-
  of-freedom dynamic amplification using the nearest mode (a bounding estimate, not a modal
  superposition); `margin(f)` the distance to the nearest mode; `campbell()` the diagram.
- **Block / LoadSpectrum** — the output: manoeuvre cycles from rainflow counting of each pattern's
  level sequence (ground → segments → ground) and vibration cycles per segment and excitation
  (`cycles = f × t`, amplitude × amplification). Saved as JSON.

## Python
```python
from vegeta import chronos

unb = chronos.Excitation("unbalance 1P", frequency_hz=137.0, amplitude=0.11, pattern="lateral")
hover = chronos.Mission("inspection hover", (
    chronos.Segment("takeoff", 10, {"thrust": 6.0}, (unb,)),
    chronos.Segment("hover", 600, {"thrust": 4.7}, (unb,)),
    chronos.Segment("landing", 5, {"thrust": 2.0, "landing": 25.0}),
))
frame = chronos.Structure(modes_hz=(210.0, 415.0, 1300.0), damping_ratio=0.03)   # from solve_modes
spec = chronos.build_spectrum(hover, frame)
spec.table(); spec.plot(); spec.save("runs/spectrum_hover.json")

# scalar check at one hotspot (stress per unit load from the unit FEA cases):
curve = chronos.SNCurve("PETG-CF", sigma_f=80.0, b=-0.11, ultimate=55.0)
chronos.hotspot_damage(spec, {"thrust": 1.9, "lateral": 4.2, "landing": 0.9}, curve)["total"]

# long term: damage per mission (from talos.assess_fatigue), hours per mission, usage mix
sim = chronos.simulate_life({"hover": 2e-4, "freestyle": 3e-3, "cruise": 5e-4},
                            {"hover": 0.2, "freestyle": 0.1, "cruise": 0.25},
                            usage={"hover": 0.5, "freestyle": 0.2, "cruise": 0.3}, n_flights=2000, seed=0)
sim.hours_to_failure; sim.plot()
```

## With Talos (the full-field version)
```python
unit = {"thrust": (talos_result_thrust, 8.0), "lateral": (talos_result_lateral, 1.0), "landing": (talos_result_landing, 40.0)}
fat = talos.assess_fatigue(unit, "runs/spectrum_hover.json", talos.FatigueCurve("PETG-CF", 80, -0.11, ultimate=55))
fat.result.metrics["passes_to_failure"], fat.hotspot            # per node: fat.damage
talos.viz.plot_damage(fat, mesh)
```

## Assumptions (stated in every result)
- Linear structure; patterns act one at a time (no phase between, say, thrust and unbalance).
- Damage is Miner's rule on an S-N curve with Goodman mean correction; no crack growth, no
  sequence effects, no temperature or humidity effects on the printed material.
- Excitation amplification is a one-mode bound; a mode within ±20 % of an excitation line is a
  design problem before it is a fatigue number.
- Nodal stresses at bolt holes and sharp corners are mesh-dependent: compare designs, validate the
  absolute life with a test.

## CLI
```bash
vegeta chronos spectrum missions.py:HOVER --modes 210,415 --damping 0.03 -o runs/spectrum_hover.json
vegeta chronos life --damage hover=2e-4 freestyle=3e-3 --hours hover=0.2 freestyle=0.1 --usage hover=0.7 freestyle=0.3
```

Notebooks: `13_quadcopter_life`, `14_fixed_wing_life`.
