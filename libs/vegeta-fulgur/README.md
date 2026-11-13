# vegeta-fulgur — the cold-plasma toolkit

`vegeta.fulgur`: the corona discharge in air and the thrust it makes (electrohydrodynamic propulsion — the
"bladeless turbine"), the spark that ends it, and the high-voltage supply, all as reusable pieces. Plan and
physics: `docs/plasma.md`. numpy + scipy only.

```python
from vegeta.fulgur import MARK4C, Air, climate

e = MARK4C.at_margin(1.0, climate.warsaw("jul"))     # the most it can do before the predicted spark, in July
e["V"], e["I"], e["thrust"], e["power"], e["v_focal"], e["limited_by"]
MARK4C.at_power(50.0, Air(rh=0.5))                   # what 50 W gives at 50 % RH
```

| module | what |
|---|---|
| `gas` | `Air(T, p, rh)`: density, ion mobility, Peek's onset field, breakdown and stability fields, net ionisation |
| `climate` | Warsaw's twelve months, a humid day, the dry room |
| `corona` | `Gap`: onset voltage, `I = C V (V − V0)`, thrust `I d / μ`, thrust per watt `1/(μE)`, the jet |
| `breakdown` | `spark_margin`: inception (Meek), propagation (stability field), uniform gap, surface flashover |
| `supply` | `HVSupply`: operating point with fold-back, input power, mass, a spark's energy |
| `thruster` | `RingThruster` and the Mark IV / 4B / 4C presets (geometry estimated from the video) |

What is a law and what is a fit is written at each constant (`gas.py`, `breakdown.py`); the one geometry factor
of the 1-D model (`collection`) is fitted on the Mark IV's 50 W, everything else is predicted. The tests are the
benchmarks: `python -m pytest libs/vegeta-fulgur`.
