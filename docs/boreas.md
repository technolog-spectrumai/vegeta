# Boreas — propeller and rotor performance (BEMT)

Boreas predicts thrust, torque, power and efficiency of a propeller or rotor from its geometry with
blade element momentum theory (Prandtl tip loss, hover included), couples it to a first-order
brushless motor and a battery, and writes one JSON file that other tools and notebooks read. It is
pure numpy: no external program, no database, every coefficient is an input you wrote.

## Geometry and section
```python
from vegeta import boreas

d, p = boreas.inches(9, 6)                                   # 9x6 -> metres
prop = boreas.Propeller.from_pitch("9x6 E", d, p, blades=2, chord_root_m=0.015, chord_max_m=0.022,
                                   chord_tip_m=0.006, mass_kg=0.010, rotor_mass_kg=0.035)
# or station by station: boreas.Propeller(name, blades, r=(...), chord=(...), beta_deg=(...))
airfoil = boreas.Airfoil(cl_alpha=5.7, alpha0_deg=-2.0, cl_max=1.1, cd0=0.02, k=0.04)   # fit to a polar
```
`Airfoil` is a linear-lift, parabolic-drag section with a stall cap; there is no Reynolds dependence.
Fit its four numbers to a measured polar of the section you use, or accept the generic defaults as a
first estimate (checked against the UIUC database: an APC 10x4.7 static point comes out ~25 % low).

## Operating points
```python
op = boreas.solve(prop, airfoil, rpm=8000, airspeed=12.0)   # OperatingPoint
op.thrust, op.torque, op.power, op.efficiency, op.ct, op.cp, op.advance_ratio, op.tip_mach, op.converged
op.r, op.dT_dr, op.alpha_deg                                 # radial distributions
boreas.rpm_for_thrust(prop, airfoil, thrust=5.0, airspeed=0.0)
grid = boreas.performance_map(prop, airfoil, rpms, airspeeds) # dict of lists, JSON-ready
```

## Motor and battery
```python
motor = boreas.Motor("2212-920KV", kv_rpm_per_volt=920, resistance_ohm=0.12, no_load_current_a=0.6, max_current_a=20)
bat = boreas.Battery("3S 2200", cells=3, capacity_ah=2.2)
sys_ = boreas.Propulsion(prop, airfoil, motor, bat)
pt = sys_.at_throttle(0.6, airspeed=12.0)   # rpm, current, electrical power, motor efficiency, aero point
pt = sys_.for_thrust(1.4, airspeed=14.0)    # throttle that gives the thrust
```
`current_limited` flags points above the motor's current limit; nothing is clipped silently.

## What a structure sees
`boreas.excitations(prop, rpm)` gives the shaft frequency (1P), blade-pass frequency (B·1P) and the
rotating unbalance force for an ISO 21940 balance grade (`G 6.3` by default): the inputs of a
vibration or fatigue assessment.

## Export (the hand-off)
```python
boreas.export("runs/prop_9x6.json", prop, airfoil, map=grid, motor=motor, battery=bat,
              points={"cruise": pt, "climb": sys_.at_throttle(1.0, 8.0)}, notes="...")
doc = boreas.load("runs/prop_9x6.json")     # geometry, airfoil, map, named points + excitations
```

## CLI
```bash
vegeta boreas point --dp 10x4.7 --rpm 6000 --speed 5          # generic planform from the inch designation
vegeta boreas for-thrust -p props.py:NINE_SIX --thrust 1.4 --speed 14
vegeta boreas map --dp 9x6 --rpm 3000 12000 10 --speed 0 20 5 -o runs/map.json
```

Notebooks: `11_quadcopter_propeller`, `12_fixed_wing_propeller`.
