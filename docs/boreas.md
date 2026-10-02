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

## Noise and cavitation (first estimates)
```python
tones = boreas.gutin_harmonics(prop, thrust=5.0, torque=0.1, rpm=6000, distance=10.0, angle_deg=90.0)  # Gutin steady-loading tones
tones["blade_pass_hz"], tones["spl_db"], tones["total_tonal_db"]          # dB re 20 uPa (air) / 1 uPa (boreas.SEA_WATER)
boreas.broadband_level(prop, thrust=5.0, rpm=6000, distance=10.0)         # empirical allowance, dB
boreas.vortex_noise(prop, rpm=9000, distance=10.0, airspeed=20.0)         # Schlegel-King-Mull broadband: spl_db, peak_hz
boreas.a_weighting([100, 1000, 4000])                                     # dB to add for dB(A): -19.1, 0.0, +1.0
boreas.add_levels([60.0, 60.0])                                           # energetic sum: 63.0 dB
boreas.cavitation(prop, rpm=3000, airspeed=2.0, depth_m=0.5, cp_min=-1.0) # cavitation number vs -Cp_min at 0.7 R
```
Gutin's formula covers the steady thrust/torque loading only (no thickness or unsteady-inflow noise, no
installation effects); the broadband term is a tip-speed⁶ allowance (its default constant reads high for small air
propellers — `vortex_noise`, Schlegel–King–Mull as in Hubbard's NASA RP-1258, is the calibrated choice in air); `cp_min` is a property of your
blade section. Use them to rank designs and operating points, not as certification numbers.

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

## A propeller in a wake (`boreas.wake`)
A propeller behind a hull, fins or struts sees a non-uniform inflow, and every blade passing a wake
deficit is a load pulse. `boreas.wake` turns a wake field into the loads, their frequencies and their noise:
```python
from vegeta.boreas import wake
w = wake.fin_wake(4, mean=0.15, depth=0.25, width_deg=12)        # or wake.sampled_wake(sampler, centre, R, V) from a hull CFD field
h = wake.load_harmonics(prop, airfoil, rpm, ship_speed, w, rho=1025.0)
h.table(16)                     # per order: frequency, blade thrust/torque, shaft thrust/torque, side forces (amplitudes)
wake.unsteady_tones(h, 1.0, boreas.SEA_WATER, angle_deg=30)       # the shaft force harmonics as dipole tones, dB
wake.rotating_tones(h, prop, 10.0, boreas.AIR, 45.0, harmonics=16) # Lowson's rotating dipoles: steady + unsteady loading at m x BPF
w_eff, op = wake.effective_inflow(prop, airfoil, rpm, V, w, rho)  # thrust-weighted mean wake and the point at V (1 - w_eff): T V / P installed
field = wake.slipstream_sampler(prop, op)                         # points -> (U, valid): a slipstream model for particle movies
```
The loads are quasi-steady (a blade-element solution at the local inflow of every blade angle; no
unsteady-aerofoil lag, so higher orders are upper estimates). The selection rules follow from the sums
over the blades: each blade sees the wake orders; the shaft thrust and torque keep only multiples of the
blade count; the side (bearing) forces keep blade orders one either side of those. With 3 blades behind
4 fins: blade orders 4, 8, 12, shaft thrust at 12, side forces at 3 and 9 (`tests/boreas/test_wake.py`).
`skew_deg=` sweeps the blade back (linear from the root, as in the Dedalus `Propeller` example's `skew_deg`):
each section meets the wake at its own angle, so every blade-load harmonic is multiplied by the thrust-weighted
phase average `Σ w_r e^(-i q skew(r)) / Σ w_r` — the higher the order, the more the sections cancel; the mean is unchanged.
The tones are compact dipoles, `p = ω F / (4 π c r)` × cos (thrust) or sin (side force) of the angle from
the axis. The slipstream model is momentum theory (induced axial velocity and swirl from the solution,
contraction by continuity), a stand-in for pictures before a CFD field exists, not a flow solution.

Notebooks: the propeller parts of `08_quadcopter` and `09a_fixed_wing_design` (BEMT, CFD check, noise, blade FEA and its video).

`rotating_tones` is the form for air propellers (the tip Mach number makes the rotating source non-compact): every
blade-load harmonic k radiates at m·BPF with `J_{mB−k}`, so the steady load (k = 0) gives Gutin's tones exactly and a
sharp wake pulse lifts a whole comb of harmonics, mostly towards the axis (on the axis only k = mB radiates, the compact
thrust dipole of `unsteady_tones` — both identities in `tests/boreas/test_wake.py`). Loading noise only (no thickness
noise, no forward-flight Doppler factor). Notebook `25_air_propeller` uses it for a tractor against a pusher behind a pylon.

## Ducted fans (`boreas.ducted`)
An electric ducted fan (EDF) is a rotor in a duct with a stator row and a nozzle. `boreas.ducted` couples 1-D duct
momentum theory with the rotor's blade elements:
```python
import math
from vegeta.boreas import ducted

rotor = boreas.Propeller("EDF 90 mm", 12, r=(...), chord=(...), beta_deg=(...))   # hub (0.45 R) to tip, constant pitch 160 mm
fan = ducted.DuctedFan("EDF 90 mm", rotor, tip_clearance_m=0.0007, exit_area_ratio=0.9, stator_vanes=7,
                       stator_loss=0.1, duct_loss=0.06, external_wetted_area_m2=0.035, duct_length_m=0.15)
section = boreas.Airfoil(cl_alpha=2 * math.pi * 0.9, alpha0_deg=-3.0, cl_max=1.1, cd0=0.018, k=0.04)
op = ducted.solve(fan, section, rpm=25000, airspeed=20.0)                 # DuctedPoint
op.thrust, op.rotor_thrust, op.duct_thrust, op.power, op.efficiency, op.figure_of_merit, op.converged
op.mass_flow, op.fan_velocity, op.exit_velocity, op.total_pressure_rise, op.tip_mach
op.r, op.c_theta, op.alpha_deg, op.cl, op.dp0                            # radial distributions
ducted.rpm_for_thrust(fan, section, thrust=22.0)                         # static, bisection on rpm
op.thrust - ducted.nacelle_drag(fan, 20.0)                               # net of the nacelle's outside friction
```
The model, all formulas in the module docstring:
- incompressible; the jet leaves the nozzle (`exit_area_ratio` σ = exit / fan annulus area) at ambient static pressure,
  so `V_fan = σ V_exit`, `ṁ = ρ A_fan V_fan` and the whole unit's thrust (rotor + duct lip + stator + nozzle) is
  `T = ṁ (V_exit − V)`; `rotor_thrust` is the axial force on the blades, `duct_thrust` the rest;
- blade elements with no inlet swirl at the mean relative velocity (`ωr − c_θ/2`, `V_fan`), with Euler's equation
  `B ½ρW²c (cl sin φ + cd cos φ) = ρ V_fan 2πr c_θ` for the swirl `c_θ` at every station;
- useful total-pressure rise per station: `ρ ω r c_θ` minus the profile loss, minus the swirl (`stator_loss · ½ρc_θ²` with
  a stator, all of `½ρc_θ²` without one), minus a tip-clearance loss of `2 · clearance / blade height` of the ideal rise
  (an empirical rule: ~2 % efficiency per 1 % clearance); mass-averaged, it balances
  `½ρ(V_exit² − V²) + duct_loss · ½ρV_fan²` (inlet, walls and nozzle);
- uniform `V_fan` (no radial equilibrium, no mixing loss), isolated-aerofoil section data (no cascade correction at a
  high-solidity hub), no Reynolds effects, axial inflow only; the stator is a loss coefficient, not a vane-row model.

With no losses it is the ideal ducted actuator disk, `P = T (V_exit + V)/2` and in hover `P = T^1.5 / (2 √(ρ σ A_fan))`:
for σ = 1 that is 1/√2 of an ideal open rotor of the same area. `figure_of_merit` is `T^1.5 / (P √(2 ρ A_fan))` — the
open-rotor definition on the fan annulus — so an ideal duct reaches `√(2σ)` (1.34 at σ = 0.9) and a real one can exceed 1.
The default losses describe a clean fan: the 90 mm, 12-blade unit above makes 22 N static at ~27 000 rpm with
FM ≈ 1.07 (only 26 % above its ideal power), which flatters a hobby EDF. Catalogue claims for 90 mm, 12-blade 6S units
(≈ 3.0–3.8 kgf from 1.9–3.1 kW electrical, at an assumed 85 % motor + ESC efficiency) give FM ≈ 0.75–1.0; a lumped
`duct_loss` of 0.25–0.5 with a 0.9 mm gap gives 0.91–0.78 (0.4 → 0.83). Do not compare a propeller with the defaults:
use such values, or fit `duct_loss` to a measured static thrust and shaft power of the unit you have,
`fan = ducted.fit_duct_loss(fan, section, thrust=29.4, power=0.85 * 1930.0)` (it lumps what the model leaves out:
inlet lip, struts, low-Reynolds blades, mixing). `rpm_for_thrust` searches 100 rpm..`rpm_max` and raises when the
target lies outside the thrusts at those ends or the point found is not converged.
Windmilling (`V_exit < V`, negative thrust) is solved like any other point;
if the fan cannot push air against the system at any flow, `solve` returns the point nearest a balance with
`converged=False`. `nacelle_drag` is turbulent flat-plate friction (Prandtl–Schlichting `Cf = 0.455 / (log10 Re)^2.58`)
on `external_wetted_area_m2` with Re on `duct_length_m`. Tests: `tests/boreas/test_ducted.py`.

## Fan noise (`boreas.fan_noise`)
A ducted fan's loudest tones come from the rotor wakes striking the stator vanes. `boreas.fan_noise` gives first
estimates of them from plain numbers (no geometry objects), to compare blade and vane counts, spacing and rpm:
```python
from vegeta.boreas import fan_noise
fan_noise.tyler_sofrin_modes(12, 7, harmonics=3)            # n = mB - sV per harmonic, lowest |n| first (s around 0 and mB/V)
fan_noise.cut_on_ratio(1, 12, -2, tip_mach=0.35)            # m B M_tip / |n|: > 1 propagates, < 1 decays
fan_noise.vane_count_study(12, [7, 11, 17, 25], 0.35)       # lowest mode per harmonic, cut on or off, per vane count
w = fan_noise.rotor_wake_harmonics(12, chord_m=0.015, cd=0.03, spacing_m=0.015, radius_m=0.036, wake_angle_deg=55)
tones = fan_noise.interaction_tones(12, 7, 25000, vane_radius_m=0.036, vane_chord_m=0.015, vane_span_m=0.025,
                                    vane_inflow_m_s=55, gust_m_s=w * 90 * 0.8,     # deficit x W x sin(W, vane chord)
                                    distance=1.0, medium=boreas.AIR, angle_deg=45, stagger_deg=20,
                                    duct_radius_m=0.045, duct_length_m=0.03)        # optional: cut-off modes decay in the duct
tones[0]["spl_db"], tones[0]["dominant_n"], tones[0]["modes"]   # per m x BPF: level, loudest mode, every mode
```
- **Modes** (Tyler & Sofrin 1962): B wakes sweeping over V vanes make, at m × BPF, only the spinning patterns
  `n = mB − sV`. A mode propagates when its pattern moves supersonically at the tip, `m B M_tip / |n| > 1` (free-field
  form; a hard-walled duct needs `k R > j'_{n,1} ≈ |n| + 0.81 |n|^{1/3}`, so the ratio errs on the noisy side).
  1 × BPF is cut off for every V > B (1 + M_tip) and never for V < 2 B M_tip (the classic "V > 2B" for sonic tips).
  With 12 blades at M_tip ≈ 0.35, V = 7 leaves `n = −2` cut on; V ≥ 17 cuts 1 × BPF off; V = 12 makes a plane wave.
- **Wakes**: Silverstein, Katzoff & Bullivant (NACA Rep. 651) fit the total-head loss `H₀/q = 2.42 √cd / (x/c + 0.3)`
  with a cos² profile of edge half width `0.68 c √(cd (x/c + 0.15))`. As a velocity deficit (`H/q ≈ 2u/W`) that is a
  Gaussian of centreline deficit `u_c/W = 1.21 √cd / (x/c + 0.3)` and half width at half deficit
  `0.34 c √(cd (x/c + 0.15))`, repeated every blade pitch and Fourier-analysed at the vane radius; its momentum deficit
  is 1.4 × / 1.1 × the section drag's `cd c/2` at x = c / 2c (reading 2.42 as the velocity deficit would make it ~6 ×
  and 1 × BPF ~10 dB too loud). `wake_angle_deg` lets the wake travel along the rotor's relative exit flow (longer path,
  wider cut). Tip-clearance and secondary-flow losses go into `cd`.
- **Vane load**: thin aerofoil in a transverse gust, `L_m = π ρ c U w_m |S(k_m)| span` with the Sears amplitude
  `|S| ≈ 1/√(1 + 2πk)` (`fan_noise.sears`), normal to the chord: `L sin ξ` axial, `L cos ξ` tangential (stagger ξ from
  the axis, towards the rotation).
- **Radiation**: the V vanes as stationary compact dipoles on a ring, vane v lagging by `mB·2πv/V`, summed exactly
  (Jacobi–Anger + Poisson) into the modes
  `p_m = ω V/(4π c r) |Σ_s i^n e^{−inφ} [F_a cos θ J_n(kR sin θ) − F_t n J_n(kR sin θ)/(kR)]|` (amplitude, rms = /√2);
  θ from the downstream axis, φ from vane 0 in the sense of rotation. Checked against the vane dipoles summed one by
  one, a single vane on the axis (`ω F / (4π c r)`), 1/r and the plane wave of V = B (`tests/boreas/test_fan_noise.py`).

Not included: duct terminations, hub and liners (the duct option is only the exponential decay of cut-off modes over a
length of hard-walled duct), rotor-alone tones (cut off at subsonic tips), potential interaction, broadband
(turbulence, tip clearance), vane non-compactness, forward-flight Doppler. Use it to rank designs, not to certify.
