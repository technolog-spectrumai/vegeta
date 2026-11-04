# Gyrus — a Hall thruster, from what we know (plan v0)

**Gyrus** (the ring the electrons circle in): a Hall-effect thruster for a small spacecraft, designed with the
tools and habits of `docs/plasma.md` — Fulgur's field solves on FiPy with Gmsh sections (here magnetostatics for
the magnetic circuit), Halka's ring-geometry CAD patterns, the supply model, and the rule that a design is run to a
*margin*, not to the edge. Branch `dev_hall`, built on `dev_plasma`.

## What a Hall thruster is, in the terms of Halka

Halka pushes **air** with **ions in air**: a corona makes them, the field drives them, collisions hand their
momentum to the neutrals, and the thrust per watt is set by the gap and the voltage (`dz/(μV)`). A Hall thruster
pushes **nothing but the ions themselves**, in **vacuum**: a propellant (xenon) flows into an annular ceramic
channel, electrons from an external cathode are drawn towards the anode at the channel's bottom but a **radial
magnetic field** across the channel traps them in an azimuthal **E × B drift** (the Hall current, hence the name);
trapped, they ionise the xenon efficiently; the ions, far too heavy to be magnetised, fall through the axial
electric field (~300 V) and leave at ~15–20 km/s. The cathode's second job is to neutralise the beam.

So the same three ingredients — an electric field, ions, a geometry that sets the field — plus one new one, the
magnetic field, and no air. The numbers change scale: thrust per watt ~ 60 mN/kW (ten times Halka's best), specific
impulse 1 300–1 600 s against a propeller's ~ 0, and the thing only works below ~10⁻⁴ mbar.

## The target: Gyrus-300

A **300 W class** thruster (the BHT-200 / SPT-50 class, the smallest that works well): small enough that the
magnetic circuit, the ceramic channel and the anode are machinable parts, large enough that the physics is the
textbook one. Xenon baseline, krypton as the cheaper option the model also runs.

| | target | how it is set |
|---|---|---|
| discharge power | 300 W | the class |
| discharge voltage | 250–300 V | the ion speed; above ~300 V the walls erode fast at this size |
| anode mass flow | ~1.1 mg/s Xe | the current the power allows over the voltage, divided by e/M, over the mass utilisation |
| thrust | ~18 mN | `γ·η_m·ṁ·v_i` |
| specific impulse | ~1 300 s | `T/(ṁ_total·g)` with the cathode's flow (~8 %) counted |
| channel | mean diameter 40 mm, width 7 mm, length 12 mm | the SPT-100 scaled: lengths ∝ √P at fixed voltage and current density |
| magnetic field | ~300 G radial at the exit plane, ~0 at the anode | electrons magnetised (Larmor radius ≪ width), ions not (≫ length); smaller thrusters need stronger fields |
| magnetic circuit | iron core and back plate, inner coil, outer coil(s), front pole pieces | ampere-turns from the gap and the field, checked by the magnetostatic solve |
| cathode | an external hollow cathode (bought: a LaB₆ or BaO unit of ~0.1 mg/s) | not designed here; a placeholder with its mass and flow |

## `vegeta.fulgur.hall` — what it holds

| piece | what | first version |
|---|---|---|
| `performance` | the 0-D model (Goebel & Katz): mass utilisation, voltage utilisation, beam divergence, the electron fraction of the discharge current → thrust, Isp, efficiency, power | the standard efficiency chain, each factor a named parameter |
| `scale` | the SPT-100 scaled to a power at fixed voltage and current density: channel diameter, width, length, flow, field | the Dannenmayer–Mazouffre exponents |
| `magnetised` | the checks: electron and ion Larmor radii against the width and the length, the Hall parameter, the electron cyclotron frequency | numbers with pass/fail |
| `circuit` | the magnetic circuit by reluctance: ampere-turns for the gap's field, the coil (turns, wire, resistance, power, mass) | 1-D reluctance |
| `magnetostatics` | **on FiPy**: the axisymmetric vector potential `∇·((1/μr) ∇(rA)) = −J` on a Gmsh section with the iron (μᵣ) and the coils (J) as regions; `B_r`, `B_z` maps; the field along the channel midline | the same machinery as `ehd` |
| `plume` (later) | the beam's divergence and the charge-exchange plume | — |

## The lab: `labs/plasma/designs/gyrus.py`

The CAD in CadQuery: the boron-nitride channel (an annular cup), the anode ring with its gas distributor at the
channel bottom, the iron back plate, the inner core with its coil, four outer coils on cores with the iron screen
(the CAD has four, the section's model one equivalent annular coil), the front pole pieces, the propellant tube,
a placeholder cathode on a bracket; `masses()`; `gyrus_draw3d.py`. `gyrus_fulgur.py`: the performance table, the
magnetised checks, the magnetostatic solve and the field map.

## Phases

| phase | what | done when |
|---|---|---|
| **H0** | this plan | pushed |
| **H1** ✓ | `fulgur.hall`: performance, scaling, the magnetised checks, the reluctance circuit; tests against the SPT-100 and BHT-200 published numbers | SPT-100: 85 mN / 1608 s vs 83 / 1600; BHT-200: 14.6 mN / 1468 s vs 12.8 / 1390 (+14 %) |
| **H2** ✓ | the Gyrus-300 CAD, masses, drawings | 8 parts build; 514 g; `gyrus_3d.png` |
| **H3** ✓ | magnetostatics on FiPy: the circuit's field map, the radial field at the exit plane against the target, the coil's ampere-turns adjusted | 410 G at the exit (target 410), 38 G at the anode, peak 414 G at 0.87 L; 293 At per coil; `gyrus_fulgur.png` |
| **H4** | a 1-D channel model (quasi-neutral fluid: ionisation, ion momentum, the electron mobility across B with an anomalous term): the discharge current, the ionisation zone, the potential profile — the step from the efficiency chain to the physics | the discharge current within 20 % of the 0-D at the design point |
| **H5** | thermal (the anode's and the walls' power), wall erosion and life, the vacuum test-stand plan (chamber, pumping, thrust balance) | a test plan |

## H1–H3 done (2026-11-04)

**Gyrus-300 at 280 V, xenon**: 1.21 mg/s through the anode, 1.07 A, 20.0 mN, 1 554 s, efficiency 0.51,
67 mN/kW. Krypton at the same power: 16.0 mN, 1 945 s (lighter ions, faster, less thrust per watt). 200 V trades
Isp for thrust (23.7 mN, 1 313 s), 350 V the other way (17.9 mN, 1 737 s). The channel scaled from the SPT-100:
Ø41.5 mm mean, 7.3 mm wide, 12.2 mm long, 410 G at the exit. Magnetised: the electrons' Larmor radius 0.26 mm
(≪ 7.3 mm), the ions' 0.67 m (≫ 12 mm), Hall parameter ~10³.

**The magnetic circuit** (`CircuitSection`: back plate, core, inner coil, channel walls, outer coil, yoke, two
front pole pieces, each an r–z rectangle) solved on FiPy — the vector potential `∇·((1/μr)∇(rA)) = −J` on a
non-uniform structured grid with the region edges on grid lines (Gmsh + OpenCASCADE fragments gave inverted cells
here; the rectangles do not need it). The radial field rises from 38 G at the anode to a peak of 414 G at 0.87 L
and 410 G at the exit: the textbook profile, with the ionisation zone where the field peaks. 293 ampere-turns per
coil do it — less than the reluctance estimate's 701, because both coils sit inside the flux loop and their
magnetomotive forces add (a first run with the coils in opposite senses needed 6 650: the sign is a test now).
Coils: 367 turns of 0.63 mm wire at 0.8 A (2.6 A/mm², cool enough for vacuum), 1.4 W, 110 g for the inner and the
four outer together.

**The CAD** (`labs/plasma/designs/gyrus.py`, the channel and the circuit's sections from the same `CircuitSection`
the solve uses): BN channel 23 g, anode with its 24-hole distributor 36 g, iron core + plate + inner pole 207 g,
screen + outer pole 124 g, coils and cores 63 g, the cathode placeholder 60 g: **514 g**. The iron is two thirds of
it — the permanent-magnet trade (H5) is the obvious diet.

## Decisions (v0)

* power class 300 W; xenon baseline, krypton evaluated; voltage 250–300 V;
* the cathode is bought, not designed;
* the magnetic circuit is designed with coils (not permanent magnets): adjustable in test, heavier — the trade is
  noted for H5;
* the thruster lives in the plasma lab (what pushes gas without blades, now also without air).

## Risks

High voltage and vacuum: the 300 V discharge is a hazard only with the chamber open; the coils' heating in vacuum
(no convection) is the usual killer of first builds — H5. Over-claiming: this is a design study with published
reference data, not a tested thruster; every performance number is the 0-D chain's.

## Sources

* Goebel, D. M., Katz, I. (2008). *Fundamentals of Electric Propulsion: Ion and Hall Thrusters*. JPL/Wiley — the
  efficiency chain, the magnetised plasma conditions.
* Dannenmayer, K., Mazouffre, S. (2011). Elementary scaling relations for Hall effect thrusters. *J. Propulsion
  and Power* 27(1).
* The SPT-100 (1.35 kW, 300 V, 4.5 A, 5.3 mg/s Xe, 83 mN, 1 600 s, channel outer Ø100 mm) and the Busek BHT-200
  (200 W, 250 V, 12.8 mN, 1 390 s, ~1 mg/s) as published by their makers and in the above.
