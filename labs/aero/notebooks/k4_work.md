# K4TQ — the work, step by step (the verbose plan)

Companion of [`k4_plan.md`](k4_plan.md): the same stages and step numbers, each step with **what** is done, **how**
(the method and the Vegeta code it reuses), **out** (the deliverable) and **check** (how it is judged). The plan's
checkboxes are the status; this file is the reasoning. Requirements R1–R12 are in `k4_plan.md`.

Conventions (as the aero lab): design modules in `labs/aero/designs/` imported by plain name, SI units (metres,
radians) unless a name says `_deg` / `_mm`, x aft from the nose tip, y to the right, z up. Every number carries its
label: ESTIMATE, CALC, MOCK, SOLVER, TEST. Under `DEBUG = True` (all of Stages 1–4, R12) the solvers are mocked by
`vegeta.mock`; a section polar under the mock is the thin-airfoil stand-in, so the sizing uses section constants
recorded from a NeuralFoil run (source and date stored next to them) and labels anything a mock produced as MOCK.

---

## Stage 1 — Engineering design and feasibility
*First-principles sizing of the whole aircraft, compared with three alternatives and a same-size quadcopter, before any CAD.*

Modules: `designs/k4tq.py` (requirements, catalogue, parameters, layout, mass) and `designs/k4tq_sizing.py` (the
analyses); report `designs/data/k4tq/phase1.md` + `phase1_spec.json`; tests `tests/test_k4tq.py`.

**1.1 Requirements and catalogue.**
- what: R1–R12 as a `REQUIREMENTS` dict; a `CATALOGUE` of the bought parts.
- how: candidate motors from datasheets (T-Motor MN4014 KV400: 67 mΩ, 6S, 15×5 → 26 N at 18.7 A static — retailer
  bench data; SunnySky X4110S KV400: 112 mΩ, 149 g), Molicel P45B cells (`og.batteries.CELLS`), flight controller,
  GNSS, telemetry, tilt servos, thin-section bearings (6704 / 61804), temperature sensors and an independent logger.
- out: one table with mass, price (EUR), source, "to verify" flag.
- check: every mass in the budget comes from this table or from a stated areal/linear density.

**1.2 Parameters and layout.**
- what: one `params(variant)` dict and `layout(p)`; the propeller and wing poses as a function of the tilt angle
  (0° cruise, 90° hover) and the fold angle.
- how: front pivot at ~0.30 L, rear at ~0.75 L; front wing low, rear high (the Kestrel-Q4T finding: the rear wing out
  of the front wake); pivot at ~30 % chord with the thrust line through the pivot axis (no thrust moment on the
  actuator); propellers near the tips, the disc separation front-to-rear checked in hover; the wing tilts about its
  spar, the fold hinge (vertical axis, inboard) is separate.
- out: stations, clearances (disc to disc, disc to fuselage, wing trailing edge to ground in the hover stance, disc
  to ground in the cruise stance), the folded envelope estimate.
- check: `checks(L)` lists every violated clearance; none for the selected configuration.

**1.3 Mass and CG.**
- what: the itemised budget with positions; the battery placed for the CG target (`og.mass.place_mass_for_cg`),
  inertia (`og.mass.Budget`).
- how: structure from areal densities (composite skins on foam, carbon tubes per metre), the wing-root cartridges
  itemised (servo, reduction, two bearings, housing, fold hinge and latch, encoder, fasteners), the medical cartridge
  (insulated shell, shock mounts, two sensors, logger), a recovery parachute, a 7 % growth margin.
- out: mass by group, MTOW, CG, inertia; the "tilt-wing tax" (wings + spars + roots + fin) as one number.
- check: battery inside its bay; CG inside the range the stability (1.5) and the hover thrust split (1.7) allow.

**1.4 Wing loading, lift, stall, section.**
- what: W/S, CL at cruise, stall speed with the tandem's stall rule, the section.
- how: candidate NACA 2412 / 4412 / 2415 / 4415 at Re ≈ 2e5, n_crit 6 (NeuralFoil, recorded): 2412 cl_max 1.23,
  cm −0.05; 4412 cl_max 1.34, cm −0.10. The all-moving wing prefers a low |cm| (hinge moment on the tilt actuator):
  2412-class baseline, 4412 as the option. Built-up wing factor 0.9 on cl_max (as `kestrel_fast`).
- out: stall speed, cruise CL, margin to stall at cruise.
- check: cruise ≥ 1.3 V_stall; the front wing reaches its cl_max first (nose-down stall).

**1.5 Tandem interference.**
- what: front downwash on the rear wing, each wing's effective lift slope, the lift split, the neutral point, the
  static margin, the induced-drag factor.
- how: `og.aero.vlm` on the four panels with per-panel incidence (the tilt actuators set the decalage — an all-moving
  pair can trim itself), as `kestrel_tandem.aero` does for two wings.
- out: CLα, k, e, x_np, front share, rear slope ratio; the CG target (static margin 8–12 % of the stagger).
- check: rear slope ratio against the Kestrel-Q4T lattice (≈0.6 flat, ≈0.9 with a gap) for plausibility.

**1.6 Cruise.**
- what: drag build-up, thrust, electrical power, endurance and range against speed; the best-range and
  best-endurance speeds; four motors against two (rear pair stopped, folding blades streamlined).
- how: `og.aero.cf_mixed`, form factors as `kestrel_tandem.drag_buildup`; the drive as a Boreas map
  (propeller + motor + pack, BEMT) evaluated at the thrust per running motor.
- out: the cruise table, L/D, best speeds; the motor no-load loss and its share at cruise.
- check: power at the drag × speed / (propulsive × motor efficiency) within 10 % of the map's value.

**1.7 Hover.**
- what: disc loading, ideal induced power, BEMT power and figure of merit, hover endurance, T/W, the thrust split
  front/rear for the CG, download on the edge-on wings.
- how: `og.rotors.ideal_hover_power`, the Boreas map at V = 0.
- out: hover electrical power, minutes of hover per pack, T/W at full and sagged voltage, motor-out statement.
- check: T/W ≥ 1.6 (full pack), the most loaded motor ≤ 75 % of its static maximum in a still-air hover.

**1.8 Propeller and motor selection.**
- what: the hover-against-cruise propeller compromise.
- how: BEMT sweep: 13×6.5, 13×8, 14×7, 15×5, 15×8 on the MN4014 KV400, 6S. First run: 13×8 hovers at 9.8 N for
  133 W (FM 0.59) and still has 11.9 N at 22 m/s; 15×5 hovers cheaper (120 W) but has no thrust left at 22 m/s
  (pitch speed too low). The sweep repeats at the real hover thrust per motor of the selected MTOW.
- out: the table and the choice with its reason.
- check: thrust at the dash speed ≥ 1.3 × the drag there; hover power within 10 % of the best propeller's.

**1.9 Transition corridor.**
- what: for each airspeed in level flight (and in a 5° descent), the wing tilt and the thrust that balance lift,
  drag and weight with the fuselage level; the wing angle of attack on the blown and the unblown part; the stall
  margin; the power; the front/rear thrust split for pitch trim.
- how: quasi-steady: momentum theory for the slipstream (`og.rotors.induced_velocity`, the disc inclined to the
  flow), strip split of each wing into the blown span (inside the disc) and the free span, the full-range
  `og.aero.Polar` (post-stall flat plate, `viterna_cd90`) on both parts. The tailsitter model
  (`aeromant.transitions`) does not apply: it pitches the body.
- out: tilt(V), thrust(V), power(V), stalled fraction(V); the corridor width; the slowest speed the wing carries
  the weight alone.
- check: a continuous corridor from hover to cruise; the power peak below the drive's maximum; flagged where the
  free span stalls (the CFD cases of Stage 3 go there — R10).

**1.10 Wing-root loads.**
- what: root bending (cruise gust, manoeuvre), the hover thrust's bending, torsion and the actuator torque
  (aerodynamic moment about the pivot, AC travel through the transition, thrust-line offset tolerance, mass offset
  of the nacelle, gyroscopic moment of the propeller during the tilt), bearing radial loads.
- how: elliptic lift on the half-wing with the tip mass (as `kestrel_tandem.cantilever`); moments into two bearings
  a known distance apart.
- out: limit and ultimate loads per case; the actuator torque class; the bearing choice.
- check: the actuator's rated torque ≥ 1.5 × the worst limit torque; bearing dynamic rating ≥ 10 × the radial load.

**1.11 Safety factors and materials.**
- what: the factors (ultimate 1.5 on limit; 2.0 on printed parts; fatigue and knock-downs for PETG/PA-CF), the
  materials (pultruded carbon tube, glass/carbon on foam skins, 7075 for the cartridge, steel pins).
- out: the spar tube from a standard list (Ø12…Ø20) by strength, tip deflection and the first bending mode against
  the 1P/2P rotor lines.
- check: reserve ≥ 1.5 on ultimate; first mode clear of the hover 1P by ≥ 20 %.

**1.12 Directional stability.**
- what: Cnβ of the fuselage, the nacelles and propellers, the wings (sweep), without a fin; the fin area for a
  target Cnβ (R9 allows one).
- how: Raymer's fuselage term, the propeller side-force term (De Young, as `kestrel_flight.prop_cn_alpha`), the fin
  as a low-aspect lifting surface with its arm.
- out: Cnβ without and with the fin; the fin's size and whether it doubles as the rear landing legs.
- check: Cnβ ≥ 0.05 /rad with the fin.

**1.13 Launch and landing (R6).**
- what:
  - hand release: hover stance, held by a handle under the fuselage, disc-to-hand distance, the thrust margin at release;
  - tube: the opened tube half as the pad (clearance of the discs to the tube walls); ejection with in-flight
    unfolding judged too (expected: rejected for a civilian platform — complexity, failure modes);
  - hospital roof: roof-edge wind speed-up and turbulence, gusts, the vertical wings' area to the wind, the ground
    hold before take-off (tilt for minimum overturning moment), downwash on people.
- out: limits (wind speed for take-off and landing), procedures, the interlocks (motors armed only in the hover stance).
- check: the overturning moment at the limit wind below the stance's restoring moment by ≥ 1.5.

**1.14 Mission energy and payload–range (R2, R3).**
- what: climb, transitions, cruise in 8 m/s wind both ways, hover, reserve; payload against range for the standard
  and the extended pack.
- out: the mission energy table, the margin, the payload–range curve.
- check: the standard mission closes with 1 kg; the extended mission's payload ≥ 0.5 kg.

**1.15 Configurations and the quadcopter.**
- what:
  - A: equal tandem, 1.4 m;
  - B: unequal tandem (larger front wing);
  - C: compact, 1.2 m, 0.5 kg payload;
  - Q-13: the same quad with no wings and no roots, on the same 13-inch propellers;
  - Q-best: the quad on the largest propellers its 1.4 m footprint takes.
- how: the same analyses; the quad in forward flight by momentum theory (Glauert, the disc tilted for the drag) +
  profile power from the BEMT hover + parasite drag.
- out: one table (mass, hover W, cruise W, speed, range, endurance, wind tolerance, parts count, cost) and the
  answer: is K4TQ better than a quad of the same size or larger — explained either way.
- check: the comparison is fair: same payload, cargo box, avionics, pack, reserve.

**1.16 Verdict and report.**
- out: `data/k4tq/phase1.md`, `phase1_spec.json`, the initial BOM (`bom_phase1.csv`); assumptions, uncertainties.
- tests: mass sums, CG placement, layout clearances, the hover momentum limit (BEMT power ≥ ideal), lattice
  symmetry, corridor continuity, mission energy consistency.

---

## Stage 2 — Parametric CAD and mechanical design
*The selected configuration as a Dedalus/CadQuery assembly in three configurations, with the wing-root tilt-and-fold assembly designed as real hardware.*

Module `designs/k4tq_cad.py` (`dedalus.Design` subclasses), parameters read from `k4tq.params` (one source of truth).
Reuse: `kestrel.Kestrel4Q._pod` (superellipse loft), `_nacelle`, `naca4`; `KestrelQ4T._tandem_wing` (rotation about a
chord station); `dedalus.export_yax`; `dedalus.viz.plot3d`.

- **2.1 Fuselage.** Loft from the nose to the tapered tail; cargo bay at the CG with a sealed hatch, the marking text
  (R8) on the hatch and both sides; battery bay on rails, avionics tray, wing-root hardpoints (frames at the pivots).
- **2.2 Wings, nacelles, propellers.** Panels lofted from the chosen section with the sweep; the spar tube; nacelle
  with the motor mount on the chord line through the pivot; the propeller as a thin disc plus folding blades.
- **2.3 Wing-root cartridge.**
  - the spar tube in two thin-section bearings, preloaded by a nut;
  - a high-torque digital servo through a 2–3:1 reduction (backlash-limited gears, or a belt), hard stops at −5° and
    95°, a magnetic encoder on the spar axis;
  - the motor power cables through the hollow spar with a ±90° service loop (no slip ring);
  - a 7075 housing machined, covers printed.
- **2.4 Fold hinge.** A vertical steel pin in bronze bushings at the fuselage hardpoint, a spring latch with a
  witness mark; folding only with the wings at 90° (edge-on); the fold angle from the clearance of the folded front
  wing to the rear root (partial fold allowed, R5).
- **2.5 Legs, fin, cartridge.** Four legs (or two legs and the ventral fin) for the hover stance; the medical
  cartridge: insulated shell (VIP or EPP), shock isolators, two temperature sensors, logger, connector.
- **2.6 Assembly.** `config` = cruise / hover / folded or any tilt and fold angles; colours white / light grey,
  text marking.
- **2.7 Checks.** Interference between solids per configuration, the disc separation, ground strike in each stance,
  the folded envelope against the tube, tolerance stacks of the bearing seats, tool access to every fastener, the
  load path from the spar to the fuselage frame.
- **2.8 Export.** STEP and STL per part and per configuration, Yax mass properties; CAD mass against Stage 1 —
  if they differ by more than 5 %, Stage 1 is rerun with the CAD masses.
- **2.9 Visualisation and tests.** pyvista views in the notebook; tests: generation, volumes > 0, the three configs,
  tilt transform keeps the tips laterally fixed, exports readable.

## Stage 3 — Aerodynamics, CFD and performance optimisation
*Lightweight numerical aerodynamics on the CAD geometry, a few optimisation iterations, and CFD cases prepared (and mocked) for the transition and cruise.*

Module `designs/k4tq_aero.py` (+ `k4tq_cfd.py`). Reuse: Zephyr, `og.aero.vlm`, `kestrel_cfd` (STL export, actuator-disc
setup, case specs), `kestrel_campaigns` (fingerprints, `require`), `kestrel_loop.IterationLog`.

- **3.1 Sections and drag** on the CAD wetted areas.
- **3.2 Lattice on the CAD planform** with tilt angles and decalage.
- **3.3 Slipstream and rotor interference**: actuator discs, the rear propeller in the front wake in cruise, the
  edge-on wings in the hover.
- **3.4 Transition** at intermediate tilt: lift, drag, moments; stability and control authority (thrust split, tilt
  differential) through the corridor.
- **3.5 Optimisation**: 3–5 iterations on the parameters named in the plan, each logged with its reason and result.
- **3.6 Against Stage 1**: a table of the differences and their causes.
- **3.7 CFD preparation**: Aeromant cases (cruise sweep, transition at 15/30/45/60/75° tilt with the rotor discs, the
  hover with ground plane), meshes, scripts, a `k4tq` book in `campaigns/side_labs/flow.py`; run on the mocks only.
- **3.8 Manual runs**: the list with priority, cost and the decision each run settles.

## Stage 4 — Structural FEA and manufacturing with Mellonia
*Load cases through Talos (mocked FEA), the make/buy plan and printability through Mellonia, and an EU-sourced BOM — revising the design where a part fails.*

Modules `designs/k4tq_structure.py`, `designs/k4tq_mfg.py`. Reuse: `kestrel_fea` (FEACase, cases, solve), `kestrel_xc`
(`panel_cad`, `fea_cases`: one cantilever spar in its socket with the tip motor), `falco_structure` (motor-mount case),
Chronos (spectra, life), the 08 notebook's Campbell and fatigue flow; the generic print checks from
`exp/mini_raptor_print.py` promoted to `common/aero_kit.py` (nothing imports `exp/`).

- **4.1 Load cases** with their source numbers (Stage 1 / 3).
- **4.2 Components**: wing, spar, bearings, rotary joint, fold hinge, actuator and locks, motor mounts, fuselage frames.
- **4.3 Buckling, modes, freeplay and flutter, fatigue.**
- **4.4 Talos models and book chapters** (mocked; real runs prepared).
- **4.5 Make/buy**: buy (motors, ESCs, cells, bearings, shafts, servos, fasteners), print (covers, fairings,
  fixtures, brackets), composite (wings, spars, fuselage shells), machine (cartridge housing if no catalogue part
  fits), hybrid (the wing roots).
- **4.6 Mellonia**: orientation, supports, tolerances, materials (PETG, PA-CF, ASA), print time and cost — mocked slicer.
- **4.7 BOM**: suppliers in Poland / the EU with links, prices, masses; rejected parts and why.
- **4.8 Revisions** forced by the structure, recorded.

## Stage 5 — Validation, documentation and graduation
*The experimental notebook, the regression tests, the manual simulation checklist, and a graduation proposal awaiting approval.*

- **5.1 Notebook** `labs/aero/exp/K4TQ.ipynb`: the bootstrap, then Parts 1–4 as the stages, every figure and table
  from the design modules (no physics in the notebook cells), labels on every result.
- **5.2 Tests**: smoke (< 5 s) and regression (pinned values with tolerances) under the mocks.
- **5.3 Manual checklist**: commands, expected outputs, CPU hours, memory, disk.
- **5.4 Conclusions**: the K4TQ-against-quad table, the risks left open.
- **5.5 Graduation** to `labs/aero/notebooks/K4TQ.ipynb`: proposed, executed only on approval.
