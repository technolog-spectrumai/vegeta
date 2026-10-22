# Auseklis — reusable LEO debris-removal servicer (space lab, `exp/`)

A speculative design in limbo: nothing outside `exp/` and the lab's tests imports it. Python-first: closed-form physics,
three closed budgets, per-removal ledgers, approach safety, capture estimates, parametric CadQuery CAD with matplotlib
renders, and *prepared* (never auto-run) CalculiX / Gmsh / MuJoCo scripts.

**Everything is a preliminary engineering estimate.** Each number is tagged in code (`provenance.py`): `VERIFIED`
(cited constant), `CALCULATED` (a script here), `ASSUMED` (a design choice or class-typical value, with its reason).
**`satellite.md` documents every design choice and describes the satellite subsystem by subsystem.** `REPORT.md` is generated from the models (`python -m auseklis.report`) and lists all 49+ assumptions. No commercial part
numbers are used yet: component selection needs datasheet verification and is not done.

## Run it

```bash
python tools/lab.py test space                 # 33 tests, mocks on, ~30 s, no solver
python -m auseklis.report [REPORT.md]          # from labs/space/exp: regenerate the tables
python -c "from auseklis.render import render; render('A', 'a.png')"     # matplotlib 3-D views (renders/ holds A, B, C)
python -c "from auseklis import cad; cad.export_step('a.step', variant='A')"
python external/bus_fea.py --variant A --workdir runs/bus_A              # prepares Talos (Gmsh+CalculiX); --run to solve
python external/mujoco_capture.py --variant A --debris 500 --out runs/c.xml    # prepares MuJoCo; --run to simulate
```
Needs `cadquery`, `numpy`, `scipy`, `matplotlib` (+ `ccx`/Gmsh and `mujoco` only for the external runs).

## Variants (`variants.py`)

| | A | B | C |
|---|---|---|---|
| propulsion | storable bipropellant, Isp 290 s | green monoprop, Isp 235 s | Hall (Isp 1500 s, ~54 mN) + green RCS (60 m/s) |
| capture | 2 m arm + 2-finger gripper | docking adapter (cooperative only) | short boom + 3-finger gripper |
| wet mass sized for 500 m/s | 355 kg | 339 kg | 311 kg |
| Δv when filled to 400 kg | ~800 m/s | ~830 m/s | ~3.6 km/s |

## What the numbers say (all CALCULATED from ASSUMED inputs; see `REPORT.md`)

1. **The 350–500 kg window closes only if the tanks are filled beyond 500 m/s.** Sized strictly to 500 m/s, B (339 kg)
   and C (311 kg) fall *below* the 350 kg floor. The bottom-up dry mass (212–234 kg nominal, +25 % margin) is itself an assumption.
2. **500 m/s is not the binding constraint — the disposal strategy and the atmosphere are.** Handing debris over on an orbit
   that decays within 5 years costs 45–64 m/s per removal and gives 10–14 removals per life (A) under a *moderate* solar-activity
   density table. At solar minimum the hand-over altitude drops from 538 to 455 km, the Δv triples and A manages **4**. A controlled
   reentry (perigee 60 km) costs ~312 m/s per removal and A manages **1–2**. A real density model (NRLMSISE-00/JB2008) is the first thing to replace.
3. **Variant B cannot remove uncooperative debris** (docking only). **Variant C cannot do a controlled reentry** (low thrust)
   and its RCS propellant caps it at ~4 removals; each takes 27–30 days because the tow is thrust-limited.
4. **Targets must already be in the servicer's plane.** A plane change is ~132 m/s per degree at 550 km; closing a RAAN gap by
   ±20 km altitude drift takes ~100 days per degree. The throughput figures assume targets within 20 km altitude and 90° phase.
5. **Approach safety is not automatic.** With the assumed navigation errors, a thrusters-off drift from 1 km fails the 500 m
   keep-out within 90 min (1st-percentile minimum 356 m); the 10 km, 100 m and 10 m gates pass. It needs a tighter velocity
   estimate (< ~0.01 m/s) or a safety-ellipse hold; this is an open design item, not a result to hide.
6. **Power and capture look comfortable:** 6.2 m² of array covers 1.5 kW EOL (rigid panels 22 kg), battery 791 Wh for a 35.6 min
   eclipse; 3 deg/s tumble of a 500 kg stage is ~32 N m s of angular momentum (≈19 g of RCS propellant to remove), arm peak
   force ~27 N at 5 cm/s. The capture *mechanics* (grappling a tumbling uncooperative body without a fixture) is the hard part, and is not modelled here.
7. **Trade (assumed weights):** A 0.69, C 0.63, B 0.22; A wins in 71 % of random weight perturbations, C in 29 %, B never
   (it is disqualified from uncooperative targets). Do not read this as a decision: the weights are assumptions.

## What is done and what is not

| # | deliverable | state |
|---|---|---|
| 1 | requirements and assumptions | partial: `variants.py` constants + the assumption register in `REPORT.md`; no formal requirement table |
| 2 | three variants + weighted trade | done (`trade.py`, with weight-sensitivity) |
| 3 | mass / power / propulsion / Δv budgets | done (`variants.py`, `physics.py`) |
| 4 | components with real specifications | **not done** (needs datasheet verification) |
| 5 | BOM and costs | **not done** |
| 6 | parametric CAD + STEP | done (`cad.py`, `renders/`); STEP is exported on demand |
| 7 | orbital / rendezvous notebooks | partial: closed-form + CW models; no J2+drag numerical propagator, no notebooks yet |
| 8 | capture mechanism and dynamics | partial: `capture.py` (momentum, contact) + prepared MuJoCo scenario |
| 9 | structural and thermal feasibility | partial: FEA prepared (`external/bus_fea.py`, not run); thermal equilibrium function only |
| 10 | 100 / 250 / 500 kg mission profiles | done as ledgers (`mission.py`) |
| 11 | reliability, collision risk, FMEA | partial: approach safety and Pc; no FMEA / reliability model |
| 12–14 | AIT roadmap, launch / regulatory, lifecycle cost | **not done** |
| 15 | consolidated specification | this README + `REPORT.md` |

## External verification needed

Bus FEA (equivalent-density solid shell; real sandwich panels, launcher loads, modes), MuJoCo capture contact (a soft weld,
not the real gripper), the atmosphere model, all navigation covariances (EKF study), solar-array and thermal models, and
every `ASSUMED` mass. Nothing here has been run on a real Gmsh/CalculiX/MuJoCo install.
