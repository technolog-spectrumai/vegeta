# RAPTORS product space: four products, four airframes

BUTEO, ARPELLA, ASTOR and SPARFELL are four **different products for different customers**. Until now each was
ARIGO-1H with a kit, and run 1 showed why that is not enough: the kits moved scores by a few points, while the things
that decide each job were outside what a kit can change. Those are an unattended station cycle, a three-hour patrol, a
70 m/s dash and a 400 mm packed length.

This document works from the application down to the configuration. All sizing numbers are **first-cut estimates**
from `designs/raptors_sizing.py`: a parabolic polar, a mass build-up, momentum-theory hover. Two anchors come from
better models: ARIGO-1H (screening CFD and run 1) and GAVILAN-X (its flight card). The estimates are there to choose
between configurations, not to quote. Each chosen configuration then gets its own CAD, aerodynamic model and
simulation.

## The line at a glance

| | BUTEO (buzzard) | ARPELLA (harrier) | ASTOR (goshawk) | SPARFELL (sparrowhawk) |
|---|---|---|---|---|
| Job | guard one site, unattended | patrol a wide area | reach a fast threat first | protect the people carrying it |
| Buyer | infrastructure operators | area and border security | air-defence and airport security | infantry, police, field teams |
| Where it lives | on its station at the site | at a base, in the air for hours | in a launcher, ready | in a backpack |
| Who launches it | nobody (automatic) | a two-person team | nobody (automatic, on a cue) | the person carrying it |
| Threat | quadcopters up to ~20 m/s, inside 1.5 km | slow drones and intruders over 10 km | fast fixed-wing drones, 30–50 m/s | a quadcopter already close |
| Deciding number | stays on station for weeks with nobody touching it | hours aloft | seconds to arrive | grams and millimetres |
| **Chosen airframe** | **lift + cruise VTOL** | **high-aspect-ratio patroller** | **propeller dart** | **compact flying wing** |

Each product also has things it is **not** for. BUTEO does not patrol. ARPELLA does not catch fast targets. ASTOR
does not loiter. SPARFELL does not defend a site. A customer with two of these problems buys two products.

---

## BUTEO: guards one site, unattended

**Applications**
- Electrical substations, solar and wind farms, refineries and tank farms.
- Prisons (contraband drops), data centres, depots, and stadiums on event days.

What these have in common: a fixed fence line, a site sensor (radar or RF) that cues the target, and **nobody on
site to throw or collect an aircraft**.

**How it is used.** Two to four units sit on stations around the site, charging. On a cue one launches, meets the
intruder inside about 1.5 km, and **returns to its station and recharges without anyone touching it**. It repeats
this for weeks.

**What decides it.** The full cycle has to be unattended: launch, intercept, return, recharge. Run 1's best BUTEO
(the fixed wing on a turning perch) launches well, but it lands on its belly in a field and waits for a person. That
breaks the product, however good its flight scores are.

| Candidate | Mass | Top speed | Endurance | Hover | Holds an 18 m/s gust in the hover | Unattended return |
|---|---|---|---|---|---|---|
| B1 fixed wing, perch drop | 1.79 kg | 46 m/s | 82 min | none | n/a | **no**: a belly landing, someone carries it back |
| B2 VTOL tailsitter (GAVILAN-X) | 1.86 kg | 48 m/s | 11 min | 5.6 min | marginal: needs 65° of the 78° of tilt it has | yes, onto its charging perch |
| **B3 VTOL lift + cruise** | 2.18 kg | 44 m/s | 58 min | 13.9 min | **yes**: needs 37° of 68° | **yes, onto a charging pad** |

**Choice: B3, lift + cruise VTOL.** It is a conventional wing with a pusher, plus four small lift rotors on two
booms.
- It takes off and lands vertically on a charging pad, with the wing level. A gust then acts on a small side area,
  not on the whole wing as it does on a tailsitter standing on its tail.
- It pays about 0.4 kg and some drag for the lift rotors. On this job that does not matter: it needs minutes of
  flight, not an hour.
- The tailsitter sprints better (0 to 25 m/s in 0.7 s), but it is marginal in gusts and has 11 minutes of flight. It
  stays as the alternative if the client's wind requirement is lower.

This **reopens VTOL**, which the ARIGO catalogue discarded on cost and mass. The reasoning there was right for a
patrol aircraft. For a product whose whole value is that nobody attends it, the lift rotors are the product.

**KPI changes this implies**
- A new gate: an unattended cycle (returns to its station and recharges).
- New KPIs: landing accuracy on the pad; time back to ready, including the recharge; hover in wind.
- The perch-drop KPIs (B3 perch margin, B7 lift-off wind) fall away.

---

## ARPELLA: patrols a wide area

**Applications**
- Pipelines and power lines, border sections, large solar parks, ports, and the perimeters of large bases.
- Convoy routes and event perimeters.

The area is too large for fixed stations, and the threat is slow: small drones, and people on the ground.

**How it is used.** A two-person team launches it by hand or bungee from a base or vehicle. It patrols for hours at
100–150 m with a stabilised camera and **relays its own radio link**. When it finds a drone it pursues; the target
is slow, so 35–40 m/s is enough. It lands back at the team.

**What decides it.** Area covered per sortie, and how rarely it has to come down. Run 1's best was ARIGO-1H with
the biggest pack: 137 minutes at 2.3 kg, because a short wing wastes the energy.

| Candidate | Span | Mass | Stall | Top speed | Endurance | Range | Packed |
|---|---|---|---|---|---|---|---|
| A1 ARIGO-1H, 3S4P (run 1 best) | 1.25 m | 2.37 kg | 11.0 m/s | 42 m/s | 136 min | 110 km | 1250 mm |
| **A2 high-aspect-ratio patroller** | 1.80 m | 2.34 kg | 10.0 m/s | 38 m/s | **206 min** | 152 km | 920 mm (two-piece) |
| A3 motor-glider | 2.40 m | 2.53 kg | 8.9 m/s | 35 m/s | 249 min | 164 km | 1000 mm (three-piece) |

**Choice: A2, the high-aspect-ratio patroller.** A 1.8 m, aspect-ratio-10 pusher wing in two pieces, with a V or T
tail and a gimbal camera under the nose.
- Same pack and mass as run 1's best, but about 50 % more time in the air and a lower stall, so it is still
  hand-launchable.
- The motor-glider adds another 40 minutes. It costs a 2.4 m wing, a bungee, 9 m/s of climb, and a top speed that
  leaves little margin over an 18 m/s target in wind.

**KPI changes this implies**
- The endurance objective moves from 120 to about 180 minutes.
- Coverage should be scored with the gimbal's swath, not a fixed 115 m.
- Communications becomes relay range with terrain (A7); this still has no model.

---

## ASTOR: reaches a fast threat first

**Applications**
- Airports and air bases (incursions, one-way attack drones).
- Protection of high-value events and convoys, and the last layer of a site's air defence.

The threat is a **fixed-wing drone at 30–50 m/s**. Nothing else in the line can catch that from behind.

**How it is used.** It waits in a launcher at the site or on a vehicle. On a cue it is launched automatically,
dashes at 60–70 m/s to meet the target 2–5 km out, and comes down under a parachute. A fresh one is loaded in
minutes.

**What decides it.** Speed, and how long it can hold it. Run 1 gave the surprise: a short-winged ARIGO-1H with a
1200 W motor matched ARIGO-20's top speed with more than twice the full-power time.

| Candidate | Mass | Top speed | Climb | Full power | Stall | Launch |
|---|---|---|---|---|---|---|
| S1 ARIGO-20, twin ducted fans | 2.57 kg | 63 m/s | 43 m/s | 2.8 min | 14.0 m/s | rail |
| **S2 propeller dart** | 1.97 kg | **69 m/s** | 41 m/s | **4.9 min** | 14.4 m/s | rail |
| S3 tube-launched folding dart | 1.93 kg | 76 m/s | 42 m/s | 4.9 min | 16.4 m/s | sealed tube |

**Choice: S2, the propeller dart.** A slim fuselage, a short swept 0.8 m wing, a cruciform tail, and a 1.4 kW
pusher with a folding propeller. It launches from a rail and recovers under a parachute.
- A propeller is far more efficient than small ducted fans at these speeds. That is why it is faster, lighter and
  lasts longer than ARIGO-20.
- **S3 is its growth version.** It is the same aircraft with wings that fold into a sealed tube, which gives
  all-weather storage and reload by swapping a canister. It flies almost identically, so the simulation of S2
  covers it. The hinges are a mechanical-design step, not a flight question.
- ARIGO-20 leaves the line. It remains the reference that S2 has to beat.

**KPI changes this implies**
- A structural gate at top speed: dynamic pressure is about 2.9 kPa, against about 1.7 kPa for ARIGO-1H.
- S6 (tail-chase a 50 m/s target) becomes a gate.
- Recovery is by parachute only.

---

## SPARFELL: protects the people carrying it

**Applications**
- Infantry sections and patrols, border and police units on foot.
- Field teams such as EOD, survey and convoy escorts, with no vehicle and no site.

The threat is a quadcopter that is **already within a kilometre**.

**How it is used.** It rides in a backpack beside everything else the person carries. When a drone is seen or
heard it comes out, is assembled in seconds and thrown. It chases, then lands on its belly nearby and goes back in
the bag.

**What decides it.** Grams and millimetres, then surviving the handling. Run 1 could not get ARIGO-1H below
1.49 kg and 720 mm, because the fuselage, the tail and the standard avionics do not shrink with the wing.

| Candidate | Mass | Packed | Stall | Throw margin | Endurance | What can break |
|---|---|---|---|---|---|---|
| P1 ARIGO-1H lite, 3S1P (run 1 best) | 1.39 kg | 720 mm | 10.3 m/s | 1.16 | 55 min | tail, long fuselage |
| **P2 compact flying wing** | **0.88 kg** | **420 mm** | 10.2 m/s | 1.18 | 86 min | very little |
| P3 small conventional, folding | 1.06 kg | 500 mm | 12.1 m/s | 0.99 | 69 min | tail |

**Choice: P2, the compact flying wing.** A tailless EPP foam wing in two halves that plug into a small pod. The pod
holds the pack, the lighter avionics and a small folding pusher.
- It is the only candidate near both the mass objective (0.9 kg) and the packing objective (500 mm).
- It has no tail and no long fuselage to snap in a bag or a hard landing.
- It still stalls slowly enough to throw.
- Its risks are pitch control and damping without a tail, and a lower maximum lift. Its own aerodynamic model has
  to show those are acceptable.

**KPI changes this implies**
- The KPIs stand as they are. They were the hardest to meet, and this is the configuration that can.

---

## What this means for the simulations

Four airframes, each with its own design file, aerodynamic model and simulation. They share the engine: the lattice
model, the MuJoCo simulator, the performance cards, the launch and wind models, and the KPI scoring.

| Product | New work | Reused |
|---|---|---|
| BUTEO (lift + cruise) | CAD; the **hover and transition simulation** (lift rotors, pad landing in wind); a station model | wing aerodynamics, chase, wind model |
| ARPELLA (patroller) | CAD (long two-piece wing, V tail, gimbal); a coverage model with the gimbal | the whole fixed-wing chain |
| ASTOR (dart) | CAD (short swept wing, cruciform tail); a structural check at top speed | fixed-wing chain; rail and parachute models |
| SPARFELL (flying wing) | CAD; a **tailless aerodynamic model** (elevons, no stabiliser) | simulator, throw and landing models |

BUTEO and SPARFELL each need new physics, so they carry the most risk. ARPELLA and ASTOR are conventional
airframes, and the existing chain covers them once the CAD exists.

**Decisions that are the client's**
1. BUTEO: is an unattended cycle really the requirement? If a person can collect the aircraft, the cheaper fixed
   wing (B1) comes back.
2. ASTOR: what is the fastest target it must catch from behind?
3. ARPELLA: hand launch only, or is a bungee acceptable? That decides how far toward the motor-glider it can go.
