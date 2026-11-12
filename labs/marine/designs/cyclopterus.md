# Cyclopterus — design brief (v0)

Named after *Cyclopterus lumpus*, the lumpfish: the cleaner fish salmon farms keep in their pens, which clings to
surfaces by a belly disc. Cyclopterus is a small resident robot for salmon pens, in the same family as Scampi and
Echensis (`scampi.md`, `echensis.md`): the same autonomy stack, cameras and water model, cut down to a farm's price.

## Decisions

| question | decision |
|---|---|
| main job | **all three, by tool**: net inspection + small repair (clip tool), dead-fish removal (grabber + basket), net brushing |
| mobility | **swim + grip the net**. A suction disc cannot hold on a mesh (water flows through it), so four soft fingers pinch the twine; the belly disc is the dock's latch and inductive charging pad |
| base | **resident**, docked on the feed barge; shifts day and night |
| price | **under €40k** a unit |

## v0 numbers (`python labs/marine/designs/cyclopterus.py`)

* 0.8 × 0.6 × 0.42 m, rounded "lumpfish" hull; **30–32 kg** in air depending on the tool; 0.3 kg buoyant with
  3.4–5.3 kg of lead in the keel (CG low, stable).
* Rated to 50 m (nets are 20–40 m deep). Six ducted, guarded thrusters (four vectored horizontal, two vertical); no
  exposed propellers or sharp edges near the fish; dim or red light at night.
* Holding on the net in a 0.5 m/s current with a ×2 safety factor: clip **+1.20**, morts **+1.43**, brush **+0.27**
  (brushing pushes hardest). Four fingers at 60 N pinch each, μ = 0.5 on wet twine (assumed: measure first).
* Inspection: a 1.5 m camera swath at 0.4 m/s, 80 % lane efficiency, **1,728 m²/h**. A 160 m pen (about 7,250 m²
  of net, wall and bottom cone) takes **4.2 h**, about one pen per charge.
* Battery 1.2 kWh LiFePO4: 4.3 h inspecting; tool shifts of 5.8 h (clip, morts) and 3.6 h (brush).
* Morts: a 60 L basket takes about 8 fish (40 kg) a trip, but dead salmon are only ~4 % heavier than seawater, so the
  load in water is just **1.7 kg**.
* Price (class prices, not quotes): **€30k** with one tool, **€34k** with all three, plus a **€6.5k** dock per site
  (charger, latch, USBL head, Wi-Fi/4G link). A first site with one robot, all three tools and its dock is
  **€40.4k**: just over the target, so the second robot at a site is where the economics work.

## The tools (front tool port)

| tool | what | price |
|---|---|---|
| clip | closes holes up to ~0.3 m with polymer clips from a 60-clip magazine | €3.0k |
| morts | grabber + 60 L basket at the net's bottom | €2.2k |
| brush | 0.5 m rotary brush: spot-cleaning between the farm's big net cleaners | €1.6k |

## Open questions

* the clip: which clip closes a knotless nylon/HDPE mesh well enough to pass the farm's escape inspection
* the fingers: pinch force and grip measured on fouled twine, and that they never cut or abrade it
* fish welfare: thruster noise and lights near the fish; how close the robot may come to a feeding school
* how many robots per site: one per 2–3 pens with daily checks, or one per pen
