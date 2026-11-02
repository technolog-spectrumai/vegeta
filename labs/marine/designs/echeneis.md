# Echeneis — design brief (v0)

*Echeneis naucrates*, the live sharksucker remora, rides on ships and sharks by a suction disc. Echeneis is a tethered hull
crawler for large merchant ships, and Scampi's sibling (`scampi.md`). It shares the Jetson autonomy, cameras, UT/CP
probe and water model, but it is a different vehicle: it crawls first and swims only between work areas.

## Decisions

| question | decision |
|---|---|
| adhesion | magnetic tracks + a suction thruster (steel first; aluminium/composite at lower grip) |
| main job | cleaning **with capture** of the fouling + inspection (UT thickness, CP/anodes, photogrammetry) on the same pass |
| first ships | bulkers, tankers, container ships: flat steel sides, 10–25 m draft |
| power | tether from the quay or a boat: power, fibre and the fouling return hose |

## v0 numbers (`python labs/marine/designs/echeneis.py`)

* 1.56 × 1.01 × 0.26 m with the cleaning head; **86 kg** in air, 2 kg heavy in water (it sinks if it lets go;
  the tether brings it back). The two magnetic tracks are 18 kg of that.
* Grip, with a ×2 safety factor and μ = 0.45 rubber on wet paint, in a 1.5 m/s (3 kn) current while brushing:
  **+2.6** on a vertical side, **+2.8** upside down under the flat bottom. Magnets give 6.3 kN, the suction thruster 0.4 kN.
* Composite or aluminium hulls (suction only): inspection without brushing in ≤ 0.5 m/s, margin +0.70; no cleaning.
* Cleaning: two Ø500 mm counter-rotating discs, a 0.95 m swath at 0.25 m/s, 85 % lane efficiency, **727 m²/h**:
  a Supramax bulker (~7,500 m²) in 10 h alone, an Aframax tanker in 15 h (2 robots for a 12 h port window), a
  14k TEU container ship in 28 h (3 robots).
* Capture: the hood's 15 mm skirt gap drawn in at 0.3 m/s, **~14 L/s** up a 100 mm, 60 m hose to a filter on the quay;
  about 5 m of head, a **1.4 kW** pump.
* Power: **3.4 kW** down the tether at 400 V DC (8.6 A): capture pump 1.4 kW, suction thruster 0.9 kW, brushes
  0.7 kW, tracks 0.3 kW, electronics 0.2 kW.

The magnet pull per metre, the brush reaction and power, the thruster pull and the filter's head loss are assumed
class values; they are the first things to measure on a prototype.

## Open questions

* where the capture pump sits: on the robot (pushes up the hose) or on the quay (sucks; limited by cavitation)
* lane planning on a real hull: frames, weld seams, bilge keels, the bulbous bow, thruster tunnels, the propeller
* how much fouling (barnacle class) the brushes can take before a cavitation-jet head is needed
* which ports allow in-water cleaning with capture, and what filtration they require (micron rating, biocide paint)
