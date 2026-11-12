# Echensis — design brief (v1)

Named after *Echeneis naucrates*, the live sharksucker remora, which rides on ships and sharks by a suction disc. Echensis is a tethered hull
crawler for large merchant ships, and Scampi's sibling (`scampi.md`). It shares the Jetson autonomy, cameras, UT/CP
probe and water model, but it is a different vehicle: it crawls first and swims only between work areas.

## Decisions

| question | decision |
|---|---|
| adhesion | magnetic tracks + a suction thruster (steel first; aluminium/composite at lower grip) |
| main job | cleaning **with capture** of the fouling + inspection (UT thickness, CP/anodes, photogrammetry) on the same pass |
| first ships | bulkers, tankers, container ships: flat steel sides, 10–25 m draft |
| power | tether from the quay or a boat: power, fibre and the fouling return hose |

## Decisions (round 2)

| question | decision |
|---|---|
| capture pump | **on the robot**, pushing the slurry up the hose: no suction lift, so no cavitation or hose collapse, and any hose length |
| heavy fouling | brushes now (grooming); a **cavitation-jet head** on the same quick-change mount for barnacles and tubeworms |
| navigation | **hull map**: track odometry + depth + the hull's inclination, corrected by camera features (weld seams, marks, draft marks); lanes planned from the ship's drawings when available; forward sonar for the bilge keel and appendages |
| name | Echensis |

### Navigation on the hull

A ship's side is a developable surface. The robot works in hull coordinates: the girth position along the frame
(from depth and the inclination of the hull's normal) and the length along the ship (from track odometry). Camera
features (weld seams, the draft marks, plate edges) reset the odometry's drift, and the forward sonar finds the bilge
keel, sea chests and thruster tunnels before the head reaches them. Lanes run fore and aft, one head width apart, with a
10 % overlap (the 85 % lane efficiency in the sizing). Seams and appendages are marked on the map as no-go strips for the
tracks.

### The two heads

| head | swath | speed | rate | robot power | quay power | for |
|---|---|---|---|---|---|---|
| brushes (2 × Ø500 mm) | 0.95 m | 0.25 m/s | 727 m²/h | 0.7 kW | – | slime, weed, light shell: grooming every few weeks |
| cavitation jet (rotating bar) | 0.60 m | 0.10 m/s | 184 m²/h | 0.12 kW | ~15 kW HP pump | barnacles, tubeworms |

Heavy fouling is slow by nature (a bulker is ~41 robot-hours with the cavitation head): it is a spot job between
groomings, or several robots, never a whole hull in one port call.

## v1 numbers (`python labs/marine/designs/echensis.py`)

* 1.56 × 1.01 × 0.26 m with the cleaning head; **99 kg** in air (brush head), 2 kg heavy in water (it sinks if it
  lets go; the tether brings it back). The magnetic tracks are 18 kg, the onboard capture pump 6 kg, the foam 21 kg.
* Grip, with a ×2 safety factor and μ = 0.45 rubber on wet paint, in a 1.5 m/s (3 kn) current while brushing:
  **+2.6** on a vertical side, **+2.8** upside down under the flat bottom. Magnets give 6.3 kN, the suction thruster 0.4 kN.
* Composite or aluminium hulls (suction only): inspection without brushing in ≤ 0.5 m/s, margin +0.70; no cleaning.
* Cleaning: two Ø500 mm counter-rotating discs, a 0.95 m swath at 0.25 m/s, 85 % lane efficiency, **727 m²/h**:
  a Supramax bulker (~7,500 m²) in 10 h alone, an Aframax tanker in 15 h (2 robots for a 12 h port window), a
  14k TEU container ship in 28 h (3 robots).
* Capture: the hood's 15 mm skirt gap drawn in at 0.3 m/s, **~14 L/s** up a 100 mm, 60 m hose to a filter on the quay;
  about 5 m of head, a **1.4 kW** pump.
* Power: **3.5 kW** down the tether at 400 V DC (8.6 A): capture pump 1.4 kW, suction thruster 0.9 kW, brushes
  0.7 kW, tracks 0.3 kW, electronics 0.2 kW.

The magnet pull per metre, the brush reaction and power, the thruster pull and the filter's head loss are assumed
class values; they are the first things to measure on a prototype.

## Open questions

* weight: 99 kg is heavy to handle on a quay; allowing more weight in water (the magnets carry it) cuts the foam
* the tether: one umbilical (power, fibre, slurry hose, the cavitation head's HP hose) and its winch on the quay
* turning on the hull: skid-steer on magnetic tracks scrubs paint; a lifted turn on the suction thruster instead
* which ports allow in-water cleaning with capture, and what filtration they require (micron rating, biocide paint)
