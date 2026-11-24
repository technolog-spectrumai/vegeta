# GOSHAWK — a small aircraft for hunting inside a forest

First sizing, 2026-11-24, branch `dev_astor_hawk`. Code: `labs/aero/designs/goshawk.py`.
The earlier open-country interceptor is stored as PROTOHAWK (`labs/aero/protohawk.md`).

**Status: a drawing, a mass budget and a flight model carried over from another aircraft.
No CFD, no FEA, no forest flight has been run. On the model's own cautious numbers it
does not yet turn tightly enough for a dense stand.**

How to read the numbers: **Section model** is NeuralFoil on the wing section. **Carried
over** is the flight model, which takes PROTOHAWK's predecessor's coarse CFD scaled by
area, aspect ratio and body size, at one third of that aircraft's Reynolds number.
**HAND** is an estimate with nothing behind it; every mass is HAND.

## 1. The bird and the aircraft

| | Northern goshawk (HAND, from memory) | GOSHAWK | PROTOHAWK |
|---|---|---|---|
| Mass | 0.7–1.5 kg | 2.05 kg | 8.25 kg |
| Span | 1.0–1.2 m | 1.50 m (the limit set) | 1.95 m |
| Wing area | about 0.2 m² | 0.45 m² | 0.51 m² |
| Wing loading | about 5 kg/m² | 4.6 kg/m² | 16 kg/m² |
| Tail, share of the wing | large | 38 % | 37 % |
| Static thrust / weight | — | 2.0 (HAND) | 2.5 (HAND) |
| Catches with | feet | two legs, tear-away feet on a line | net launcher, three nets |

GOSHAWK is heavier than the bird because it carries a vision computer, two motors and a
battery; the wider span buys the bird's wing loading back.

## 2. Layout

- **Wing:** 1.50 m span, 360 mm chord at the centre to 240 mm at the tip, aspect ratio
  5.0, section NACA 6412. Flaperons on the outer pieces, plain flaps on the inner pieces
  behind the propellers.
- **Wing tuck in flight:** the outer pieces swing back on pivots 250 mm out. Spread
  1.50 m, drawn in 1.04 m. Rolled on its side the aircraft is 0.3 m wide.
- **Tail:** two all-moving halves, 330 mm out and 340 mm long at the root, each with a
  geared flap, low in the propeller wash; one fin.
- **Power:** two 600 W motors with 8-inch propellers on the inner wing.
- **Body:** 980 mm long, 60 mm diameter; overall length 1.32 m.
- **Legs:** two sprung legs under the balance point, smaller copies of ASTOR-T's, with
  feet that tear away on a line to foul the rotors.
- **Pack:** 4S 2200 mAh, 33 Wh.

It carries no net launcher: GAVILAN's weighs 1 kg with a single net.

## 3. Can it turn between the trees?

Mean distance between stems, on a square grid: 5.0 m at 400 stems per hectare, 3.8 m at
700, 3.2 m at 1000 (stand densities are HAND).

| Speed | Wing, as the flight model has it | Wing, if the section's lift holds | Nose 40° up, on the propellers |
|---|---|---|---|
| 8 m/s | 5.6 m | 3.7 m | 3.6 m |
| 10 m/s | 5.7 m | 3.8 m | 4.5 m |
| 12 m/s | 5.9 m | 4.0 m | 5.3 m |
| 14 m/s | 6.1 m | 4.1 m | 6.0 m |
| 17 m/s | 6.4 m | 5.0 m | 6.7 m |

- **As modelled (pessimistic):** maximum lift coefficient 0.98 with flaps, carried over
  from a coarse CFD of a different wing. Turn radius 5.6–6.4 m: wider than the stems'
  spacing in every stand above.
- **If the section holds:** NeuralFoil gives NACA 6412 a maximum of 1.52 flaps up
  (confidence 0.97). Times 0.9 for the wing, plus the flap gain the CFD measured, that
  is 1.57. Turn radius 3.7–4.1 m: enough for a stand of 400–700 stems per hectare.
- **On the propellers:** nose 40° up, past the stall. 3.6 m at 8 m/s, but thrust only
  holds the speed up to about 10 m/s, there is no measured aerodynamics there, and no
  controller flies it.

Stall: 8.5 m/s flaps down as modelled, 6.9 m/s if the section holds.

**The wing's real maximum lift decides this aircraft.** The two estimates differ by 60 %.

## 4. Stability and balance

- Balance point 300 mm from the nose, static margin 5 %.
- **The tail does not carry lift in trim** (about 2 %). A lifting tail needs the balance
  point at or behind the neutral point. Tried at 305 mm on the first sizing: the tail
  carried 7 %, and hands off the aircraft lost 60 m in 20 s. The bird does this with a
  brain; here it needs an autopilot loop that does not exist yet.
- Angle-of-attack augmentation is in the controller for this design. Its gains are
  PROTOHAWK's, moved to GOSHAWK's speeds, **not tuned**.
- One debug stability job at 14 m/s: stable hands off, Dutch-roll damping 0.32, roll
  rate 240 °/s through the first turn. A 3 m/s gust reaches the stall at that speed.

## 5. What is not known or not done

- **Maximum lift** (section 3). CFD at this aircraft's speed is prepared.
- **Reynolds number.** The carried-over data is from 3.5 × 10⁵; GOSHAWK flies at 2 × 10⁵.
- **No forest flight simulation exists.** Nothing here flies between obstacles. The turn
  table is a steady-turn calculation.
- **Perception.** Flying a wood by sight is assumed (95 g of computer and cameras) and
  not designed.
- **The catch.** The legs are drawn; no catch has been simulated on this aircraft.
  ASTOR-T's catches closed at 17 m/s and more; GOSHAWK can fly at the prey's speed.
- **Mass** is 2.05 kg on HAND estimates.
- **Propellers** are momentum theory, not a chosen part. Disc clearance to the body and
  to the pivot is 8 mm each, which is tight.
- **Tucked span** is 1.04 m, not the 0.5 m a bird manages. The motors on the wing set it.
- **Endurance** is not computed; 33 Wh at 1200 W is 1.4 minutes at full power.

## 6. To run

```bash
git checkout -- labs/aero/designs/data/astor_v_cfd_screening.yaml
```

```bash
git pull origin dev_astor_hawk
```

CFD at 12 m/s, 35 cases: flaps up and down 20° to 60° angle of attack, flaps 35°, wings
drawn in.

```bash
bash labs/aero/sims/astor.sh -j 14 --design goshawk --cfd-only
```

Stability and performance card, 8 jobs, after the CFD:

```bash
bash labs/aero/sims/astor.sh -j 14 --design goshawk --skip-cfd
```

Files: drawings and STEP in `labs/aero/designs/data/goshawk/`; the turn table is
`goshawk_forest.yaml` there, from `goshawk.forest()`.
