# Walrus — design brief (v0)

The walrus finds clams with its whiskers and digs them out with its tusks and flippers. Walrus is a **subsurface drone that picks clams**: not a dredge but a tracked seabed crawler that sees each clam's
show, loosens it with one jet pulse, lifts it with a soft gripper, sizes it and keeps the legal ones — undersize and
the wrong species go back into the sand behind the tracks. Family: Scampi's autonomy stack (`scampi.md`), Mysis's
lane map (`mysis.md`) laid on the seabed.

## Decisions (v0)

| question | decision |
|---|---|
| how it picks | **selectively, one clam at a time**: camera → jet pulse → soft gripper → sizing chute → basket or back to the bed. The bed stays intact: no fluidising a whole swath, no bycatch |
| digging | **two simple arms** on the front corners, one joint and a flat paddle each: they scrape the top ~6 cm aside where the clams are deep or the sand packed, and clear weed and stones from the lane; the arms dig about 0.1 m³/h on a third of the lane at ~15 % duty — cheap (2.4 kg, €900) and the difference between a bed it can work and one it cannot |
| how it moves | **tracks** on the bed (wide grousers for mud); in the water **eight big, slow ducted thrusters** (round 2, below): it swims, hovers and trims in all six degrees of freedom |
| power | **tethered to the harvest boat**: 48 V and fibre on 150 m of neutral tether; no battery to carry, video and the lane map live |
| getting there | **it swims itself to the bed**: a GPS fix at the surface, a dive, then DVL dead reckoning corrected by the boat's USBL fixes to the waypoint, the tether paying out behind; two horizontal thrusters and enough foam to sit only 1.8 kg heavy. 120 m in 8 minutes at 0.42 m/s (half the tether broadside is most of the drag), arriving within ~2 m, holding a 0.5 m/s current with a ×2 margin |
| the catch | **baskets on float lines**: a full 20 kg basket is released and hauled by the crew while the crawler keeps working on the spare — the crawler never surfaces for the catch |
| pickers | **two** grippers on the belly gantry, side by side: the pick cycle, not the bed, sets the rate, so a second picker doubles it |
| shifts | **two** 8 h shifts a day: the crawler does not care about daylight, the boat's crew does |
| price | under **€30k** |

## v0 numbers (`python labs/marine/designs/walrus.py`)

* 900 × 700 × 380 mm; the crawler **35 kg** (round 2) (45 kg with the tether and reel, which stay on the boat); sits 1.8 kg
  heavy on the bed; the vertical pair lifts it empty with a ×3 margin; swims at 0.42 m/s.
* **Picking-limited**: a 6 s cycle per picker, two pickers → **1 200 clams/h** at any bed density above ~5/m² (it
  crawls at the pace the pickers set, ~0.01 m/s on a rich bed); **36 kg/h** of Manila clams; a basket every 33 min;
  **543 kg a day** in two shifts — **3.6× a hand-harvest diver** — over ~380 m² of bed.
* Power from the boat: 370 W, a ×1.7 margin on a 1 kW tether supply.
* Price (class prices): **€28.8k** with the arms, the swim and the USBL; at €6/kg the day's take pays it back in about a week of harvest days — the
  season, not the machine, is the limit.

## Round 2: the thrusters - big, slow, and every direction

v0's four T200-class thrusters (76 mm props at ~5000 rpm; two vertical, two pushing forward) wasted power and could
not sway. Actuator-disc (momentum) theory: the ideal bollard power for thrust T through a disc of area A is
T^1.5 / sqrt(2 rho A) - double the diameter and the power for the same thrust halves; the real figure divides by the
figure of merit (0.55 for a small fast open prop, 0.65 with a duct round a larger slow one) and the motor and
driver's 0.8. Every candidate is ducted and sized at one disc loading (30 N on a 160 mm disc, ~1490 N/m2), checked
for fit in the 900 x 700 x 380 mm envelope, and its allocation matrix (thrusts -> surge, sway, heave, roll, pitch,
yaw) checked for rank and for the weakest pure force over all directions (`compare_layouts`):

| layout | thrusters | props | mass | fits | rank (of 6) | controlled | weakest force | power, lift 30 N | power, surge 60 N | N/W at full |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 2 vertical + 2 horizontal (fwd), T200 class | 76 mm, ~5100 rpm, open | 1.4 kg | yes | 4 | surge, heave, pitch, yaw | 0 N (no sway) | 87 W | 245 W | 0.24 |
| A: 4 vectored, canted 35 deg up | 4 ducted | 160 mm, 1157/1157/1157/1157 rpm | 2.2 kg | yes | 4 | surge, sway, heave, yaw | 40 N | 57 W | 158 W | 0.61 |
| B: 4 vectored + 2 vertical | 6 ducted | 140/160 mm, 1323/1157 rpm | 3.0 kg | yes | 5 | surge, sway, heave, yaw, pitch | 46 N | 35 W | 164 W | 0.61 |
| **C: 4 vectored + 4 vertical (chosen)** | 8 ducted | 120/140 mm, 1323/1543 rpm | 3.5 kg | yes | 6 | surge, sway, heave, yaw, roll, pitch | 46 N | 33 W | 170 W | 0.61 |

**C** is chosen: it is the only layout with full rank (roll and pitch trim with a basket on one side, or levelling on
a slope), its weakest direction is as strong as B's, it lifts for the least power, and it costs 1.3 kg and ~10 % more
power ahead than A. The horizontal four sit in the deck's corners vectored 45 deg (front pair toward the bow, rear
pair toward the stern, crossed); the vertical four over the tracks. Thrust per watt at full: 0.61 N/W against v0's
0.24 (x2.5), at 1323 rpm instead of ~5100. The crawler gains 2.1 kg (sits 2.9 kg on the bed, lift margin
+1.37), swims at 0.43 m/s, and the price rises to EUR 30.4k: **EUR 0.4k over the 30k target, approved by the owner** (layout C kept; was the owner's
call - layout B would have stayed under it, without roll control).

## What the selectivity buys

A dredge takes everything and leaves the bed turned over; a diver takes the right ones but 150 kg a day, in cold
water, with a bend risk. Walrus takes the right ones at a diver's selectivity and a dredge's hours, and leaves a
count and a size map of what it did not take: the stock estimate the next season's plan is made from.

## Open questions (next round)

1. Razor clams (deep, fast) need the arms to dig deeper and a longer jet — a second paddle shape, or a second machine?
2. Mud beds: the tracks' sinkage and the visibility after the jet pulse set the real crawl speed; a test on a
   real bed before anything else.
3. Rules: which fisheries license a mechanical selective picker where they ban dredging? The first customer is where
   the answer is yes.
4. One crawler per boat, or two on one tether reel sharing the lanes (Mysis's group on the bed)?
