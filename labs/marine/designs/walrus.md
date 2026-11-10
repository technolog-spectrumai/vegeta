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
| how it moves | **tracks** on the bed (wide grousers for mud); two vertical thrusters only to come up at the end of the shift |
| power | **tethered to the harvest boat**: 48 V and fibre on 150 m of neutral tether; no battery to carry, video and the lane map live |
| getting there | **it swims itself to the bed**: a GPS fix at the surface, a dive, then DVL dead reckoning corrected by the boat's USBL fixes to the waypoint, the tether paying out behind; two horizontal thrusters and enough foam to sit only 1.8 kg heavy. 120 m in 8 minutes at 0.42 m/s (half the tether broadside is most of the drag), arriving within ~2 m, holding a 0.5 m/s current with a ×2 margin |
| the catch | **baskets on float lines**: a full 20 kg basket is released and hauled by the crew while the crawler keeps working on the spare — the crawler never surfaces for the catch |
| pickers | **two** grippers on the belly gantry, side by side: the pick cycle, not the bed, sets the rate, so a second picker doubles it |
| shifts | **two** 8 h shifts a day: the crawler does not care about daylight, the boat's crew does |
| price | under **€30k** |

## The hull (`walrus_hull_3d.png`)

A rounded HDPE fairing in two halves over the frame between the tracks, 12 mm thick, 420 mm to the top: open at
the nose for the camera, the lights and the arms; bored for the two vertical thrusters; the basket hatch on top (the
basket releases through it on its float line); the tether port at the back; flanks flared over the tracks so the
crawler slides over weed and stones instead of catching. 3 kg, €950; it is what it swims in and what the crew
grabs.

## v0 numbers (`python labs/marine/designs/walrus.py`)

* 900 × 700 × 420 mm over the hull; the crawler **36 kg** (48 kg with the tether and reel, which stay on the boat);
  sits 2.8 kg heavy on the bed; the vertical pair lifts it empty with a ×2 margin; swims at 0.42 m/s.
* **Picking-limited**: a 6 s cycle per picker, two pickers → **1 200 clams/h** at any bed density above ~5/m² (it
  crawls at the pace the pickers set, ~0.01 m/s on a rich bed); **36 kg/h** of Manila clams; a basket every 33 min;
  **543 kg a day** in two shifts — **3.6× a hand-harvest diver** — over ~380 m² of bed.
* Power from the boat: 370 W, a ×1.7 margin on a 1 kW tether supply.
* Price (class prices): **€29.8k** with the arms, the swim, the USBL and the hull; at €6/kg the day's take pays it back in about a week of harvest days — the
  season, not the machine, is the limit.

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
