# Odobenus — design brief (v0)

Named after *Odobenus rosmarus*, the walrus, which finds clams with its whiskers and sucks them out of the sand.
Odobenus is a **subsurface drone that picks clams**: not a dredge but a tracked seabed crawler that sees each clam's
show, loosens it with one jet pulse, lifts it with a soft gripper, sizes it and keeps the legal ones — undersize and
the wrong species go back into the sand behind the tracks. Family: Scampi's autonomy stack (`scampi.md`), Mysis's
lane map (`mysis.md`) laid on the seabed.

## Decisions (v0)

| question | decision |
|---|---|
| how it picks | **selectively, one clam at a time**: camera → jet pulse → soft gripper → sizing chute → basket or back to the bed. The bed stays intact: no fluidising a whole swath, no bycatch |
| how it moves | **tracks** on the bed (wide grousers for mud); two vertical thrusters only to come up at the end of the shift |
| power | **tethered to the harvest boat**: 48 V and fibre on 100 m; no battery to carry, video and the lane map live |
| the catch | **baskets on float lines**: a full 20 kg basket is released and hauled by the crew while the crawler keeps working on the spare — the crawler never surfaces for the catch |
| pickers | **two** grippers on the belly gantry, side by side: the pick cycle, not the bed, sets the rate, so a second picker doubles it |
| shifts | **two** 8 h shifts a day: the crawler does not care about daylight, the boat's crew does |
| price | under **€30k** |

## v0 numbers (`python labs/marine/designs/odobenus.py`)

* 900 × 700 × 380 mm; the crawler **29 kg** (38 kg with the tether and reel on the boat); sits 4 kg heavy on the
  bed; the vertical pair lifts it empty with a 53 % margin.
* **Picking-limited**: a 6 s cycle per picker, two pickers → **1 200 clams/h** at any bed density above ~5/m² (it
  crawls at the pace the pickers set, ~0.01 m/s on a rich bed); **36 kg/h** of Manila clams; a basket every 33 min;
  **543 kg a day** in two shifts — **3.6× a hand-harvest diver** — over ~380 m² of bed.
* Power from the boat: 330 W, three times the margin on a 1 kW tether supply.
* Price (class prices): **€23.6k**; at €6/kg the day's take pays it back in about a week of harvest days — the
  season, not the machine, is the limit.

## What the selectivity buys

A dredge takes everything and leaves the bed turned over; a diver takes the right ones but 150 kg a day, in cold
water, with a bend risk. Odobenus takes the right ones at a diver's selectivity and a dredge's hours, and leaves a
count and a size map of what it did not take: the stock estimate the next season's plan is made from.

## Open questions (next round)

1. Razor clams (deep, fast) need a longer jet and a different grip — a second head, or a second machine?
2. Mud beds: the tracks' sinkage and the visibility after the jet pulse set the real crawl speed; a test on a
   real bed before anything else.
3. Rules: which fisheries license a mechanical selective picker where they ban dredging? The first customer is where
   the answer is yes.
4. One crawler per boat, or two on one tether reel sharing the lanes (Mysis's group on the bed)?
