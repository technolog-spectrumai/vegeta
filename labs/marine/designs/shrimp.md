# Shrimp — design brief (v0)

The smallest of the family (`scampi.md`, `echensis.md`, `cyclopterus.md`, `nephro.md`, `squilla.md`): a
**backpack-held craft for aquaculture work**. Where Cyclopterus is a 30 kg resident robot with a dock on the barge,
Shrimp is what one person carries to the pen, drops in by hand, flies from a tablet on a 50 m tether, and packs
back. The rethink: the resident robot's economics only work from the second unit at a site; a backpack craft works
from the first day, on any site, with the same autonomy stack scaled to a Jetson Orin Nano.

## Decisions (v0)

| question | decision |
|---|---|
| jobs | net inspection (holes, fouling, mooring lines), mortality spotting and counting, feeding and behaviour observation, water sampling and probing at depth — **not** mort retrieval or brushing (those stay with Cyclopterus: too heavy for a hand-launched craft) |
| carry | one backpack under 10 kg all in: the vehicle under 5 kg, the tether on a hand reel, a rugged tablet, a spare battery |
| link | **tethered**, 50 m, 3 mm neutral: live 4K video and a hard safety line; no acoustic positioning (the tether and the net-following do the job) |
| compute | a Jetson Orin Nano aboard: net following, hole and mort detection, so the operator supervises |
| tools | a port under the nose: a 250 mL syringe **sampler**, a dissolved-oxygen / temperature / salinity **probe** pod, a six-clip **magazine** for small holes (the big ones are reported) |
| price | under **€6k** a kit |

## v0 numbers (`python labs/marine/designs/shrimp.py`)

* 360 × 230 × 130 mm, a rounded shell round one Ø90 × 250 mm tube (dome front, tether port back); **4.2–4.5 kg
  in air** with the tool, trimmed with ~0.3 kg of lead; the whole kit **8.6–8.9 kg**.
* Six 60 mm ducted thrusters at 6 N each (four vectored at 45°, two vertical): 17 N forward, **1.26 m/s** flat out.
* Battery 100 Wh (swappable): **2.5–2.6 h** a dive at the inspection speed (the brain on the surface, round 2). A 160 m pen's net (7 250 m²) at a
  1 m swath and 0.3 m/s takes 8.4 h: **four dives** — this is an inspection craft, not a one-charge-per-pen one.
* **The tether sets the envelope, not the hull.** In 0.5 m/s with 25 m of tether out, the tether's drag (14 N)
  is five times the hull's (3 N) and the thrusters hold with a 19 % margin; with the full 50 m out they cannot. So:
  0.5 m/s on up to **31 m** of tether, 0.55 m/s at 25 m, or slack water for the deep end of the net. The operator
  pays out only what the depth needs.
* Price (class prices, round 2): **€5.4k** the base kit with the probe pod, €5.7k with all tools; the vehicle alone
  €2.6k, the surface box with the Jetson, tablet and reel the rest.

## Decisions (round 2: the open questions answered, and some ideas added)

| question | decision | why |
|---|---|---|
| rated depth | **50 m stays**; a 100 m variant is a thicker tube (+0.3 kg, +€150) for the odd deep site, not the default | salmon nets are 20–40 m; mussel and seaweed lines shallower; the tether, not the tube, limits the deep end anyway |
| where the brain is | **on the surface**: the Jetson Orin Nano moves into the tether box; the vehicle keeps a microcontroller (thrusters, IMU, depth, the video encoder) and an **auto-surface reflex** | the video goes up the tether anyway (100 Mb/s, no added latency to speak of); the vehicle gets lighter, cheaper (€5.4k → the vehicle itself €2.6k) and dumber, which is what you want in the thing that gets lost; a vehicle that loses its tether floats up on its 50 g of positive trim, strobes, and holds a heading on its MCU |
| the clip tool | **dropped**; replaced by a magazine of six numbered **clip-on tags** | a hand-launched 4 kg craft should not be mending nets in a current; it should *find and mark* — a tag at the hole and a GPS-less "tether length + heading + depth" fix in the report send the divers straight there |
| fish safety | **in software, enforced in firmware**: a near-fish thrust cap (≤ 0.3 m/s when the detector sees fish inside 1 m), red light by the clock at night; the lights' colour is the visible state, no switch to forget | a switch is one more thing to leave in the wrong position |

Ideas added:

* **The probe pod is standard**, not optional: a 40 m **oxygen profile** takes 4 minutes, and a daily O₂ profile
  is the single most valuable number a farm can get from a craft this size (oxygen crashes kill pens). Base kit
  **€5.4k**; all tools €5.7k.
* **The net is its own ruler**: the mesh pitch is known (25–30 mm), so the camera's view of it gives the distance
  to the net without a sonar — the net-following loop uses it; a €2.8k sensor stays out.
* **Lanes with the tether paid out by the operator**: the autonomy flies the lane at constant net distance; the
  operator only manages the reel (and is told "pay out" / "take in" on the tablet from the depth and the tether
  tension), which is what keeps the exposed length — and its drag — minimal.
* A **top strobe** and the auto-surface reflex: the two cheapest things that stop a craft being lost.
* The report: holes, morts (counted per lane, not per sighting: the lanes overlap 20 %), fouling score per lane,
  the O₂ profile — one page per dive, sent from the tablet before the backpack is closed.

## Open questions (round 3)

1. A farm's fleet: one kit per site, or one per technician? (The kit price makes the second one easy.)
2. Krill-sized next: is there a case for a 1 kg "look only" version with no tool port at all?
