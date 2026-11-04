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

* 360 × 230 × 130 mm, a rounded shell round one Ø90 × 250 mm tube (dome front, tether port back); **4.2 kg in
  air** (4.5–4.7 with a tool), trimmed with 0.1–0.3 kg of lead; the whole kit **8.4–8.9 kg**.
* Six 60 mm ducted thrusters at 6 N each (four vectored at 45°, two vertical): 17 N forward, **1.26 m/s** flat out.
* Battery 100 Wh (swappable): **2.0–2.2 h** a dive at the inspection speed. A 160 m pen's net (7 250 m²) at a
  1 m swath and 0.3 m/s takes 8.4 h: **four dives** — this is an inspection craft, not a one-charge-per-pen one.
* **The tether sets the envelope, not the hull.** In 0.5 m/s with 25 m of tether out, the tether's drag (14 N)
  is five times the hull's (3 N) and the thrusters hold with a 19 % margin; with the full 50 m out they cannot. So:
  0.5 m/s on up to **31 m** of tether, 0.55 m/s at 25 m, or slack water for the deep end of the net. The operator
  pays out only what the depth needs.
* Price (class prices): **€4.7k** camera-only, €5.3k with the probe, **€5.9k with all three tools**.

## Open questions (next round)

1. Rated depth: 50 m covers salmon pens; mussel and seaweed lines are shallower; is there a deeper site?
2. Should the tablet do the inference instead (a cheaper vehicle, no Jetson aboard, the video goes up anyway)?
3. The clip tool at this size: worth it, or inspection-and-report only?
4. Red light at night and a slow mode near fish: fixed in software, or a physical "fish mode" switch?
