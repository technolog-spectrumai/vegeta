# K4TQ — lessons learnt: K4TQ-H3 "Hanna 3" and K4TQ-J3 "Joanna 3" (2026-10-22)

What went wrong, looked odd, or surprised us while growing Ginna 3 (2.1 kg) into Hanna 3 (3.5 kg) and Joanna 3 (4.5 kg).
Numbers are CALC / ESTIMATE on the Stage 1 models (no CAD, no CFD, no FEA at these scales).

## Design findings

1. **No single off-the-shelf drive works.** Under the hover rules (T/W ≥ 1.6 sagged, most loaded motor ≤ 75 %) the best
   single drive (MN605-S / MN801-S on Aeronaut CAM 18x13) reached T/W 1.37 (Hanna) and 1.16 (Joanna). The reason: no
   folding propeller is both **big and high-pitch**. The big folding ones (T-Motor MF2009 20.4x7.3, FA22.2/24.2/29.2) are
   low pitch and cannot cruise (pitch speed < 23 m/s); the high-pitch folding ones (Aeronaut CAM) stop at 18".
2. **The way out: two drives.** The front pair never stops (it cruises), so it does not need to fold — **rigid APC 22x12E**
   (cruise + hover); the rear pair only hovers and transitions, so it can be **big, low-pitch and folding** (T-Motor FA24.2).
   K4TQ's "rear pair stopped in cruise" architecture is what makes this possible — worth keeping.
3. **The front pair is the hover bottleneck**, not the total thrust: with the CG near mid-stagger the front pair carries
   about half the weight; on the small cruise props it saturated first (most loaded motor 0.80–0.93) while the total T/W
   already looked fine. Check the per-pair loading, not only T/W.
4. **Mass snowballs with payload.** Hanna is +8.9 kg over Ginna 3 for +1.4 kg payload; Joanna +10.2 kg for +2.4 kg. The
   drivers: the 12S6P pack (5.4 kg, needed to keep 83 km), the airframe grown to keep the 18 m/s stall (×1.74 / ×1.77 of
   Freddy), the middle root hardware. Joanna (22.7 kg) is back near Bertha's 24 kg, which was parked as "too heavy".
5. **A step in the root hardware.** From about ×1.48 the small wing-root set (2.9 N·m servo, 6804 bearings) no longer
   holds and the middle set (6.5 N·m, 6806) adds ≈1.2 kg at once — a discontinuity the sweep had to step over.
6. **The stall cap sets the scale.** A smaller airframe would carry the payload lighter but stall faster; every heavy
   candidate sat on the 18 m/s line. Joanna ends at 18.1 m/s (0.1 over) to keep T/W ≥ 1.6 — the two rules fight.
7. **Spar sized by stiffness, not strength.** The spar comes from the first bending mode (≥ 12 Hz): Ø32–35 mm at a strength
   reserve of 4–6. A lighter spar is a Stage 4 (FEA, modes) question.
8. **Rigid props stopped are expensive.** Rigid APCs on the *stopped* pair cost range (stopped-prop drag taken ×4); they only
   pay off on the running pair.
9. **Carrying a 15" FPV quad**: inside the cartridge it is just payload (2.34 kg < 3.5 / 4.5 kg — if it fits folded:
   cartridge ≈ 420 × 210 × 190 mm outside); slung outside, it more than doubles the drag area (0.03 → 0.08 m²): range
   86 → 51 km, top speed → 29 m/s. Carry it inside or not at all.

## Data and model problems met on the way

10. **Catalogue errors found by checking datasheets**: MN801-S KV150 is 480 g (not 320 g), 50 mΩ, 60 A; the "U8 II KV170"
    used for Ginna / G2 is not a real product (the U8 II comes as KV85/100/150/190) — its real twin is the MN601-S KV170;
    MN505-S KV380 is rated 6S only. All corrected and flagged (`off_the_shelf`, `cells_max`).
11. **The model assumed one drive for all four motors** (hover split, cruise with four running, root loads, acceleration).
    It now takes a rear drive (`motor_rear`, `prop_rear`); the top speed of a two-drive design is taken on the front pair
    alone (the rear's low pitch would cap the four-motor sweep).
12. **Candidates reading "hover 0 min" / "range 0 km"** in the sweep tables are not results: they are drives that cannot
    hold the weight (hover) or cannot make cruise thrust (range) — the model returns infinite power; filter them, don't
    plot them.
13. **The renders are Freddy's CAD scaled** — the earlier scaled renders flagged a gear / frame clash and the cartridge
    outside the skin; the CAD of these scales is still to do (todo K4.11).
14. **Above 20 kg** the operation needs a specific-category authorisation (SORA) even under 25 kg — a programme cost the
    performance tables don't show.
