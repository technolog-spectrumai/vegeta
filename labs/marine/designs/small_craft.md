# Small craft — the contacts round the platform (v0)

These are four simple craft that a simulator shows as **contacts** around an offshore platform at night, for the
guard carrier Esmeralda: a fishing boat, a pilot boat, a sailing yacht and a skiff. They are not robots, but they are
physically honest:

- the **displacement** comes from the hull's dimensions;
- the **top speed** comes from the installed power, against the hull speed (displacement craft) or Savitsky's planing
  resistance (planing craft, the same method as Barracuda);
- the **speed in a sea** comes from the added resistance, or from the impacts the crew takes;
- the **turning circle** and **acceleration** come from class limits.

Every input is a **typical class value with its basis** (`small_craft.BASIS`, and `basis` on each card). **No weapons
on any craft.** The sizing and CAD are in `small_craft.py`; the render is `small_craft_3d.png`
(`small_craft_draw3d.py`). The cards (`kind: craft`) are `exports/ketos/{fishing_boat,pilot_boat,yacht,skiff}.card.json`,
and the Yax files are `exports/*.yx2`, each with its card for Ketos.

## The craft

| | fishing boat | pilot boat | yacht | skiff |
|---|---|---|---|---|
| what | 20 m steel trawler / longliner | 16 m aluminium pilot boat, twin diesels | 12 m GRP sloop, fin keel, 40 kW auxiliary | 7 m open GRP skiff, 2 outboards |
| L x B x T | 20 x 6.2 x 2.6 m | 16 x 5.0 x 1.4 m | 12 x 3.9 x 2.0 m (keel) | 7 x 2.0 x 0.35 m |
| displacement | 157 t | 26.6 t | 12.1 t | 1.65 t loaded (6 aboard), 1.23 t light |
| power | 300 kW | 2 x 450 kW | 40 kW + sail | 2 x 75 kW outboards |
| max / cruise | **9.7** / 8 kn (hull speed); trawling 3 kn | **22.4** / 20 kn | **8.1** / 6 kn under engine; 4–8 sailing | **35.2** / 25 kn loaded; 36.3 light |
| Hs 0.5 / 1.0 / 1.5 / 2.0 / 2.5 m | 9.3 / 9.0 / 8.7 / 8.4 / 8.0 kn | 21.6 / 21.0 / 20.4 / 19.9 / 19.4 kn | 7.7 / 7.4 / 7.0 / 6.7 / 6.3 kn | 24.9 / **15.4** / 12.9 / 11.3 / 10.1 kn |
| limited by | hull speed | power (added resistance) | hull speed | slamming: its crew's 1.5 g, from Hs 0.5 m |
| turning circle | 70 m (3.5 L) | 48 m slow, 75 m at 22 kn | 36–43 m | 18 m slow, 75 m at 25 kn, 147 m at 35 kn |
| 0 to cruise | 11 s | 6 s | 4 s | 7.5 s |
| RCS broadside / bow / stern | ~300 / 60 / 120 m² | ~30 / 8 / 12 m² | ~3 / 1 / 1.5 m² (10 with a reflector) | **~1 / 0.3 / 0.5 m²** |
| lights at night | trawling lights, floodlit deck | white over red, sidelights | tricolour, often dim | **often none** |
| IR | warm stack and wheelhouse, ~6 K | two exhausts, ~5 K | cold under sail, ~2 K | outboards and crew, ~4 K |
| AIS | class A (EU rule, ≥ 15 m) | class A, always on | class B, or none | **none** |
| crew | 3–8 | 2–3 | 1–6 | 2–8 |
| behaviour | slow, long dwell, gear astern | fast, direct, talks on VHF | wind-set course, may not answer | drifts or idles, then runs |

## What it means for the guard

- **The skiff beats Barracuda only in flat water**: 35 kn against 47 kn at Hs 0, so Barracuda is still faster.
  - At **Hs 1 m** the skiff's crew holds it to **15 kn** while Barracuda sustains 29 kn.
  - At **Hs 2.5 m** it is 10 kn against 16 kn.
  - Barracuda can get ahead of it to block at every sea state in the table (`test_barracuda_outruns_the_skiff_in_a_sea`).
- **The 12 m workboat cannot catch it** (6 m/s, 11.7 kn, its card's speed in calm water) below Hs ~2 m. That is the Ketos
  ladder's problem.
- **The skiff's radar cross-section** (~1 m² broadside) puts it at Barracuda's radar range of 4.8 km and Esmeralda's
  X-band range of 7.5 km in Hs 0.5 m, and much less bow-on (~0.3 m², roughly ×0.74 on range).
- **No lights and no AIS** leaves the IR as the only passive cue: the outboards and the crew at ~4 K over the sea.
- **The pilot boat** holds ~20 kn into Hs 2.5 m (heavy, a seated crew) where Barracuda sustains 16. It is no
  threat, but it cannot be paced in a big sea.

## Assumed or not modelled

- **Class values**: every input in `CRAFT`, with its basis in `BASIS`.
- **The crew's impact limits** (1.25 g pilot boat, 1.5 g skiff raid crew) are assumed.
- **Displacement craft**: their speed loss with Hs (0.30 Hs/√L) is a class fit.
- **Turning and acceleration**: class limits (lateral g, lengths) and a simple thrust average.
- **Not modelled**:
  - sail performance (a speed band only);
  - broaching;
  - the trawl's drag beyond its 3 kn speed;
  - each craft's radar reflector, if fitted;
  - lights' intensities.
