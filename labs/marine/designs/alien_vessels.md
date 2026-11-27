# Fictional alien vessels — exterior models for the sea game

Two **fictional extraterrestrial vessels**. They are **non-functional exterior models**: simple solids, lofts,
extrusions, chamfers and fillets, with nothing inside, for a real-time 3D naval simulation. They fly the owner's
**flag**: black, with a white alien head with large almond-shaped black eyes above two crossed bones. The emblem was
traced from the owner's image (`emblem_2d`: the head's width is the unit; the eyes are almonds slanted 38 deg; the
bones cross at 30 deg under the chin).

- **Design and CAD**: `alien_vessels.py` (`AlienShip`, `AlienSubmarine`).
- **Render**: `alien_vessels_3d.png` (`alien_vessels_draw3d.py`).
- **Ketos cards**: `exports/ketos/alien_ship.card.json` and `alien_submarine.card.json` (`kind: vessel`,
  `fictional: true`), with one mesh per component in its colour.
- **Yax files**: `exports/alien_ship.yx2` and `exports/alien_submarine.yx2` (Ketos and Montezuma served).

All dimensions are parameters. Frames: x forward, the ship's waterline at z = 0, the submarine's axis at z = 0.

## Alien ship

| | |
|---|---|
| overall | **140 m** long, **17 m** wide, **8 m** of hull above the water at midship (4.5 m draught below), **31 m** overall above the water (the flag's top) |
| hull (dark graphite) | an elongated loft through 15 angular sections: a flat keel, a hard chine at the maximum beam, **sloping sides** (tumblehome to 84 % of the beam at the deck edge), a **sharp raised bow** (the sheer rising 3.5 m, the stem raked 32 deg) and a **broad flat transom** (88 % of the beam) |
| superstructure (medium grey) | **56 x 13 m**, two stepped tiers with chamfered (faceted) edges and raked ends |
| bridge (medium grey) | a faceted block forward on the top tier, its front raked |
| windows (black) | a **narrow panoramic band** round the bridge's front and sides |
| tower (grey) | the **geometric sensor tower**: an octagonal frustum, a faceted hexagonal sensor head, a mast |
| flag (black) | 6 x 4 m (3:2) on the tower's mast, the emblem on both faces |
| emblem | on **both sides of the bow**, 6 m tall, on the sloping side's plane (white, with black eye inserts) |
| displacement | 5346 t (the hull's volume below the waterline: the mass of its rigid body) |

## Alien submarine

| | |
|---|---|
| overall | **95 m** long, **10 m** hull width and height, **17 m** with the sail |
| hull (matte graphite-black) | a body of revolution: a **rounded** elliptic bow (14 m), a **cylindrical** middle, a **tapered** stern (30 m) to a small cone |
| sail (graphite-black) | **12 x 4 x 7 m** above the hull, angular: a hexagonal section lofted to a smaller crown **swept back 12 deg** |
| fins (dark grey) | sail planes two thirds up the sail, **cruciform** stern fins |
| masts (dark grey) | decorative sensor masts: a periscope-like column, a faceted sensor mast, a whip |
| emblem | on **both sides of the sail**, 4 m tall |
| flag | a small 2.4 x 1.6 m flag on a mast on the sail: **surfaced configuration only** (drop `flag` and `flag_emblem` submerged) |
| displacement | 5759 t submerged (the hull's and the sail's volume) |

## Components (separate meshes)

- ship: `hull`, `superstructure`, `bridge`, `windows`, `tower`, `emblem`, `emblem_eyes`, `flag`, `flag_emblem`;
- submarine: `sub_hull`, `sail`, `fins`, `masts`, `emblem`, `emblem_eyes`, `flag`, `flag_emblem`.

There is no performance on the cards: these are fictional looks. The only physics is the displacement, for a rigid
body.
