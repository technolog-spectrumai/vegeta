# Published machines (Yax exchange files)

Vegeta's machines as `.yx2` files, for the platforms that simulate or draw them. A consumer takes them at a Vegeta
commit and needs nothing else of Vegeta: no checkout of its code and no CadQuery or MuJoCo, only these files and Yax.

* `<machine>.yx2` holds `tools/export.py`'s full export of each machine: links, joints, masses and inertias,
  materials, motors and connections where the design states them, and the CAD design as the root link's visual
  mesh (`cad`).
* `MANIFEST.json` records:
  * the Vegeta commit the files were exported from, and the Yax revision;
  * each file's sha256 and size, its lab, links, joints, mass and visual triangles;
  * whether the file serves each consumer in `tools/consumers/`.

Written by `python tools/publish.py` (every machine with a CAD design, or the ones named, or `--all`), after the
designs or the export change. `python tools/publish.py --check` checks that the files match the manifest and are
valid YX2. Do not edit the files by hand.

Consumers: Montezuma (`scripts/update_designs.sh` fetches this folder at a Vegeta ref); Piccolo and Perkun through
their requirements files.
