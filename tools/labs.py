"""Vegeta's layout rules: which lab (or layer) a file belongs to, the lifecycle stage it is in, and the imports that
break the rules.

    python tools/labs.py check          # exit 1 on any violation (tests/smoke/test_boundaries.py runs it)
    python tools/labs.py stages         # every machine's lifecycle stage, per lab

The layers::

    libs/   vegeta-* tool packages, ogma, labkit       never import labs/ or tools/
    labs/<lab>/{studies,designs,sims,campaigns,exp,common,data,tests}
            a lab imports libs/ and itself, never another lab; nothing imports exp/ but exp/ and the lab's tests
    tools/  tests/  scripts/                           may import anything (they discover the labs)
    vendor/                                            libraries (Yax, Ozaru)

Labs exchange machines only as files written by Yax (``tools/export.py``), as Piccolo does.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABS = ("aero", "land", "legged", "marine", "studio")
STAGES = ("studies", "designs", "sims", "campaigns", "exp", "common", "data", "tests")

# the old top-level layout, mapped to the labs until it is gone (phase 3 of the split)
_LEGACY = (("legged", r"robot_dog|myropod|apheloria|cleopatra|persephone|gait|stability_plots|benchmark/"),
           ("marine", r"survey_boat|submarine|boat"),
           ("land", r"rover|pekari|perseus|onager|sweeper|atlas|manus|sentinel"),
           ("aero", r"kestrel|nisus|falco|merlin|peregrine|aguya|velutina|drongo|fixed_wing|quad_frame|frame_study|"
                    r"prop_lab|propulsor|air_propeller|ducted_fan|turbojet|microjet|mini_ra|xwing|quadcopter|exotic|"
                    r"races/|side_labs/|\bxc\b|propeller"))
_LEGACY_DIRS = ("designs", "notebooks", "scenarios", "components", "assemblies", "side_labs", "races", "benchmark")


def place(path: str) -> tuple[str, str | None]:
    """(group, stage): group is a lab name, ``libs``, ``tools`` or ``vendor``; stage the lab folder (or None)."""
    parts = path.split("/")
    if parts[0] == "labs" and len(parts) > 2:
        return parts[1], parts[2] if len(parts) > 3 else None
    if parts[0] == "libs" or parts[0].startswith("vegeta-") or parts[0] == "ogma":
        return "libs", None
    if parts[0] == "vendor":
        return "vendor", None
    if parts[0] in _LEGACY_DIRS:
        if parts[0] == "notebooks" and (parts[1] == "demo" or re.match(r"notebooks/old/0[0-5]_", path)):
            return "studio", "studies"
        for lab, rx in _LEGACY:
            if re.search(rx, path):
                return lab, "exp" if "/exp/" in path or "/old/" in path else "legacy"
        return "libs" if parts[0] in ("components", "assemblies") else "unplaced", "legacy"
    return "tools", None


def violation(src: str, dst: str) -> str | None:
    """Why the import ``src`` → ``dst`` breaks the rules (None when it does not)."""
    (gs, ss), (gd, sd) = place(src), place(dst)
    if gs in ("tools", "vendor") or gs == "unplaced" or gd == "unplaced":
        return None
    if gs == "libs" and gd not in ("libs", "vendor"):
        return f"libs import {gd}"
    if gs in LABS and gd in LABS and gs != gd:
        return f"lab {gs} imports lab {gd}"
    if gd in LABS and sd == "exp" and not (gs == gd and ss in ("exp", "tests")):
        return f"{gs}/{ss} imports {gd}'s exp (limbo is imported by nothing)"
    return None


def scan(root: Path = ROOT):
    """Every internal import edge (src, dst, line) of the repository, resolved as ``tools/dump_ast.py`` resolves them."""
    import ast
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import dump_ast as d
    files = list(d.walk(root))
    res = d.Resolver(root, files)
    for f in files:
        rel = f.relative_to(root)
        try:
            src = d.notebook_source(f) if f.suffix == ".ipynb" else f.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src)
        except (SyntaxError, ValueError, OSError):
            continue
        for name, level, line, dynamic, names in d.imports(tree):
            if name == "*":
                continue
            cands = [f"{name}.{n}" for n in (names or [])] if not level else \
                [(f"{name}.{n}" if name else n) for n in (names or [])]
            hits = [res.resolve(c, f, level) for c in cands] if names else []
            hits = [h for h in hits if h[0] is not None] or [res.resolve(name, f, level)]
            for hit, how in hits:
                if hit is not None and hit != f:
                    yield rel.as_posix(), hit.relative_to(root).as_posix(), line


def check(root: Path = ROOT) -> list[str]:
    out = []
    for src, dst, line in scan(root):
        why = violation(src, dst)
        if why:
            out.append(f"{src}:{line} -> {dst}: {why}")
    return sorted(set(out))


def stages(root: Path = ROOT) -> dict:
    """lab -> machine -> the stages its files are in (and ``export`` when the lab registers it)."""
    out = defaultdict(lambda: defaultdict(set))
    for lab_dir in sorted((root / "labs").glob("*")):
        if not lab_dir.is_dir():
            continue
        lab = lab_dir.name
        mach = _machines_of(lab_dir)
        for stage in ("studies", "designs", "sims", "campaigns", "exp"):
            for f in (lab_dir / stage).rglob("*"):
                if f.suffix in (".py", ".ipynb") and "tests" not in f.parts:
                    for m in mach:
                        if re.search(rf"(^|[_/\d-]){re.escape(m)}", f.relative_to(lab_dir).as_posix()):
                            out[lab][m].add(stage)
        reg = lab_dir / "designs" / "machines.py"
        if reg.exists():
            for m in re.findall(r'register\("(\w+)"', reg.read_text()) + \
                    re.findall(r'"(\w+)": "\w+:\w+"', reg.read_text()):
                out[lab][m].add("export")
    return {lab: {m: sorted(s, key=_stage_order) for m, s in sorted(ms.items())} for lab, ms in out.items()}


def _stage_order(s):
    return ("studies", "designs", "sims", "campaigns", "export", "exp").index(s)


def _machines_of(lab_dir: Path) -> list[str]:
    """The machines a lab declares in ``lab.toml`` (``machines = [...]``)."""
    import tomllib
    f = lab_dir / "lab.toml"
    return tomllib.loads(f.read_text()).get("machines", []) if f.exists() else []


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else "check"
    if cmd == "check":
        v = check()
        print("\n".join(v) if v else "layout ok: no import breaks the lab rules")
        return 1 if v else 0
    if cmd == "stages":
        for lab, ms in stages().items():
            print(f"[{lab}]")
            for m, s in ms.items():
                print(f"  {m:<18} {' → '.join(s)}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
