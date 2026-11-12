"""Publish Vegeta's machines as Yax exchange files for the platforms that consume them: ``exports/<machine>.yx2`` and
``exports/MANIFEST.json``, committed with Vegeta. A consumer (Montezuma, Piccolo, …) takes the files at a Vegeta commit
and needs nothing of Vegeta: no checkout of its code, no CadQuery, no MuJoCo — only the files and Yax.

    python tools/publish.py                    # every machine registered with its CAD design (its looks)
    python tools/publish.py pekari quad        # some (the others stay as published)
    python tools/publish.py --all              # every registered machine
    python tools/publish.py --check            # the files match the manifest (sha256, valid YX2); exports nothing

Each file is ``tools/export.py``'s full export (``--full``: every property, the CAD visual included). The manifest says,
per machine, the file's sha256 and size, its lab, links, mass, visual triangles and, per consumer requirements file in
``tools/consumers/``, whether the file serves that consumer — and which Vegeta commit (and Yax revision) the files were
exported from. ``--strict`` exits 1 when a published file does not serve a consumer that names it in ``machines``."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exports"
CONSUMERS = ROOT / "tools" / "consumers"
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _git(*args) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _yax_revision() -> str | None:
    f = ROOT / "vendor" / "YAX_REVISION"
    if not f.exists():
        return None
    for line in f.read_text().splitlines():
        if line.startswith("revision:"):
            return line.split(":", 1)[1].strip()
    return None


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_path(out: Path = OUT) -> Path:
    return out / "MANIFEST.json"


def load_manifest(out: Path = OUT) -> dict:
    p = manifest_path(out)
    return json.loads(p.read_text()) if p.exists() else {"format": "vegeta-exports/1", "machines": {}}


def default_machines(registry, all_machines: bool = False) -> list:
    return sorted(n for n in registry.names() if all_machines or registry.get(n).visual is not None)


def describe(m, path: Path, consumers) -> dict:
    """One machine's manifest entry: the file and what it holds, and the consumers' verdicts on it."""
    from yax import tags as tg
    tris = sum(len(s.faces) for lk in m.links for s in lk.shapes if s.kind == "mesh" and s.role == "visual")
    entry = {"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size,
             "lab": (m.producer or {}).get("lab"), "links": len(m.links), "joints": len(m.joints),
             "mass_kg": round(sum(lk.mass or 0.0 for lk in m.links), 4), "visual_triangles": int(tris), "consumers": {}}
    for c in consumers:
        req = tg.Requirements.load(c)
        entry["consumers"][req.consumer] = bool(tg.check(m, req, "yx2").ok)
    return entry


def publish(names, out: Path = OUT, consumers=None, strict: bool = False) -> dict:
    """Export ``names`` into ``out`` and update its manifest (other machines' entries stay); the manifest."""
    import export
    out.mkdir(parents=True, exist_ok=True)
    consumers = sorted(CONSUMERS.glob("*.json")) if consumers is None else list(consumers)
    man = load_manifest(out)
    failed = []
    for name in names:
        m, _ = export.machine(name)
        path = out / f"{name}.yx2"
        export.write(m, path)
        man["machines"][name] = describe(m, path, consumers)
        for c in consumers:
            wants = json.loads(Path(c).read_text()).get("machines")
            consumer = json.loads(Path(c).read_text()).get("consumer", Path(c).stem)
            if wants and name in wants and not man["machines"][name]["consumers"].get(consumer):
                failed.append(f"{name}: does not serve {consumer}")
        print(f"{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}: {man['machines'][name]['bytes']} bytes, "
              f"{man['machines'][name]['visual_triangles']} visual triangles, "
              + ", ".join(f"{k} {'served' if v else 'NOT served'}" for k, v in man['machines'][name]['consumers'].items()))
    dirty = bool(_git("status", "--porcelain", "--", "tools", "libs", "labs"))
    man.update({"format": "vegeta-exports/1",
                "vegeta": {"repo": "technolog-spectrumai/vegeta", "commit": _git("rev-parse", "HEAD"),
                           "branch": _git("rev-parse", "--abbrev-ref", "HEAD"), "dirty": dirty},
                "yax": _yax_revision(), "published": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    man["machines"] = dict(sorted(man["machines"].items()))
    manifest_path(out).write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    if dirty:
        print("note: tools/, libs/ or labs/ have uncommitted changes: commit them, then publish again, so the "
              "manifest's commit is the files' source")
    if failed:
        print("\n".join(failed))
        if strict:
            raise SystemExit(1)
    return man


def check(out: Path = OUT) -> list:
    """What is wrong with the published files: a file missing or changed since the manifest, an invalid YX2, a file
    the manifest does not list. [] when they agree."""
    from yax.formats import yx2
    man = load_manifest(out)
    problems = []
    for name, e in man["machines"].items():
        p = out / e["file"]
        if not p.exists():
            problems.append(f"{name}: {e['file']} is missing")
            continue
        if sha256(p) != e["sha256"]:
            problems.append(f"{name}: {e['file']} differs from the manifest (sha256)")
        problems += [f"{name}: {x}" for x in yx2.validate(p)]
    listed = {e["file"] for e in man["machines"].values()}
    problems += [f"{p.name}: not in the manifest" for p in sorted(out.glob("*.yx2")) if p.name not in listed]
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Publish Vegeta's machines as Yax exchange files (exports/).")
    ap.add_argument("machines", nargs="*", help="machines to publish (default: every machine with a CAD design)")
    ap.add_argument("--all", action="store_true", help="every registered machine")
    ap.add_argument("--check", action="store_true", help="only check exports/ against its manifest")
    ap.add_argument("--strict", action="store_true", help="exit 1 when a file does not serve a consumer that names it")
    ap.add_argument("--out", default=str(OUT), help="the folder (default exports/)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.check:
        problems = check(out)
        print("\n".join(problems) if problems else f"{out}: every file matches its manifest")
        return 1 if problems else 0
    import export
    names = a.machines or default_machines(export.registry(), a.all)
    publish(names, out, strict=a.strict)
    problems = check(out)
    if problems:
        print("\n".join(problems))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
