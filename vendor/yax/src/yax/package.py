"""A Yax package on disk: write it atomically, load it, check it."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from . import stl, units
from .manifest import BODY, NAME, VERSION, identity, validate


class Invalid(Exception):
    """The package is not a valid Yax package; ``problems`` lists why."""

    def __init__(self, problems):
        super().__init__("; ".join(problems))
        self.problems = list(problems)


@dataclass(frozen=True)
class Design:
    dir: Path
    manifest: dict

    @property
    def name(self) -> str:
        return self.manifest["name"]

    @property
    def length_unit(self) -> str:
        return self.manifest["units"]["length"]

    @property
    def frame(self) -> dict:
        return self.manifest["frame"]

    @property
    def geometry(self) -> dict:
        return self.manifest["geometry"]

    @property
    def mass(self) -> dict | None:
        return self.manifest.get("mass")

    @property
    def parts(self) -> list:
        return self.manifest.get("parts", [])

    def path(self, name: str = BODY) -> Path:
        return self.dir / name

    def mesh(self, name: str = BODY) -> list:
        """Triangles of a listed STL, in the package's length unit."""
        return stl.read(self.path(name))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def describe(triangles, length_unit: str = "mm", mass_kg: float | None = None,
             density_kg_m3: float | None = None) -> dict:
    """``geometry`` and (with a mass or a density) ``mass`` sections computed from a closed mesh — for producers
    without an exact CAD kernel, and for checking those with one."""
    props = stl.mass_properties(triangles)
    out = {"geometry": {"bbox": stl.bbox(triangles), "volume": props["volume"], "com": props["com"]}}
    if mass_kg is None and density_kg_m3 is not None:
        mass_kg = density_kg_m3 * props["volume"] * units.LENGTH[length_unit] ** 3
    if mass_kg is not None:
        scale = mass_kg / props["volume"] * units.LENGTH[length_unit] ** 2      # per unit volume, length^2 -> m^2
        out["mass"] = {"kg": mass_kg, "com": props["com"],
                       "inertia": [[v * scale for v in row] for row in props["inertia"]],
                       "source": "given" if density_kg_m3 is None else "density"}
    return out


def write(directory, manifest: dict, files: dict) -> Design:
    """Publish a package: ``files`` maps package paths (``body.stl``, ``parts/arm.stl``) to source paths or bytes.

    Hashes and the identity are filled in, the manifest is validated, and the directory appears complete or not at
    all (staged next to it, then renamed; an existing package is replaced)."""
    directory = Path(directory)
    m = {"yax": VERSION, **manifest}
    stage = directory.parent / f".{directory.name}.{uuid.uuid4().hex}.tmp"
    stage.mkdir(parents=True)
    try:
        digests = {}
        for name, source in files.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(source, (bytes, bytearray)):
                target.write_bytes(source)
            else:
                shutil.copyfile(source, target)
            digests[name] = _sha256(target)
        m["files"] = digests
        if "producer" in m:
            m["identity"] = identity(m["producer"])
        problems = validate(m)
        if problems:
            raise Invalid(problems)
        (stage / NAME).write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
        old = None
        if directory.exists():
            old = directory.parent / f".{directory.name}.{uuid.uuid4().hex}.old"
            os.rename(directory, old)
        os.rename(stage, directory)
        if old is not None:
            shutil.rmtree(old)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return Design(directory, m)


def check(directory) -> list:
    """Problems with a package on disk (manifest, missing files, hash mismatches, unreadable STL); ``[]`` if none."""
    directory = Path(directory)
    try:
        m = json.loads((directory / NAME).read_text())
    except (OSError, ValueError) as e:
        return [f"{NAME}: {e}"]
    problems = validate(m)
    for name, digest in m.get("files", {}).items():
        path = directory / name
        if not path.is_file():
            problems.append(f"{name}: missing")
        elif _sha256(path) != digest:
            problems.append(f"{name}: SHA-256 does not match the manifest")
        elif name.lower().endswith(".stl"):
            try:
                stl.read(path)
            except ValueError as e:
                problems.append(str(e))
    return problems


def load(directory, verify: bool = True) -> Design:
    """The package at ``directory``; ``Invalid`` if ``verify`` finds problems."""
    directory = Path(directory)
    if verify:
        problems = check(directory)
        if problems:
            raise Invalid(problems)
    return Design(directory, json.loads((directory / NAME).read_text()))
