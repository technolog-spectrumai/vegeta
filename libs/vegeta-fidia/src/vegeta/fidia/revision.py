"""One revision of a prompt-to-3D run: a directory of files, sealed by ``revision.json`` (written once).

::

    rev-003/design.py  generation.json  execution.json  runner.log  result.json  parts.npz  model.step
            checks.json  renders/{front,right,top,iso,rear_iso,sheet}.png  review.json
            export/{model.glb, model.gltf, model.bin, model.obj, model.mtl, model.stl, model.step, manifest.json,
                    reimport.json}
            revision.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EXPORTS = ("glb", "gltf", "obj", "stl", "step")


class RevisionSealed(RuntimeError):
    """A sealed revision's record cannot be rewritten."""


class Revision:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    # -- files -------------------------------------------------------------------------------------------------
    def _json(self, name: str) -> Any:
        f = self.path / name
        return json.loads(f.read_text()) if f.is_file() else None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def number(self) -> int:
        return int(self.path.name.split("-")[-1]) if self.path.name.startswith("rev-") else 0

    @property
    def record(self) -> dict:
        return self._json("revision.json") or {}

    @property
    def sealed(self) -> bool:
        return (self.path / "revision.json").is_file()

    def seal(self, record: dict) -> None:
        """Write ``revision.json`` once; a sealed revision is never rewritten (status lives in ``run.json``)."""
        f = self.path / "revision.json"
        if f.exists():
            raise RevisionSealed(f"{self.name} is sealed; its record is not rewritten")
        tmp = self.path / "revision.json.tmp"
        tmp.write_text(json.dumps(record, indent=2, default=str))
        tmp.replace(f)

    @property
    def source(self) -> str | None:
        f = self.path / "design.py"
        return f.read_text() if f.is_file() else None

    generation = property(lambda self: self._json("generation.json"))
    execution = property(lambda self: self._json("execution.json"))
    checks = property(lambda self: self._json("checks.json"))
    review = property(lambda self: self._json("review.json"))
    manifest = property(lambda self: self._json("export/manifest.json"))
    reimport = property(lambda self: self._json("export/reimport.json"))

    @property
    def status(self) -> str | None:
        return self.record.get("status")

    @property
    def valid(self) -> bool:
        return bool(self.record.get("valid"))

    @property
    def score(self) -> int | None:
        return self.record.get("score")

    @property
    def done(self) -> bool:
        return bool(self.record.get("done"))

    @property
    def summary(self) -> str:
        return self.record.get("summary", "")

    def files(self) -> dict[str, Path]:
        """Exported model files that exist, by format (``glb``, ``gltf``, ``obj``, ``stl``, ``step``)."""
        m = self.manifest or {}
        out = {}
        for fmt in EXPORTS:
            name = (m.get("files") or {}).get(fmt)
            if name and (self.path / "export" / name).is_file():
                out[fmt] = self.path / "export" / name
        return out

    @property
    def sheet_path(self) -> Path | None:
        f = self.path / "renders" / "sheet.png"
        return f if f.is_file() else None

    # -- views -------------------------------------------------------------------------------------------------
    def sheet(self, width: int | None = None):
        """The contact sheet: an ``IPython.display.Image`` in a notebook, else a PIL image (None if not rendered)."""
        f = self.sheet_path
        if f is None:
            return None
        try:
            from IPython import get_ipython
            from IPython.display import Image

            if get_ipython() is not None:
                return Image(filename=str(f), width=width)
        except ImportError:
            pass
        from PIL import Image as PILImage

        return PILImage.open(f)

    def parts(self):
        from .mesh import load_parts

        return load_parts(self.path)

    def preview(self, **kwargs):
        """Interactive 3D view of the parts (see ``vegeta.fidia.preview.show``)."""
        from .preview import show

        return show(self, **kwargs)

    def text(self) -> str:
        """A few lines: status, checks, review."""
        r = self.record
        lines = [f"{self.name}: {r.get('status')}, {'valid' if r.get('valid') else 'invalid'}"
                 + (f", score {r['score']}/10 ({r.get('verdict')})" if r.get("score") is not None else "")
                 + (" — done" if r.get("done") else ""), f"  {r.get('summary', '')}"]
        if r.get("error"):
            lines.append(f"  error: {r['error']}")
        lines += [f"  {p}" for p in (r.get("problems") or [])[:8]]
        rv = self.review
        if rv:
            lines.append(f"  review: {rv.get('summary', '')}")
            lines += [f"  - [{i['severity']}] {i['part']}: {i['problem']}" for i in rv.get("issues", [])[:6]]
        return "\n".join(lines)

    def __repr__(self) -> str:
        r = self.record
        return (f"<Revision {self.name} {r.get('status')} valid={r.get('valid')} score={r.get('score')} "
                f"done={r.get('done')}>")
