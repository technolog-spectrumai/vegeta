"""Parametric designs: Python code + explicit parameters -> Geometry."""
from __future__ import annotations

import hashlib
import inspect
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .geometry import Geometry, utc_now
from .parameters import Parameter, ParameterSet
from .result import Result


class BuildError(RuntimeError):
    """The design's build code raised; the message names the design and parameters."""


class Design:
    """Base class for parametric designs.

    Subclass it, list ``parameters`` and implement ``build(self, p)`` returning a CadQuery
    ``Workplane`` or ``Shape``; ``p`` is a dict of resolved parameter values::

        class Plate(Design):
            parameters = [Parameter("width", 40.0, "mm", min=1)]
            def build(self, p):
                return cq.Workplane().box(p["width"], 20, 3)
    """

    name: str | None = None
    description: str = ""
    units: str = "mm"
    parameters: Sequence[Parameter] = ()

    def __init__(self):
        self.params = ParameterSet(self.parameters)
        if self.name is None:
            self.name = type(self).__name__

    def build(self, p: dict[str, Any]):  # pragma: no cover - abstract
        raise NotImplementedError

    # -- parameters -------------------------------------------------------------------------
    def resolve(self, **overrides) -> dict[str, Any]:
        return self.params.resolve(overrides)

    def _build_callable(self) -> Callable:
        return type(self).build

    # -- identity ---------------------------------------------------------------------------
    def source_identity(self) -> dict[str, Any]:
        """Module, qualified name, source file and a SHA-256 of the build source code."""
        fn = self._build_callable()
        try:
            src = textwrap.dedent(inspect.getsource(fn))
        except (OSError, TypeError):
            src = None
        try:
            src_file = inspect.getsourcefile(fn)
        except TypeError:
            src_file = None
        return {
            "design": self.name,
            "module": getattr(fn, "__module__", None),
            "qualname": getattr(fn, "__qualname__", None),
            "source_file": src_file,
            "source_sha256": hashlib.sha256(src.encode()).hexdigest() if src is not None else None,
        }

    # -- generation -------------------------------------------------------------------------
    def generate(self, **overrides) -> Geometry:
        """Build the geometry. Invalid parameters raise ``ValueError``; build failures ``BuildError``."""
        values = self.resolve(**overrides)
        try:
            obj = self._call_build(values)
        except Exception as exc:
            raise BuildError(f"design {self.name!r} failed to build with {values}: {exc!r}") from exc
        if obj is None:
            raise BuildError(f"design {self.name!r} build() returned None")
        return Geometry.from_cadquery(
            obj, parameters=values, source=self.source_identity(), units=self.units, name=self.name
        )

    def _call_build(self, values):
        return self.build(values)

    def run(
        self,
        outdir: str | Path,
        formats: Iterable[str] = ("step", "stl"),
        *,
        stl_tolerance: float = 0.01,
        stl_angular_tolerance: float = 0.1,
        **overrides,
    ) -> Result:
        """Generate and export in one explicit call; never raises for build failures."""
        t0 = time.monotonic()
        try:
            geometry = self.generate(**overrides)
        except (ValueError, BuildError) as exc:
            res = Result(kind="dedalus.generate", metadata={
                "design": self.name, "overrides": overrides, "source": self.source_identity(),
                "created_at": utc_now(),
            })
            res.fail(str(exc))
            if isinstance(exc, BuildError) and exc.__cause__ is not None:
                res.metadata["traceback"] = "".join(traceback.format_exception(exc.__cause__))
            res.duration_s = time.monotonic() - t0
            return res
        res = geometry.export(
            outdir, formats=formats, stl_tolerance=stl_tolerance, stl_angular_tolerance=stl_angular_tolerance
        )
        res.duration_s = time.monotonic() - t0
        res.save_json(Path(outdir) / "summary.json")
        return res

    def __repr__(self) -> str:
        ps = ", ".join(f"{p.name}={p.default!r}" for p in self.params)
        return f"<Design {self.name}({ps})>"


class FunctionDesign(Design):
    """A design defined by a plain function ``build(p)`` (see :func:`design`)."""

    def __init__(self, fn: Callable, parameters: Sequence[Parameter], name=None, units="mm", description=""):
        self._fn = fn
        self.parameters = list(parameters)
        self.name = name or fn.__name__
        self.units = units
        self.description = description or (inspect.getdoc(fn) or "")
        super().__init__()

    def _build_callable(self):
        return self._fn

    def _call_build(self, values):
        return self._fn(values)


def design(parameters: Sequence[Parameter], *, name: str | None = None, units: str = "mm", description: str = ""):
    """Decorator turning ``def build(p): ...`` into a :class:`Design`::

        @design(parameters=[Parameter("size", 20.0, "mm", min=1)])
        def cube(p):
            return cq.Workplane().box(p["size"], p["size"], p["size"])
    """

    def wrap(fn: Callable) -> FunctionDesign:
        return FunctionDesign(fn, parameters, name=name, units=units, description=description)

    return wrap
