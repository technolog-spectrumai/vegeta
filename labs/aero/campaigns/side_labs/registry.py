"""registry — the side labs' books, kinds and chapters (``side_labs/flow.py`` declares them, ``library.py`` reads them).

Host-local and small: a book is a design we simulate (``kestrel``, code ``K``), a kind is one side-lab script, a
chapter is a numbered study of one kind (``K1``) that may build on other chapters of the same book (``Run`` takes).
What a chapter needs on the machine uses Ozaru's Yanus need types (``Tool``, ``Package``, ``Env``, ``AnyOf``)."""
from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

try:
    from yanus import AnyOf, Env, Need, Package  # noqa: F401
    from yanus import Tool as _Tool
except ImportError:                                            # a source tree without the editable install
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "vendor" / "ozaru" / "yanus" / "src"))
    from yanus import AnyOf, Env, Need, Package  # noqa: F401
    from yanus import Tool as _Tool


@dataclass(frozen=True)
class Tool(_Tool):
    @classmethod
    def openfoam(cls, probe: Callable[[], str | None] | None = None) -> "Tool":
        return cls("OpenFOAM", why="the CFD solver", fix="install OpenFOAM (docs/installation.md), then source activate.sh",
                   version_cmd=("foamVersion",), probe=probe)


class FlowError(ValueError):
    pass


@dataclass(frozen=True)
class Run:
    """A prerequisite: a completed run of another chapter of the same book (``Run("frame")``)."""
    chapter: str
    variant: str | None = None
    required: bool = False
    states: tuple = ("completed",)


@dataclass
class Kind:
    """A kind of chapter = one side-lab script (``cfd`` → ``side_labs/kestrel/airframe_cfd.py``)."""
    name: str
    title: str
    script: str
    workflow: str
    needs: list = field(default_factory=list)
    time: str = ""
    options: list = field(default_factory=list)
    needs_text: str = ""
    screening: tuple = ()

    def as_dict(self) -> dict:
        return {"title": self.title, "script": self.script, "workflow": self.workflow, "needs": self.needs_text or ", ".join(n.label for n in self.needs),
                "time": self.time, "options": list(self.options)}


@dataclass
class Chapter:
    name: str
    needs: list = field(default_factory=list)
    takes: list = field(default_factory=list)
    title: str = ""
    kind: Kind | None = None
    campaign: str = ""
    book: "Book | None" = field(default=None, repr=False)
    n: int = 0

    @property
    def code(self) -> str:
        return f"{self.book.code}{self.n}"

    @property
    def ref(self) -> str:
        return f"{self.book.name}/{self.name}"

    @property
    def prerequisites(self) -> tuple:
        return tuple(t.chapter.split("/", 1)[-1] for t in self.takes if isinstance(t, Run))

    def all_needs(self) -> list:
        return list(self.kind.needs if self.kind else []) + list(self.needs)


@dataclass
class Import:
    """A one-off script that brings outside results into a book's run folders (``import_legacy``)."""
    name: str
    title: str
    script: str
    marker: str = ""

    def as_dict(self) -> dict:
        return {"title": self.title, "script": self.script}


@dataclass
class Book:
    """A design we simulate. ``fingerprint(variant)`` is the design as it is now; ``compatible(expected, found,
    chapter)`` compares it with a run manifest's ``design``; both come from the design modules."""
    name: str
    code: str
    title: str = ""
    design: str = ""
    variants: tuple = ()
    fingerprint: Callable | None = field(default=None, repr=False)
    compatible: Callable | None = field(default=None, repr=False)
    describe: Callable | None = field(default=None, repr=False)
    kinds: dict = field(default_factory=dict)
    chapters: dict = field(default_factory=dict)

    def kind(self, name: str, **kw) -> Kind:
        self.kinds[name] = Kind(name, **kw)
        return self.kinds[name]

    def chapter(self, name: str, *, kind: str, campaign: str, takes=(), needs=(), title: str = "") -> Chapter:
        if name in self.chapters:
            raise FlowError(f"book {self.name!r}: chapter {name!r} declared twice")
        if kind not in self.kinds:
            raise FlowError(f"book {self.name!r}: chapter {name!r} of unknown kind {kind!r} (kinds: {', '.join(self.kinds)})")
        for t in takes:
            if not isinstance(t, Run):
                raise FlowError(f"book {self.name!r}: chapter {name!r} takes {t!r}: only Run (a prerequisite chapter)")
        ch = Chapter(name, needs=list(needs), takes=list(takes), title=title, kind=self.kinds[kind], campaign=campaign, book=self, n=len(self.chapters) + 1)
        self.chapters[name] = ch
        return ch

    def by_campaign(self, kind: str, campaign: str) -> Chapter | None:
        return next((ch for ch in self.chapters.values() if ch.kind.name == kind and ch.campaign == campaign), None)


class Flow:
    """The books and the import scripts."""

    def __init__(self, name: str = "flow", root=None):
        self.name = name
        self.root = Path(root) if root else None
        self.books: dict[str, Book] = {}
        self.imports: dict[str, Import] = {}

    def book(self, name: str, code: str, **kw) -> Book:
        if name in self.books:
            raise FlowError(f"book {name!r} declared twice")
        self.books[name] = Book(name, code, **kw)
        return self.books[name]

    def add_import(self, name: str, **kw) -> Import:
        self.imports[name] = Import(name, **kw)
        return self.imports[name]

    def chapter(self, ref: str) -> Chapter:
        """``"kestrel/airframe-sweep"``, ``"K1"`` / ``"k1"``."""
        s = ref.strip()
        m = re.fullmatch(r"([A-Za-z]+)\s*(\d+)", s)
        if m:
            letter, n = m.group(1).upper(), int(m.group(2))
            for b in self.books.values():
                if b.code.upper() == letter or b.name.upper() == letter:
                    for ch in b.chapters.values():
                        if ch.n == n:
                            return ch
                    raise KeyError(f"book {b.name!r} has chapters {b.code}1–{b.code}{len(b.chapters)}, not {s}")
            raise KeyError(f"no book with code {letter!r} (books: {', '.join(f'{k} = {v.code}' for k, v in self.books.items())})")
        if "/" in s:
            book, name = s.split("/", 1)
            if book in self.books and name in self.books[book].chapters:
                return self.books[book].chapters[name]
        raise KeyError(f"unknown chapter {ref!r}: use a code such as K1 or book/chapter")

    def validate(self) -> "Flow":
        errors = [f"{ch.ref}: takes {pre!r}, not a chapter of book {b.name!r}"
                  for b in self.books.values() for ch in b.chapters.values() for pre in ch.prerequisites if pre not in b.chapters]
        if errors:
            raise FlowError("the side-lab registry has errors:\n  " + "\n  ".join(errors))
        return self


_LOADED: dict = {}


def load_flow(path=None) -> Flow:
    """The ``flow`` object of ``side_labs/flow.py`` (cached per path)."""
    p = Path(path or Path(__file__).resolve().parent / "flow.py").resolve()
    if p not in _LOADED:
        spec = importlib.util.spec_from_file_location("side_labs_flow", p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        flow = getattr(mod, "flow", None)
        if not isinstance(flow, Flow):
            raise FlowError(f"{p} defines no Flow named 'flow'")
        _LOADED[p] = flow.validate()
    return _LOADED[p]
