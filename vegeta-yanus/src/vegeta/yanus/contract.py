"""contract — the books of workflows and their chapters, as ``wf``, the GUI and the scripts read them. Plain
declarations, no I/O.

A **book** is a design we simulate (``kestrel``, code ``K``); a **kind** is one workflow script (``cfd`` →
``workflows/kestrel/airframe_cfd.py``); a **chapter** is one numbered study of the book (``K1`` =
``kestrel/airframe-sweep``) run by its kind's script. A chapter names what it **needs** on the machine (``Tool``,
``Package``, ``Env``) and its prerequisite chapters (``Run``). The ``Flow`` holds them all; it is written in Python in the
project (``workflows/flow.py``), never in a configuration file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


class FlowError(ValueError):
    """The registry itself is wrong: an unknown chapter, a cycle, a duplicate name."""


# ================================================================================================= needs
@dataclass(frozen=True)
class Need:
    name: str
    why: str = ""
    fix: str = ""

    @property
    def label(self) -> str:
        return self.name


@dataclass(frozen=True)
class Tool(Need):
    """A program on the PATH (``ccx``) — or, with ``probe``, anything a callable can find: ``probe() -> version
    string`` (raising or returning ``None`` = not found). ``version_cmd`` is run (2 s timeout) for the version."""
    version_cmd: tuple = ()
    probe: Callable[[], str | None] | None = field(default=None, compare=False)

    @classmethod
    def openfoam(cls, probe: Callable[[], str | None] | None = None) -> "Tool":
        return cls("OpenFOAM", why="the CFD solver", fix="install OpenFOAM (docs/installation.md), then source activate.sh",
                   version_cmd=("foamVersion",), probe=probe)


@dataclass(frozen=True)
class Package(Need):
    """A Python module importable in this interpreter (found, not imported: ``vegeta.aeromant``, ``gmsh``)."""


@dataclass(frozen=True)
class AnyOf(Need):
    """One of several needs is enough: ``AnyOf("section backend", Package("neuralfoil"), Tool("xfoil"))``."""
    options: tuple = ()

    def __init__(self, name: str, *options: Need, why: str = "", fix: str = ""):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "why", why)
        object.__setattr__(self, "fix", fix)
        object.__setattr__(self, "options", tuple(options))

    @property
    def label(self) -> str:
        return f"{self.name} ({' or '.join(o.label for o in self.options)})"


@dataclass(frozen=True)
class Env(Need):
    """An environment variable that must be set (its value is not shown)."""


@dataclass(frozen=True)
class Run:
    """A prerequisite of a chapter: a completed, compatible run of another chapter of the same book
    (``Run("airframe-sweep")``)."""
    chapter: str


# ================================================================================================= books
@dataclass
class Kind:
    """A kind of chapter = one workflow script (``cfd`` → ``workflows/kestrel/airframe_cfd.py``)."""
    name: str
    title: str
    script: str
    workflow: str
    needs: list = field(default_factory=list)
    time: str = ""
    options: list = field(default_factory=list)
    needs_text: str = ""

    def as_dict(self) -> dict:
        return {"title": self.title, "script": self.script, "workflow": self.workflow, "needs": self.needs_text or ", ".join(n.label for n in self.needs),
                "time": self.time, "options": list(self.options)}

    def takes_option(self, flag: str) -> bool:
        return any(o[0] == flag for o in self.options)


@dataclass
class Chapter:
    name: str
    kind: Kind
    campaign: str
    book: "Book" = field(repr=False)
    n: int
    needs: list = field(default_factory=list)
    takes: list = field(default_factory=list)
    title: str = ""

    @property
    def code(self) -> str:
        return f"{self.book.code}{self.n}"

    @property
    def ref(self) -> str:
        return f"{self.book.name}/{self.name}"

    @property
    def prerequisites(self) -> tuple:
        """The chapters of the same book this one needs a completed run of, by name."""
        return tuple(t.chapter.split("/", 1)[-1] for t in self.takes)

    def all_needs(self) -> list:
        return list(self.kind.needs) + list(self.needs)


@dataclass
class Import:
    """A one-off script that brings outside results into a book's run folders (``import_legacy``). ``marker`` = the
    text in a run directory name that says the run came from it."""
    name: str
    title: str
    script: str
    marker: str = ""

    def as_dict(self) -> dict:
        return {"title": self.title, "script": self.script}


@dataclass
class Book:
    """A design we simulate: ``kestrel`` (code ``K``). ``compatible(expected, found, chapter) -> {"level":
    "compatible"|"warnings"|"incompatible", "reasons": [...]}`` compares a fingerprint the caller built with a run
    manifest's ``design``; it comes from the project (Yanus knows no physics)."""
    name: str
    code: str
    title: str = ""
    design: str = ""
    variants: tuple = ()
    compatible: Callable | None = field(default=None, repr=False)
    describe: Callable | None = field(default=None, repr=False)
    kinds: dict = field(default_factory=dict)
    chapters: dict = field(default_factory=dict)

    def kind(self, name: str, **kw) -> Kind:
        k = Kind(name, **kw)
        self.kinds[name] = k
        return k

    def chapter(self, name: str, *, kind: str, campaign: str, takes=(), needs=(), title: str = "") -> Chapter:
        if name in self.chapters:
            raise FlowError(f"book {self.name!r}: chapter {name!r} declared twice")
        if kind not in self.kinds:
            raise FlowError(f"book {self.name!r}: chapter {name!r} of unknown kind {kind!r} (kinds: {', '.join(self.kinds)})")
        ch = Chapter(name, self.kinds[kind], campaign, self, len(self.chapters) + 1, needs=list(needs), takes=list(takes), title=title)
        self.chapters[name] = ch
        return ch

    def by_campaign(self, kind: str, campaign: str) -> Chapter | None:
        return next((ch for ch in self.chapters.values() if ch.kind.name == kind and ch.campaign == campaign), None)


# ================================================================================================= the flow
class Flow:
    """Every book once. ``root`` is the repository root every path is relative to."""

    def __init__(self, name: str = "flow", root=None):
        self.name = name
        self.root = Path(root) if root else None
        self.books: dict[str, Book] = {}
        self.imports: dict[str, Import] = {}

    def book(self, name: str, code: str, **kw) -> Book:
        if name in self.books:
            raise FlowError(f"book {name!r} declared twice")
        b = Book(name, code, **kw)
        self.books[name] = b
        return b

    def add_import(self, name: str, **kw) -> Import:
        imp = Import(name, **kw)
        self.imports[name] = imp
        return imp

    def chapters(self) -> list[Chapter]:
        return [ch for b in self.books.values() for ch in b.chapters.values()]

    def chapter(self, ref) -> Chapter:
        """``"kestrel/airframe-sweep"``, ``"K1"`` / ``"k1"`` or ``"kestrel 1"`` (a ``Chapter`` passes through)."""
        if isinstance(ref, Chapter):
            return ref
        s = str(ref).strip()
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

    def prerequisite(self, take: Run, owner: Chapter) -> Chapter:
        return self.chapter(take.chapter if "/" in take.chapter else f"{owner.book.name}/{take.chapter}")

    def validate(self) -> "Flow":
        errors = []
        for ch in self.chapters():
            for t in ch.takes:
                if not isinstance(t, Run):
                    errors.append(f"{ch.ref}: takes {t!r}: a chapter takes Run(<chapter>) prerequisites only")
                    continue
                try:
                    pre = self.prerequisite(t, ch)
                except KeyError as exc:
                    errors.append(f"{ch.ref}: takes {t.chapter}: {exc}")
                    continue
                if pre.book is not ch.book:
                    errors.append(f"{ch.ref}: a chapter's prerequisites are chapters of the same book only ({t.chapter})")
        errors += self._cycles()
        if errors:
            raise FlowError("the flow registry has errors:\n  " + "\n  ".join(errors))
        return self

    def _cycles(self) -> list[str]:
        edges = {}
        for ch in self.chapters():
            deps = set()
            for t in ch.takes:
                try:
                    deps.add(self.prerequisite(t, ch).ref)
                except (KeyError, AttributeError):
                    pass
            edges[ch.ref] = deps
        out, state = [], {}

        def visit(n, path):
            if state.get(n) == 1:
                out.append("cycle: " + " → ".join(path[path.index(n):] + [n]))
                return
            if state.get(n) == 2:
                return
            state[n] = 1
            for d in sorted(edges.get(n, ())):
                visit(d, path + [n])
            state[n] = 2

        for n in sorted(edges):
            visit(n, [])
        return out
