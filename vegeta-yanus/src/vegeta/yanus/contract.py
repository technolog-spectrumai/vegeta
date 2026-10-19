"""contract — what a unit (a notebook or a workflow chapter) needs, takes and gives. Plain declarations, no I/O.

A **unit** is a notebook (``43_kestrel_study``) or a chapter of a book (``K3`` = ``kestrel/airframe-cruise``). Its
contract names what it **needs** on the machine (``Tool``, ``Package``, ``Env``), what it **takes** from other units
(``Run`` of a chapter, ``File`` another unit gives) and what it **gives** (``File``). A ``Flow`` holds every unit
once; it is written in Python in the project (``side_labs/flow.py``), never in a configuration file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


class FlowError(ValueError):
    """The registry itself is wrong: an unknown unit, a cycle, a file nobody gives, a duplicate name."""


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


# ================================================================================================= takes and gives
@dataclass(frozen=True)
class Run:
    """A completed run of a chapter: ``Run("kestrel/airframe-sweep", variant="4Q-2", required=True)``. Inside a book a
    chapter may write the bare chapter name. ``states`` are the run states that count as usable."""
    chapter: str
    variant: str | None = None
    required: bool = False
    states: tuple = ("completed",)

    def key(self, book: str | None = None) -> str:
        ch = self.chapter if "/" in self.chapter or book is None else f"{book}/{self.chapter}"
        return ch + (f"@{self.variant}" if self.variant else "")


@dataclass(frozen=True)
class File:
    """A file one unit gives and another takes. On the giving side ``File(path, schema=2, keys=(…))``; on the taking
    side ``File("38_prop_lab:designs/data/prop_lab.json", min_schema=2)`` — the producer, a colon, the path
    (relative to the repository root). ``keys`` = the top-level JSON keys that must be there. On the giving side
    ``optional=True`` = the unit may not write it on every run (a mode that skips it: no OpenFOAM, a quick pass)."""
    ref: str
    schema: int | None = None
    min_schema: int | None = None
    keys: tuple = ()
    required: bool = False
    optional: bool = False

    @property
    def producer(self) -> str | None:
        return self.ref.split(":", 1)[0] if ":" in self.ref else None

    @property
    def path(self) -> str:
        return self.ref.split(":", 1)[1] if ":" in self.ref else self.ref

    def key(self, book: str | None = None) -> str:
        return self.ref


@dataclass(frozen=True)
class Data:
    """A data file a unit reads that no unit gives: committed source data (``required``, the default: the gate closes
    when it is missing or lacks its ``keys``), or a **cache** a design module rebuilds when it is missing
    (``cache=True``: never closes the gate; it says the first use will compute it, and ``note`` says how long)."""
    path: str
    required: bool = True
    cache: bool = False
    keys: tuple = ()
    note: str = ""

    def key(self, book: str | None = None) -> str:
        return "data:" + self.path


# ================================================================================================= units
@dataclass
class Kind:
    """A kind of chapter = one workflow script (``cfd`` → ``side_labs/kestrel/airframe_cfd.py``)."""
    name: str
    title: str
    script: str
    workflow: str
    needs: list = field(default_factory=list)
    time: str = ""
    options: list = field(default_factory=list)
    needs_text: str = ""
    screening: tuple = ()           # the script arguments that make a run coarse and short (auto mode): ("--quality", "screening")

    def as_dict(self) -> dict:
        return {"title": self.title, "script": self.script, "workflow": self.workflow, "needs": self.needs_text or ", ".join(n.label for n in self.needs),
                "time": self.time, "options": list(self.options)}


@dataclass
class Unit:
    name: str
    needs: list = field(default_factory=list)
    takes: list = field(default_factory=list)
    gives: list = field(default_factory=list)
    title: str = ""

    kind_of_unit = "unit"

    @property
    def ref(self) -> str:
        return self.name

    def all_needs(self) -> list:
        return list(self.needs)


@dataclass
class Notebook(Unit):
    """A notebook under ``notebooks/`` (or ``notebooks/<folder>/``): ``screening_env`` names the variable that turns
    every required take optional. ``auto`` (the ``yanus_auto`` flag) = the notebook runs by itself what it is missing,
    in screening quality, and its gate never closes (``auto.py``)."""
    screening_env: str | None = None
    folder: str = ""
    auto: bool = False
    kind_of_unit = "notebook"

    @property
    def path(self) -> str:
        return f"notebooks/{self.folder + '/' if self.folder else ''}{self.name}.ipynb"


@dataclass
class Chapter(Unit):
    kind: Kind | None = None
    campaign: str = ""
    book: "Book | None" = field(default=None, repr=False)
    n: int = 0
    kind_of_unit = "chapter"

    @property
    def code(self) -> str:
        return f"{self.book.code}{self.n}"

    @property
    def ref(self) -> str:
        return f"{self.book.name}/{self.name}"

    @property
    def prerequisites(self) -> tuple:
        """The chapters of the same book whose runs this one takes (the ``Run`` takes), by name."""
        return tuple(t.chapter.split("/", 1)[-1] for t in self.takes if isinstance(t, Run))

    def all_needs(self) -> list:
        return list(self.kind.needs if self.kind else []) + list(self.needs)


@dataclass
class Import:
    """A one-off script that brings outside results into a book's run folders (``import_legacy``). ``marker`` = the
    text in a run directory name that says the run came from it (a pin to such a missing run → this command)."""
    name: str
    title: str
    script: str
    marker: str = ""

    def as_dict(self) -> dict:
        return {"title": self.title, "script": self.script}


@dataclass
class Book:
    """A design we simulate: ``kestrel`` (code ``K``). ``fingerprint(variant) -> dict`` is the design as it is now;
    ``compatible(expected, found, chapter) -> {"level": "compatible"|"warning"|"incompatible", "reasons": [...]}``
    compares it with a run manifest's ``design``. Both come from the project (Yanus knows no physics)."""
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
        k = Kind(name, **kw)
        self.kinds[name] = k
        return k

    def chapter(self, name: str, *, kind: str, campaign: str, takes=(), needs=(), gives=(), title: str = "") -> Chapter:
        if name in self.chapters:
            raise FlowError(f"book {self.name!r}: chapter {name!r} declared twice")
        if kind not in self.kinds:
            raise FlowError(f"book {self.name!r}: chapter {name!r} of unknown kind {kind!r} (kinds: {', '.join(self.kinds)})")
        ch = Chapter(name, needs=list(needs), takes=list(takes), gives=list(gives), title=title, kind=self.kinds[kind], campaign=campaign,
                     book=self, n=len(self.chapters) + 1)
        self.chapters[name] = ch
        return ch

    def by_campaign(self, kind: str, campaign: str) -> Chapter | None:
        for ch in self.chapters.values():
            if ch.kind.name == kind and ch.campaign == campaign:
                return ch
        return None


# ================================================================================================= the flow
class Flow:
    """Every unit once. ``root`` is the repository root every path is relative to."""

    def __init__(self, name: str = "flow", root=None):
        from pathlib import Path
        self.name = name
        self.root = Path(root) if root else None
        self.books: dict[str, Book] = {}
        self.notebooks: dict[str, Notebook] = {}
        self.imports: dict[str, Import] = {}

    # ---- declaring
    def book(self, name: str, code: str, **kw) -> Book:
        if name in self.books:
            raise FlowError(f"book {name!r} declared twice")
        b = Book(name, code, **kw)
        self.books[name] = b
        return b

    def notebook(self, name: str, *, needs=(), takes=(), gives=(), title: str = "", screening_env: str | None = None, folder: str = "",
                 auto: bool = False) -> Notebook:
        if name in self.notebooks:
            raise FlowError(f"notebook {name!r} declared twice")
        nb = Notebook(name, needs=list(needs), takes=list(takes), gives=list(gives), title=title, screening_env=screening_env, folder=folder, auto=auto)
        self.notebooks[name] = nb
        return nb

    def add_import(self, name: str, **kw) -> Import:
        imp = Import(name, **kw)
        self.imports[name] = imp
        return imp

    # ---- looking up
    def units(self) -> list[Unit]:
        return [ch for b in self.books.values() for ch in b.chapters.values()] + list(self.notebooks.values())

    def chapter(self, ref: str) -> Chapter:
        """``"kestrel/airframe-sweep"``, ``"K1"`` / ``"k1"`` or ``"kestrel 1"``."""
        import re
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

    def unit(self, ref: str) -> Unit:
        """A notebook by name (``43_kestrel_study``, or its number ``43`` when unique) or a chapter (see ``chapter``)."""
        s = ref.strip()
        if s.endswith(".ipynb"):
            s = s.rsplit("/", 1)[-1][:-6]
        if s in self.notebooks:
            return self.notebooks[s]
        hits = [n for n in self.notebooks if n.split("_", 1)[0] == s]
        if len(hits) == 1:
            return self.notebooks[hits[0]]
        try:
            return self.chapter(s)
        except KeyError as exc:
            raise KeyError(f"unknown unit {ref!r}: a notebook ({', '.join(self.notebooks) or 'none declared'}) or a chapter code ({exc})") from None

    def run_take_chapter(self, take: Run, owner: Unit) -> Chapter:
        if "/" in take.chapter:
            return self.chapter(take.chapter)
        if isinstance(owner, Chapter):
            return self.chapter(f"{owner.book.name}/{take.chapter}")
        raise FlowError(f"{owner.name}: Run({take.chapter!r}) needs book/chapter outside a book")

    def giver(self, path: str) -> Unit | None:
        for u in self.units():
            if any(isinstance(g, File) and g.path == path for g in u.gives):
                return u
        return None

    # ---- checking the registry
    def validate(self) -> "Flow":
        errors = []
        for u in self.units():
            for t in u.takes:
                if isinstance(t, Run):
                    try:
                        ch = self.run_take_chapter(t, u)
                    except (KeyError, FlowError) as exc:
                        errors.append(f"{u.ref}: takes {t.key()}: {exc}")
                        continue
                    if isinstance(u, Chapter) and ch.book is not u.book:
                        errors.append(f"{u.ref}: a chapter's prerequisites are chapters of the same book only ({t.chapter})")
                    if t.variant and ch.book.variants and t.variant not in ch.book.variants:
                        errors.append(f"{u.ref}: takes {t.key()}: book {ch.book.name!r} has no variant {t.variant!r}")
                elif isinstance(t, File):
                    if not t.producer:
                        errors.append(f"{u.ref}: takes File({t.ref!r}) without its producer ('<unit>:<path>')")
                        continue
                    try:
                        prod = self.unit(t.producer)
                    except KeyError as exc:
                        errors.append(f"{u.ref}: takes {t.ref}: {exc}")
                        continue
                    if not any(isinstance(g, File) and g.path == t.path for g in prod.gives):
                        errors.append(f"{u.ref}: takes {t.path} from {prod.ref}, which does not give it")
                elif isinstance(t, Data):
                    g = self.giver(t.path)
                    if g is not None:
                        errors.append(f"{u.ref}: takes Data({t.path!r}), which {g.ref} gives: take File('{g.name}:{t.path}') instead")
                else:
                    errors.append(f"{u.ref}: takes {t!r}: not a Run, a File or a Data")
            for g in u.gives:
                if not isinstance(g, File) or g.producer:
                    errors.append(f"{u.ref}: gives {g!r}: a File with a bare path")
        errors += self._cycles()
        if errors:
            raise FlowError("the flow registry has errors:\n  " + "\n  ".join(errors))
        return self

    def _cycles(self) -> list[str]:
        edges: dict[str, set] = {}
        for u in self.units():
            deps = set()
            for t in u.takes:
                try:
                    if isinstance(t, Run):
                        deps.add(self.run_take_chapter(t, u).ref)
                    elif isinstance(t, File) and t.producer:
                        deps.add(self.unit(t.producer).ref)
                except (KeyError, FlowError):
                    pass
            edges[u.ref] = deps
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
