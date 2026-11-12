"""A notebook's (or a script's) first lines in a lab::

    DEBUG = True                                   # the expensive solvers mocked (vegeta.mock); False for real results
    from vegeta.labkit import bootstrap
    DEBUG = bootstrap.lab("aero", DEBUG)

puts the lab's ``designs/``, ``common/``, ``exp/`` and ``sims/`` on ``sys.path`` (the modules are imported by their plain names,
``import nisus_flight``) and installs the mocks when the debug switch is on (``VEGETA_DEBUG`` overrides it)."""
from __future__ import annotations

import sys
from pathlib import Path

FOLDERS = ("designs", "common", "exp", "sims")       # sims/ last: its runnable scripts never shadow a design


def root() -> Path:
    """The repository: the folder that holds ``labs/`` (from this file, else from the working directory up)."""
    here = Path(__file__).resolve()
    for p in list(here.parents) + [Path.cwd(), *Path.cwd().parents]:
        if (p / "labs").is_dir() and (p / "libs").is_dir():
            return p
    raise FileNotFoundError("no Vegeta checkout (a folder with labs/ and libs/) above " + str(here))


def folders(name: str) -> list[Path]:
    lab = root() / "labs" / name
    if not lab.is_dir():
        raise FileNotFoundError(f"no lab {name!r}: {sorted(p.name for p in (root() / 'labs').iterdir() if p.is_dir())}")
    return [lab / f for f in FOLDERS if (lab / f).is_dir()]


def lab(name: str, debug: bool = True) -> bool:
    """The lab's folders on ``sys.path``, the mocks per the debug switch; returns the switch (``vegeta.mock.setup``)."""
    for p in reversed(folders(name)):                 # designs/ first
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from vegeta import mock
    return mock.setup(debug)
