"""The one command line every workflow shares: ``python -m assemblies.workflows.<name> [options]``."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

from . import vida


def parser(description: str, *, cfd: bool = True, fea: bool = True) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--fidelity", choices=("smoke", "quick", "full"), default="full",
                    help="mesh and solver settings (smoke: seconds per case, only to see it work; full: the real thing)")
    ap.add_argument("-j", "--processors", type=int, default=1, help="cores per CFD solver (MPI); default 1")
    ap.add_argument("--jobs", type=int, default=1, help="CFD cases side by side; default 1")
    ap.add_argument("--threads", type=int, default=1, help="CalculiX threads per FEA solve; default 1")
    if cfd:
        ap.add_argument("--no-cfd", dest="run_cfd", action="store_false", help="run no CFD (solved cases are still read back)")
    if fea:
        ap.add_argument("--no-fea", dest="run_fea", action="store_false", help="run no FEA (solved models are still read back)")
    ap.add_argument("--redo", action="append", default=[], metavar="PATH",
                    help="compute this sub-assembly again even if the saved .vida has it (repeatable), e.g. --redo compressor")
    ap.add_argument("--force", action="store_true", help="ignore the saved .vida and delete the run directory first")
    ap.add_argument("--out", type=Path, default=None, help="run directory (solver cases, meshes); default runs/assemblies/<name>")
    ap.add_argument("--vida", dest="vida_path", type=Path, default=None, help="the .vida to read and write; default assemblies/data/<name>.vida")
    ap.add_argument("--include", choices=vida.LEVELS, default="results",
                    help="what the .vida carries: results (default), geometry, mesh, cases (to move it to another machine)")
    ap.add_argument("--no-export", dest="export", action="store_false", help="do not write the JSON export in assemblies/data")
    ap.add_argument("--show", action="store_true", help="only print the tree of the saved .vida and exit")
    return ap


def main(run: Callable, ap: argparse.ArgumentParser, default_vida: Path, argv=None) -> int:
    args = ap.parse_args(argv)
    if args.show:
        path = args.vida_path or default_vida
        if not path.is_file():
            print(f"no {path}: run the workflow first", file=sys.stderr)
            return 2
        root = vida.load(path)
        print(f"{path}  (written {root.meta['manifest']['written_at']}, include={root.meta['manifest']['include']})")
        print(root)
        return 0
    kw = {k: v for k, v in vars(args).items() if k != "show"}
    t0 = time.monotonic()
    root = run(**kw)
    print(root)
    print(f"saved {root.meta.get('saved_to')}  ({time.monotonic() - t0:.0f} s)")
    if root.meta.get("exported_to"):
        print(f"exported {root.meta['exported_to']}")
    if root.meta.get("results_to"):
        print(f"results {root.meta['results_to']}  (assemblies.results.load)")
    return 0
