"""``vegeta`` command: one entry point for the engineering tools.

``vegeta dedalus ...`` is the same as ``dedalus ...`` (likewise talos, aeromant, mellonia).
Each tool is imported only when its subcommand is used.
"""
from __future__ import annotations

import importlib
import sys
from importlib import metadata

TOOLS = {
    "dedalus": "parametric CAD (CadQuery): designs -> STEP/STL, measurements",
    "talos": "linear static FEA (Gmsh + CalculiX): STEP -> mesh -> results",
    "aeromant": "aerodynamics (OpenFOAM template cases): STL -> Cd/Cl/Cm",
    "mellonia": "3D-print manufacturability (PrusaSlicer): STL -> G-code, time, material",
}


def _version() -> str:
    try:
        return metadata.version("vegeta-cli")
    except metadata.PackageNotFoundError:  # running from a source tree
        return "unknown"


def usage() -> str:
    lines = ["usage: vegeta <tool> [arguments]", "", "Engineering tools:"]
    lines += [f"  {name:<10} {text}" for name, text in TOOLS.items()]
    lines += ["", "Run 'vegeta <tool> --help' for the tool's commands.",
              "The tools are also available as the commands dedalus, talos, aeromant and mellonia."]
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage())
        return 0
    if argv[0] in ("-V", "--version"):
        print(f"vegeta-cli {_version()}")
        return 0
    tool, rest = argv[0], argv[1:]
    if tool not in TOOLS:
        print(f"vegeta: unknown tool {tool!r}\n\n{usage()}", file=sys.stderr)
        return 2
    cli = importlib.import_module(f"vegeta.{tool}.cli")
    return cli.main(rest, prog=f"vegeta {tool}")
