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
    "boreas": "propeller/rotor performance (BEMT) with motor and battery: thrust, power, rpm, excitations",
}


def _version() -> str:
    try:
        return metadata.version("vegeta-cli")
    except metadata.PackageNotFoundError:  # running from a source tree
        return "unknown"


def plugins() -> dict:
    """Command groups contributed by other installed distributions (entry point group ``vegeta.commands``)."""
    try:
        eps = metadata.entry_points(group="vegeta.commands")
    except TypeError:  # Python < 3.10 style API
        eps = metadata.entry_points().get("vegeta.commands", [])
    return {ep.name: ep for ep in eps if ep.name not in TOOLS}


def usage() -> str:
    lines = ["usage: vegeta <command> [arguments]", "", "Engineering tools:"]
    lines += [f"  {name:<10} {text}" for name, text in TOOLS.items()]
    extra = plugins()
    if extra:
        lines += ["", "Workbench commands:"]
        lines += [f"  {name:<10} (from {ep.dist.name if ep.dist else ep.value})" for name, ep in sorted(extra.items())]
    lines += ["", "Run 'vegeta <command> --help' for details.",
              "The tools are also available as the commands dedalus, talos, aeromant, mellonia and boreas."]
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
        extra = plugins()
        if tool in extra:
            return extra[tool].load()(rest, prog=f"vegeta {tool}")
        print(f"vegeta: unknown command {tool!r}\n\n{usage()}", file=sys.stderr)
        return 2
    cli = importlib.import_module(f"vegeta.{tool}.cli")
    return cli.main(rest, prog=f"vegeta {tool}")
