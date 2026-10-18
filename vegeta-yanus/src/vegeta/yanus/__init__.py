"""Yanus — finds, reads and checks the saved runs of Vegeta's workflows. It never starts a computation.

    from vegeta import yanus
    sel = yanus.find("K1", variant="4Q-2", fingerprint=FP, quality="fine")      # the newest usable run (or run=<dir>)
    display(sel)                       # which run, its state, quality and compatibility — or why none, with the wf command
    summary = sel.require()            # RunUnusable when it is missing or unusable
    report_meta["runs"] = {"K1": sel.provenance()}

The workflow registry (``workflows/flow.py``: books, kinds, chapters, their needs and prerequisites) is read by ``wf``,
the GUI and the scripts; ``check_chapter`` is a script's entry check; ``check_need`` the machine diagnostics.
"""
from .chapters import check_chapter
from .contract import AnyOf, Book, Chapter, Env, Flow, FlowError, Import, Kind, Need, Package, Run, Tool
from .needs import check_need, check_needs
from .registry import current, find_flow_file, load_flow, use
from .select import RunUnusable, Selection, check, command, find, runs_root, table

__all__ = ["AnyOf", "Book", "Chapter", "Env", "Flow", "FlowError", "Import", "Kind", "Need", "Package", "Run", "RunUnusable", "Selection", "Tool",
           "check", "check_chapter", "check_need", "check_needs", "command", "current", "find", "find_flow_file", "load_flow", "runs_root", "table", "use"]
__version__ = "0.2.0"
