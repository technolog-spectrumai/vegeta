"""Yanus — the door keeper between Vegeta's notebooks and workflows.

Every unit (a notebook, a workflow chapter) declares in the project's flow registry (``workflows/flow.py``) what it
**needs** on the machine, what it **takes** from other units and what it **gives**. ``enter`` checks the needs and
the takes before the unit starts and raises one ``GateClosed`` with every fix; ``Gate.leave`` checks the gives and
stamps them. Yanus guards; it drives only in **auto mode** (a notebook's ``auto=True`` flag, ``YANUS_AUTO=1``), where
a missing take is produced in screening quality before the gate opens, and the gate never closes (``auto.py``).

    from vegeta import yanus
    gate = yanus.enter("43_kestrel_study")        # the gate table; GateClosed with the commands when closed
    ...
    gate.leave()                                  # the promised files written and stamped
"""
from .contract import AnyOf, Book, Data, Chapter, Env, File, Flow, FlowError, Import, Kind, Need, Notebook, Package, Run, Tool, Unit
from . import auto
from .gate import Gate, GateClosed, Row, check_chapter, enter
from .needs import check_need, check_needs
from .registry import current, find_flow_file, load_flow, use

__all__ = ["AnyOf", "Book", "Data", "Chapter", "Env", "File", "Flow", "FlowError", "Gate", "GateClosed", "Import", "Kind", "Need", "Notebook", "Package", "Row",
           "Run", "Tool", "Unit", "auto", "check_chapter", "check_need", "check_needs", "current", "enter", "find_flow_file", "load_flow", "use"]
__version__ = "0.1.0"
