# vegeta-yanus

Finds, reads and checks the saved runs of Vegeta's workflows. It also holds the workflow registry that `wf` reads.
**It never starts a computation.** The design and the rules are in `docs/yanus.md`.

```python
from vegeta import yanus
sel = yanus.find("K1", variant="4Q-2", fingerprint=FP, quality="fine")   # newest usable run; run=<dir> for an explicit one
display(sel)                 # the run chosen, or why none, with the wf command that would make one
summary = sel.require()      # yanus.RunUnusable when missing or unusable
meta["runs"] = {"K1 4Q-2": sel.provenance()}
```

A run is usable when all of these hold:
- it completed and its `summary.json` reads;
- its workflow and variant are the requested ones;
- its quality is the requested one;
- it is not a mock (Mimesis's `"mock": true`);
- the book's compatibility check does not reject it.

With no fingerprint to compare, the run is accepted and shown as **unverified**.

```
yn find K1 --variant 4Q-2 [--quality fine] [--run DIR] [--json]
yn doctor          the chapters' needs on this machine
yn validate        the registry, workflows/flow.py
yn new chapter <book>/<name> --kind K [--new-kind] [--after CH]
```

The package depends only on the standard library (it uses pandas for `yanus.table` when pandas is installed) and
imports no other Vegeta package. Tests: `pytest vegeta-yanus/tests`.
