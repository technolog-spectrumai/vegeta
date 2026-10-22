# Yanus

Requirement checks, standard library only: is a tool, a package, an environment variable there on this machine?

```python
from yanus import AnyOf, Env, Package, Tool, check_needs, table
print(table(check_needs([Tool("ccx"), Package("numpy"), Env("HOME"), AnyOf("backend", Package("neuralfoil"), Tool("xfoil"))])))
```

`Tool(name, version_cmd=(…), probe=callable)` — found on the PATH or by the probe, version asked with a 2 s timeout.
`Package(name)` — found, not imported. `Env(name)` — set. `AnyOf(label, *needs)` — one is enough. Each need may
carry `why` and `fix`. Results are cached per process (`clear_cache()`). Yanus never installs or launches work; the
caller decides what a missing need means. Tests: `pytest` here.
