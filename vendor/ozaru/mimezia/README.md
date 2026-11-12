# Mimezia — imitating expensive APIs

A small, standard-library-only base for running whole programs and notebooks against **stubs** of slow or heavy APIs
(finite-element solvers, external tools) before any long run is made — without the stubs ever passing for results.

| piece | what it does |
|---|---|
| `Mode(env, default="mock")` | the switch `mock` / `off` / `real`; the environment variable overrides the program's default and is exported to child processes; `Mode.launches(mode)` (off launches nothing) |
| `@mimic(name)`, `call(name, …, real=fn, mock=…)` | a stub per named real API; the real function or the stub, by mode |
| `label(result, name)`, `is_mock(result)` | every stub result carries `"mock": true`, the mimic's name and a note |
| `isolated(path)`, `isolate_env({VAR: default})` | mock outputs in separate places (`runs` → `runs-mock`) |
| `record_schema(real, path)`, `check(stub, schema)` | a real result's shape recorded once; stubs checked against it so they do not drift |
| `banner(mode, text)` | a loud notice (HTML box in Jupyter, a starred line elsewhere) |
| `interpreter(mock=…, env=…, candidates=…)` | the external interpreter a real API needs, or this one under mock |

```python
import mimezia as mm
MODE = mm.Mode("MYAPP_SOLVER", default="mock")

@mm.mimic("solver.plate")
def plate_stub(spec):                     # same keys, types and shapes as the real result; dumb numbers
    return {"T_max": 300.0 + spec["power"] / 10, "converged": True}

result = mm.call("solver.plate", spec, real=real_plate, mock=MODE.is_mock())
```

Install: `pip install -e mimezia` (package `mimezia-mock`, import `mimezia`). Tests: `pytest mimezia/tests`.

Status: proof of concept, part of Ozaru (a common layer shared by Vegeta and Alabastron). Hosts decide how their mocks use it.
