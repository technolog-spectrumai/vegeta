# Result shape

Every operation that produces engineering output returns a result object with the same fields.
Each package defines its own class (`dedalus.Result`, `talos.Result`, ...); the shape is a convention.

| Field | Type | Meaning |
|-------|------|---------|
| `kind` | `str` | what produced it, e.g. `"dedalus.generate"`, `"talos.solve"` |
| `status` | `str` | `"success"`, `"failed"` or `"cancelled"` |
| `metrics` | `dict[str, Any]` | numeric/text quantities; unavailable ones are `None` |
| `artifacts` | `dict[str, Path]` | named files and directories produced (never deleted) |
| `messages` | `list[str]` | human-readable notes, warnings and errors |
| `duration_s` | `float` | wall-clock duration of the operation |
| `execution` | `list[CommandRecord]` | every external command that was run |
| `metadata` | `dict[str, Any]` | inputs, configuration, units, tool versions, timestamps |

Methods: `ok` (property), `to_dict()`, `save_json(path)`, `raise_for_status()`, and `_repr_html_()`
for a compact notebook view.

## CommandRecord

| Field | Meaning |
|-------|---------|
| `command` | argv list exactly as executed |
| `cwd` | working directory |
| `returncode` | process exit code, `None` if it never ran or was killed |
| `duration_s` | wall-clock seconds |
| `stdout`, `stderr` | captured text |
| `started_at` | ISO-8601 UTC timestamp |
| `error` | why it did not run or finish (missing executable, timeout, cancelled), else `None` |
| `log_file` | path of the log written next to the artifacts |

A command that returns zero is not by itself a success: packages also check that the expected
output exists and can be parsed.
