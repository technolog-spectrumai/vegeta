"""``workflows/runkit.py`` importable from the design modules (the notebooks put ``designs/`` on the path, not
``workflows/``): one source, re-exported here."""
import importlib.util as _ilu
import sys as _sys
from pathlib import Path as _Path

_src = _Path(__file__).resolve().parents[1] / "workflows" / "runkit.py"
_spec = _ilu.spec_from_file_location("runkit", _src)
_mod = _ilu.module_from_spec(_spec)
_sys.modules.setdefault("runkit", _mod)
_spec.loader.exec_module(_mod)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
