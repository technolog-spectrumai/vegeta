"""Porunga — a small, generic local cache.

Four calls::

    k = porunga.key({"what": "word-count", "text_sha256": sha, "lowercase": True})
    if porunga.check(k):
        entry = porunga.load(k)          # entry.data, entry.files, entry.metadata
    else:
        porunga.save(k, data, metadata={...}, files={"table.csv": path})

Porunga stores and returns; it never interprets. The caller decides which parameters identify a result (software
versions, input hashes, settings, a mock flag …), what the optional metadata means, and what to do on a miss. Porunga
checks storage integrity only: an entry is visible once it is complete and intact, never before.

Entries live under a local root — ``root=`` on each call, else ``configure(root)``, else ``$PORUNGA_DIR`` — one
directory per key with the files and a small ``manifest.json``. Standard library only.
"""
from .store import Corrupt, Entry, check, clean_partial, configure, key, load, save

__all__ = ["Corrupt", "Entry", "check", "clean_partial", "configure", "key", "load", "save"]
__version__ = "0.1.0"
