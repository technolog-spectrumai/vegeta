# Porunga

A small, generic local cache. Standard library only. It understands JSON, keys, stored data and files, and
storage integrity — nothing about software, versions, scientific meaning, notebooks or mocks.

```python
import porunga

k = porunga.key(identity)              # SHA-256 of the canonical JSON of the caller's identity (text or value)
porunga.check(k)                       # True only for a complete, intact entry
entry = porunga.load(k)                # entry.data, entry.files {name: Path}, entry.metadata
porunga.save(k, data, metadata=None, files=None)   # atomic publish; the first intact writer wins
```

* **Keys** (`docs/identity.md`: canonical form, reference implementation, test vectors). JSON text is
  parsed first, so formatting never matters; object key order never matters; tuples are lists, sets are sorted, paths are POSIX strings, numpy values
  their `.tolist()`. Anything else is a `TypeError`, NaN/inf a `ValueError`. No `hash()`, no UUIDs.
* **What goes in the key is the caller's choice**: software versions, input file hashes, numerical settings, a mock
  flag. Metadata is stored and returned as given, never read.
* **Storage.** `<root>/<key>/manifest.json`, `data.json`, `files/…`. Root: `root=` on a call, else
  `porunga.configure(path)`, else `$PORUNGA_DIR`.
* **Safety.** A write is staged in `<root>/.tmp/` and published with one `rename`: an interrupted write is never
  visible, concurrent writers of one key leave exactly one intact entry, a failed replacement leaves the old entry.
  `check` compares sizes (`verify=True` re-hashes). `load` raises `KeyError` on a miss, `porunga.Corrupt` on damage.
  `clean_partial()` removes leftovers of interrupted writes, on request only. No eviction.
* **Never** runs software, picks between entries or decides what a miss means — the caller does.

Example: `python3 example.py <text file>` (counts words once, then reuses the count). Tests: `pytest` here.
