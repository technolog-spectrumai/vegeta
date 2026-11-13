#!/usr/bin/env python3
"""Porunga, standalone: count the words of a text file once, reuse the count afterwards.

    python3 example.py some.txt            # first run: miss → count → save; second run: hit → load
    python3 example.py some.txt --upper    # different parameters → a different key → counted again

The caller decides what identifies the result (the file's content, the option, the counting code's version);
Porunga only turns those parameters into a key and stores what the caller hands it.
"""
import argparse
import collections
import hashlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import porunga  # noqa: E402

COUNTER_VERSION = 1


def count_words(text: str, upper: bool) -> dict:
    words = (text.upper() if upper else text.lower()).split()
    return dict(collections.Counter(words).most_common(10))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file", type=Path)
    ap.add_argument("--upper", action="store_true", help="fold to upper case instead of lower case")
    ap.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / "porunga-example")
    a = ap.parse_args(argv)
    raw = a.file.read_bytes()
    k = porunga.key({"what": "top-10 words", "text_sha256": hashlib.sha256(raw).hexdigest(), "upper": a.upper,
                     "counter_version": COUNTER_VERSION})
    if porunga.check(k, root=a.cache):
        entry = porunga.load(k, root=a.cache)
        print(f"hit  {k[:12]}  (counted {entry.metadata['counted_at']})")
    else:
        print(f"miss {k[:12]}  counting …")
        table = count_words(raw.decode("utf-8", "replace"), a.upper)
        import time
        entry = porunga.save(k, table, metadata={"counted_at": time.strftime("%H:%M:%S"), "source": str(a.file)}, root=a.cache)
    for w, n in entry.data.items():
        print(f"  {n:5d}  {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
