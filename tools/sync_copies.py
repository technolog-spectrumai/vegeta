"""The tools' deliberate copies (each tool stays standalone: ``test_independence`` forbids importing another vegeta
package), kept identical. ``tests/smoke/test_tool_copies.py`` fails when they drift; this copies the master over.

    python tools/sync_copies.py            # report
    python tools/sync_copies.py --write    # copy each master over its copies
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = "libs/vegeta-cli/src/vegeta"
# master → copies (the master is where a fix is made first)
COPIES = {
    f"{SRC}/talos/result.py": [f"{SRC}/{t}/result.py" for t in ("aeromant", "atropos", "boreas", "chronos", "dedalus",
                                                                  "mellonia", "zephyr")]
                              + ["libs/vegeta-fidia/src/vegeta/fidia/result.py"],
    f"{SRC}/talos/_process.py": [f"{SRC}/aeromant/_process.py", f"{SRC}/mellonia/_process.py"],
    f"{SRC}/talos/cache.py": [f"{SRC}/aeromant/cache.py"],
    f"{SRC}/talos/_progress.py": [f"{SRC}/aeromant/_progress.py"],
    f"{SRC}/talos/_watermark.py": [f"{SRC}/aeromant/_watermark.py"],
}


def drift() -> list[str]:
    out = []
    for master, copies in COPIES.items():
        text = (ROOT / master).read_bytes()
        out += [f"{c} differs from {master}" for c in copies if (ROOT / c).read_bytes() != text]
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    d = drift()
    if "--write" in argv:
        for master, copies in COPIES.items():
            for c in copies:
                (ROOT / c).write_bytes((ROOT / master).read_bytes())
        print(f"{len(d)} copies rewritten from their masters")
        return 0
    print("\n".join(d) if d else "the tools' copies are identical")
    return 1 if d else 0


if __name__ == "__main__":
    sys.exit(main())
