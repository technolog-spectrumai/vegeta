# Keys: the canonical form and the hash

A Porunga key is computed from an **identity**: a JSON description of a result that the caller writes. Two parties
that write the same description — in any formatting, with object keys in any order, in different processes or
programs — get the same key.

## Algorithm

1. Parse the identity as JSON (when given as text). `NaN`, `Infinity` and `-Infinity` are refused.
2. Write it back canonically:
   * objects: keys sorted by Unicode code point, every key a string;
   * no whitespace: separators `,` and `:`;
   * strings: JSON escapes only where JSON requires them (`"`, `\`, control characters); other characters as
     themselves (no `\uXXXX` for non-ASCII);
   * numbers: integers as integers; floats in the shortest form that reads back to the same double (Python's
     `repr`), so `1` and `1.0` are different identities, and `0.1` stays `0.1`;
   * `true`, `false`, `null` as JSON.
3. Encode the text as UTF-8 and take its **SHA-256**; the key is the 64 lowercase hexadecimal characters.

From Python values (not text), tuples are lists, sets are sorted lists, paths are POSIX strings and numpy values
their `.tolist()`; anything else is refused rather than guessed.

## Reference implementation

```python
import hashlib, json

def key(identity_json_text: str) -> str:
    value = json.loads(identity_json_text)
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

## Test vectors (checked by `tests/test_identity_vectors.py`)

| Identity | Key |
|---|---|
| `{}` | `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a` |
| `{"b": [1, 2.5, "é"], "a": {"y": null, "x": true}}` | `1ec5114d36593eba3ac85d60b13d0046b42734255834ce2b5366c78cd108ea57` |
| `{"identity": "example.word-count", "version": 1, "text_sha256": "0000", "lowercase": true}` | `35db20f89af3a257e1ff4e983ca82730a62f799ba2ff38b4252cb7df3536d551` |

## What belongs in an identity

Everything that changes the result, and nothing else: what is computed, the inputs (or their content hashes), the
numerical settings, the versions of the software that computes it, whether it is a mock. Provenance that does not
change the result (who ran it, when, which commit) goes into the entry's metadata instead — putting it in the
identity would make every unrelated change a cache miss. Porunga never looks inside: the meaning is the caller's.
