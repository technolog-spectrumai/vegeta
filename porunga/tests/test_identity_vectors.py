"""The key specification in docs/identity.md: its vectors, its reference implementation, JSON text input."""
import hashlib
import json
import re
from pathlib import Path

import pytest

import porunga

DOC = (Path(__file__).resolve().parents[1] / "docs" / "identity.md").read_text()


def _vectors():
    return re.findall(r"^\| `(\{.*\})` \| `([0-9a-f]{64})` \|$", DOC, flags=re.M)


def _reference(text: str) -> str:
    value = json.loads(text)
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_documented_vectors_hold_for_porunga_and_the_reference():
    vs = _vectors()
    assert len(vs) == 3
    for text, k in vs:
        assert porunga.key(text) == k == _reference(text)


def test_json_text_formatting_and_key_order_do_not_matter():
    a = '{"b": [1, 2.5], "a": {"x": true}}'
    b = '{\n  "a" : { "x":true },\n\t"b":[ 1,2.5 ]\n}'
    assert porunga.key(a) == porunga.key(b) == porunga.key({"a": {"x": True}, "b": [1, 2.5]})
    assert porunga.key(b.encode()) == porunga.key(a)
    assert porunga.key('{"n": 1}') != porunga.key('{"n": 1.0}')


def test_json_text_constants_are_refused():
    with pytest.raises(ValueError):
        porunga.key('{"x": NaN}')
