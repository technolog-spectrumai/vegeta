import pytest

from vegeta.dedalus import Parameter, ParameterSet


def test_defaults_and_overrides():
    ps = ParameterSet([Parameter("w", 10.0, "mm", min=1), Parameter("n", 3, min=1, max=10), Parameter("f", True)])
    assert ps.resolve() == {"w": 10.0, "n": 3, "f": True}
    assert ps.resolve({"w": 12, "n": 4.0}) == {"w": 12.0, "n": 4, "f": True}


@pytest.mark.parametrize("override, msg", [
    ({"w": 0.5}, "below minimum"),
    ({"n": 11}, "above maximum"),
    ({"n": 2.5}, "integer"),
    ({"w": "wide"}, "number"),
    ({"f": 1}, "bool"),
    ({"zzz": 1}, "unknown parameter"),
])
def test_invalid_overrides_raise(override, msg):
    ps = ParameterSet([Parameter("w", 10.0, min=1), Parameter("n", 3, min=1, max=10), Parameter("f", True)])
    with pytest.raises(ValueError, match=msg):
        ps.resolve(override)


def test_parse_cli_strings():
    assert Parameter("w", 1.0).parse("2.5") == 2.5
    assert Parameter("n", 1).parse("4") == 4
    assert Parameter("f", False).parse("yes") is True
    assert Parameter("s", "a", choices=("a", "b")).parse("b") == "b"
    with pytest.raises(ValueError):
        Parameter("s", "a", choices=("a", "b")).parse("c")


def test_bad_definitions():
    with pytest.raises(ValueError):
        Parameter("not valid", 1.0)
    with pytest.raises(ValueError):
        Parameter("w", 0.0, min=1)
    with pytest.raises(ValueError, match="duplicate"):
        ParameterSet([Parameter("a", 1), Parameter("a", 2)])
