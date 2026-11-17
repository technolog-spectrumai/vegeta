"""Command parsing, normalisation, sampling and the sources."""
import numpy as np

from kerchak_rl.commands import COMMAND_DIM, Command, CommandRanges, CommandSampler, ConstantCommand, SampledCommand


def test_vector_roundtrip():
    c = Command(0.3, -0.1, 0.2, 0.6, False)
    v = c.vector()
    assert v.shape == (COMMAND_DIM,)
    assert Command.from_vector(v) == c


def test_stop_zeroes_speeds():
    c = Command(0.3, 0.1, 0.2, 0.6, True).effective()
    assert (c.vx, c.vy, c.yaw_rate) == (0.0, 0.0, 0.0) and c.stop
    assert Command(0.01, 0.0, 0.0).is_still()


def test_normalisation_within_unit_range():
    r = CommandRanges((-0.3, 0.4), (-0.15, 0.15), (-0.5, 0.5), (0.5, 0.7))
    n = r.normalize(Command(0.4, -0.15, 0.5, 0.7, False))
    assert np.allclose(n, [1.0, -1.0, 1.0, 1.0, 0.0])
    n = r.normalize(Command(0.4, 0.0, 0.0, 0.6, True))
    assert np.allclose(n, [0.0, 0.0, 0.0, 0.0, 1.0])                        # a stop hides the speeds and flags itself


def test_clip_and_sampler():
    r = CommandRanges((-0.3, 0.4), (-0.15, 0.15), (-0.5, 0.5), (0.5, 0.7))
    c = r.clip(Command(2.0, -2.0, 3.0, 0.1))
    assert (c.vx, c.vy, c.yaw_rate, c.height) == (0.4, -0.15, 0.5, 0.5)
    s = CommandSampler(r, stop_probability=0.0, hold_still_speed=0.0, rng=np.random.default_rng(0))
    for _ in range(50):
        c = s.sample()
        assert -0.3 <= c.vx <= 0.4 and -0.15 <= c.vy <= 0.15 and 0.5 <= c.height <= 0.7 and not c.stop
    s = CommandSampler(r, stop_probability=1.0, rng=np.random.default_rng(0))
    assert all(s.sample().stop for _ in range(10))


def test_sources():
    r = CommandRanges()
    src = SampledCommand(CommandSampler(r, rng=np.random.default_rng(1)), resample_seconds=2.0)
    a, b = src.current(0.0), src.current(1.0)
    assert a == b
    assert ConstantCommand(Command(0.1)).current(5.0).vx == 0.1
