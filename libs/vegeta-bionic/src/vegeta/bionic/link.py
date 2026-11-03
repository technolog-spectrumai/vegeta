"""The glove-to-host link: a versioned binary message for ``GloveSample`` and ``InProcessLink``, a transport with a
latency, jitter and loss model — so the teleop's latency budget can be measured in simulation (``docs/bionic.md``:
"latency measured first") before the real transport (serial / UDP) is chosen.

Message, little-endian::

    magic 'VB' | version u8 | side 'R'/'L' | seq u32 | t f64 | n u8 | hall u16 x n | gyro f32x3 | acc f32x3 | mag f32x3
"""
from __future__ import annotations

import heapq
import struct
from dataclasses import dataclass, field

import numpy as np

from .glove import GloveSample

MAGIC, VERSION = b"VB", 1
_HEAD = struct.Struct("<2sBcIdB")


def encode(s: GloveSample) -> bytes:
    n = len(s.hall)
    return (_HEAD.pack(MAGIC, VERSION, s.side.encode(), s.seq & 0xFFFFFFFF, float(s.t), n)
            + struct.pack(f"<{n}H", *[int(v) for v in s.hall])
            + struct.pack("<9f", *np.r_[s.gyro, s.acc, s.mag].astype(float)))


def decode(b: bytes) -> GloveSample:
    magic, ver, side, seq, t, n = _HEAD.unpack_from(b, 0)
    if magic != MAGIC:
        raise ValueError("not a vegeta.bionic glove message")
    if ver != VERSION:
        raise ValueError(f"glove message version {ver}, this reader knows {VERSION}")
    o = _HEAD.size
    hall = np.array(struct.unpack_from(f"<{n}H", b, o), dtype=np.int64)
    o += 2 * n
    v = np.array(struct.unpack_from("<9f", b, o), dtype=float)
    return GloveSample(float(t), int(seq), side.decode(), hall, v[0:3], v[3:6], v[6:9])


@dataclass
class InProcessLink:
    """Messages arrive ``latency_s`` (+ |N(0, jitter_s)|) after they are sent; a share ``loss`` never arrives. Jitter
    can reorder them; ``receive(t)`` returns what has arrived by ``t`` in arrival order, and the receiver drops a
    message older (by ``seq``) than one it already has (``stale``)."""

    latency_s: float = 0.005
    jitter_s: float = 0.0
    loss: float = 0.0
    seed: int = 0
    _q: list = field(default_factory=list)
    sent: int = 0
    lost: int = 0
    stale: int = 0
    delays: list = field(default_factory=list)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self._last_seq = {}

    def send(self, t: float, payload: bytes) -> None:
        self.sent += 1
        if self.rng.random() < self.loss:
            self.lost += 1
            return
        d = self.latency_s + abs(self.rng.normal(0.0, self.jitter_s)) if self.jitter_s > 0 else self.latency_s
        heapq.heappush(self._q, (t + d, self.sent, t, payload))

    def receive(self, t: float) -> list:
        """[(arrival time, GloveSample)] that arrived by ``t``, newest per glove side wins over older ones."""
        out = []
        while self._q and self._q[0][0] <= t:
            ta, _, ts, payload = heapq.heappop(self._q)
            s = decode(payload)
            last = self._last_seq.get(s.side, -1)
            if s.seq <= last:
                self.stale += 1
                continue
            self._last_seq[s.side] = s.seq
            self.delays.append(ta - ts)
            out.append((ta, s))
        return out

    def stats(self) -> dict:
        d = np.array(self.delays) if self.delays else np.array([np.nan])
        return {"sent": self.sent, "delivered": len(self.delays), "lost": self.lost, "stale": self.stale,
                "delay_mean_s": float(np.nanmean(d)), "delay_p99_s": float(np.nanpercentile(d, 99))}
