"""A minimal in-process message bus and a deterministic executor (stands in for ROS 2 topics and timers).

``Bus``: named topics. ``publish(topic, msg)`` stores the latest message and appends it to every subscriber's queue;
``latest(topic)`` reads the last one; ``take(topic, key)`` drains the queue of subscriber ``key``. With ``record``
every message is kept (topic, msg) for replay, plots and the video.

``Executor``: nodes with their own rates, driven by the caller's clock (the simulator's or the flight computer's).
``spin_until(t)`` runs every due node tick in time order (ties in the order the nodes were added). Nothing is
threaded, so a run is repeatable to the bit.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque

__all__ = ["Bus", "Node", "Executor"]


class Bus:
    def __init__(self, record: bool = True):
        self._latest: dict = {}
        self._queues: dict = defaultdict(dict)       # topic -> {subscriber: deque}
        self.record = record
        self.log: list = []                          # (topic, msg)

    def publish(self, topic: str, msg) -> None:
        self._latest[topic] = msg
        for q in self._queues[topic].values():
            q.append(msg)
        if self.record:
            self.log.append((topic, msg))

    def latest(self, topic: str, default=None):
        return self._latest.get(topic, default)

    def subscribe(self, topic: str, key: str, maxlen: int | None = None) -> None:
        self._queues[topic].setdefault(key, deque(maxlen=maxlen))

    def take(self, topic: str, key: str) -> list:
        q = self._queues[topic].get(key)
        if q is None:
            raise KeyError(f"{key!r} is not subscribed to {topic!r}")
        out = list(q)
        q.clear()
        return out

    def messages(self, topic: str) -> list:
        """Every recorded message of ``topic`` (needs ``record``)."""
        return [m for tp, m in self.log if tp == topic]


class Node:
    """A unit of the mission software. Subclasses set ``name``, declare ``subscriptions`` (topics they queue) and
    implement ``step(t, bus)``."""
    name = "node"
    subscriptions: tuple = ()

    def attach(self, bus: Bus) -> None:
        for topic in self.subscriptions:
            bus.subscribe(topic, self.name)

    def step(self, t: float, bus: Bus) -> None:          # pragma: no cover - interface
        raise NotImplementedError


class Executor:
    def __init__(self, bus: Bus):
        self.bus = bus
        self.nodes: list = []                        # [node, period, next_t]
        self.t = 0.0
        self.ticks = defaultdict(int)

    def add(self, node: Node, rate_hz: float, t0: float = 0.0) -> Node:
        node.attach(self.bus)
        self.nodes.append([node, 1.0 / rate_hz, t0])
        return node

    def node(self, name: str) -> Node:
        return next(n for n, _, _ in self.nodes if n.name == name)

    def spin_until(self, t: float) -> None:
        while True:
            due = [(nt, k) for k, (_, _, nt) in enumerate(self.nodes) if nt <= t + 1e-9]
            if not due:
                break
            nt, k = min(due)
            node, period, _ = self.nodes[k]
            self.t = nt
            node.step(nt, self.bus)
            self.ticks[node.name] += 1
            self.nodes[k][2] = nt + period
        self.t = max(self.t, t)

    def reset_clock(self, t0: float) -> None:
        for entry in self.nodes:
            entry[2] = t0
        self.t = t0


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
