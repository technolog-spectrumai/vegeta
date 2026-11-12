"""Progress reporting that works for scripts, notebooks (tqdm) and a future GUI (callbacks)."""
from __future__ import annotations

import threading
from typing import Callable, Protocol


class Progress(Protocol):
    def __call__(self, stage: str, fraction: float | None = None, message: str = "") -> None: ...


def _noop(stage: str, fraction: float | None = None, message: str = "") -> None:
    return None


class TqdmProgress:
    """Adapter mapping progress callbacks onto a tqdm bar (0..100 %).

    A background ticker refreshes the bar every second, so the elapsed time keeps moving during long
    steps that report nothing in between (a Gmsh 3D mesh, a solver iteration)."""

    def __init__(self, desc: str = "", tick: float = 1.0):
        from tqdm.auto import tqdm

        self._bar = tqdm(total=100, desc=desc, bar_format="{desc}: {percentage:3.0f}%|{bar}| {elapsed} {postfix}")
        self._stage = ""
        self._stop = threading.Event()
        self._ticker = threading.Thread(target=self._tick, args=(tick,), daemon=True)
        self._ticker.start()

    def _tick(self, every: float) -> None:
        while not self._stop.wait(every):
            self._bar.refresh()

    def __call__(self, stage: str, fraction: float | None = None, message: str = "") -> None:
        if stage != self._stage:
            self._stage = stage
            self._bar.set_description_str(stage)
        if fraction is not None:
            target = max(0.0, min(1.0, fraction)) * 100
            self._bar.update(target - self._bar.n)
        if message:
            self._bar.set_postfix_str(message[:60])

    def close(self) -> None:
        self._stop.set()
        self._ticker.join(timeout=2)
        self._bar.close()


def resolve_progress(progress: bool | Callable | None, desc: str = "") -> tuple[Callable, Callable[[], None]]:
    """Return ``(callback, close)`` for ``progress`` = True (tqdm), False/None, or a callable."""
    if progress is True:
        bar = TqdmProgress(desc)
        return bar, bar.close
    if callable(progress):
        return progress, lambda: None
    return _noop, lambda: None
