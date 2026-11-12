"""Demonstration recordings for teaching robots (``docs/bionic.md``): one folder per recording.

* ``meta.json`` — who and what (robot, gloves, calibrations, the mapping and its mode and clutch events, the latencies,
  the software revision, free fields), the streams and their columns, the cameras;
* ``<stream>.npz`` — one per stream, columnar, all on one clock ``t`` [s]: ``glove_<side>`` (raw samples),
  ``hand_<side>`` (the estimated and the true hand), ``robot_command``, ``robot_state``, ``contacts``, ``objects``;
* ``camera_<name>.mp4`` with ``camera_<name>_frames.npz`` (frame timestamps) — when a renderer is available; when it is
  not, the camera's definition and its frame times are still recorded and ``meta.json`` marks it ``not rendered``, so
  a replay on a machine with OpenGL renders it from the states.

``Recording.load(folder)`` reads it back for replay or training.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FORMAT = "vegeta-bionic-recording/1"


@dataclass
class Recorder:
    meta: dict = field(default_factory=dict)
    streams: dict = field(default_factory=dict)
    cameras: dict = field(default_factory=dict)

    def add(self, stream: str, t: float, **values) -> None:
        """One row: ``t`` and named scalars or arrays (the same names every row of a stream)."""
        self.streams.setdefault(stream, []).append({"t": float(t), **{k: np.asarray(v, float) for k, v in values.items()}})

    def camera(self, name: str, spec: dict) -> None:
        """A camera's definition (pose or body it rides on, field of view, size, rate)."""
        self.cameras[name] = {"spec": spec, "frame_t": [], "frames": [], "rendered": None}

    def frame(self, name: str, t: float, image=None) -> None:
        c = self.cameras[name]
        c["frame_t"].append(float(t))
        if image is not None:
            c["frames"].append(np.asarray(image, np.uint8))

    def save(self, folder) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        cols = {}
        for name, rows in self.streams.items():
            keys = list(rows[0])
            arrays = {k: np.stack([r[k] for r in rows]) for k in keys}
            np.savez_compressed(folder / f"{name}.npz", **arrays)
            cols[name] = {k: list(arrays[k].shape[1:]) for k in keys}
        cams = {}
        for name, c in self.cameras.items():
            np.savez_compressed(folder / f"camera_{name}_frames.npz", t=np.array(c["frame_t"]))
            rendered = bool(c["frames"])
            if rendered:
                _write_mp4(folder / f"camera_{name}.mp4", c["frames"], c["spec"].get("fps", 25))
            cams[name] = {"spec": c["spec"], "frames": len(c["frame_t"]),
                          "video": f"camera_{name}.mp4" if rendered else "not rendered (no OpenGL renderer here)"}
        meta = {"format": FORMAT, "saved": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **self.meta,
                "streams": cols, "cameras": cams}
        (folder / "meta.json").write_text(json.dumps(meta, indent=1, default=_json))
        return folder


def _json(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def _write_mp4(path, frames, fps):
    import subprocess
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                          "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()


@dataclass
class Recording:
    meta: dict
    streams: dict

    @classmethod
    def load(cls, folder) -> "Recording":
        folder = Path(folder)
        meta = json.loads((folder / "meta.json").read_text())
        if meta.get("format") != FORMAT:
            raise ValueError(f"{folder}: not a {FORMAT} recording")
        streams = {name: dict(np.load(folder / f"{name}.npz")) for name in meta["streams"]}
        return cls(meta, streams)
