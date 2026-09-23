"""How to invoke OpenFOAM executables. Explicit; ``detect()`` is an opt-in helper."""
from __future__ import annotations

import glob
import os
import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OpenFOAMEnvironment:
    """How OpenFOAM commands are launched.

    ``bashrc``: source this OpenFOAM ``etc/bashrc`` before each command.
    ``prefix``: launcher put in front of each command, e.g. ``["micromamba", "run", "-p", "/opt/foam"]``
    (see :meth:`conda`).
    ``env``: extra environment variables (e.g. ``{"WM_PROJECT_DIR": "/usr/share/openfoam"}`` for the
    Debian/Ubuntu package). With none of these, executables are taken from ``PATH`` as is.
    """

    bashrc: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    prefix: list[str] = field(default_factory=list)

    @classmethod
    def conda(cls, env_prefix: str, runner: str = "micromamba") -> "OpenFOAMEnvironment":
        """OpenFOAM installed in a conda/mamba environment (e.g. conda-forge ``openfoam``)."""
        return cls(prefix=[runner, "run", "-p", str(env_prefix)])

    def command(self, argv: list[str]) -> list[str]:
        argv = list(argv)
        if self.bashrc:
            inner = " ".join(shlex.quote(a) for a in argv)
            argv = ["bash", "-c", f"source {shlex.quote(self.bashrc)} >/dev/null 2>&1; exec {inner}"]
        return list(self.prefix) + argv

    def available(self, executable: str = "blockMesh") -> bool:
        if self.prefix:
            return shutil.which(self.prefix[0]) is not None or Path(self.prefix[0]).is_file()
        if self.bashrc:
            return Path(self.bashrc).is_file()
        return shutil.which(executable, path=self.env.get("PATH")) is not None

    def describe(self) -> dict:
        return {"bashrc": self.bashrc, "env": dict(self.env), "prefix": list(self.prefix)}

    @classmethod
    def detect(cls) -> "OpenFOAMEnvironment":
        """Look in well-known places: an already sourced environment, official openfoam.com/.org
        packages under /usr/lib/openfoam and /opt, then the Debian/Ubuntu package."""
        if os.environ.get("WM_PROJECT_DIR") and shutil.which("blockMesh"):
            return cls()
        for pattern in ("/usr/lib/openfoam/openfoam*/etc/bashrc", "/opt/openfoam*/etc/bashrc",
                        "/opt/OpenFOAM-*/etc/bashrc", "/usr/lib/openfoam*/etc/bashrc"):
            hits = sorted(glob.glob(pattern))
            if hits:
                return cls(bashrc=hits[-1])
        if shutil.which("blockMesh") and Path("/usr/share/openfoam/etc/controlDict").is_file():
            return cls(env={"WM_PROJECT_DIR": "/usr/share/openfoam"})
        raise RuntimeError(
            "OpenFOAM not found: no blockMesh on PATH, no /usr/share/openfoam, no /opt/openfoam*/etc/bashrc; "
            "create OpenFOAMEnvironment(bashrc=...) explicitly"
        )
