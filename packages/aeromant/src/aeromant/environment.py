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
    def conda(cls, env_prefix: str | Path, runner: str | None = None) -> "OpenFOAMEnvironment":
        """OpenFOAM installed in a conda/mamba environment (e.g. conda-forge ``openfoam``).

        By default the environment's own activation script is sourced; pass ``runner="micromamba"``
        (or ``"conda"``) to launch through ``<runner> run -p <env_prefix>`` instead.
        """
        env_prefix = str(Path(env_prefix))
        if runner:
            return cls(prefix=[runner, "run", "-p", env_prefix])
        script = Path(env_prefix) / "etc" / "conda" / "activate.d" / "openfoam_activate.sh"
        return cls(bashrc=str(script) if script.is_file() else None,
                   env={"CONDA_PREFIX": env_prefix,
                        "PATH": str(Path(env_prefix) / "bin") + os.pathsep + os.environ.get("PATH", "")})

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
        packages under /usr/lib/openfoam and /opt, conda environments, then the Debian/Ubuntu package."""
        if os.environ.get("WM_PROJECT_DIR") and shutil.which("blockMesh"):
            return cls()
        for pattern in ("/usr/lib/openfoam/openfoam*/etc/bashrc", "/opt/openfoam*/etc/bashrc",
                        "/opt/OpenFOAM-*/etc/bashrc", "/usr/lib/openfoam*/etc/bashrc"):
            hits = sorted(glob.glob(pattern))
            if hits:
                return cls(bashrc=hits[-1])
        conda_envs = [os.environ.get("CONDA_PREFIX", ""), "/opt/foam"]
        for pattern in ("/opt/conda/envs/*", "~/micromamba/envs/*", "~/miniforge3/envs/*", "~/.conda/envs/*"):
            conda_envs += sorted(glob.glob(os.path.expanduser(pattern)))
        for env_dir in conda_envs:
            if env_dir and (Path(env_dir) / "bin" / "simpleFoam").is_file():
                return cls.conda(env_dir)
        if shutil.which("blockMesh") and Path("/usr/share/openfoam/etc/controlDict").is_file():
            return cls(env={"WM_PROJECT_DIR": "/usr/share/openfoam"})
        raise RuntimeError(
            "OpenFOAM not found: no sourced environment, no /usr/lib/openfoam or /opt/openfoam* bashrc, no conda "
            "environment with simpleFoam, no /usr/share/openfoam; "
            "create OpenFOAMEnvironment(bashrc=...) explicitly"
        )
