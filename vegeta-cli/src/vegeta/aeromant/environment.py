"""How to invoke OpenFOAM executables. Explicit; ``detect()`` is an opt-in helper."""
from __future__ import annotations

import glob
import os
import re
import shlex
import shutil
import subprocess
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
        """True when ``executable`` can actually be started through this environment."""
        if self.prefix and not (shutil.which(self.prefix[0]) or Path(self.prefix[0]).is_file()):
            return False
        if self.bashrc and not (Path(self.bashrc).is_file() and shutil.which("bash")):
            return False
        if not self.prefix and not self.bashrc:
            return shutil.which(executable, path=self.env.get("PATH")) is not None
        try:
            # the installation's own bin directory must provide it (not another OpenFOAM earlier on PATH)
            script = (f'if [ -n "$FOAM_APPBIN" ]; then test -x "$FOAM_APPBIN/{executable}"; '
                      f'else command -v {shlex.quote(executable)} >/dev/null; fi')
            r = subprocess.run(self.command(["bash", "-c", script]), env={**os.environ, **self.env},
                               capture_output=True, text=True, timeout=60)
            return r.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    def describe(self) -> dict:
        return {"bashrc": self.bashrc, "env": dict(self.env), "prefix": list(self.prefix),
                "version": self.version(), "flavor": self.flavor()}

    def version(self) -> str | None:
        """``WM_PROJECT_VERSION`` of this installation (e.g. ``v2412`` for openfoam.com, ``14`` for
        openfoam.org), read from the environment the commands run in; ``None`` if unknown."""
        if self.bashrc and not self.prefix:
            m = re.search(r"^\s*(?:export\s+)?WM_PROJECT_VERSION=\s*[\"']?([^\s\"';]+)", Path(self.bashrc).read_text(errors="replace"), re.M)
            if m and "$" not in m.group(1):
                return m.group(1)
        try:
            r = subprocess.run(self.command(["bash", "-c", 'echo "$WM_PROJECT_VERSION"']),
                               env={**os.environ, **self.env}, capture_output=True, text=True, timeout=60)
            v = r.stdout.strip().splitlines()[-1].strip() if r.stdout.strip() else ""
            if v:
                return v
        except (OSError, subprocess.SubprocessError):
            pass
        for key in ("WM_PROJECT_VERSION",):
            if self.env.get(key) or os.environ.get(key):
                return self.env.get(key) or os.environ.get(key)
        d = self.env.get("WM_PROJECT_DIR") or os.environ.get("WM_PROJECT_DIR")
        if d and (Path(d) / "etc" / "bashrc").is_file():
            m = re.search(r"WM_PROJECT_VERSION=\s*[\"']?([\w.+-]+)", (Path(d) / "etc" / "bashrc").read_text(errors="replace"))
            if m:
                return m.group(1)
        return None

    def flavor(self) -> str | None:
        """``"openfoam.com"`` (versions ``v1912``, ``v2412`` ...), ``"openfoam.org"`` (``9``, ``14`` ...) or ``None``."""
        return flavor_of(self.version())

    @classmethod
    def candidates(cls) -> list["OpenFOAMEnvironment"]:
        """Installations found in well-known places: an already sourced environment, official
        openfoam.com/.org packages under /usr/lib/openfoam and /opt, conda environments, the
        Debian/Ubuntu package."""
        found: list[OpenFOAMEnvironment] = []
        if os.environ.get("WM_PROJECT_DIR") and shutil.which("blockMesh"):
            found.append(cls())
        for pattern in ("/usr/lib/openfoam/openfoam*/etc/bashrc", "/opt/openfoam*/etc/bashrc",
                        "/opt/OpenFOAM-*/etc/bashrc", "/usr/lib/openfoam*/etc/bashrc"):
            found += [cls(bashrc=h) for h in sorted(glob.glob(pattern), reverse=True)]
        conda_envs = [os.environ.get("CONDA_PREFIX", ""), "/opt/foam"]
        for pattern in ("/opt/conda/envs/*", "~/micromamba/envs/*", "~/miniforge3/envs/*", "~/.conda/envs/*"):
            conda_envs += sorted(glob.glob(os.path.expanduser(pattern)))
        for env_dir in conda_envs:
            if env_dir and (Path(env_dir) / "bin" / "simpleFoam").is_file():
                found.append(cls.conda(env_dir))
        if shutil.which("blockMesh") and Path("/usr/share/openfoam/etc/controlDict").is_file():
            found.append(cls(env={"WM_PROJECT_DIR": "/usr/share/openfoam"}))
        return found

    @classmethod
    def detect(cls, flavor: str | None = "openfoam.com") -> "OpenFOAMEnvironment":
        """The first installation of the wanted ``flavor`` (the templates are written for
        ``"openfoam.com"``); with ``flavor=None`` or when none matches, the first one found."""
        found = cls.candidates()
        if flavor:
            for env in found:
                if env.flavor() == flavor:
                    return env
        if found:
            return found[0]
        raise RuntimeError(
            "OpenFOAM not found: no sourced environment, no /usr/lib/openfoam or /opt/openfoam* bashrc, no conda "
            "environment with simpleFoam, no /usr/share/openfoam; "
            "create OpenFOAMEnvironment(bashrc=...) explicitly"
        )


def flavor_of(version: str | None) -> str | None:
    if not version:
        return None
    v = version.strip().lower()
    if re.match(r"^v\d{4}", v):
        return "openfoam.com"
    if re.match(r"^\d+(\.\d+)?(-|$)", v) or v in ("dev",):
        return "openfoam.org"
    return None
