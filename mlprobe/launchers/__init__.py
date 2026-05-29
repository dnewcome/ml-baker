"""Probe launchers — interchangeable backends for running one probe.

The runner accepts either a launcher *name* (``"subprocess"``, ``"in_process"``,
``"docker"``) or a configured ``ProbeLauncher`` instance. Name-based lookup
covers the zero-config launchers; launchers that need configuration (Docker
needs an image; SageMaker needs a role/image URI) are passed as instances::

    mlprobe.run(spec, run_dir, launcher="subprocess")
    mlprobe.run(spec, run_dir, launcher=DockerLauncher(image="my-img:latest"))
"""

from __future__ import annotations

from mlprobe.launchers.base import ProbeLauncher
from mlprobe.launchers.docker import DockerLauncher
from mlprobe.launchers.in_process import InProcessLauncher
from mlprobe.launchers.sagemaker import SagemakerLauncher
from mlprobe.launchers.subprocess import SubprocessLauncher

# Zero-config launchers resolvable by name.
_REGISTRY: dict[str, type] = {
    SubprocessLauncher.name: SubprocessLauncher,
    InProcessLauncher.name: InProcessLauncher,
}


def get_launcher(launcher: "str | ProbeLauncher") -> ProbeLauncher:
    """Resolve a launcher name or pass through an already-built launcher.

    Configured launchers (Docker, SageMaker) must be passed as instances —
    they can't be conjured from a bare name because they need an image, role,
    etc. A helpful error nudges toward that."""
    if isinstance(launcher, str):
        cls = _REGISTRY.get(launcher)
        if cls is None:
            if launcher in ("docker", "sagemaker"):
                raise ValueError(
                    f"launcher {launcher!r} needs configuration; pass an instance, "
                    f"e.g. launcher=DockerLauncher(image=...)"
                )
            raise ValueError(
                f"unknown launcher {launcher!r}; known: {sorted(_REGISTRY)} "
                f"(or pass a DockerLauncher/SagemakerLauncher instance)"
            )
        return cls()
    return launcher


__all__ = [
    "ProbeLauncher",
    "SubprocessLauncher",
    "InProcessLauncher",
    "DockerLauncher",
    "SagemakerLauncher",
    "get_launcher",
]
