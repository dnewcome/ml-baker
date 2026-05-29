"""Docker launcher — run the probe binary inside a container.

The probe contract is unchanged: ``python -m mlprobe.probe <input.json>``. The
only additions are a bind mount and a ``docker run`` wrapper. The directory
holding the probe's input/spec/output/result files is mounted **at the same
absolute path** inside the container, so the absolute paths already baked into
the input JSON resolve identically in and out of the container — no path
rewriting, and the result file the probe writes appears back on the host.

The image must have mlprobe **and the user's train/evaluate/load code**
importable (the probe imports the callables by dotted path). Point ``image`` at
a build that satisfies that; see ``examples/docker/`` for reference Dockerfiles.

GPU caveat (from the issue): ``--gpus`` exercises the *host's* GPUs, not the
target instance's. Useful for build-time correctness checks; true cost/perf
measurement on the target hardware needs the SageMaker launcher (#24).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Callable

from mlprobe.launchers.base import (
    PROBE_MODULE,
    error_result,
    result_or_error,
    write_probe_input,
)
from mlprobe.probe import ProbeInput, ProbeResult
from mlprobe.targets import resolve


class DockerLauncher:
    """Run probes via ``docker run``.

    Parameters
    ----------
    image : default container image (must have mlprobe + user code importable).
    gpus : ``"auto"`` (default) adds ``--gpus all`` when the resolved target has
        GPUs; ``True`` forces it, ``False`` never, or pass a literal string like
        ``"device=0"`` for ``--gpus <str>``.
    mounts : extra bind mounts as ``(host, container, mode)`` tuples — e.g. the
        repo so ``examples.*`` resolves, or a model cache.
    env : environment variables to pass through with ``-e``.
    workdir : container working directory (``-w``).
    image_for : optional ``instance_type -> image`` override for per-target images.
    docker_bin : the docker CLI (default ``"docker"``).
    extra_run_args : appended to ``docker run`` verbatim (escape hatch).
    """

    name = "docker"

    def __init__(
        self,
        image: str,
        *,
        gpus: "str | bool" = "auto",
        mounts: list[tuple[str, str, str]] | None = None,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
        image_for: Callable[[str], str] | None = None,
        docker_bin: str = "docker",
        extra_run_args: list[str] | None = None,
    ):
        self.image = image
        self.gpus = gpus
        self.mounts = list(mounts or [])
        self.env = dict(env or {})
        self.workdir = workdir
        self.image_for = image_for
        self.docker_bin = docker_bin
        self.extra_run_args = list(extra_run_args or [])

    # -- image / gpu resolution --

    def _image(self, instance_type: str) -> str:
        return self.image_for(instance_type) if self.image_for else self.image

    def _gpu_args(self, instance_type: str) -> list[str]:
        if self.gpus == "auto":
            try:
                has_gpu = resolve(instance_type).gpu_count > 0
            except KeyError:
                has_gpu = False
            return ["--gpus", "all"] if has_gpu else []
        if self.gpus is True:
            return ["--gpus", "all"]
        if isinstance(self.gpus, str):
            return ["--gpus", self.gpus]
        return []   # gpus is False

    # -- command construction (pure; unit-tested without a daemon) --

    def build_run_command(self, probe: ProbeInput, input_path: Path, *, timeout: int) -> list[str]:
        """Build the full ``docker run ...`` argv for a probe. Pure function of
        its inputs, so tests can assert the command without a Docker daemon."""
        mount_root = _mount_root(probe, input_path)
        argv = [
            self.docker_bin, "run", "--rm",
            "--name", _container_name(probe),
            "--stop-timeout", str(timeout),
            "-v", f"{mount_root}:{mount_root}",
        ]
        for host, container, mode in self.mounts:
            argv += ["-v", f"{host}:{container}:{mode}"]
        argv += self._gpu_args(probe.instance_type)
        if self.workdir:
            argv += ["-w", self.workdir]
        for k, v in self.env.items():
            argv += ["-e", f"{k}={v}"]
        argv += self.extra_run_args
        argv += [self._image(probe.instance_type)]
        argv += ["python", "-m", PROBE_MODULE, str(input_path)]
        return argv

    # -- execution --

    def launch(self, probe: ProbeInput, *, timeout: int) -> ProbeResult:
        input_path = write_probe_input(probe)
        argv = self.build_run_command(probe, input_path, timeout=timeout)
        try:
            proc = subprocess.run(argv, capture_output=True, timeout=timeout, text=True)
        except subprocess.TimeoutExpired:
            _kill_container(self.docker_bin, _container_name(probe))
            return error_result(probe, "TimeoutExpired", f"probe exceeded {timeout}s")
        except FileNotFoundError:
            return error_result(
                probe, "DockerNotFound",
                f"docker binary {self.docker_bin!r} not found on PATH",
            )
        return result_or_error(probe, proc.returncode, proc.stderr)


def _mount_root(probe: ProbeInput, input_path: Path) -> str:
    """The directory to bind-mount: the common ancestor of every path the probe
    touches (spec, input, output dir, result). Mounting it at the same absolute
    path keeps the JSON's paths valid inside the container."""
    paths = [
        str(Path(probe.spec_path).resolve()),
        str(Path(input_path).resolve()),
        str(Path(probe.output_dir).resolve()),
        str(Path(probe.result_path).resolve()),
    ]
    return os.path.commonpath(paths)


def _container_name(probe: ProbeInput) -> str:
    """Stable, filesystem-derived container name so a timed-out run can be killed."""
    stem = Path(probe.result_path).name.replace(".result.json", "").replace(".", "-")
    return f"mlprobe-{stem or 'probe'}"


def _kill_container(docker_bin: str, name: str) -> None:
    try:
        subprocess.run([docker_bin, "kill", name], capture_output=True, timeout=30)
    except Exception:
        pass   # best-effort cleanup; the --rm container may already be gone
