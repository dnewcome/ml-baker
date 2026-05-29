"""The ProbeLauncher protocol + shared launch helpers.

A *launcher* takes one ``ProbeInput`` and returns one ``ProbeResult``. Where the
probe actually runs is the only thing that varies:

  - ``InProcessLauncher`` — same Python process (debug; no isolation).
  - ``SubprocessLauncher`` — a fresh ``python -m mlprobe.probe`` subprocess (default).
  - ``DockerLauncher``     — the same command inside a container.
  - ``SagemakerLauncher``  — a SageMaker training job (scaffold).

They are interchangeable because the probe binary's contract is just *read an
input JSON, write a result JSON*. The subprocess and Docker launchers literally
run the same command (``python -m mlprobe.probe <input.json>``); Docker only
adds a mount + ``docker run`` wrapper. That shared contract is what makes the
subprocess↔docker parity test meaningful.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol, runtime_checkable

from mlprobe.probe import ProbeInput, ProbeResult


@runtime_checkable
class ProbeLauncher(Protocol):
    """Runs one probe. ``name`` is the registry key (e.g. ``"docker"``)."""

    name: str

    def launch(self, probe: ProbeInput, *, timeout: int) -> ProbeResult: ...


# ---- Shared helpers (used by subprocess + docker; same on-disk contract) ----

def write_probe_input(probe: ProbeInput) -> Path:
    """Serialize a probe to its ``<result>.input.json`` next to where its result
    will land. Both the subprocess and Docker launchers feed the probe binary
    from this file."""
    input_path = Path(probe.result_path).with_suffix(".input.json")
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(json.dumps(asdict(probe), indent=2))
    return input_path


def load_result(path: Path) -> ProbeResult:
    return ProbeResult(**json.loads(Path(path).read_text()))


def error_result(probe: ProbeInput, error: str, traceback: str | None = None) -> ProbeResult:
    """A ProbeResult carrying just the probe's identity + a failure reason, for
    when the probe never wrote its own result (crash, timeout, launch failure)."""
    return ProbeResult(
        config=probe.config,
        instance_type=probe.instance_type,
        subset_fraction=probe.subset_fraction,
        repetition=probe.repetition,
        error=error,
        traceback=traceback,
    )


def result_or_error(probe: ProbeInput, returncode: int, stderr: str | None) -> ProbeResult:
    """Read the probe's result file if it was written; otherwise synthesize an
    error record so the runner never loses track of a probe slot."""
    result_path = Path(probe.result_path)
    if result_path.exists():
        return load_result(result_path)
    return error_result(
        probe,
        error=f"probe exited {returncode} without writing result",
        traceback=(stderr or "")[-4000:],
    )


PROBE_MODULE = "mlprobe.probe"   # invoked as: python -m mlprobe.probe <input.json>
