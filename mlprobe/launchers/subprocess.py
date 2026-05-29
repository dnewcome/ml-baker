"""Subprocess launcher (default) — runs the probe in a fresh Python process.

This is the reference implementation of the launch contract: write the input
JSON, invoke ``python -m mlprobe.probe <input.json>``, read the result JSON the
probe wrote. The Docker launcher runs the *same* command inside a container.
"""

from __future__ import annotations

import subprocess
import sys

from mlprobe.launchers.base import (
    PROBE_MODULE,
    error_result,
    result_or_error,
    write_probe_input,
)
from mlprobe.probe import ProbeInput, ProbeResult


class SubprocessLauncher:
    name = "subprocess"

    def __init__(self, python_executable: str | None = None):
        self.python = python_executable or sys.executable

    def launch(self, probe: ProbeInput, *, timeout: int) -> ProbeResult:
        input_path = write_probe_input(probe)
        cmd = [self.python, "-m", PROBE_MODULE, str(input_path)]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=timeout, text=True)
        except subprocess.TimeoutExpired:
            return error_result(probe, "TimeoutExpired", f"probe exceeded {timeout}s")
        return result_or_error(probe, proc.returncode, proc.stderr)
