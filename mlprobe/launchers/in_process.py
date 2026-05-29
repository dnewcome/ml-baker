"""In-process launcher — runs the probe in the current process. Debug/testing
only: no isolation, the user's framework is imported into the runner's
interpreter. Convenient for tests because there is no subprocess to coordinate.
"""

from __future__ import annotations

from mlprobe.probe import ProbeInput, ProbeResult, run_probe


class InProcessLauncher:
    name = "in_process"

    def launch(self, probe: ProbeInput, *, timeout: int) -> ProbeResult:
        # timeout is unenforceable in-process (no subprocess to kill); accepted
        # for interface parity and ignored.
        return run_probe(probe)
