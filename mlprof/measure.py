"""External measurement of a training probe.

Measurement is *external* on purpose — mlprof stays framework-agnostic by
not asking torch/spacy/sklearn to self-report. A background sampler thread
polls ``psutil`` (CPU/RSS) and ``nvidia-ml-py`` (VRAM/util, if available)
while the user's ``train()`` runs in the foreground.

GPU instrumentation is optional. When ``pynvml`` is missing or fails to
init, ``Measurement.peak_vram_mb`` and ``gpu_util_avg`` stay ``None``.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import psutil


@dataclass
class Measurement:
    wall_clock_s: float = 0.0
    peak_rss_mb: float = 0.0           # this process + its children
    peak_vram_mb: float | None = None  # summed across visible GPUs
    gpu_util_avg: float | None = None  # 0-100, averaged across visible GPUs+samples
    samples_collected: int = 0
    gpu_devices: list[str] = field(default_factory=list)


_SAMPLE_INTERVAL_S = 0.1   # polling period for the sampler thread


@contextmanager
def measure() -> Iterator[Measurement]:
    """Context manager: ``with measure() as m: ...`` populates ``m`` on exit."""
    m = Measurement()
    proc = psutil.Process(os.getpid())
    nvml = _NvmlSampler.try_init()
    if nvml:
        m.gpu_devices = nvml.device_names

    stop = threading.Event()
    state = {
        "peak_rss": 0.0,
        "peak_vram": 0.0 if nvml else None,
        "util_sum": 0.0 if nvml else None,
        "samples": 0,
    }

    def sampler() -> None:
        while not stop.is_set():
            try:
                rss_mb = _process_tree_rss_mb(proc)
                if rss_mb > state["peak_rss"]:
                    state["peak_rss"] = rss_mb
                if nvml:
                    vram, util = nvml.sample()
                    if vram > state["peak_vram"]:
                        state["peak_vram"] = vram
                    state["util_sum"] += util
                state["samples"] += 1
            except Exception:
                # Don't let a transient sampling error kill the measurement;
                # the training run is what matters.
                pass
            stop.wait(_SAMPLE_INTERVAL_S)

    t = threading.Thread(target=sampler, daemon=True)
    t_start = time.monotonic()
    t.start()
    try:
        yield m
    finally:
        stop.set()
        t.join(timeout=2 * _SAMPLE_INTERVAL_S)
        m.wall_clock_s = time.monotonic() - t_start
        m.peak_rss_mb = state["peak_rss"]
        m.samples_collected = state["samples"]
        if nvml:
            m.peak_vram_mb = state["peak_vram"]
            if state["samples"] > 0:
                # util_sum is sum of (avg-util-across-devices) per sample
                m.gpu_util_avg = state["util_sum"] / state["samples"]
            nvml.shutdown()


def _process_tree_rss_mb(proc: psutil.Process) -> float:
    """RSS in MB for the process and its descendants. Subprocess-launched
    training (e.g. torchrun spawning workers) won't be undercounted."""
    total = 0
    try:
        total = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0
    return total / (1024 * 1024)


class _NvmlSampler:
    """Tiny pynvml wrapper. Returns (summed_vram_mb, avg_util_pct) per sample."""

    def __init__(self, pynvml, handles, device_names: list[str]):
        self._pynvml = pynvml
        self._handles = handles
        self.device_names = device_names

    @classmethod
    def try_init(cls) -> "_NvmlSampler | None":
        try:
            import pynvml  # type: ignore[import-not-found]
        except ImportError:
            return None
        try:
            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            if n == 0:
                pynvml.nvmlShutdown()
                return None
            handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
            names = [pynvml.nvmlDeviceGetName(h) for h in handles]
            names = [n.decode() if isinstance(n, bytes) else n for n in names]
            return cls(pynvml, handles, names)
        except Exception:
            return None

    def sample(self) -> tuple[float, float]:
        vram_mb = 0.0
        util_sum = 0.0
        for h in self._handles:
            mi = self._pynvml.nvmlDeviceGetMemoryInfo(h)
            ut = self._pynvml.nvmlDeviceGetUtilizationRates(h)
            vram_mb += mi.used / (1024 * 1024)
            util_sum += ut.gpu
        return vram_mb, util_sum / len(self._handles)

    def shutdown(self) -> None:
        try:
            self._pynvml.nvmlShutdown()
        except Exception:
            pass
