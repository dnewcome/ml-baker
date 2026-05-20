"""Orchestrates probe execution.

The runner plans the probe matrix (sweep × subset × target × repetition),
launches each probe as a subprocess (today; Docker/SageMaker tomorrow), and
collects the resulting ``ProbeResult`` records.

A probe is a single training run. The matrix is intentionally explicit and
flat — it is easy to inspect ahead of time, easy to filter, and the same
shape is what would be enqueued for a remote launcher.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from ml_baker.probe import ProbeInput, ProbeResult, run_probe
from ml_baker.runtime import resolve_runtime
from ml_baker.spec import ModelSpec
from ml_baker.sweep import expand_sweeps
from ml_baker.targets import resolve


@dataclass(frozen=True)
class RunPlan:
    """Materialized list of probe inputs the runner intends to execute."""
    probes: list[ProbeInput]

    def __len__(self) -> int:
        return len(self.probes)


def plan_run(spec: ModelSpec, run_dir: Path) -> RunPlan:
    """Expand a spec into the full probe matrix.

    Layout under ``run_dir``::

        run_dir/
          spec.json
          probes/
            probe-0000.input.json
            probe-0000.result.json   (written by probe)
            probe-0000.output/        (artifact directory for that probe)
            ...
    """
    run_dir = Path(run_dir)
    (run_dir / "probes").mkdir(parents=True, exist_ok=True)

    spec_path = run_dir / "spec.json"
    spec_path.write_text(spec.model_dump_json(indent=2))

    configs = expand_sweeps(spec)
    probes: list[ProbeInput] = []
    idx = 0
    for target in spec.targets:
        # Resolve here so the audit's instance-unknown findings have already
        # surfaced upstream; a planning failure here is a programmer error.
        instance = resolve(target.instance_type)
        runtime = resolve_runtime(spec.capabilities, instance)
        for config in configs:
            for subset_fraction in spec.probe.subset_fractions:
                for repetition in range(spec.probe.repetitions):
                    seed = _probe_seed(spec, idx)
                    probes.append(ProbeInput(
                        spec_path=str(spec_path),
                        config=config,
                        instance_type=instance.instance_type,
                        subset_fraction=subset_fraction,
                        repetition=repetition,
                        runtime=asdict(runtime),
                        output_dir=str(run_dir / "probes" / f"probe-{idx:04d}.output"),
                        result_path=str(run_dir / "probes" / f"probe-{idx:04d}.result.json"),
                        seed=seed,
                    ))
                    idx += 1
    return RunPlan(probes=probes)


def _probe_seed(spec: ModelSpec, probe_idx: int) -> int | None:
    """Deterministic per-probe seed only when the spec asks for determinism;
    otherwise None so the user's loader can pick its own randomness."""
    if not spec.capabilities.deterministic:
        return None
    return 1337 + probe_idx


# ---- Execution -----------------------------------------------------------

@dataclass
class RunResults:
    """Aggregated probe outcomes."""
    plan: RunPlan
    results: list[ProbeResult]
    run_dir: Path

    @property
    def succeeded(self) -> list[ProbeResult]:
        return [r for r in self.results if r.error is None]

    @property
    def failed(self) -> list[ProbeResult]:
        return [r for r in self.results if r.error is not None]


def run(
    spec: ModelSpec,
    run_dir: Path,
    *,
    launcher: str = "subprocess",
    progress: bool = True,
) -> RunResults:
    """Plan + execute. ``launcher`` is currently ``"subprocess"`` or
    ``"in_process"``; ``"docker"`` will join the family later."""
    plan = plan_run(spec, run_dir)
    results: list[ProbeResult] = []
    for i, probe in enumerate(plan.probes):
        if progress:
            _emit_progress(i, len(plan.probes), probe)
        result = _launch(probe, launcher, timeout=spec.probe.timeout_seconds)
        results.append(result)
    return RunResults(plan=plan, results=results, run_dir=Path(run_dir))


def _launch(probe: ProbeInput, launcher: str, *, timeout: int) -> ProbeResult:
    if launcher == "in_process":
        return run_probe(probe)
    if launcher == "subprocess":
        return _launch_subprocess(probe, timeout=timeout)
    raise ValueError(f"unknown launcher {launcher!r}")


def _launch_subprocess(probe: ProbeInput, *, timeout: int) -> ProbeResult:
    """Run the probe in a fresh Python subprocess. Identical contract to
    what a Docker launcher will use — mount input file, invoke the probe
    binary, read the result file."""
    input_path = Path(probe.result_path).with_suffix(".input.json")
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(json.dumps(asdict(probe), indent=2))

    cmd = [sys.executable, "-m", "ml_baker.probe", str(input_path)]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return ProbeResult(
            config=probe.config,
            instance_type=probe.instance_type,
            subset_fraction=probe.subset_fraction,
            repetition=probe.repetition,
            error="TimeoutExpired",
            traceback=f"probe exceeded {timeout}s",
        )

    result_path = Path(probe.result_path)
    if result_path.exists():
        return _load_result(result_path)

    # Probe crashed before writing result — synthesize an error record so the
    # runner doesn't lose track of this slot.
    return ProbeResult(
        config=probe.config,
        instance_type=probe.instance_type,
        subset_fraction=probe.subset_fraction,
        repetition=probe.repetition,
        error=f"probe exited {proc.returncode} without writing result",
        traceback=(proc.stderr or "")[-4000:],
    )


def _load_result(path: Path) -> ProbeResult:
    data = json.loads(path.read_text())
    return ProbeResult(**data)


def _emit_progress(i: int, total: int, probe: ProbeInput) -> None:
    print(
        f"[{i + 1}/{total}] {probe.instance_type} "
        f"subset={probe.subset_fraction:.3f} "
        f"rep={probe.repetition} "
        f"config={probe.config}",
        flush=True,
    )


def iter_probe_records(results: RunResults) -> Iterator[dict]:
    """Convenience: flat dict-of-scalars per probe, handy for tabular dumps."""
    for r in results.results:
        base = {
            "instance_type": r.instance_type,
            "subset_fraction": r.subset_fraction,
            "repetition": r.repetition,
            "wall_clock_s": r.wall_clock_s,
            "peak_rss_mb": r.peak_rss_mb,
            "peak_vram_mb": r.peak_vram_mb,
            "gpu_util_avg": r.gpu_util_avg,
            "cost_usd": r.cost_usd,
            "error": r.error,
        }
        for k, v in r.config.items():
            base[f"cfg.{k}"] = v
        for k, v in r.eval_metrics.items():
            base[f"eval.{k}"] = v
        yield base
