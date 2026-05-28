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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from mlprobe.probe import ProbeInput, ProbeResult, _import_dotted, run_probe
from mlprobe.runtime import resolve_runtime
from mlprobe.spec import ModelSpec
from mlprobe.sweep import expand_sweeps
from mlprobe.targets import InstanceSpec, resolve

if TYPE_CHECKING:
    from mlprobe.protocol import EvalResult, RuntimeConfig


@dataclass(frozen=True)
class RunPlan:
    """Materialized list of probe inputs the runner intends to execute."""
    probes: list[ProbeInput]

    def __len__(self) -> int:
        return len(self.probes)


def write_spec(spec: ModelSpec, run_dir: Path, *, name: str = "spec") -> Path:
    """Dump a spec to ``run_dir/<name>.json`` and ensure the probes/ dir exists.

    Returns the spec path. Scenarios that probe multiple specs pass distinct
    ``name``s so each lands in its own file.
    """
    run_dir = Path(run_dir)
    (run_dir / "probes").mkdir(parents=True, exist_ok=True)
    spec_path = run_dir / f"{name}.json"
    spec_path.write_text(spec.model_dump_json(indent=2))
    return spec_path


def make_probe_input(
    *,
    spec_path: Path | str,
    config: dict,
    instance: InstanceSpec,
    runtime: "RuntimeConfig",
    subset_fraction: float,
    idx: int,
    run_dir: Path,
    repetition: int = 0,
    seed: int | None = None,
    resume_from: str | None = None,
) -> ProbeInput:
    """Build one ``ProbeInput`` with conventional output/result paths.

    The single source of truth for how a probe slot is laid out under
    ``run_dir`` — used by both ``plan_run`` (the sweep matrix) and the
    scenarios package (which vary their own axes)."""
    run_dir = Path(run_dir)
    return ProbeInput(
        spec_path=str(spec_path),
        config=config,
        instance_type=instance.instance_type,
        subset_fraction=subset_fraction,
        repetition=repetition,
        runtime=asdict(runtime),
        output_dir=str(run_dir / "probes" / f"probe-{idx:04d}.output"),
        result_path=str(run_dir / "probes" / f"probe-{idx:04d}.result.json"),
        seed=seed,
        resume_from=resume_from,
    )


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
    spec_path = write_spec(spec, run_dir)

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
                    probes.append(make_probe_input(
                        spec_path=spec_path,
                        config=config,
                        instance=instance,
                        runtime=runtime,
                        subset_fraction=subset_fraction,
                        idx=idx,
                        run_dir=run_dir,
                        repetition=repetition,
                        seed=_probe_seed(spec, idx),
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
    results = execute_probes(
        plan.probes,
        launcher=launcher,
        timeout=spec.probe.timeout_seconds,
        progress=progress,
    )
    return RunResults(plan=plan, results=results, run_dir=Path(run_dir))


def execute_probes(
    probes: list[ProbeInput],
    *,
    launcher: str = "subprocess",
    timeout: int = 1800,
    progress: bool = True,
) -> list[ProbeResult]:
    """Execute an explicit list of probes and collect their results.

    Decoupled from ``plan_run`` so callers that build their own probe lists
    (notably the scenarios package, which varies axes other than the sweep
    matrix) reuse the exact same launcher contract."""
    results: list[ProbeResult] = []
    for i, probe in enumerate(probes):
        if progress:
            _emit_progress(i, len(probes), probe)
        results.append(_launch(probe, launcher, timeout=timeout))
    return results


def evaluate_existing(
    spec: ModelSpec,
    artifact_path: str | Path,
    *,
    eval_set: Any = None,
    seed: int | None = None,
) -> "EvalResult":
    """Run ``evaluate()`` against an existing artifact, skipping ``train()``.

    The runner normally chains ``train() → evaluate()`` as one atomic probe.
    When training is expensive (a 45-minute SageMaker job) but you want to
    evaluate the resulting artifact against a gold standard — or re-evaluate it
    under several eval configurations — there is no reason to pay for training
    again. ``TrainResult.artifact_path`` is already the explicit handoff
    between the two steps; this entrypoint just starts from an artifact you
    already have.

    Parameters
    ----------
    spec : the ModelSpec — only its ``evaluate_callable`` and (for auto-loading
        the eval set) ``dataset`` are used here; no training is run.
    artifact_path : path to the trained artifact. A local path is passed to
        ``evaluate()`` as a ``Path``; a URI (``s3://...``, ``gs://...``) is
        passed through unchanged as a string for the user's loader to resolve.
    eval_set : the held-out set to evaluate on. If ``None``, it is materialized
        via the spec's ``LoadDatasetFn`` at full size on ``dataset.eval_split``.
    seed : forwarded to the loader when ``eval_set`` is auto-loaded.

    Returns the user's ``EvalResult`` — the same shape produced inside a full
    probe. Runs in-process: call it from the environment where your
    ``evaluate_callable`` and its dependencies are importable.
    """
    evaluate_fn = _import_dotted(spec.evaluate_callable)
    if eval_set is None:
        load_fn = _import_dotted(spec.dataset.loader)
        eval_set = load_fn(subset_fraction=1.0, split=spec.dataset.eval_split, seed=seed)
    return evaluate_fn(artifact_path=_as_artifact(artifact_path), eval_set=eval_set)


def _as_artifact(artifact_path: str | Path) -> str | Path:
    """Local paths become ``Path``; remote URIs stay strings so the user's
    loader (boto3, gcsfs, ...) can resolve them itself."""
    s = str(artifact_path)
    return s if "://" in s else Path(s)


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

    cmd = [sys.executable, "-m", "mlprobe.probe", str(input_path)]
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
