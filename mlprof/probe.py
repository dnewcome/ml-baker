"""The probe binary: a self-contained entry point that runs ONE training+eval.

Invocation::

    python -m mlprof.probe <probe_input.json>

Reads the input JSON, imports the user's train/evaluate/load_dataset by
dotted path, runs them wrapped in external measurement, and writes a result
JSON to the path specified in the input. Same binary works in three
launchers — local subprocess (today), Docker container, SageMaker training
job — because the contract is just "read input file, write result file".

The runner (``mlprof.runner``) is what produces probe_input.json files and
collects the resulting probe_result.json files. The probe itself does not
talk to the runner.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mlprof.measure import Measurement, measure
from mlprof.protocol import RuntimeConfig
from mlprof.spec import ModelSpec
from mlprof.targets import resolve


@dataclass
class ProbeInput:
    """Everything one probe needs to execute. Serializable to JSON."""
    spec_path: str                   # path to spec.json (pydantic dump)
    config: dict[str, Any]           # resolved hyperparams for this run
    instance_type: str               # target instance (cost lookup)
    subset_fraction: float           # passed to LoadDatasetFn
    repetition: int                  # noise-reduction replicate index
    runtime: dict[str, Any]          # RuntimeConfig dump
    output_dir: str                  # where train() writes its artifact
    result_path: str                 # where THIS probe writes its result JSON
    resume_from: str | None = None   # for incremental-train probes
    seed: int | None = None          # passed to LoadDatasetFn for repeatability


@dataclass
class ProbeResult:
    """What one probe records. Serializable to JSON."""
    config: dict[str, Any]
    instance_type: str
    subset_fraction: float
    repetition: int

    # External measurements. None if probe failed before measure() exited.
    wall_clock_s: float | None = None
    peak_rss_mb: float | None = None
    peak_vram_mb: float | None = None
    gpu_util_avg: float | None = None

    # Outcomes
    train_metrics: dict[str, float] = field(default_factory=dict)
    eval_metrics: dict[str, float] = field(default_factory=dict)
    cost_usd: float | None = None

    # Failure mode (None on success)
    error: str | None = None
    traceback: str | None = None


def run_probe(probe_input: ProbeInput) -> ProbeResult:
    """In-process probe execution. The CLI just deserializes and calls this —
    keeping it as a function makes it directly testable without a subprocess."""

    spec = ModelSpec.model_validate_json(Path(probe_input.spec_path).read_text())
    runtime = RuntimeConfig(**probe_input.runtime)

    result = ProbeResult(
        config=probe_input.config,
        instance_type=probe_input.instance_type,
        subset_fraction=probe_input.subset_fraction,
        repetition=probe_input.repetition,
    )

    try:
        train_fn = _import_dotted(spec.train_callable)
        evaluate_fn = _import_dotted(spec.evaluate_callable)
        load_fn = _import_dotted(spec.dataset.loader)

        output_dir = Path(probe_input.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset_subset = load_fn(
            subset_fraction=probe_input.subset_fraction,
            split=None,
            seed=probe_input.seed,
        )
        eval_set = load_fn(
            subset_fraction=1.0,
            split=spec.dataset.eval_split,
            seed=probe_input.seed,
        )

        with measure() as m:
            train_result = train_fn(
                config=probe_input.config,
                dataset_subset=dataset_subset,
                output_dir=output_dir,
                resume_from=Path(probe_input.resume_from) if probe_input.resume_from else None,
                runtime=runtime,
            )

        eval_result = evaluate_fn(
            artifact_path=Path(train_result.artifact_path),
            eval_set=eval_set,
        )

        _populate_measurements(result, m)
        result.train_metrics = dict(train_result.metrics)
        result.eval_metrics = _filter_eval_metrics(spec, eval_result.metrics)
        result.cost_usd = _compute_cost(probe_input.instance_type, m.wall_clock_s)

    except Exception as e:  # noqa: BLE001 — we want all failures captured
        result.error = f"{type(e).__name__}: {e}"
        result.traceback = traceback.format_exc()

    return result


def _populate_measurements(result: ProbeResult, m: Measurement) -> None:
    result.wall_clock_s = m.wall_clock_s
    result.peak_rss_mb = m.peak_rss_mb
    result.peak_vram_mb = m.peak_vram_mb
    result.gpu_util_avg = m.gpu_util_avg


def _filter_eval_metrics(spec: ModelSpec, metrics: dict[str, float]) -> dict[str, float]:
    declared = {m.name for m in spec.eval_metrics}
    return {k: v for k, v in metrics.items() if k in declared}


def _compute_cost(instance_type: str, wall_clock_s: float) -> float:
    """Cost = wall_clock_hours * on-demand price. Approximate catalog prices
    for now; AWS Pricing API integration is a planned follow-up."""
    inst = resolve(instance_type)
    return (wall_clock_s / 3600.0) * inst.on_demand_usd_per_hour


def _import_dotted(path: str) -> Any:
    """Resolve a 'pkg.module:attr' (or 'pkg.module.attr') reference."""
    if ":" in path:
        module_name, attr = path.split(":", 1)
    else:
        module_name, _, attr = path.rpartition(".")
        if not module_name:
            raise ValueError(f"cannot import dotted path {path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _cli() -> int:
    parser = argparse.ArgumentParser(prog="mlprof.probe")
    parser.add_argument("input_path", type=Path, help="path to probe input JSON")
    args = parser.parse_args()

    data = json.loads(args.input_path.read_text())
    probe_input = ProbeInput(**data)
    result = run_probe(probe_input)
    Path(probe_input.result_path).write_text(json.dumps(asdict(result), indent=2))
    return 0 if result.error is None else 2


if __name__ == "__main__":
    sys.exit(_cli())
