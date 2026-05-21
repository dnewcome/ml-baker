"""End-to-end demo: spec → audit → probe run → report.

Runs probes in subprocess mode against the synthetic ``fake_trainable``,
fits scaling curves, and prints the combined report. Designed to finish in
a handful of seconds on a laptop.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Make repo root importable so `examples.fake_trainable` resolves.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mlprof import (
    Capabilities,
    CategoricalSweep,
    DatasetSpec,
    EvalMetric,
    FrameworkHints,
    HyperParam,
    ModelSpec,
    NumericSweep,
    ProbeConfig,
    TargetInstance,
    audit,
    build_report,
    run,
)


spec = ModelSpec(
    name="fake-classifier-demo",
    train_callable="examples.fake_trainable:train",
    evaluate_callable="examples.fake_trainable:evaluate",
    dataset=DatasetSpec(loader="examples.fake_trainable:load", total_size=50_000),
    eval_metrics=[
        EvalMetric(name="f1_macro", higher_is_better=True, primary=True),
        EvalMetric(name="inference_latency_ms", higher_is_better=False),
    ],
    hyperparameters=[
        HyperParam(name="lr", type="float", default=1e-3, min=1e-5, max=1e-1),
        HyperParam(name="batch_size", type="int", default=32, min=8, max=128),
        HyperParam(name="epochs", type="int", default=1, min=1, max=10),
        HyperParam(name="encoder", type="categorical", default="small",
                   values=["small", "medium", "large"],
                   sweep=CategoricalSweep(values=["small", "medium", "large"])),
    ],
    capabilities=Capabilities(
        supports_checkpointing=True,
        supports_parallel_data_loading=True,
        supports_mixed_precision=True,
        supports_multi_gpu_data_parallel=True,
        max_useful_gpus=4,
        deterministic=True,
    ),
    framework_hints=FrameworkHints(framework="pytorch", expected_device="gpu"),
    targets=[
        TargetInstance(instance_type="g5.xlarge"),    # 1x A10G
        TargetInstance(instance_type="g5.12xlarge"),  # 4x A10G
    ],
    probe=ProbeConfig(
        subset_fractions=[0.02, 0.1, 0.4],
        repetitions=1,
        timeout_seconds=60,
        max_variants=3,
    ),
)


def main() -> None:
    report_pre = audit(spec)

    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td) / "run"
        results = run(spec, run_dir=run_dir, launcher="subprocess", progress=True)
        report = build_report(spec, results, report_pre)
        print()
        print(report.format())


if __name__ == "__main__":
    main()
