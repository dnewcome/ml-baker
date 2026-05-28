"""End-to-end AG News demo across two architectures.

Two ModelSpecs, two architectures, multiple target instances:

  1. sklearn TF-IDF + LogReg  — cheap CPU baseline, ~92% acc
  2. DistilBERT fine-tune      — heavier, GPU-friendly, ~94-95% acc

Each runs through audit + probes + report. The combined output makes the
mlprobe value props concrete: the audit flags wasted-GPU-$ on the sklearn
spec's g5 target and flags idle-GPU-$ on DistilBERT's g5.12xlarge target;
the Pareto frontier across both architectures shows where the cost/quality
tradeoff actually lives.

Usage:

    pip install -e ".[demo]"
    python examples/agnews/demo.py

First run downloads AG News (~10MB) and distilbert-base-uncased (~250MB)
into the HuggingFace cache.

Expected runtime:
  - sklearn variant (5 probes): ~30s-2min on CPU.
  - DistilBERT variant (4 probes): ~3-5min on GPU (g5/A10G class);
    ~30-60min on CPU. If running CPU-only, consider commenting out the
    DistilBERT call in main() or lowering ``subset_fractions``.

IMPORTANT CAVEAT: probes run as local subprocesses, so wall-clock is
measured on YOUR machine. Cost numbers apply that throughput to the
target instance's $/hr. The relative ordering across architectures and
configs is faithful; absolute "what would this cost on a real g5.xlarge"
needs the Docker launcher (issue #1) or SageMaker.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mlprobe import (
    Capabilities,
    DatasetSpec,
    EvalMetric,
    FrameworkHints,
    HyperParam,
    ModelSpec,
    ProbeConfig,
    TargetInstance,
    audit,
    build_report,
    run,
)


AGNEWS_TRAIN_ROWS = 120_000


def sklearn_spec() -> ModelSpec:
    """sklearn TF-IDF + LogReg. CPU-bound by design.

    Targets include a GPU instance specifically to demonstrate the
    ``cpu_bound_on_gpu_target`` audit warning — the probe will still run
    (sklearn ignores the GPU) but the audit calls out that you'd be paying
    GPU prices for CPU work.
    """
    return ModelSpec(
        name="agnews-sklearn-tfidf-logreg",
        train_callable="examples.agnews.sklearn_logreg:train",
        evaluate_callable="examples.agnews.sklearn_logreg:evaluate",
        dataset=DatasetSpec(
            loader="examples.agnews.data:load",
            total_size=AGNEWS_TRAIN_ROWS,
            eval_split="test",
        ),
        eval_metrics=[
            EvalMetric(name="accuracy", higher_is_better=True, primary=True),
            EvalMetric(name="f1_macro", higher_is_better=True),
        ],
        hyperparameters=[
            HyperParam(name="C", type="float", default=1.0, min=0.01, max=10.0),
            HyperParam(name="max_features", type="int", default=20000,
                       min=1000, max=100000),
        ],
        capabilities=Capabilities(
            deterministic=True,
            supports_parallel_data_loading=True,  # sklearn n_jobs covers this
            # Intentionally bare on GPU/checkpointing — sklearn doesn't do
            # either. The audit's INFO findings (no_incremental_training,
            # no_mixed_precision, etc.) are accurate, not problems to fix.
        ),
        framework_hints=FrameworkHints(
            framework="sklearn",
            expected_device="cpu",
            cpu_bound=True,
        ),
        targets=[
            TargetInstance(instance_type="c5.4xlarge"),
            TargetInstance(instance_type="g5.xlarge"),   # surfaces cpu_bound_on_gpu_target
        ],
        probe=ProbeConfig(
            subset_fractions=[0.02, 0.1, 0.4],
            timeout_seconds=600,
        ),
    )


def distilbert_spec() -> ModelSpec:
    """DistilBERT fine-tune. GPU strongly preferred but not strictly required.

    Targets include g5.12xlarge specifically to demonstrate the
    ``exceeds_max_useful_gpus`` warning (4-GPU instance vs ``max_useful_gpus=2``
    — three GPUs would actually be paid for, only two would be useful).
    """
    return ModelSpec(
        name="agnews-distilbert-finetune",
        train_callable="examples.agnews.distilbert:train",
        evaluate_callable="examples.agnews.distilbert:evaluate",
        dataset=DatasetSpec(
            loader="examples.agnews.data:load",
            total_size=AGNEWS_TRAIN_ROWS,
            eval_split="test",
        ),
        eval_metrics=[
            EvalMetric(name="accuracy", higher_is_better=True, primary=True),
            EvalMetric(name="f1_macro", higher_is_better=True),
        ],
        hyperparameters=[
            HyperParam(name="lr", type="float", default=5e-5, min=1e-6, max=1e-3),
            HyperParam(name="batch_size", type="int", default=16, min=4, max=64),
            HyperParam(name="epochs", type="int", default=1, min=1, max=5),
        ],
        capabilities=Capabilities(
            supports_checkpointing=True,
            supports_parallel_data_loading=True,
            supports_multi_gpu_data_parallel=True,
            max_useful_gpus=2,   # honest: DDP on AG News + DistilBERT plateaus fast
            supports_mixed_precision=True,
            deterministic=True,
        ),
        framework_hints=FrameworkHints(
            framework="pytorch",
            expected_device="gpu",
            requires_gpu=False,   # technically possible on CPU, just very slow
            min_vram_gb=6,
        ),
        targets=[
            TargetInstance(instance_type="g5.xlarge"),     # 1x A10G; sweet spot
            TargetInstance(instance_type="g5.12xlarge"),   # 4x A10G; 2 wasted
        ],
        probe=ProbeConfig(
            subset_fractions=[0.002, 0.01, 0.04],
            timeout_seconds=1800,
        ),
    )


def run_one(spec: ModelSpec) -> None:
    print()
    print("=" * 78)
    print(f"DEMO: {spec.name}")
    print("=" * 78)
    pre = audit(spec)
    print("AUDIT:")
    print(pre.format())
    print()

    with tempfile.TemporaryDirectory(prefix="mlbaker-demo-") as td:
        results = run(spec, run_dir=Path(td) / "run",
                      launcher="subprocess", progress=True)
        report = build_report(spec, results, pre)
        print()
        print(report.format())


def main(argv: list[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="AG News demo for mlprobe.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sklearn-only", action="store_true",
                       help="Run only the cheap sklearn variant (~30s-2min).")
    group.add_argument("--distilbert-only", action="store_true",
                       help="Run only the DistilBERT variant (slow on CPU).")
    opts = parser.parse_args(argv)

    if not opts.distilbert_only:
        run_one(sklearn_spec())
    if not opts.sklearn_only:
        run_one(distilbert_spec())


if __name__ == "__main__":
    main()
