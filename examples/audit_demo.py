"""Audit-only demo: spec → audit + sweep expansion + runtime resolution per target.
Runs in milliseconds with no probe execution — useful as a quick sanity check
and as a reference for what the audit output looks like."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlprobe import (
    ModelSpec, DatasetSpec, EvalMetric, HyperParam,
    Capabilities, FrameworkHints, TargetInstance, ProbeConfig,
    NumericSweep, CategoricalSweep,
    audit, expand_sweeps, resolve, resolve_runtime,
)


# Pretend this is a notebook script someone just dragged in: minimal
# capabilities, a couple of targets, a small sweep.
spec = ModelSpec(
    name="intent-classifier-distilbert",
    train_callable="my_project.training:train",
    evaluate_callable="my_project.training:evaluate",
    dataset=DatasetSpec(loader="my_project.data:load", total_size=1_200_000),
    eval_metrics=[
        EvalMetric(name="f1_macro", higher_is_better=True, primary=True),
        EvalMetric(name="inference_latency_ms", higher_is_better=False),
    ],
    hyperparameters=[
        HyperParam(name="lr", type="float", default=2e-5, min=1e-6, max=1e-2,
                   sweep=NumericSweep(min=1e-5, max=1e-3, log_step=10)),
        HyperParam(name="batch_size", type="int", default=32, min=8, max=128,
                   sweep=NumericSweep(min=8, max=64, step=8)),
        HyperParam(name="encoder", type="categorical", default="distilbert",
                   values=["distilbert", "minilm", "tfidf+logreg"],
                   sweep=CategoricalSweep(values=["distilbert", "minilm"])),
    ],
    capabilities=Capabilities(
        # Deliberately bare — simulating naive notebook code.
        supports_checkpointing=False,
        supports_incremental_training=False,
        supports_parallel_data_loading=False,
        supports_multi_gpu_data_parallel=False,
        supports_mixed_precision=False,
        deterministic=False,
    ),
    framework_hints=FrameworkHints(
        framework="pytorch", expected_device="gpu",
        requires_gpu=True, min_vram_gb=20,
    ),
    targets=[
        TargetInstance(instance_type="ml.g5.xlarge"),     # 1x A10G 24GB
        TargetInstance(instance_type="g5.12xlarge"),      # 4x A10G - idle GPUs incoming
        TargetInstance(instance_type="p3.2xlarge"),       # V100 16GB - VRAM warning incoming
        TargetInstance(instance_type="c5.4xlarge"),       # CPU - hard incompatibility
    ],
    probe=ProbeConfig(subset_fractions=[0.005, 0.02, 0.08], max_variants=12),
)


print("=" * 70)
print("AUDIT")
print("=" * 70)
report = audit(spec)
print(report.format())

print()
print("=" * 70)
print("SWEEP EXPANSION (capped at max_variants=12)")
print("=" * 70)
configs = expand_sweeps(spec)
print(f"{len(configs)} variants (cartesian would be 3 lr × 8 batch × 2 encoder = 48)")
for i, c in enumerate(configs[:5]):
    print(f"  [{i}] {c}")
print(f"  ... ({len(configs) - 5} more)")

print()
print("=" * 70)
print("RUNTIME RESOLUTION per target")
print("=" * 70)
# Flip on more capabilities to show the runtime resolver gating
spec2 = spec.model_copy(deep=True)
spec2.capabilities.supports_multi_gpu_data_parallel = True
spec2.capabilities.supports_mixed_precision = True
spec2.capabilities.max_useful_gpus = 4
for t in spec.targets:
    try:
        inst = resolve(t.instance_type)
    except KeyError:
        print(f"  {t.instance_type}: unknown")
        continue
    rt = resolve_runtime(spec2.capabilities, inst)
    print(f"  {t.instance_type:18s} → n_gpus={rt.n_gpus} n_cpus={rt.n_cpus} "
          f"precision={rt.precision} strategy={rt.multi_gpu_strategy}")
