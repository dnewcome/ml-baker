# mlprof

> Estimate the cost, time, and quality of an ML training run *before* you
> commit to it — and audit the training code for production-readiness while
> you're at it.

mlprof is a framework-agnostic profiler for ML training workloads. You
describe your model (libraries used, hyperparameters, eval metrics,
candidate target instances), point at your `train()` / `evaluate()` /
`load_dataset()` functions, and mlprof:

1. **Audits** the code's production-readiness — calls out missing
   checkpointing, single-process data loading, idle GPUs on multi-GPU
   instances, VRAM overshoot, framework/target mismatches — *before*
   spending any compute.
2. **Probes** the model on small subsets of the data across the configured
   targets, measuring wall-clock, RAM, VRAM, GPU utilization, and cost
   externally (no framework instrumentation needed).
3. **Fits scaling curves** so you can predict full-dataset training time and
   quality from cheap small-N probes.
4. **Emits a Pareto frontier** over `(cost, quality)` so the cost-vs-quality
   tradeoff across different instance types and hyperparameter choices is
   explicit, not buried in raw numbers.

It is **framework-agnostic** by design — you can use PyTorch, spaCy,
sklearn, HuggingFace, or anything else. mlprof never imports your
framework; you implement three small callables and mlprof orchestrates.

## Status

Early. The core spec/audit/probe/report pipeline works end-to-end (see
`examples/probe_demo.py`). Probes run locally as subprocesses; Docker /
SageMaker launchers are on the roadmap. See [open issues](#roadmap) for
what is planned next.

## Installation

```bash
git clone https://github.com/dnewcome/mlprof.git
cd mlprof
pip install -e .
```

Optional GPU measurement (nvidia-ml-py for VRAM/util sampling):

```bash
pip install -e ".[gpu]"
```

Requires Python 3.10+.

## The problem

You have an ML model that came out of a Jupyter notebook or a naive
training script. You need to deploy it to a real training pipeline — likely
SageMaker, with a half-dozen possible instance types ranging from
`ml.c5.4xlarge` (CPU) to `ml.p4d.24xlarge` (8x A100). A few questions you
*can't* answer cheaply:

- Will the training even fit in VRAM? At what batch size?
- Will it actually use all 8 GPUs, or will 7 sit idle paying $30/hr?
- How long will the full dataset take to train? How much will it cost?
- Is the extra cost of the bigger model worth the quality gain?
- Is the code production-ready (checkpointing, incremental training,
  deterministic, parallel data loading)?

mlprof answers these by combining a declarative audit (cheap) with
empirical small-scale probes (cheap-ish) and extrapolation.

## Quickstart

### 1. Implement three callables

mlprof calls these. You implement them however you like.

```python
# my_project/training.py
from pathlib import Path
from mlprof import TrainResult, EvalResult, RuntimeConfig


def load(subset_fraction: float = 1.0, split: str | None = None,
         seed: int | None = None):
    """Return a dataset (or a subset). Opaque to mlprof."""
    ...


def train(config: dict, dataset_subset, output_dir: Path,
          resume_from: Path | None = None,
          runtime: RuntimeConfig = RuntimeConfig()) -> TrainResult:
    """Run training. `runtime` carries n_gpus, precision, multi_gpu_strategy,
    seed, etc. — only the fields your spec opts into via Capabilities are
    populated. Save the artifact to output_dir and return its path."""
    ...
    return TrainResult(
        artifact_path=output_dir / "model",
        metrics={"train_loss": 0.12},
    )


def evaluate(artifact_path: Path, eval_set) -> EvalResult:
    """Load the artifact and return quality metrics keyed by name."""
    ...
    return EvalResult(metrics={"f1_macro": 0.87})
```

### 2. Write a `ModelSpec`

```python
from mlprof import (
    ModelSpec, DatasetSpec, EvalMetric, HyperParam,
    Capabilities, FrameworkHints, TargetInstance, ProbeConfig,
    NumericSweep, CategoricalSweep,
)

spec = ModelSpec(
    name="intent-classifier-distilbert",
    train_callable="my_project.training:train",
    evaluate_callable="my_project.training:evaluate",
    dataset=DatasetSpec(
        loader="my_project.training:load",
        total_size=1_200_000,
        subset_strategy="stratified",
        stratify_column="intent",
        eval_split="val",
    ),
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
        supports_checkpointing=True,
        supports_incremental_training=True,
        supports_parallel_data_loading=True,
        supports_multi_gpu_data_parallel=True,
        max_useful_gpus=4,
        supports_mixed_precision=True,
        deterministic=True,
    ),
    framework_hints=FrameworkHints(
        framework="pytorch", expected_device="gpu",
        requires_gpu=True, min_vram_gb=20,
    ),
    targets=[
        TargetInstance(instance_type="ml.g5.xlarge"),    # 1x A10G, 24GB
        TargetInstance(instance_type="ml.g5.12xlarge"),  # 4x A10G, 96GB
        TargetInstance(instance_type="ml.p3.2xlarge"),   # 1x V100, 16GB
    ],
    probe=ProbeConfig(
        subset_fractions=[0.005, 0.02, 0.08],
        max_variants=12,
    ),
)
```

### 3. Audit (no compute spent)

```python
from mlprof import audit

report = audit(spec)
print(report.format())
```

Sample output:

```
INCOMPATIBLE (1):
  [gpu_required_cpu_target] c5.4xlarge: Model declares requires_gpu=True
  but target 'c5.4xlarge' is CPU-only.

WARNING (3):
  [no_checkpointing] No checkpointing declared. Long jobs are not
  crash-resumable; consider adding periodic state saves before running on
  spot instances.
  [idle_gpus_no_multi_gpu] g5.12xlarge: 4 GPUs but no multi-GPU strategy
  declared. 3 GPU(s) will sit idle.
  [vram_estimate_exceeds_gpu] p3.2xlarge: min_vram_gb=20.0 exceeds per-GPU
  VRAM of 16GB (V100). OOM is likely unless gradient accumulation or model
  parallelism is used.

INFO (4):
  [no_incremental_training], [no_mixed_precision],
  [non_deterministic], [no_gradient_accumulation]
```

### 4. Probe and report

```python
from mlprof import audit, build_report, run

with tempfile.TemporaryDirectory() as td:
    results = run(spec, run_dir=td, launcher="subprocess")
    report = build_report(spec, results, audit(spec))
    print(report.format())
```

Sample report:

```
=== Report for 'intent-classifier-distilbert' ===

AUDIT: ...

EXTRAPOLATED AT FULL DATASET:
  g5.xlarge   [encoder=distilbert] → t=2.3h cost=$2.31  f1_macro=0.847
              (time=linear@R²0.99, qual=log@R²0.94)
  g5.xlarge   [encoder=minilm]     → t=0.9h cost=$0.91  f1_macro=0.798
              (time=linear@R²1.00, qual=log@R²0.92)
  g5.12xlarge [encoder=distilbert] → t=0.7h cost=$3.97  f1_macro=0.847
              ...

PARETO FRONTIER (cost ↓, f1_macro optimum):
  g5.xlarge   [encoder=minilm]     → cost=$0.91  f1_macro=0.798
  g5.xlarge   [encoder=distilbert] → cost=$2.31  f1_macro=0.847
```

## How it composes

```
ModelSpec (pydantic)            ────┐
  Capabilities (declared)           │
  FrameworkHints                    ▼
  HyperParam + Sweep        ┌──────────────┐
  TargetInstance            │     audit    │  pure-data, no compute
  DatasetSpec               └──────┬───────┘
  EvalMetric                       │
  ProbeConfig                      ▼
                            ┌──────────────┐
                            │ sweep expand │  Cartesian + max_variants cap
                            └──────┬───────┘
                                   │
                            ┌──────▼───────┐
                            │ runtime res. │  capability-gated RuntimeConfig
                            └──────┬───────┘
                                   │
User callables ───────────┐        ▼
  train / evaluate / load │ ┌──────────────┐
                          └►│   probe.py   │  subprocess (today),
                            │              │  Docker / SageMaker later
                            └──────┬───────┘
                                   │  external measurement
                                   │  via psutil + (optional) NVML
                            ┌──────▼───────┐
                            │   scaling    │  power / linear / log fits
                            └──────┬───────┘
                                   ▼
                            ┌──────────────┐
                            │   report     │  Pareto frontier + audit
                            └──────────────┘
```

### Key design choices

- **Framework-agnostic.** mlprof never imports your training framework.
  It calls user-supplied callables referenced by dotted-path string. The
  audit can run without even importing them.
- **External measurement.** Wall-clock, RAM, VRAM, and GPU utilization are
  measured by a sampler thread reading process-level data (psutil, NVML) —
  not by asking the framework to self-report. This is the only portable
  way to compare PyTorch, spaCy, sklearn, and custom code under one roof.
- **Capability-gated runtime.** The runner only enables a runtime feature
  (mixed precision, DDP, deterministic seeding) when *both* your
  `Capabilities` block opts in *and* the resolved target hardware supports
  it. bf16 on a V100 silently becomes fp16; DDP on a single-GPU instance
  silently becomes none.
- **Docker-ready by design.** The probe binary reads its input from a JSON
  file and writes results to a JSON file. The eventual `DockerLauncher`
  just mounts those files and runs the same command — no probe-code
  changes needed.
- **No magic.** The runner does not wrap your `train()` with `torchrun` or
  inject distributed-init boilerplate. It passes `runtime.n_gpus` and
  `runtime.multi_gpu_strategy` into your function; you are responsible for
  honoring them. This keeps the contract honest.

## The user-implemented protocol

Three callables. Full signatures in [`mlprof/protocol.py`](mlprof/protocol.py).

```python
class LoadDatasetFn(Protocol):
    def __call__(self, subset_fraction: float = 1.0,
                 split: str | None = None,
                 seed: int | None = None) -> Any: ...

class TrainFn(Protocol):
    def __call__(self, config: dict[str, Any], dataset_subset: Any,
                 output_dir: Path, resume_from: Path | None = None,
                 runtime: RuntimeConfig = RuntimeConfig()) -> TrainResult: ...

class EvaluateFn(Protocol):
    def __call__(self, artifact_path: Path, eval_set: Any) -> EvalResult: ...
```

`RuntimeConfig` fields the runner may populate (gated by `Capabilities`):

```python
@dataclass(frozen=True)
class RuntimeConfig:
    n_gpus: int = 0
    n_cpus: int = 1
    precision: "fp32" | "fp16" | "bf16" = "fp32"
    multi_gpu_strategy: "ddp" | "model_parallel" | "pipeline" | None = None
    gradient_accumulation_steps: int | None = None
    seed: int | None = None
```

## Target catalog

[`mlprof/targets.py`](mlprof/targets.py) ships with ~16 common AWS
instance types covering CPU (`c5`, `c6i`, `m5`), T4 (`g4dn`), A10G (`g5`),
V100 (`p3`), A100 (`p4d`, `p4de`), and H100 (`p5`). Both EC2 names
(`g5.xlarge`) and SageMaker names (`ml.g5.xlarge`) resolve to the same
entry.

Each entry knows vCPUs, RAM, GPU count, GPU model with VRAM and precision
support (fp16/bf16), and an approximate us-east-1 on-demand price. Prices
are stamped with `PRICE_AS_OF` so it is obvious when they need refreshing.

Add your own:

```python
from mlprof.targets import InstanceSpec, GpuSpec, register

register(InstanceSpec(
    instance_type="g5.24xlarge",
    vcpus=96, ram_gb=384,
    gpu_count=4, gpu=GpuSpec(model="A10G", vram_gb=24,
                              supports_fp16=True, supports_bf16=True),
    on_demand_usd_per_hour=8.144,
    family="gpu",
))
```

## Examples

- [`examples/audit_demo.py`](examples/audit_demo.py) — audit-only run
  (milliseconds, no probe execution).
- [`examples/probe_demo.py`](examples/probe_demo.py) — full pipeline with
  a synthetic trainable: spec → audit → 18 probes via subprocess → report.
  Requires no ML libraries.
- [`examples/fake_trainable.py`](examples/fake_trainable.py) — a
  reference implementation of the user-supplied callables (synthetic).
- [`examples/agnews/`](examples/agnews) — **realistic demo on the AG News
  text-classification dataset**. Two architectures (sklearn TF-IDF + LogReg
  vs DistilBERT fine-tune) across multiple target instances. Has its own
  [README](examples/agnews/README.md) with setup, expected runtimes, and
  caveats. Quick start:

  ```bash
  ./examples/agnews/run.sh --sklearn-only   # fast (~30s-2min on CPU)
  ./examples/agnews/run.sh                  # full demo (both architectures)
  ```

## Roadmap

Tracked as open issues:

1. Docker launcher for probes
2. AWS Pricing API integration for live cost estimates
3. Saturating power-law fit for quality extrapolation (needs scipy)
4. Repetition averaging in the report layer
5. LLM- / hill-climbing-driven sweep expander
6. Data-sampling tricks for higher-accuracy small-N quality estimates

## License

TBD.
