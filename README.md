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

**Scenarios are the recommended entry point.** Engineers don't think in
hyperparameter grids — they have questions: *does this scale linearly?*,
*where does adding CPUs stop helping?*, *is algorithm A or B faster at my
data size?*. A [`Scenario`](mlprof/scenarios/) answers one such question
directly, planning its own probes and returning a plain-language answer +
recommendation instead of a raw frontier. The Cartesian sweep + Pareto
frontier is still available as the lower-level surface (and is itself the
implementation of the `scaling_with_n` scenario).

## Status

Early but capable. The core spec → audit → probe → report pipeline works
end-to-end (`examples/probe_demo.py`), and on top of it:

- **Scenarios** — question-shaped probes: `scaling_with_n`,
  `parallelization_effect`, `algorithm_selection`, and the exploration-phase
  `baseline_compare`.
- **Library mode** — `mlprof.profile()` to instrument a real training run from
  the inside, plus `evaluate_existing()` for eval-only against an artifact.
- **Dataset profiling** — surface data-shape roadblocks (block-size pair
  explosion, class imbalance, similarity separation, outliers) and a
  spec-integrated pre-flight `data_audit()`.

Probes run locally as subprocesses; Docker / SageMaker launchers are still on
the roadmap. See [open issues](#roadmap) for what's next.

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

### 5. Or ask a question with a Scenario

Instead of running the full sweep and reading a frontier, point a scenario at
the same spec and get a direct answer:

```python
from mlprof.scenarios import (
    scaling_with_n, parallelization_effect, algorithm_selection,
)

print(scaling_with_n.run(spec, target="g5.xlarge").format())
# === scaling_with_n ===
# Q: Does training time scale linearly with data size N?
# A: Scales as O(N^1.04) (best fit: power, R²=0.99 over 3 sizes).
#    Extrapolated full run (N=1,200,000): 2.3h, ~$2.31.
# → Linear scaling confirmed — full-dataset cost extrapolates predictably.

print(parallelization_effect.run(
    spec, target="c5.9xlarge", n_cpus=[1, 2, 4, 8, 16, 36]).format())
# === parallelization_effect ===
# Q: Where does adding parallel workers stop helping?
# A: Speedup vs 1 CPUs: 1.0x / 1.6x / 2.3x / 2.9x / 3.4x / 3.7x.
#    Amdahl P≈0.75 (theoretical ceiling ~4.0x).
# → Plateaus around 16 CPUs. Beyond that each added worker buys <10% —
#   pick the instance that gives ~16 CPUs, not the biggest one.

# Compare algorithm variants at a specific data size:
print(algorithm_selection.run(
    [spec_agglomerative, spec_lsh], target="g5.8xlarge", n=88_000).format())

# Judge a candidate against a baseline you converge on as you probe:
from mlprof.scenarios import baseline_compare

r = baseline_compare.run(naive_result)                 # establishes the baseline
base = r.data["baseline"]
r = baseline_compare.run(candidate, baseline=base)     # tradeoff vs baseline
# → "f1 +2.3% (better); cost_usd +40% (worse) — Tradeoff: gaining f1 at the
#    cost of cost_usd. Worthwhile if f1 matters more than cost_usd."
if worthwhile:
    base = r.data["candidate"]                         # promote the challenger
```

`baseline_compare` is for the *exploration* phase: rather than gating against a
fixed bar, you converge on a baseline and judge each candidate as a tradeoff
across quality **and** cost/time/memory. It's input-agnostic — feed it an
`EvalResult`, a `ProfileReport`, a `ProbeResult`, a dict, a list of those
merged, or an artifact path it evaluates for you.

Each scenario returns a `ScenarioResult` with `.question`, `.answer`,
`.recommendation`, `.data` (raw probe results for further analysis), and an
optional `.passed` gate. Scenarios default to running their probes on **one**
target over the axis they care about, so they're cheap enough to iterate on —
run one, read the answer, decide what to run next.

See [`examples/scenario_demo.py`](examples/scenario_demo.py) for a runnable
end-to-end demo of all three (no ML libraries required).

The baseline scenarios shipping today are `scaling_with_n`,
`parallelization_effect`, and `algorithm_selection`; the
[roadmap](#roadmap) tracks the rest of the catalog (`gpu_vs_cpu`,
`vram_headroom`, `regression_guard`, `cheapest_instance`, ...).

## Library mode — instrument a real training run

Everything above is *probe mode*: mlprof predicts cost/time/memory before you
commit by running small external probes. That requires your training code to
be importable in mlprof's environment, which is awkward for complex production
models. **Library mode** sidesteps it — you `import mlprof` *inside* the
training script you're already running, so you're already in the right env,
with the right data, on the right hardware. The measurements are of the real
run, not an extrapolation.

The low-level primitive is `measure()` (already public) for timing one block:

```python
import mlprof

with mlprof.measure() as m:
    labels = run_agglomerative(distances, threshold)

print(m.wall_clock_s, m.peak_rss_mb)   # also peak_vram_mb / gpu_util_avg with [gpu]
```

`profile()` is the unified entry point. It optionally runs the pre-flight
`audit()` (and can **fail fast** on hard blockers before you spend compute),
captures per-stage timings, builds a single-run report, and — given an MLflow
run — logs all of it there:

```python
with mlprof.profile(spec=spec, mlflow_run=mlflow.active_run(),
                    on_blocker="raise") as p:
    with p.stage("load"):
        data = load_everything()
    with p.stage("cluster"):
        labels = run_agglomerative(data, threshold)

print(p.report().format())
# === Profile: brand-clustering ===
#   wall_clock: 42.3m
#   peak_rss:   18.4GB
# STAGES:
#   load                        3.1m  ( 7.3%)  rss=6.20GB
#   cluster                    38.9m  (92.0%)  rss=18.40GB
#   (unstaged)                  0.3m  ( 0.7%)
```

MLflow logging is optional (`pip install 'mlprof[mlflow]'`); without it, a
passed `mlflow_run` is ignored with a warning and the report is still built.
The single-run report intentionally omits the scaling fits and Pareto frontier
— those need multiple data sizes, which a single real run doesn't have (use a
scenario or the sweep runner for that).

### Eval-only mode

When training is expensive but evaluation is cheap, you often want to evaluate
an artifact you *already have* — against a gold standard, or under several eval
configurations — without paying to retrain. `evaluate_existing()` runs just the
`evaluate()` step against an existing artifact:

```python
result = mlprof.evaluate_existing(
    spec,
    "s3://my-bucket/run-123/model.tar.gz",   # local Path or remote URI
    # eval_set=...  # optional; auto-loaded via LoadDatasetFn(split=eval_split) if omitted
)
print(result.metrics)   # same EvalResult shape as inside a full probe
```

No protocol change — `TrainResult.artifact_path` is already the handoff between
train and eval. Local paths are passed to your `evaluate()` as a `Path`; remote
URIs (`s3://`, `gs://`) are passed through unchanged for your loader to resolve.
Runs in-process, so call it where your `evaluate_callable` is importable.

## Dataset profiling — spot roadblocks before you train

mlprof is data-blind by design, but the *shape* of the training set often
predicts trouble — and some of it is computable cheaply, with no training run.
The clearest case is **block sizes**: blocked / all-pairs-within-group methods
(dedup, blocked clustering, entity resolution) do work proportional to
`Σ C(block_size, 2)`, so one giant block (a common token, a huge near-duplicate
cluster) dominates wall-clock. That's an O(n) group-by away — you don't need to
discover it by crashing a full run.

You supply the block sizes (you own the blocking key); mlprof does the reusable
pair-cost analysis and surfaces the roadblock as **neutral facts** — it tells
you *where the cost is*, not which algorithm to use (that decision is yours):

```python
from mlprof import block_size_profile

profile = block_size_profile(df.groupby("blocking_key").size().to_dict())
print(profile.format())
# WARNING:
#   [pair_explosion_largest_block] Largest block (40,000 items) generates
#   799,980,000 pairs = 94% of all candidate pairs. All-pairs-within-block
#   work is dominated by this one block.
# INFO:
#   [blocking_reduction] Blocking yields 850,233,000 candidate pairs —
#   4x fewer than all-pairs (3,120,460,500).
```

Block sizes are one signal; the same data-blind, surface-don't-prescribe shape
applies across a small family of profilers (see
[`examples/dataset_profile_demo.py`](examples/dataset_profile_demo.py)):

| Profiler | Surfaces |
|---|---|
| `block_size_profile` | pair-explosion cost of blocked/all-pairs methods (`Σ C(size,2)`) |
| `class_balance_profile` | imbalance ratio, entropy, and the small-class floor that limits how far you can subsample |
| `stratified_plan` | per-group draw counts for a representative subsample (mlprof computes the plan; your loader draws it) |
| `similarity_profile` | distribution of pairwise similarities — clean separation vs a blur where FP/FN concentrate (Sarle bimodality coefficient) |
| `outlier_profile` | IQR/σ outlier fractions and tail heaviness |

Each returns a `DatasetProfile` (scalar `stats`, structured `data`, and neutral
`findings`). For a spec-integrated **pre-flight** — the explicit parallel to the
static capability `audit(spec)` — set `DatasetSpec.analyze_callable` to a
function returning profiles, and call `data_audit(spec)` before probing:

```python
profiles = mlprof.data_audit(spec)        # runs your analyze() over cheaply-loaded data
print(mlprof.format_data_audit(profiles)) # neutral shape facts, no training spent
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
- [`examples/scenario_demo.py`](examples/scenario_demo.py) — the three
  baseline scenarios (`scaling_with_n`, `parallelization_effect`,
  `algorithm_selection`) against a synthetic trainable. No ML libraries.
- [`examples/library_mode_demo.py`](examples/library_mode_demo.py) —
  library mode: `mlprof.profile()` instrumenting a run from the inside with
  per-stage timings. No ML libraries.
- [`examples/baseline_compare_demo.py`](examples/baseline_compare_demo.py) —
  converging on a baseline across a probing session: establish → tradeoff →
  promote. No ML libraries.
- [`examples/dataset_profile_demo.py`](examples/dataset_profile_demo.py) —
  block-size profiling: surface a pair-explosion roadblock before training.
  No ML libraries.
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

**Shipped recently:** scenarios framework
([#21](https://github.com/dnewcome/mlprof/issues/21)), library mode + eval-only
([#20](https://github.com/dnewcome/mlprof/issues/20),
[#18](https://github.com/dnewcome/mlprof/issues/18)), the `baseline_compare`
exploration scenario ([#19](https://github.com/dnewcome/mlprof/issues/19)), and
dataset profiling (block-size / class-balance / similarity / outlier + a
pre-flight `data_audit`).

**Open** (tracked as issues):

- **Cost** — live AWS pricing ([#2](https://github.com/dnewcome/mlprof/issues/2)),
  spot-pricing with interruption overhead
  ([#12](https://github.com/dnewcome/mlprof/issues/12))
- **Checkpointing / incremental** — passive
  ([#9](https://github.com/dnewcome/mlprof/issues/9)) and full empirical
  ([#11](https://github.com/dnewcome/mlprof/issues/11)) verification, chained
  warm-start probes ([#10](https://github.com/dnewcome/mlprof/issues/10))
- **Inference / deployment** — inference profiling
  ([#8](https://github.com/dnewcome/mlprof/issues/8)), model-size measurement
  ([#15](https://github.com/dnewcome/mlprof/issues/15))
- **MLflow** — `from_mlflow_run()` read side
  ([#17](https://github.com/dnewcome/mlprof/issues/17))
- **Profiling depth** — per-stage GPU util + bottleneck findings
  ([#13](https://github.com/dnewcome/mlprof/issues/13))
- **Sampling** — loader-side `subset_strategy` plumbing
  ([#6](https://github.com/dnewcome/mlprof/issues/6))
- **Quality gates** — hard pass/fail gate across variants
  ([#14](https://github.com/dnewcome/mlprof/issues/14)), sub-group gates
  ([#23](https://github.com/dnewcome/mlprof/issues/23)), subset-fraction guard
  ([#22](https://github.com/dnewcome/mlprof/issues/22))
- **Report math** — saturating power-law quality fit
  ([#3](https://github.com/dnewcome/mlprof/issues/3)), repetition averaging
  ([#4](https://github.com/dnewcome/mlprof/issues/4))
- **Infra / spec** — Docker launcher
  ([#1](https://github.com/dnewcome/mlprof/issues/1); lower priority now that
  library mode exists), YAML spec loading
  ([#7](https://github.com/dnewcome/mlprof/issues/7)), sagebaker relationship
  doc ([#16](https://github.com/dnewcome/mlprof/issues/16))

## License

TBD.
