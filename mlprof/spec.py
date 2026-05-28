"""Declarative spec for a model under evaluation.

The spec is intentionally framework-agnostic. The user points at training/eval
callables via dotted paths so the spec can be parsed and audited *without*
importing torch / spacy / sklearn. The probe runner is what actually imports
and invokes those callables.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


HyperType = Literal["int", "float", "categorical", "bool"]


class NumericSweep(BaseModel):
    """Range-based sweep with explicit resolution. Exactly one of
    ``step`` (arithmetic), ``log_step`` (multiplicative — typical for lr),
    or ``count`` (N evenly spaced) must be set."""

    kind: Literal["numeric"] = "numeric"
    min: float
    max: float
    step: float | None = None
    log_step: float | None = None
    count: int | None = None

    @model_validator(mode="after")
    def _check_one_resolution(self) -> NumericSweep:
        set_count = sum(x is not None for x in (self.step, self.log_step, self.count))
        if set_count != 1:
            raise ValueError("NumericSweep needs exactly one of step / log_step / count")
        if self.max <= self.min:
            raise ValueError(f"NumericSweep max ({self.max}) must exceed min ({self.min})")
        if self.log_step is not None and self.min <= 0:
            raise ValueError("log_step sweeps require strictly positive min")
        return self


class CategoricalSweep(BaseModel):
    """Explicit subset of values to probe across."""

    kind: Literal["categorical"] = "categorical"
    values: list[Any]

    @model_validator(mode="after")
    def _check_nonempty(self) -> CategoricalSweep:
        if not self.values:
            raise ValueError("CategoricalSweep needs at least one value")
        return self


Sweep = NumericSweep | CategoricalSweep


class HyperParam(BaseModel):
    """A tunable knob. Structural hypers change model arch (and so change
    memory/quality dramatically); training-only hypers (lr, batch_size, epochs)
    typically don't change arch but do change throughput and convergence."""

    name: str
    type: HyperType
    default: Any
    values: list[Any] | None = None          # categorical / bool enumeration
    min: float | int | None = None           # numeric range (full allowed)
    max: float | int | None = None
    structural: bool = False                 # affects model architecture
    sweep: Sweep | None = None               # how the runner expands this param

    @model_validator(mode="after")
    def _check_bounds(self) -> HyperParam:
        if self.type in ("int", "float") and self.values is None:
            if self.min is None or self.max is None:
                raise ValueError(f"numeric hyperparam {self.name!r} needs min/max or values")
        if self.type in ("categorical", "bool") and not self.values:
            raise ValueError(f"hyperparam {self.name!r} needs 'values'")
        return self

    @model_validator(mode="after")
    def _check_sweep_matches_type(self) -> HyperParam:
        if self.sweep is None:
            return self
        if isinstance(self.sweep, NumericSweep):
            if self.type not in ("int", "float"):
                raise ValueError(f"{self.name!r}: NumericSweep requires numeric hyperparam type")
            if self.min is not None and self.sweep.min < self.min:
                raise ValueError(f"{self.name!r}: sweep.min below hyperparam min")
            if self.max is not None and self.sweep.max > self.max:
                raise ValueError(f"{self.name!r}: sweep.max above hyperparam max")
        else:  # CategoricalSweep
            if self.type not in ("categorical", "bool"):
                raise ValueError(f"{self.name!r}: CategoricalSweep requires categorical/bool type")
            if self.values is not None:
                stray = [v for v in self.sweep.values if v not in self.values]
                if stray:
                    raise ValueError(f"{self.name!r}: sweep values not in declared values: {stray}")
        return self


class Capabilities(BaseModel):
    """Production-readiness capabilities the user *declares* about their
    training code. The runner audits these up front and (where possible)
    empirically verifies them. Missing capabilities surface as warnings —
    e.g. "no checkpointing on an estimated 12h job → spot-instance risk",
    or "target instance has 8 GPUs but no multi-GPU support declared → 7 idle".
    """

    # Resumability / fault tolerance
    supports_incremental_training: bool = False  # train(resume_from=...) works
    supports_checkpointing: bool = False         # periodic state save during training

    # Data-loading parallelism (CPU side: workers, prefetch). Distinct from
    # data-parallel training below.
    supports_parallel_data_loading: bool = False

    # Multi-GPU strategies. These are independent — a model can support
    # several. max_useful_gpus lets the runner warn when a target instance
    # has more GPUs than the model can efficiently use.
    supports_multi_gpu_data_parallel: bool = False    # DDP-style: same model, sharded batches
    supports_multi_gpu_model_parallel: bool = False   # tensor/layer sharding across GPUs
    supports_multi_gpu_pipeline_parallel: bool = False  # pipelined stages across GPUs
    max_useful_gpus: int | None = None                  # diminishing returns past this count

    # Multi-node
    supports_distributed_training: bool = False  # multi-node (typically on top of DDP)

    # Numerics / memory
    supports_mixed_precision: bool = False       # fp16 / bf16
    supports_gradient_accumulation: bool = False # large effective batch on small VRAM

    # Reproducibility
    deterministic: bool = False                  # seedable; same input -> same output

    notes: str | None = None

    @property
    def any_multi_gpu(self) -> bool:
        return (
            self.supports_multi_gpu_data_parallel
            or self.supports_multi_gpu_model_parallel
            or self.supports_multi_gpu_pipeline_parallel
        )


class FrameworkHints(BaseModel):
    """Optional hints that drive fast pre-flight warnings *before* any compute
    is spent. When missing, the runner falls back to empirical detection
    (e.g. observing GPU utilization during a probe)."""

    framework: str | None = None                  # "pytorch" | "spacy" | "sklearn" | ...
    expected_device: Literal["cpu", "gpu", "either"] = "either"
    requires_gpu: bool = False
    min_vram_gb: float | None = None              # user's best guess, validated empirically
    cpu_bound: bool = False                       # e.g. spaCy pipelines; GPU is wasted


class DatasetSpec(BaseModel):
    """How the runner gets data. The user's loader is responsible for
    materializing a subset at a requested fraction so probes can be cheap."""

    loader: str                                   # dotted path to LoadDatasetFn
    total_size: int | None = None                 # full row count if known (cost/time extrap.)
    subset_strategy: Literal["random", "stratified", "head"] = "random"
    eval_split: str | None = None                 # named split or path for held-out eval
    stratify_column: str | None = None            # required if subset_strategy == "stratified"
    analyze_callable: str | None = None           # optional dotted path: analyze(dataset) ->
                                                  # DatasetProfile | list[DatasetProfile]. Run by
                                                  # data_audit() as a pre-flight (no training).


class EvalMetric(BaseModel):
    """A quality metric. The user's evaluate callable returns a dict keyed by
    these names. The metric flagged `primary=True` is used as the quality
    axis on the Pareto frontier."""

    name: str                                     # e.g. "f1_macro", "accuracy", "inference_latency_ms"
    higher_is_better: bool
    primary: bool = False


class TargetInstance(BaseModel):
    """An infrastructure target to evaluate against. Right now SageMaker
    instance types, but the shape is generic enough to grow."""

    instance_type: str                            # e.g. "ml.p3.2xlarge", "ml.c5.4xlarge"
    region: str = "us-east-1"
    spot: bool = False                            # use spot pricing in cost estimate


class ProbeConfig(BaseModel):
    """How the runner sizes its calibration jobs."""

    subset_fractions: list[float] = Field(       # used to fit a scaling curve
        default_factory=lambda: [0.01, 0.05, 0.1]
    )
    repetitions: int = 1                          # noise reduction; >1 averages runs
    timeout_seconds: int = 1800                   # hard cap per probe
    max_variants: int | None = None               # cap on Cartesian sweep expansion;
                                                  # None = no cap. Probes scale as
                                                  # max_variants * len(subset_fractions)
                                                  # * len(targets) * repetitions.
    extrapolation_models: list[Literal["linear", "loglinear", "power"]] = Field(
        default_factory=lambda: ["linear", "loglinear", "power"]
    )


class ModelSpec(BaseModel):
    """The single source of truth for one model variant under evaluation.

    A user generally writes several of these (or generates them by sweeping
    hyperparameters) and the runner evaluates each one across the configured
    targets, producing a Pareto frontier of (cost, time, quality).
    """

    name: str
    version: str = "0.0.1"

    # User-implemented callables (dotted paths so spec parsing is import-free).
    train_callable: str                           # mlprof.protocol.TrainFn
    evaluate_callable: str                        # mlprof.protocol.EvaluateFn

    dataset: DatasetSpec
    eval_metrics: list[EvalMetric]
    hyperparameters: list[HyperParam] = Field(default_factory=list)

    capabilities: Capabilities = Field(default_factory=Capabilities)
    framework_hints: FrameworkHints = Field(default_factory=FrameworkHints)

    targets: list[TargetInstance] = Field(default_factory=list)
    probe: ProbeConfig = Field(default_factory=ProbeConfig)

    @model_validator(mode="after")
    def _check_primary_metric(self) -> ModelSpec:
        primaries = [m for m in self.eval_metrics if m.primary]
        if len(primaries) != 1:
            raise ValueError(
                f"exactly one eval_metric must be marked primary=True (got {len(primaries)})"
            )
        if self.dataset.subset_strategy == "stratified" and not self.dataset.stratify_column:
            raise ValueError("stratified subset_strategy requires dataset.stratify_column")
        return self
