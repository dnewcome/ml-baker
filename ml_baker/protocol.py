"""The user-implemented protocol.

ml-baker is framework-agnostic. The user implements three callables that
the runner orchestrates. Everything the runner needs to measure (cost, time,
memory, GPU utilization) is captured *externally* by the probe layer — the
user's callables only need to return what they uniquely know (the trained
artifact and any framework-reported metrics).

Callables are referenced from a ModelSpec by dotted path (e.g.
"my_project.training:train") so the spec can be parsed and audited without
importing the user's framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol


Precision = Literal["fp32", "fp16", "bf16"]
MultiGpuStrategy = Literal["ddp", "model_parallel", "pipeline"]


@dataclass(frozen=True)
class RuntimeConfig:
    """Runner-injected, capability-gated runtime knobs for one train() call.

    The runner populates a field only if (a) the spec's Capabilities declares
    support, AND (b) the resolved target instance supports it. The user's
    train() inspects these and acts on them — there is no magic wrapping.

    Defaults match the safest fallback: single device, fp32, no parallelism.
    """

    n_gpus: int = 0                              # 0 means CPU only
    n_cpus: int = 1                              # cores the user should attempt to use
    precision: Precision = "fp32"
    multi_gpu_strategy: MultiGpuStrategy | None = None
    gradient_accumulation_steps: int | None = None
    seed: int | None = None                      # set when capabilities.deterministic


@dataclass
class TrainResult:
    """What a train() invocation returns. External measurements (wall-clock,
    peak VRAM, GPU util, cost) are added by the probe runner — the user does
    not need to populate them."""

    artifact_path: Path                            # serialized model, opaque to ml-baker
    metrics: dict[str, float] = field(default_factory=dict)   # loss, train_acc, etc.
    steps_completed: int | None = None             # for incremental-train resumption
    checkpoint_paths: list[Path] = field(default_factory=list)  # if checkpointing on


@dataclass
class EvalResult:
    """What evaluate() returns. Keys must match the names declared in
    ModelSpec.eval_metrics; extra keys are ignored with a warning."""

    metrics: dict[str, float] = field(default_factory=dict)


class TrainFn(Protocol):
    """The user's training function.

    Parameters
    ----------
    config :
        Resolved hyperparameter values for this run. Keys match HyperParam.name
        entries in the spec.
    dataset_subset :
        Whatever ``LoadDatasetFn`` returned for the requested subset fraction.
        Opaque to ml-baker.
    output_dir :
        A clean directory the user writes the artifact (and any checkpoints) to.
    resume_from :
        If supports_incremental_training is True, the path of a prior artifact
        to warm-start from. None means train from scratch.
    runtime :
        Runner-decided runtime knobs (gpu/cpu counts, precision, multi-gpu
        strategy, seed, ...). Only fields the spec opted into via Capabilities
        will be populated; the rest stay at their safe defaults. The user is
        responsible for actually honoring them (e.g. launching DDP when
        runtime.multi_gpu_strategy == "ddp").
    """

    def __call__(
        self,
        config: dict[str, Any],
        dataset_subset: Any,
        output_dir: Path,
        resume_from: Path | None = None,
        runtime: RuntimeConfig = RuntimeConfig(),
    ) -> TrainResult: ...


class EvaluateFn(Protocol):
    """Loads the artifact and reports quality metrics on a held-out set."""

    def __call__(self, artifact_path: Path, eval_set: Any) -> EvalResult: ...


class LoadDatasetFn(Protocol):
    """Materializes a dataset (or a subset of one).

    The runner calls this once per probe with the requested ``subset_fraction``
    so it can build a scaling curve. ``split`` selects a named split (e.g.
    "train", "eval") when the spec uses one.
    """

    def __call__(
        self,
        subset_fraction: float = 1.0,
        split: str | None = None,
        seed: int | None = None,
    ) -> Any: ...
