"""Resolve a RuntimeConfig from (spec capabilities + target instance).

This is what the runner injects into the user's ``train()`` call. It is
intentionally conservative: only enable a runtime feature when *both* the
spec opts in (Capabilities) *and* the resolved target supports it. The user
keeps full control over what is opt-in via their declared capabilities.

This module is also consumed by the audit so the audit can report exactly
what the runner *would* inject — no divergence between predicted and actual.
"""

from __future__ import annotations

from mlprof.protocol import MultiGpuStrategy, Precision, RuntimeConfig
from mlprof.spec import Capabilities
from mlprof.targets import InstanceSpec


DEFAULT_SEED = 1337


def resolve_runtime(
    capabilities: Capabilities,
    instance: InstanceSpec,
    preferred_precision: Precision | None = None,
) -> RuntimeConfig:
    """Build a RuntimeConfig appropriate for this (model, target) pair.

    Parameters
    ----------
    capabilities :
        From the model spec — what the model says it supports.
    instance :
        Resolved target hardware (vCPUs, GPU model, VRAM, etc.).
    preferred_precision :
        Optional user/runner override. Will be silently downgraded if the
        target can't support it (e.g. bf16 requested on V100 → fp16).
        If None, picks the best available given capabilities + hardware.
    """

    n_gpus = _pick_n_gpus(capabilities, instance)
    n_cpus = max(1, instance.vcpus)
    precision = _pick_precision(capabilities, instance, preferred_precision)
    strategy = _pick_strategy(capabilities, n_gpus)
    seed = DEFAULT_SEED if capabilities.deterministic else None

    return RuntimeConfig(
        n_gpus=n_gpus,
        n_cpus=n_cpus,
        precision=precision,
        multi_gpu_strategy=strategy,
        gradient_accumulation_steps=None,   # exposed to manual override later
        seed=seed,
    )


def _pick_n_gpus(caps: Capabilities, inst: InstanceSpec) -> int:
    if inst.gpu_count == 0:
        return 0
    if not caps.any_multi_gpu:
        # Model can't use multiple GPUs — use one, rest are idle (the audit
        # is what warns about that wasted capacity).
        return 1
    if caps.max_useful_gpus is not None:
        return min(inst.gpu_count, caps.max_useful_gpus)
    return inst.gpu_count


def _pick_precision(
    caps: Capabilities,
    inst: InstanceSpec,
    preferred: Precision | None,
) -> Precision:
    if not caps.supports_mixed_precision or inst.gpu is None:
        return "fp32"
    if preferred == "bf16" and inst.gpu.supports_bf16:
        return "bf16"
    if preferred == "fp16" and inst.gpu.supports_fp16:
        return "fp16"
    if preferred is None:
        # Prefer bf16 when available — wider dynamic range than fp16.
        if inst.gpu.supports_bf16:
            return "bf16"
        if inst.gpu.supports_fp16:
            return "fp16"
    return "fp32"


def _pick_strategy(caps: Capabilities, n_gpus: int) -> MultiGpuStrategy | None:
    if n_gpus <= 1:
        return None
    # Preference order: data > model > pipeline. Most models that support
    # multiple support DDP, which is the simplest and lowest-overhead.
    if caps.supports_multi_gpu_data_parallel:
        return "ddp"
    if caps.supports_multi_gpu_model_parallel:
        return "model_parallel"
    if caps.supports_multi_gpu_pipeline_parallel:
        return "pipeline"
    return None
