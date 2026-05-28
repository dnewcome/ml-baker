"""Static capability + target-compatibility audit.

The audit consumes a ModelSpec and emits structured findings *without
running any training*. It surfaces things like:
  - notebook code with no checkpointing → spot-instance risk on long jobs
  - GPU target paired with a CPU-bound framework (e.g. spaCy) → wasted $
  - 8-GPU instance paired with no multi-GPU strategy → 7 GPUs idle
  - bf16 declared but target is V100 → silent downgrade to fp16
  - hard incompatibilities (requires_gpu on a CPU-only target)

This is the highest-leverage output for the "notebook script → production"
use case: it is useful before the user has paid for any compute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mlprobe.runtime import resolve_runtime
from mlprobe.spec import ModelSpec
from mlprobe.targets import InstanceSpec, resolve


Severity = Literal["incompatible", "warning", "info"]


@dataclass(frozen=True)
class AuditFinding:
    severity: Severity
    code: str                            # stable identifier, e.g. "no_checkpointing"
    message: str                         # human-readable
    target: str | None = None            # instance_type if target-specific
    capability: str | None = None        # capability name if capability-driven


@dataclass
class AuditReport:
    findings: list[AuditFinding] = field(default_factory=list)

    @property
    def incompatible(self) -> list[AuditFinding]:
        return [f for f in self.findings if f.severity == "incompatible"]

    @property
    def warnings(self) -> list[AuditFinding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def has_blockers(self) -> bool:
        return any(f.severity == "incompatible" for f in self.findings)

    def format(self) -> str:
        if not self.findings:
            return "No findings."
        lines = []
        for sev in ("incompatible", "warning", "info"):
            group = [f for f in self.findings if f.severity == sev]
            if not group:
                continue
            lines.append(f"\n{sev.upper()} ({len(group)}):")
            for f in group:
                prefix = f"  [{f.code}]"
                if f.target:
                    prefix += f" {f.target}:"
                lines.append(f"{prefix} {f.message}")
        return "\n".join(lines).strip()


def estimate_training_vram_gb(
    params_b: float,
    *,
    precision: str = "bf16",
    method: str = "full",
    overhead: float = 0.25,
) -> dict[str, float | str]:
    """Rough training-VRAM estimate (GB) for a transformer, as decision support.

    Sums the predictable components — weights, gradients, optimizer state — and
    adds a coarse ``overhead`` allowance for activations/fragmentation
    (activations depend on batch × seq_len × depth and are reduced a lot by
    gradient checkpointing, so they are not modeled precisely here).

    method:
      - ``"full"``  — all params trainable; AdamW mixed-precision ≈ 16 B/param.
      - ``"lora"``  — base frozen in working precision; adapter params negligible.
      - ``"qlora"`` — 4-bit quantized frozen base (~0.5 B/param), adapter negligible.

    This is a ballpark for choosing an instance, not a guarantee — validate with
    a probe.
    """
    bytes_w = 4 if precision == "fp32" else 2
    p = params_b * 1e9
    if method == "qlora":
        weights = p * 0.5
        trainable = 0.0
    elif method == "lora":
        weights = p * bytes_w
        trainable = 0.0
    else:  # full
        weights = p * bytes_w
        trainable = p
    grads = trainable * bytes_w
    optimizer = trainable * 12.0  # AdamW: fp32 master copy + m + v
    total_gb = (weights + grads + optimizer) * (1.0 + overhead) / 1e9
    return {
        "params_b": params_b,
        "precision": precision,
        "method": method,
        "weights_gb": weights / 1e9,
        "grad_optimizer_gb": (grads + optimizer) / 1e9,
        "estimated_total_gb": total_gb,
    }


def audit(spec: ModelSpec) -> AuditReport:
    """Run the full audit. Resolves each declared target against the catalog."""
    report = AuditReport()
    _audit_capabilities(spec, report)
    for target in spec.targets:
        try:
            instance = resolve(target.instance_type)
        except KeyError as e:
            report.findings.append(AuditFinding(
                severity="incompatible",
                code="unknown_instance",
                message=str(e),
                target=target.instance_type,
            ))
            continue
        _audit_target_compatibility(spec, instance, report)
    return report


# --- capability-driven (target-independent) findings -----------------------

def _audit_capabilities(spec: ModelSpec, report: AuditReport) -> None:
    caps = spec.capabilities

    if not caps.supports_checkpointing:
        report.findings.append(AuditFinding(
            severity="warning",
            code="no_checkpointing",
            message=(
                "No checkpointing declared. Long jobs are not crash-resumable; "
                "consider adding periodic state saves before running on spot instances."
            ),
            capability="supports_checkpointing",
        ))

    if not caps.supports_incremental_training:
        report.findings.append(AuditFinding(
            severity="info",
            code="no_incremental_training",
            message=(
                "No incremental training declared. Cannot warm-start on new data; "
                "each retrain is from scratch."
            ),
            capability="supports_incremental_training",
        ))

    if not caps.supports_parallel_data_loading:
        report.findings.append(AuditFinding(
            severity="warning",
            code="no_parallel_data_loading",
            message=(
                "No parallel data loading declared. Data loading is likely "
                "single-process and may bottleneck multi-GPU training."
            ),
            capability="supports_parallel_data_loading",
        ))

    if not caps.supports_mixed_precision:
        report.findings.append(AuditFinding(
            severity="info",
            code="no_mixed_precision",
            message=(
                "Mixed precision (fp16/bf16) not declared. Modern GPUs leave "
                "1.5–3x throughput on the table running fp32."
            ),
            capability="supports_mixed_precision",
        ))

    if not caps.deterministic:
        report.findings.append(AuditFinding(
            severity="info",
            code="non_deterministic",
            message=(
                "Determinism not declared. Quality measurements across probes "
                "will be noisier; consider seeding."
            ),
            capability="deterministic",
        ))

    if not caps.supports_gradient_accumulation:
        report.findings.append(AuditFinding(
            severity="info",
            code="no_gradient_accumulation",
            message=(
                "No gradient accumulation. Effective batch size is capped by VRAM; "
                "small-VRAM targets may force smaller batches than ideal."
            ),
            capability="supports_gradient_accumulation",
        ))

    hints = spec.framework_hints
    if hints.param_count_b:
        method = hints.finetune_method or "full"
        precision = "bf16" if caps.supports_mixed_precision else "fp32"
        est = estimate_training_vram_gb(hints.param_count_b, precision=precision, method=method)
        report.findings.append(AuditFinding(
            severity="info",
            code="estimated_training_vram",
            message=(
                f"Estimated training VRAM ~{est['estimated_total_gb']:.0f}GB for a "
                f"{hints.param_count_b:g}B {method} run in {precision} "
                f"(weights ~{est['weights_gb']:.0f}GB + grad/optimizer "
                f"~{est['grad_optimizer_gb']:.0f}GB + overhead). Rough estimate — "
                f"validate empirically."
            ),
            capability="param_count_b",
        ))


# --- target-compatibility findings -----------------------------------------

def _audit_target_compatibility(
    spec: ModelSpec, instance: InstanceSpec, report: AuditReport
) -> None:
    caps = spec.capabilities
    hints = spec.framework_hints
    tt = instance.instance_type

    # Hard incompatibilities.
    if hints.requires_gpu and instance.gpu_count == 0:
        report.findings.append(AuditFinding(
            severity="incompatible",
            code="gpu_required_cpu_target",
            message=(
                f"Model declares requires_gpu=True but target {tt!r} is CPU-only."
            ),
            target=tt,
        ))
        return  # rest of GPU checks moot

    # Wasted-resource warnings.
    if hints.cpu_bound and instance.gpu_count > 0:
        report.findings.append(AuditFinding(
            severity="warning",
            code="cpu_bound_on_gpu_target",
            message=(
                f"Framework hint says cpu_bound=True but {tt!r} provisions "
                f"{instance.gpu_count} GPU(s) at ${instance.on_demand_usd_per_hour:.2f}/hr. "
                f"A comparable CPU instance is likely far cheaper."
            ),
            target=tt,
        ))

    if instance.gpu_count > 1 and not caps.any_multi_gpu:
        idle = instance.gpu_count - 1
        report.findings.append(AuditFinding(
            severity="warning",
            code="idle_gpus_no_multi_gpu",
            message=(
                f"{tt!r} has {instance.gpu_count} GPUs but no multi-GPU strategy is "
                f"declared in capabilities. {idle} GPU(s) will sit idle — consider "
                f"a smaller single-GPU instance or adding DDP support."
            ),
            target=tt,
        ))

    if (
        caps.any_multi_gpu
        and caps.max_useful_gpus is not None
        and instance.gpu_count > caps.max_useful_gpus
    ):
        idle = instance.gpu_count - caps.max_useful_gpus
        report.findings.append(AuditFinding(
            severity="warning",
            code="exceeds_max_useful_gpus",
            message=(
                f"{tt!r} has {instance.gpu_count} GPUs but max_useful_gpus="
                f"{caps.max_useful_gpus}. {idle} GPU(s) will be underutilized."
            ),
            target=tt,
        ))

    # VRAM headroom check (best-effort: user's estimate vs per-GPU VRAM).
    if hints.min_vram_gb is not None and instance.gpu is not None:
        if hints.min_vram_gb > instance.gpu.vram_gb:
            report.findings.append(AuditFinding(
                severity="warning",
                code="vram_estimate_exceeds_gpu",
                message=(
                    f"min_vram_gb={hints.min_vram_gb} exceeds per-GPU VRAM of "
                    f"{instance.gpu.vram_gb}GB ({instance.gpu.model}). OOM is likely "
                    f"unless gradient accumulation or model parallelism is used."
                ),
                target=tt,
            ))

    # Precision downgrade notice.
    if caps.supports_mixed_precision and instance.gpu is not None:
        resolved = resolve_runtime(caps, instance)
        if resolved.precision == "fp16" and not instance.gpu.supports_bf16:
            report.findings.append(AuditFinding(
                severity="info",
                code="precision_downgrade",
                message=(
                    f"{instance.gpu.model} does not support bf16; runtime will use "
                    f"fp16 instead. Watch for loss-scaling instability."
                ),
                target=tt,
            ))

    # Estimated training-VRAM check (only when a param count is declared).
    if hints.param_count_b and instance.gpu is not None:
        resolved = resolve_runtime(caps, instance)
        est = estimate_training_vram_gb(
            hints.param_count_b, precision=resolved.precision,
            method=hints.finetune_method or "full",
        )
        if est["estimated_total_gb"] > 0.9 * instance.gpu.vram_gb:
            report.findings.append(AuditFinding(
                severity="warning",
                code="estimated_vram_exceeds_gpu",
                message=(
                    f"Estimated training VRAM ~{est['estimated_total_gb']:.0f}GB exceeds ~90% "
                    f"of {tt!r}'s {instance.gpu.vram_gb}GB per-GPU ({instance.gpu.model}). "
                    f"Expect OOM without sharding (FSDP/DeepSpeed), quantization (QLoRA), or "
                    f"gradient accumulation."
                ),
                target=tt,
            ))
