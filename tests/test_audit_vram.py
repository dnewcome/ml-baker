"""Tests for the LLM-aware training-VRAM estimate in the audit."""

from __future__ import annotations

from mlprobe import (
    Capabilities,
    DatasetSpec,
    EvalMetric,
    FrameworkHints,
    ModelSpec,
    TargetInstance,
    audit,
    estimate_training_vram_gb,
)


def test_vram_estimate_math():
    lora = estimate_training_vram_gb(8, precision="bf16", method="lora")
    assert 18 <= lora["estimated_total_gb"] <= 22          # ~16GB weights + overhead
    full = estimate_training_vram_gb(8, precision="bf16", method="full")
    assert full["estimated_total_gb"] > 100                # full FT of 8B is huge
    qlora = estimate_training_vram_gb(8, method="qlora")
    assert qlora["estimated_total_gb"] < 10                # 4-bit frozen base


def _spec(**hint_overrides):
    fh = dict(requires_gpu=True, param_count_b=8.0, finetune_method="lora")
    fh.update(hint_overrides)
    return ModelSpec(
        name="ft", train_callable="x:t", evaluate_callable="x:e",
        dataset=DatasetSpec(loader="x:l", total_size=1000),
        eval_metrics=[EvalMetric(name="f1", higher_is_better=True, primary=True)],
        capabilities=Capabilities(supports_mixed_precision=True),
        framework_hints=FrameworkHints(**fh),
        targets=[TargetInstance(instance_type="g5.xlarge"),    # A10G 24GB
                 TargetInstance(instance_type="p3.2xlarge")],  # V100 16GB
    )


def test_audit_emits_vram_estimate_and_per_target_warning():
    rep = audit(_spec())
    codes = [f.code for f in rep.findings]
    assert "estimated_training_vram" in codes

    warned = {f.target for f in rep.findings if f.code == "estimated_vram_exceeds_gpu"}
    assert "p3.2xlarge" in warned     # ~20GB estimate > 16GB V100
    assert "g5.xlarge" not in warned  # fits (tight) in 24GB A10G


def test_audit_no_vram_estimate_without_param_count():
    rep = audit(_spec(param_count_b=None))
    assert "estimated_training_vram" not in [f.code for f in rep.findings]
    assert "estimated_vram_exceeds_gpu" not in [f.code for f in rep.findings]
