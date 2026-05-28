"""Drive mlprobe against a LoRA instruction-fine-tune spec.

Two paths:

  python examples/lora_finetune/run.py
      Pre-flight `audit()` only — no compute, no ML libraries. This is the
      highest-value output for fine-tuning: will an 8B+LoRA bf16 run fit in
      this GPU's VRAM? bf16 on a V100? idle GPUs? It even prints an estimated
      training-VRAM figure from the declared param count.

  python examples/lora_finetune/run.py --simulate
      Also runs scenarios (scaling_with_n, algorithm_selection) against a
      synthetic LoRA trainable — still no ML libraries.

The *real* callables (HF + PEFT) are in lora_ft.py and need ``.[demo]`` + peft
+ a GPU. Run from the repo root so ``examples.lora_finetune.*`` resolves in the
probe subprocess.
"""

from __future__ import annotations

import argparse

from mlprobe import (
    Capabilities,
    DatasetSpec,
    EvalMetric,
    FrameworkHints,
    HyperParam,
    ModelSpec,
    NumericSweep,
    ProbeConfig,
    TargetInstance,
    audit,
)

TOTAL_EXAMPLES = 120_000
_PARAMS_B = {"Llama-3.1-8B": 8.0, "Llama-3.2-3B": 3.0}


def build_spec(base_model: str = "Llama-3.1-8B", *, simulate: bool = False) -> ModelSpec:
    mod = "examples.lora_finetune.lora_sim" if simulate else "examples.lora_finetune.lora_ft"
    return ModelSpec(
        name=f"lora-{base_model}" + ("-sim" if simulate else ""),
        train_callable=f"{mod}:train",
        evaluate_callable=f"{mod}:evaluate",
        dataset=DatasetSpec(loader=f"{mod}:load", total_size=TOTAL_EXAMPLES,
                            subset_strategy="random", eval_split="validation"),
        eval_metrics=[
            EvalMetric(name="exact_match", higher_is_better=True, primary=True),
            EvalMetric(name="gen_latency_ms", higher_is_better=False),
        ],
        hyperparameters=[
            HyperParam(name="base_model", type="categorical", default=base_model,
                       values=list(_PARAMS_B), structural=True),
            HyperParam(name="lora_r", type="int", default=16, min=4, max=64,
                       sweep=NumericSweep(min=8, max=64, step=8)),
            HyperParam(name="lora_alpha", type="int", default=32, min=8, max=128),
            HyperParam(name="batch_size", type="int", default=4, min=1, max=16),
            HyperParam(name="epochs", type="int", default=3, min=1, max=5),
        ],
        capabilities=Capabilities(
            supports_checkpointing=True, supports_incremental_training=True,
            supports_mixed_precision=True, supports_gradient_accumulation=True,
            supports_multi_gpu_data_parallel=True, max_useful_gpus=8, deterministic=True),
        framework_hints=FrameworkHints(
            framework="pytorch", expected_device="gpu", requires_gpu=True,
            min_vram_gb=20, param_count_b=_PARAMS_B[base_model], finetune_method="lora"),
        targets=[
            TargetInstance(instance_type="g5.xlarge"),     # 1x A10G 24GB
            TargetInstance(instance_type="g5.12xlarge"),   # 4x A10G
            TargetInstance(instance_type="p3.2xlarge"),    # 1x V100 16GB (no bf16) — instructive
            TargetInstance(instance_type="p4d.24xlarge"),  # 8x A100 40GB
        ],
        probe=ProbeConfig(subset_fractions=[0.02, 0.1, 0.3], max_variants=4, timeout_seconds=120),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="mlprobe LoRA fine-tune example")
    ap.add_argument("--simulate", action="store_true",
                    help="run scenarios against a synthetic LoRA trainable (no ML deps)")
    args = ap.parse_args()

    spec = build_spec(simulate=args.simulate)
    print("=== Pre-flight audit (no compute, no ML deps) ===")
    print(audit(spec).format())

    if not args.simulate:
        print("\nAudit only. Add --simulate to run scenarios on a synthetic trainable, "
              "or wire lora_ft.py (needs .[demo] + peft + a GPU) for a real fine-tune.")
        return

    from mlprobe.scenarios import algorithm_selection, scaling_with_n

    print("\n=== scaling_with_n on g5.xlarge (synthetic) ===")
    print(scaling_with_n.run(spec, target="g5.xlarge", progress=False).format())

    print("\n=== algorithm_selection: 8B vs 3B base at 24k examples ===")
    result = algorithm_selection.run(
        [build_spec("Llama-3.1-8B", simulate=True), build_spec("Llama-3.2-3B", simulate=True)],
        target="g5.12xlarge", n=24_000, progress=False)
    print(result.format())


if __name__ == "__main__":
    main()
