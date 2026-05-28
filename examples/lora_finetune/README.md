# LoRA fine-tune example

How mlprobe applies to **LLM instruction fine-tuning with LoRA**. A LoRA
fine-tune slots onto the same three-callable protocol as any other model — no
LLM special-casing — so this example is mostly a realistic `ModelSpec` plus the
callables.

Run everything from the **repo root** (so `examples.lora_finetune.*` resolves
in the probe subprocess).

## Pre-flight audit only (no compute, no ML libraries)

```bash
python examples/lora_finetune/run.py
```

This is the highest-value output for fine-tuning — it answers *before you spend
a GPU-hour*: will an 8B+LoRA bf16 run fit in this GPU's VRAM? bf16 on a V100?
idle GPUs on a multi-GPU box? It also prints an **estimated training-VRAM**
figure derived from the declared param count + fine-tune method (`audit()` uses
`FrameworkHints.param_count_b` / `finetune_method`). Note the `p3.2xlarge`
(V100, 16GB, no bf16) target — it surfaces a VRAM warning and a bf16→fp16
downgrade notice, the classic LoRA gotchas.

## Run scenarios on a synthetic trainable (still no ML libraries)

```bash
python examples/lora_finetune/run.py --simulate
```

Runs `scaling_with_n` (full-run time/cost extrapolation) and
`algorithm_selection` (8B vs 3B base at a fixed data size — a real speed/quality
tradeoff) against `lora_sim.py`, which *simulates* LoRA cost/quality so the flow
runs in seconds with no torch/peft/GPU.

## Real fine-tune (HF + PEFT)

`lora_ft.py` is the reference template for the actual callables. It needs the
heavy deps and a GPU:

```bash
pip install -e ".[demo]" peft
```

Adapt the tokenization / prompt formatting to your data and point `_HF_IDS` at
models you can access (for a CPU smoke test, map a base_model to a tiny model
like `hf-internal-testing/tiny-random-LlamaForCausalLM`). Then swap the spec's
callables from `lora_sim` to `lora_ft` (drop the `--simulate` flag once wired).

## What maps to what

| Fine-tuning question | mlprobe |
|---|---|
| Will 8B-LoRA bf16 fit in 24GB? bf16 on this GPU? | `audit()` pre-flight (+ estimated training VRAM, + empirical VRAM-overshoot finding after a probe) |
| Full-run time & cost at N examples? | `scaling_with_n` (time/cost extrapolate; quality is a hint) |
| Does 4× GPUs actually 4× the fine-tune? | `parallelization_effect` |
| 8B-LoRA vs 3B-full at my data size — best per dollar? | `algorithm_selection` |
| `lora_r=16` vs `32`: worth the extra cost/VRAM? | `baseline_compare` |
| Eval the adapter against many judge prompts | `evaluate_existing()` (adapter is tiny → re-eval is ~free) |
| Padding waste / truncation from sequence lengths | `token_length_profile` (dataset profiling) |

## Caveats

- **Quality** extrapolation from subset curves is rougher for LLMs (judge/gen
  metrics are noisy) — treat it as a hint, not a prediction.
- **QLoRA / 4-bit**: mlprobe measures wall-clock + VRAM fine (external
  measurement) but won't reason about the quantization tradeoff — your
  `train()` owns that.
- **Multi-node / from-scratch pretraining** is out of scope (single-process
  measurement model; quality there is scaling-law territory).
