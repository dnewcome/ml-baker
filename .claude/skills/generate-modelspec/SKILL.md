---
name: generate-modelspec
description: Analyze a training script, notebook, or MLflow run and generate a mlprof ModelSpec for it, then run audit() and print findings. Use when the user wants to bootstrap a ModelSpec for an existing model without writing it by hand. Triggers on phrases like "generate a mlprof spec", "audit my training code", "make a ModelSpec for this model", or when pointed at a training file / MLflow run with the intent to use mlprof on it.
---

# generate-modelspec

Bootstraps a `mlprof.ModelSpec` from existing training code so the user can
run `audit()` on a real model in minutes instead of hand-writing a spec.

## When to use

- User asks to create / generate / bootstrap a mlprof ModelSpec
- User says "audit my training code" or "make a spec for this model" or
  "set this model up for mlprof"
- User points at a training script / notebook / MLflow run and wants to
  use mlprof on it

## Inputs

Ask the user which of these to analyze (or use what they provided):

1. **A directory** containing training code. You'll need to find the
   training entry point inside it.
2. **A specific `.py` file** containing the training logic.
3. **A `.ipynb` notebook** (use Read — Jupyter notebooks are supported).
4. **An MLflow run ID** (requires `mlflow` Python package). If unavailable,
   fall back to asking for a file.

If unclear, ask once. Do not assume.

## Workflow

### 1. Identify framework

Read the target file(s) and grep for imports:

| Framework signal | Conclusion |
|---|---|
| `import torch`, `from torch` | `framework="pytorch"` |
| `from transformers` | `framework="pytorch"`, note transformers usage |
| `import sklearn`, `from sklearn` | `framework="sklearn"`, `cpu_bound=True` |
| `import spacy` | `framework="spacy"`, `cpu_bound=True` |
| `import lightgbm`, `import xgboost` | `framework="lightgbm"` or `"xgboost"`, `cpu_bound=True` |
| `import tensorflow`, `import keras` | `framework="tensorflow"` |
| Mixed (torch + transformers) | `framework="pytorch"` — transformers is built on torch |

### 2. Find the entry points

Look for these in order:

- A function literally named `train`, `main`, `fit`, or `run_training`
- A `Trainer.train()` call (HuggingFace pattern)
- The `if __name__ == "__main__"` block
- For notebooks: cells with `model.fit(...)`, `trainer.train()`, or explicit training loops

Also find:

- An `evaluate` / `eval` function (often separate from training)
- A dataset loader (often `load_dataset`, `load_data`, `get_dataloader`)

If the codebase has separate train.py / evaluate.py / data.py files, use
those as the dotted-path targets in the generated spec.

### 3. Extract hyperparameters

Look for, in this order of reliability:

- `argparse.ArgumentParser().add_argument("--lr", type=float, default=...)` 
- `@click.option("--lr", type=float, default=...)`
- Hydra / OmegaConf config files (read the YAML/JSON to extract defaults)
- HuggingFace `TrainingArguments(...)` field values
- Hardcoded constants that look hyperparameter-shaped:
  `LEARNING_RATE`, `BATCH_SIZE`, `NUM_EPOCHS`, `LR`, etc.

For each hyperparameter, capture: `name`, `type` (int/float/categorical/bool),
`default`. Infer `min`/`max` only when there's an obvious constraint
(e.g. probability ∈ [0, 1], dropout ∈ [0, 1]); otherwise pick reasonable
generous bounds around the default and leave a `# TODO: verify range` comment.

**Do not invent hyperparameters that aren't in the code.** Missing one is
better than fabricating.

### 4. Detect Capabilities (the high-leverage step)

For each capability, grep for the listed evidence. Set `True` only if
evidence is found. Set `False` if there's evidence of the opposite (e.g.
explicit `num_workers=0`). Otherwise leave as `False` and add a TODO
comment listing what to look for.

| Capability | Evidence patterns |
|---|---|
| `supports_checkpointing` | `save_strategy != "no"` in TrainingArguments; `ModelCheckpoint` callback; `torch.save(... state_dict ...)` inside a training loop with periodic call; `keras.callbacks.ModelCheckpoint` |
| `supports_incremental_training` | `resume_from_checkpoint` arg passed to Trainer; explicit `resume_from=` param in train function; warm-start logic loading prior weights |
| `supports_parallel_data_loading` | `DataLoader(..., num_workers=N)` with N > 0; sklearn estimator with `n_jobs != 1`; `tf.data` with `num_parallel_calls` |
| `supports_multi_gpu_data_parallel` | `DistributedDataParallel`, `DDP`; `accelerate launch`; `torchrun` in entrypoint scripts; HF `Trainer` (auto-DDP capable on multi-GPU); `tf.distribute.MirroredStrategy` |
| `supports_multi_gpu_model_parallel` | `device_map="auto"` (transformers); `FullyShardedDataParallel`/`FSDP`; tensor-parallelism libraries (`megatron`, `parallelformers`) |
| `supports_mixed_precision` | `fp16=True` or `bf16=True` in TrainingArguments; `torch.cuda.amp.autocast`; `GradScaler`; explicit `dtype=torch.float16` or `bfloat16`; `tf.keras.mixed_precision` |
| `supports_gradient_accumulation` | `gradient_accumulation_steps` field set in TrainingArguments; manual accumulation loops dividing loss before backward |
| `deterministic` | `torch.manual_seed(...)`; `random.seed(...)`; `np.random.seed(...)`; HF `set_seed(...)`; `torch.use_deterministic_algorithms(True)` |

If the user has multiple training files, scan all of them — capabilities
can be set in a setup file and used elsewhere.

### 5. Framework hints

- `requires_gpu`: `True` only if there's hardcoded `cuda` usage with no
  CPU fallback path (rare; usually False).
- `expected_device`: `"gpu"` if torch + transformers and not cpu_bound;
  `"cpu"` if sklearn / spacy / lightgbm; `"either"` if unclear.
- `cpu_bound`: `True` for sklearn / spacy / lightgbm / xgboost; `False`
  for torch with GPU usage detected; leave as TODO otherwise.
- `min_vram_gb`: nearly impossible to estimate statically. Leave as TODO
  unless the model is a known size (e.g. `distilbert-base-uncased` → 6;
  `bert-base` → 8; `bert-large` → 16; `gpt2-medium` → 12).

### 6. Suggest target instances

Based on framework + cpu_bound, suggest 2–3 targets from the mlprof
catalog (see `mlprof/targets.py`). Examples:

- `cpu_bound=True` → `c5.4xlarge`, `c5.9xlarge`, optionally one GPU
  instance like `g5.xlarge` to demonstrate the `cpu_bound_on_gpu_target`
  warning the audit will fire.
- GPU usage detected (single GPU sufficient) → `g5.xlarge` (sweet spot),
  `p3.2xlarge` (older comparison), `g5.12xlarge` (multi-GPU; will trigger
  `idle_gpus_no_multi_gpu` if multi-GPU not declared — that's diagnostic).
- Large model with multi-GPU declared → `g5.12xlarge`, `p3.8xlarge`,
  `p4d.24xlarge`.

Don't dump the whole catalog. 2–3 instances is enough to surface
interesting tradeoffs.

### 7. Eval metrics

Grep for: `accuracy_score`, `f1_score`, `roc_auc`, `mean_squared_error`,
`precision`/`recall`, perplexity calculations, custom `evaluate()`
functions and inspect their return values.

Always mark exactly one metric `primary=True`. If detection is ambiguous,
pick the first detected metric and leave a TODO comment.

### 8. Generate `model_spec.py`

Write to the user's current working directory unless they specify
otherwise. Use this template (fill in fields from your detection):

```python
"""mlprof ModelSpec generated from <source path>.

This spec was auto-generated. TODO comments mark fields that need human
verification — read them carefully before running probes. Audit findings
below this comment block reflect ONLY the declared capabilities; if you
fix a TODO, re-run the audit.
"""

from mlprof import (
    Capabilities,
    CategoricalSweep,
    DatasetSpec,
    EvalMetric,
    FrameworkHints,
    HyperParam,
    ModelSpec,
    NumericSweep,
    ProbeConfig,
    TargetInstance,
)


spec = ModelSpec(
    name="<inferred-from-file-or-dir>",
    train_callable="<dotted.module:train>",          # TODO: verify path
    evaluate_callable="<dotted.module:evaluate>",    # TODO: verify path
    dataset=DatasetSpec(
        loader="<dotted.module:load>",               # TODO: verify path + confirm
                                                     # the loader accepts subset_fraction
        total_size=<int>,                            # TODO: replace with actual row count
        eval_split="test",                           # TODO: verify split name
    ),
    eval_metrics=[
        EvalMetric(name="<metric>", higher_is_better=True, primary=True),
        # ... additional metrics
    ],
    hyperparameters=[
        # ... detected hyperparams (no sweeps by default; add NumericSweep / CategoricalSweep
        # later if you want to probe a range)
    ],
    capabilities=Capabilities(
        # All False by default. Each True below has evidence noted in the
        # `# detected:` comment. Each TODO below means evidence is absent
        # — verify in your code before flipping to True.
    ),
    framework_hints=FrameworkHints(
        framework="<detected>",
        expected_device="<detected>",
        cpu_bound=<detected_or_TODO>,
        requires_gpu=<detected>,
        # min_vram_gb=...,  # TODO: estimate based on your model
    ),
    targets=[
        # ... 2-3 suggested targets
    ],
    probe=ProbeConfig(
        subset_fractions=[0.005, 0.02, 0.08],
        timeout_seconds=1800,
    ),
)


if __name__ == "__main__":
    from mlprof import audit
    print(audit(spec).format())
```

For each detected capability or hint, add an inline comment like:

```python
supports_mixed_precision=True,   # detected: TrainingArguments(fp16=True) at train.py:42
supports_checkpointing=False,    # TODO: not detected — look for save_strategy / ModelCheckpoint
```

### 9. Run the audit

After writing the file, run:

```python
from mlprof import audit
# reload spec by importing the generated file or constructing inline
print(audit(spec).format())
```

Capture the output verbatim. This is the most valuable artifact the user
gets from running the skill.

### 10. Summary message

Print a structured summary to the user, in this shape:

```
GENERATED: model_spec.py

DETECTED (with evidence):
  - framework=pytorch (import torch at train.py:1)
  - supports_mixed_precision (TrainingArguments(fp16=True) at train.py:42)
  - deterministic (set_seed(42) at train.py:18)
  - hyperparams: lr (float, default=2e-5), batch_size (int, default=16), epochs (int, default=3)

NEEDS VERIFICATION (TODOs in the file):
  - dataset.total_size — replace placeholder with actual row count
  - dataset.loader — verify it accepts a subset_fraction parameter
  - capabilities.supports_checkpointing — search train.py for save_strategy / ModelCheckpoint
  - framework_hints.min_vram_gb — estimate based on your model size

AUDIT FINDINGS (from the generated spec as-is):
  <paste full audit output>

NEXT:
  1. Resolve the TODOs in model_spec.py
  2. Re-run: python model_spec.py
  3. If audit looks good, add probe configuration and run: mlprof.run(spec, ...)
```

## Hard constraints

- **Be honest.** Never set a capability to `True` without grep-able
  evidence in the user's code. A spec that overstates capabilities makes
  the audit lie, which is worse than no audit.
- **Don't fabricate hyperparameters.** If you can't find `lr`, omit it.
- **Don't guess** `min_vram_gb`, `total_size`, or `eval_split` — these
  are facts the user knows. Leave as TODO.
- **Always run `audit()` after generating.** The audit output is the
  point. Without it the skill just produces a stub.
- **The generated spec must validate** (pydantic-wise). If you can't pick
  a sensible primary metric, pick the first detected and TODO-comment it.

## Limitations to communicate

When summarizing, tell the user:

- This is static analysis. It misses capabilities set in helper
  functions, conditional branches, or external config that wasn't read.
- TODOs aren't optional — the audit is only as honest as the spec.
- If hyperparameters live in YAML/Hydra config files outside the script,
  point Claude at those too and re-run.
- For MLflow runs: only params, metrics, and tags from the run are
  used. If the run logged a model artifact, suggest the user point Claude
  at the underlying training script for capability detection — MLflow
  alone can't reveal whether the code checkpoints, uses mixed precision,
  etc.
