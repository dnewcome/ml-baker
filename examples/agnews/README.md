# AG News demo

End-to-end ml-baker demo on a real NLP dataset
([AG News](https://huggingface.co/datasets/ag_news): 120k news articles,
4 topic classes). Two architectures at very different points on the
cost/quality curve:

- **sklearn TF-IDF + LogReg** — cheap CPU baseline (~92% accuracy)
- **DistilBERT fine-tune** — heavier, GPU-preferred (~94–95% accuracy)

Each runs through the standard ml-baker pipeline: declarative audit →
probe on small subsets → fit scaling curves → predict cost / memory /
quality at the full dataset → Pareto frontier.

The specs are deliberately set up to surface key audit findings:

- The sklearn spec targets a GPU instance (`g5.xlarge`) so the audit
  fires `cpu_bound_on_gpu_target` — "you'd be paying GPU prices for
  CPU-bound code."
- The DistilBERT spec declares `max_useful_gpus=2` and targets
  `g5.12xlarge` (4 GPUs) so the audit fires `exceeds_max_useful_gpus`
  — "two of those four GPUs would sit idle."

See the [top-level README](../../README.md) for the conceptual
introduction to ml-baker.

## Prerequisites

- Python 3.10+
- ~2 GB free disk space for first-time HuggingFace model + dataset downloads
- For DistilBERT in reasonable time: a CUDA GPU. CPU works but is ~30×
  slower per epoch.

## Quickstart

```bash
# From the repo root
./examples/agnews/run.sh                  # full demo (both architectures)
./examples/agnews/run.sh --sklearn-only   # fast variant only (~30s–2min)
./examples/agnews/run.sh --distilbert-only
```

The script installs the `[demo]` extras (`datasets`, `scikit-learn`,
`torch`, `transformers`, `accelerate`) on first run if they aren't
importable yet. Subsequent runs skip the install probe.

If `python3` on your `PATH` is not the interpreter you want:

```bash
PYTHON=~/.pyenv/versions/3.11.13/bin/python ./examples/agnews/run.sh
```

## Manual install + run

If you'd rather not use the script:

```bash
pip install -e ".[demo]"
python examples/agnews/demo.py [--sklearn-only | --distilbert-only]
```

## Expected runtime

| Variant | CPU laptop (M-series Mac or similar) | GPU (A10G class) |
|---|---|---|
| `--sklearn-only` (6 probes) | ~30s – 2 min | ~30s |
| `--distilbert-only` (6 probes) | ~30 – 60 min | ~3 – 5 min |
| full demo | ~30 – 60 min | ~5 min |

First run also downloads AG News (~10 MB) and `distilbert-base-uncased`
(~250 MB) into the HuggingFace cache (`~/.cache/huggingface`); subsequent
runs reuse them.

## What you'll see

The script prints, for each spec:

1. The **audit** — capability and target-compatibility findings derived
   from the spec alone (no compute).
2. A **progress line** per probe as the runner launches subprocesses.
3. The **report** — extrapolated time, cost, accuracy, memory at the full
   dataset for each (config, target); post-probe warnings if predicted
   memory exceeds target capacity; the Pareto frontier across
   `(cost, accuracy)`.

A trimmed example from the sklearn variant (full demo's report is longer):

```
EXTRAPOLATED AT FULL DATASET:
  c5.4xlarge  [C=1.0, max_features=20000] → t=16.2s  cost=$0.003
              accuracy=0.9244  ram=938MB
      (time=loglinear@R²1.00, qual=log@R²0.99, ram=linear@R²1.00)
  g5.xlarge   [C=1.0, max_features=20000] → t=18.8s  cost=$0.005
              accuracy=0.9273  ram=946MB
      (time=loglinear@R²1.00, qual=log@R²0.99, ram=linear@R²1.00)

PARETO FRONTIER (cost ↓, accuracy optimum):
  c5.4xlarge  [C=1.0, max_features=20000] → cost=$0.003 accuracy=0.9244
  g5.xlarge   [C=1.0, max_features=20000] → cost=$0.005 accuracy=0.9273
```

## Files

- `data.py` — shared `LoadDatasetFn`. Wraps HF `datasets.load_dataset`,
  supports subset fraction + split + seed, reuses the HF on-disk cache so
  subprocess probes don't re-download.
- `sklearn_logreg.py` — `TrainFn` / `EvaluateFn` for the sklearn variant.
  Pickles a `Pipeline(TfidfVectorizer, LogisticRegression)` artifact.
- `distilbert.py` — `TrainFn` / `EvaluateFn` for the DistilBERT variant.
  Uses the HuggingFace `Trainer`. Honors `runtime.precision` (fp16/bf16),
  `runtime.seed`, `runtime.n_cpus` for the DataLoader, and auto-detects
  GPU via `torch.cuda.is_available()`. Lazy-imports torch/transformers so
  the module can be imported without those deps installed.
- `demo.py` — builds both `ModelSpec`s and runs them. CLI:
  `--sklearn-only`, `--distilbert-only`, or no flag for both.
- `run.sh` — install-and-run wrapper around `demo.py`.

## Caveats

- **Probes run as local subprocesses.** Wall-clock is measured on *your*
  machine. The cost numbers apply that throughput to the target
  instance's `$/hr`. Relative ordering across architectures and configs
  is faithful; absolute "what would this cost on a real `g5.xlarge`"
  needs the Docker launcher
  ([issue #1](https://github.com/dnewcome/ml-baker/issues/1)) or a
  SageMaker launcher.
- **The instance-type catalog uses approximate `us-east-1` on-demand
  prices** stamped with `PRICE_AS_OF = "2026-05-20"`. Live region-aware
  pricing is
  [issue #2](https://github.com/dnewcome/ml-baker/issues/2).
- **DistilBERT spec doesn't declare a CPU target** because running it on
  CPU for the demo would take hours. The audit *would* still flag a CPU
  target if you added one (and would not block the run — DistilBERT
  declares `requires_gpu=False`).
