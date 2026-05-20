"""A synthetic trainable that exercises the full ml-baker pipeline without
needing any ML library. Use it as a runnable reference for what a real
user's TrainFn / EvaluateFn / LoadDatasetFn look like.

Behavior:
  - Training time scales roughly linearly with rows × epochs / batch_size,
    so the scaling fit should pick the linear or power model with exponent ≈ 1.
  - Quality follows a logarithmic curve in data size — diminishing returns —
    so the quality fit should pick the log model.
  - ``encoder`` choice acts as a quality knob so the Pareto frontier has
    actual structure across configs.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from ml_baker import EvalResult, RuntimeConfig, TrainResult


TOTAL_ROWS = 50_000   # pretend the full dataset has this many rows


def load(subset_fraction: float = 1.0, split: str | None = None, seed: int | None = None):
    n_rows = max(1, int(TOTAL_ROWS * subset_fraction))
    return {"n_rows": n_rows, "split": split or "train", "seed": seed}


def train(
    config: dict,
    dataset_subset: dict,
    output_dir: Path,
    resume_from: Path | None = None,
    runtime: RuntimeConfig = RuntimeConfig(),
) -> TrainResult:
    n = dataset_subset["n_rows"]
    bs = int(config.get("batch_size", 32))
    epochs = int(config.get("epochs", 1))
    encoder = str(config.get("encoder", "small"))

    # Encoder choice affects throughput multiplier.
    encoder_cost = {"small": 1.0, "medium": 2.0, "large": 4.0}.get(encoder, 1.0)

    # Simulated training time. Linear in steps; touch enough wall-clock that
    # the sampler captures something.
    steps = max(1, (n // bs) * epochs)
    seconds = steps * 1e-4 * encoder_cost
    time.sleep(min(seconds, 5.0))   # cap so demos don't drag

    artifact = output_dir / "model.json"
    artifact.write_text(json.dumps({
        "trained_on_rows": n,
        "encoder": encoder,
        "lr": config.get("lr"),
    }))

    final_loss = 1.0 / math.log(n + 10) * (1.0 + 0.1 / encoder_cost)
    return TrainResult(
        artifact_path=artifact,
        metrics={"final_loss": float(final_loss)},
        steps_completed=steps,
    )


def evaluate(artifact_path: Path, eval_set: dict) -> EvalResult:
    info = json.loads(Path(artifact_path).read_text())
    n = info["trained_on_rows"]
    encoder = info["encoder"]

    # Logarithmic learning curve with an encoder-dependent asymptote and base.
    encoder_quality = {"small": 0.70, "medium": 0.80, "large": 0.88}.get(encoder, 0.70)
    f1 = min(encoder_quality, 0.45 + 0.045 * math.log(max(n, 1)))
    latency = {"small": 5.0, "medium": 9.0, "large": 18.0}.get(encoder, 5.0)
    return EvalResult(metrics={"f1_macro": f1, "inference_latency_ms": latency})
