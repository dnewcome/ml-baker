"""Synthetic LoRA fine-tune trainable — no ML libraries, runs in seconds.

Lets the mlprobe scenarios run end-to-end without torch/peft/a GPU, by
*simulating* the cost/quality behaviour of a LoRA instruction fine-tune:

  - wall-clock scales with tokens (rows × seq_len), the base model's cost
    multiplier, and a mild LoRA-rank factor, divided across runtime.n_gpus;
  - quality (exact_match) follows a diminishing-returns curve in data size,
    capped by the base model's ceiling;
  - the bigger base model is slower + higher-quality + higher inference
    latency — so algorithm_selection / baseline_compare have a real tradeoff.

The real callables are in lora_ft.py; this mirrors their interface exactly.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from mlprobe import EvalResult, RuntimeConfig, TrainResult


TOTAL_EXAMPLES = 120_000
AVG_SEQ_LEN = 256
TIME_PER_1K_TOKENS = 3e-4          # simulated seconds; tiny so demos are fast
MAX_SLEEP_S = 3.0

# base_model -> (cost_multiplier, quality_ceiling, gen_latency_ms)
_BASE = {
    "Llama-3.1-8B": (1.0, 0.82, 90.0),
    "Llama-3.2-3B": (0.45, 0.74, 45.0),
}


def load(subset_fraction: float = 1.0, split: str | None = None, seed: int | None = None):
    return {"n_examples": max(1, int(TOTAL_EXAMPLES * subset_fraction))}


def train(config, dataset_subset, output_dir, resume_from=None, runtime=RuntimeConfig()):
    n = dataset_subset["n_examples"]
    base = str(config.get("base_model", "Llama-3.1-8B"))
    cost_mult, _, _ = _BASE.get(base, _BASE["Llama-3.1-8B"])
    r = int(config.get("lora_r", 16))

    tokens = n * AVG_SEQ_LEN
    gpus = max(1, runtime.n_gpus)
    serial = (tokens / 1000.0) * TIME_PER_1K_TOKENS * cost_mult * (1.0 + r / 128.0)
    sim_seconds = serial * (0.15 + 0.85 / gpus)     # Amdahl-ish data-parallel speedup
    time.sleep(min(sim_seconds, MAX_SLEEP_S))

    artifact = Path(output_dir) / "adapter.json"
    artifact.write_text(json.dumps({"n": n, "base": base, "r": r}))
    return TrainResult(
        artifact_path=artifact,
        metrics={"train_loss": 1.0 / math.log(n + 10), "input_rows": float(n)},
    )


def evaluate(artifact_path, eval_set) -> EvalResult:
    info = json.loads(Path(artifact_path).read_text())
    _, ceiling, latency = _BASE.get(info["base"], _BASE["Llama-3.1-8B"])
    em = min(ceiling, 0.35 + 0.05 * math.log(max(info["n"], 1)))
    return EvalResult(metrics={"exact_match": em, "gen_latency_ms": latency})
