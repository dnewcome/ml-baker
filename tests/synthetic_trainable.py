"""A synthetic, dependency-free trainable for exercising scenarios in tests.

Its wall-clock is a *controlled simulation* so the analyzers have clean
signal to fit:

  - Time scales linearly in N (rows), so ``scaling_with_n`` should recover an
    exponent near 1.0.
  - Time responds to ``runtime.n_cpus`` via Amdahl's law with a known
    parallelizable fraction P, so ``parallelization_effect`` should recover
    P ≈ that value.
  - The ``algorithm`` config knob trades speed for quality, so
    ``algorithm_selection`` has a real decision to make.

Times are deliberately tiny (sub-second) so the whole suite runs fast.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from mlprobe import EvalResult, RuntimeConfig, TrainResult


TOTAL_ROWS = 10_000
TIME_PER_ROW = 6e-5        # serial seconds per row at the baseline algorithm
PARALLEL_FRACTION = 0.8    # Amdahl P the parallelization scenario should recover
MAX_SLEEP_S = 2.0          # safety cap so a misconfigured probe can't hang tests

# Per-algorithm (speed multiplier, quality ceiling). "accurate" is 2x slower
# but reaches a higher quality asymptote than "fast".
_ALGORITHMS = {
    "fast":     (1.0, 0.80),
    "accurate": (2.0, 0.92),
}


def load(subset_fraction: float = 1.0, split: str | None = None, seed: int | None = None):
    return {"n_rows": max(1, int(TOTAL_ROWS * subset_fraction)), "split": split}


def train(
    config: dict,
    dataset_subset: dict,
    output_dir: Path,
    resume_from: Path | None = None,
    runtime: RuntimeConfig = RuntimeConfig(),
) -> TrainResult:
    n = dataset_subset["n_rows"]
    algo = str(config.get("algorithm", "fast"))
    speed_mult, _ = _ALGORITHMS.get(algo, _ALGORITHMS["fast"])

    serial = n * TIME_PER_ROW * speed_mult
    cpus = max(1, runtime.n_cpus)
    # Amdahl: a fixed serial part plus a perfectly-parallel part / cpus.
    sim_seconds = serial * ((1.0 - PARALLEL_FRACTION) + PARALLEL_FRACTION / cpus)
    time.sleep(min(sim_seconds, MAX_SLEEP_S))

    artifact = output_dir / "model.json"
    artifact.write_text(json.dumps({"n_rows": n, "algorithm": algo}))
    return TrainResult(artifact_path=artifact, metrics={"input_rows": float(n)})


def evaluate(artifact_path: Path, eval_set: dict) -> EvalResult:
    info = json.loads(Path(artifact_path).read_text())
    n = info["n_rows"]
    _, ceiling = _ALGORITHMS.get(info["algorithm"], _ALGORITHMS["fast"])
    # Logarithmic learning curve toward the algorithm's quality ceiling.
    f1 = min(ceiling, 0.40 + 0.05 * math.log(max(n, 1)))
    return EvalResult(metrics={"f1": f1})


def evaluate_echo(artifact_path, eval_set) -> EvalResult:
    """Reports how it received its arguments — used to verify that
    evaluate_existing passes local paths as Path and URIs as str, and that the
    eval_set was materialized."""
    return EvalResult(metrics={
        "artifact_is_str": float(isinstance(artifact_path, str)),
        "eval_set_rows": float((eval_set or {}).get("n_rows", -1)),
    })


def analyze(dataset):
    """A data_audit analyze callable returning a single DatasetProfile."""
    from mlprobe import block_size_profile

    n = dataset["n_rows"]
    return block_size_profile([n // 2, n // 2])


def analyze_multi(dataset):
    """A data_audit analyze callable returning multiple DatasetProfiles."""
    from mlprobe import block_size_profile, class_balance_profile

    n = dataset["n_rows"]
    return [block_size_profile([n]), class_balance_profile({"pos": n // 2, "neg": n // 2})]
