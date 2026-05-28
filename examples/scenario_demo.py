"""End-to-end demo of the scenarios framework.

Runs the three baseline scenarios against a synthetic trainable that needs no
ML libraries:

  - ``scaling_with_n``        — does training time scale linearly with N?
  - ``parallelization_effect`` — where does adding CPUs stop helping?
  - ``algorithm_selection``    — which of two algorithms wins at a given N?

The trainable's wall-clock is a controlled simulation: linear in rows,
Amdahl-parallel in ``runtime.n_cpus``, with a "fast" vs "accurate" speed/quality
knob. Probes run via the default subprocess launcher (cwd must be the repo
root so ``examples.scenario_demo`` is importable in the probe subprocess).

    python examples/scenario_demo.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from mlprof import (
    Capabilities,
    DatasetSpec,
    EvalMetric,
    EvalResult,
    FrameworkHints,
    HyperParam,
    ModelSpec,
    ProbeConfig,
    RuntimeConfig,
    TrainResult,
)
from mlprof.scenarios import (
    algorithm_selection,
    parallelization_effect,
    scaling_with_n,
)


TOTAL_ROWS = 20_000
TIME_PER_ROW = 3e-5
PARALLEL_FRACTION = 0.75
_ALGORITHMS = {"fast": (1.0, 0.80), "accurate": (2.0, 0.92)}


def load(subset_fraction: float = 1.0, split: str | None = None, seed: int | None = None):
    return {"n_rows": max(1, int(TOTAL_ROWS * subset_fraction))}


def train(config, dataset_subset, output_dir, resume_from=None, runtime=RuntimeConfig()):
    n = dataset_subset["n_rows"]
    speed_mult, _ = _ALGORITHMS.get(str(config.get("algorithm", "fast")), _ALGORITHMS["fast"])
    serial = n * TIME_PER_ROW * speed_mult
    cpus = max(1, runtime.n_cpus)
    seconds = serial * ((1.0 - PARALLEL_FRACTION) + PARALLEL_FRACTION / cpus)
    time.sleep(min(seconds, 5.0))
    artifact = Path(output_dir) / "model.json"
    artifact.write_text(json.dumps({"n_rows": n, "algorithm": config.get("algorithm", "fast")}))
    return TrainResult(artifact_path=artifact, metrics={"input_rows": float(n)})


def evaluate(artifact_path, eval_set) -> EvalResult:
    info = json.loads(Path(artifact_path).read_text())
    _, ceiling = _ALGORITHMS.get(info["algorithm"], _ALGORITHMS["fast"])
    f1 = min(ceiling, 0.40 + 0.05 * math.log(max(info["n_rows"], 1)))
    return EvalResult(metrics={"f1": f1})


def _spec(name: str, algorithm: str = "fast") -> ModelSpec:
    return ModelSpec(
        name=name,
        train_callable="examples.scenario_demo:train",
        evaluate_callable="examples.scenario_demo:evaluate",
        dataset=DatasetSpec(loader="examples.scenario_demo:load", total_size=TOTAL_ROWS),
        eval_metrics=[EvalMetric(name="f1", higher_is_better=True, primary=True)],
        hyperparameters=[
            HyperParam(name="algorithm", type="categorical", default=algorithm,
                       values=["fast", "accurate"])
        ],
        capabilities=Capabilities(),
        framework_hints=FrameworkHints(expected_device="cpu"),
        probe=ProbeConfig(subset_fractions=[0.2, 0.5, 1.0], timeout_seconds=60),
    )


def main() -> None:
    spec = _spec("synthetic-demo")

    print("\n# scaling_with_n -----------------------------------------------")
    r = scaling_with_n.run(spec, target="c5.xlarge", progress=False)
    print(r.format())

    print("\n# parallelization_effect ---------------------------------------")
    r = parallelization_effect.run(
        spec, target="c5.9xlarge", n_cpus=[1, 2, 4, 8, 16, 36],
        subset_fraction=1.0, progress=False,
    )
    print(r.format())

    print("\n# algorithm_selection ------------------------------------------")
    r = algorithm_selection.run(
        [_spec("fast-algo", "fast"), _spec("accurate-algo", "accurate")],
        target="c5.4xlarge", subset_fraction=1.0, progress=False,
    )
    print(r.format())


if __name__ == "__main__":
    main()
