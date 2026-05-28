"""Tests for the scenarios framework.

All probes run with ``launcher="in_process"`` against the synthetic trainable
so the suite needs no ML libraries and no subprocess/path coordination. The
synthetic times are a controlled simulation, so the analyzers have clean
signal: a near-linear scaling law, an Amdahl speedup curve with a known
parallel fraction, and a speed/quality tradeoff between two algorithms.
"""

from __future__ import annotations

import pytest

from mlprobe import (
    Capabilities,
    DatasetSpec,
    EvalMetric,
    FrameworkHints,
    HyperParam,
    ModelSpec,
    ProbeConfig,
)
from mlprobe.scenarios import (
    ScenarioResult,
    algorithm_selection,
    parallelization_effect,
    scaling_with_n,
)

import synthetic_trainable as syn


def build_spec(
    name: str = "syn",
    *,
    algorithm: str = "fast",
    total_size: int = syn.TOTAL_ROWS,
    subset_fractions=(0.2, 0.6, 1.0),
    train_callable: str = "synthetic_trainable:train",
) -> ModelSpec:
    return ModelSpec(
        name=name,
        train_callable=train_callable,
        evaluate_callable="synthetic_trainable:evaluate",
        dataset=DatasetSpec(loader="synthetic_trainable:load", total_size=total_size),
        eval_metrics=[EvalMetric(name="f1", higher_is_better=True, primary=True)],
        hyperparameters=[
            HyperParam(
                name="algorithm", type="categorical", default=algorithm,
                values=["fast", "accurate"],
            )
        ],
        capabilities=Capabilities(),
        framework_hints=FrameworkHints(expected_device="cpu"),
        probe=ProbeConfig(subset_fractions=list(subset_fractions), timeout_seconds=30),
    )


# ---- scaling_with_n -------------------------------------------------------

def test_scaling_with_n_recovers_linear_exponent():
    spec = build_spec(subset_fractions=(0.2, 0.6, 1.0))
    result = scaling_with_n.run(
        spec, target="c5.xlarge", launcher="in_process", progress=False
    )

    assert isinstance(result, ScenarioResult)
    assert result.scenario == "scaling_with_n"
    assert result.data["fit_model"] is not None
    # Synthetic time is linear in N → exponent should land near 1.0.
    assert 0.6 <= result.data["exponent"] <= 1.4
    assert "O(N^" in result.answer
    # total_size is known → full-dataset extrapolation present.
    assert result.data["extrapolated_time_s"] is not None
    assert result.passed is None


def test_scaling_with_n_gate_on_max_exponent():
    spec = build_spec()
    result = scaling_with_n.run(
        spec, target="c5.xlarge", max_exponent=1.5,
        launcher="in_process", progress=False,
    )
    assert result.passed is True  # linear data is well under exponent 1.5

    strict = scaling_with_n.run(
        spec, target="c5.xlarge", max_exponent=0.5,
        launcher="in_process", progress=False,
    )
    assert strict.passed is False


def test_scaling_with_n_inconclusive_on_total_failure():
    spec = build_spec(train_callable="synthetic_trainable:does_not_exist")
    result = scaling_with_n.run(
        spec, target="c5.xlarge", launcher="in_process", progress=False
    )
    assert result.data.get("fit_model") is None
    assert "Inconclusive" in result.answer


# ---- parallelization_effect ----------------------------------------------

def test_parallelization_effect_recovers_amdahl_fraction():
    spec = build_spec()
    result = parallelization_effect.run(
        spec, target="c5.4xlarge", n_cpus=[1, 2, 4, 8],
        subset_fraction=1.0, launcher="in_process", progress=False,
    )

    assert result.data["axis"] == "n_cpus"
    points = result.data["points"]
    assert len(points) >= 3
    # More workers → faster (speedup grows from the 1-CPU baseline).
    assert points[-1]["speedup"] > points[0]["speedup"] > 0.9
    # Recovered parallel fraction should be in the neighbourhood of the
    # simulated PARALLEL_FRACTION (0.8); allow slack for measurement overhead.
    p = result.data["amdahl_parallel_fraction"]
    assert p is not None and 0.5 <= p <= 0.95


def test_parallelization_effect_clamps_to_instance_vcpus():
    spec = build_spec()
    # c5.xlarge has 4 vCPUs; the 16/32 requests must be dropped.
    result = parallelization_effect.run(
        spec, target="c5.xlarge", n_cpus=[1, 2, 4, 16, 32],
        subset_fraction=1.0, launcher="in_process", progress=False,
    )
    workers = [pt["workers"] for pt in result.data["points"]]
    assert max(workers) <= 4


# ---- algorithm_selection --------------------------------------------------

def test_algorithm_selection_picks_higher_quality_when_it_matters():
    fast = build_spec(name="syn-fast", algorithm="fast")
    accurate = build_spec(name="syn-accurate", algorithm="accurate")

    result = algorithm_selection.run(
        [fast, accurate], target="c5.xlarge", subset_fraction=1.0,
        launcher="in_process", progress=False,
    )

    rows = {r["name"]: r for r in result.data["rows"]}
    assert rows["syn-accurate"]["quality"] > rows["syn-fast"]["quality"]
    assert rows["syn-accurate"]["wall_clock_s"] > rows["syn-fast"]["wall_clock_s"]
    # Quality edge is meaningful here, so the higher-quality variant wins.
    assert result.data["winner"] == "syn-accurate"


def test_algorithm_selection_reports_failed_variant():
    good = build_spec(name="syn-good", algorithm="fast")
    broken = build_spec(name="syn-broken", train_callable="synthetic_trainable:nope")

    result = algorithm_selection.run(
        [good, broken], target="c5.xlarge", subset_fraction=1.0,
        launcher="in_process", progress=False,
    )
    rows = {r["name"]: r for r in result.data["rows"]}
    assert rows["syn-broken"]["error"] is not None
    assert result.data["winner"] == "syn-good"
    assert "FAILED" in result.answer


def test_algorithm_selection_requires_two_specs():
    with pytest.raises(ValueError):
        algorithm_selection.run([build_spec()], launcher="in_process", progress=False)


# ---- ScenarioResult formatting -------------------------------------------

def test_scenario_result_format_includes_gate():
    r = ScenarioResult(
        scenario="x", question="q?", answer="a", recommendation="r", passed=True
    )
    out = r.format()
    assert "GATE: PASS" in out
    assert "Q: q?" in out

    r2 = ScenarioResult(scenario="x", question="q?", answer="a", recommendation="r")
    assert "GATE" not in r2.format()
