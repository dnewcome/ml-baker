"""Tests for the baseline_compare scenario.

Covers the normalize-to-vector core (any result type → one metric vector), the
running-baseline lifecycle (establish → compare → promote), and the
direction-aware tradeoff verdicts.
"""

from __future__ import annotations

import pytest

from mlprof import (
    DatasetSpec,
    EvalMetric,
    ModelSpec,
    ProfileReport,
)
from mlprof.protocol import EvalResult
from mlprof.scenarios import Baseline, baseline_compare, to_vector


def _spec() -> ModelSpec:
    return ModelSpec(
        name="bc",
        train_callable="x:train",
        evaluate_callable="x:evaluate",
        dataset=DatasetSpec(loader="x:load", total_size=1000),
        eval_metrics=[
            EvalMetric(name="f1", higher_is_better=True, primary=True),
            EvalMetric(name="latency_ms", higher_is_better=False),
        ],
    )


# ---- to_vector normalization ----------------------------------------------

def test_to_vector_from_eval_result_and_dict():
    assert to_vector(EvalResult(metrics={"f1": 0.9})) == {"f1": 0.9}
    assert to_vector({"f1": 0.9, "cost_usd": 1.2}) == {"f1": 0.9, "cost_usd": 1.2}


def test_to_vector_from_profile_report_is_resources():
    pr = ProfileReport(spec_name="r", wall_clock_s=42.0, peak_rss_mb=2048.0, gpu_util_avg=70.0)
    vec = to_vector(pr)
    assert vec["wall_clock_s"] == 42.0
    assert vec["peak_rss_mb"] == 2048.0
    assert vec["gpu_util_avg"] == 70.0


def test_to_vector_merges_a_list_quality_plus_resources():
    quality = EvalResult(metrics={"f1": 0.9})
    profile = ProfileReport(spec_name="r", wall_clock_s=10.0, peak_rss_mb=512.0)
    vec = to_vector([quality, profile])
    assert vec["f1"] == 0.9 and vec["wall_clock_s"] == 10.0 and vec["peak_rss_mb"] == 512.0


def test_to_vector_artifact_path_without_spec_raises():
    with pytest.raises(ValueError):
        to_vector("s3://bucket/model.tar.gz")  # no spec → can't evaluate


# ---- direction handling ---------------------------------------------------

def test_resource_metric_is_lower_is_better_and_spec_overrides():
    spec = _spec()
    # cost_usd not declared → resource default lower-is-better → an increase is worse.
    r = baseline_compare.run({"cost_usd": 1.68}, baseline={"cost_usd": 1.20}, spec=spec)
    d = {x.metric: x for x in r.data["deltas"]}["cost_usd"]
    assert d.direction == "lower"
    assert d.verdict == "worse"

    # latency_ms declared higher_is_better=False → lower better → a decrease is better.
    r2 = baseline_compare.run({"latency_ms": 38.0}, baseline={"latency_ms": 45.0}, spec=spec)
    d2 = {x.metric: x for x in r2.data["deltas"]}["latency_ms"]
    assert d2.verdict == "better"


# ---- running-baseline lifecycle -------------------------------------------

def test_establishes_baseline_when_none_given():
    r = baseline_compare.run(EvalResult(metrics={"f1": 0.9}), spec=_spec())
    assert r.data["established"] is True
    assert isinstance(r.data["baseline"], Baseline)
    assert r.data["baseline"].vector == {"f1": 0.9}
    assert "No baseline yet" in r.answer


def test_promote_candidate_to_next_baseline():
    spec = _spec()
    base = baseline_compare.run({"f1": 0.90}, spec=spec).data["baseline"]
    r = baseline_compare.run({"f1": 0.93}, baseline=base, spec=spec)
    promoted = r.data["candidate"]
    assert isinstance(promoted, Baseline)
    assert promoted.vector == {"f1": 0.93}
    # The promoted baseline works as the next comparison's baseline.
    r2 = baseline_compare.run({"f1": 0.95}, baseline=promoted, spec=spec)
    assert r2.data["baseline"].vector == {"f1": 0.93}


# ---- verdicts -------------------------------------------------------------

def test_strict_improvement_recommends_adoption():
    spec = _spec()
    r = baseline_compare.run(
        {"f1": 0.95, "latency_ms": 30.0}, baseline={"f1": 0.90, "latency_ms": 45.0}, spec=spec
    )
    assert "Strict improvement" in r.recommendation
    assert all(d.verdict == "better" for d in r.data["deltas"])


def test_regression_recommends_keeping_baseline():
    spec = _spec()
    r = baseline_compare.run(
        {"f1": 0.85, "latency_ms": 60.0}, baseline={"f1": 0.90, "latency_ms": 45.0}, spec=spec
    )
    assert "Regression" in r.recommendation


def test_mixed_is_framed_as_a_tradeoff():
    spec = _spec()
    # f1 up (better), cost up (worse) → tradeoff, no dominance.
    r = baseline_compare.run(
        {"f1": 0.93, "cost_usd": 1.68}, baseline={"f1": 0.91, "cost_usd": 1.20}, spec=spec
    )
    assert "Tradeoff" in r.recommendation
    verdicts = {d.metric: d.verdict for d in r.data["deltas"]}
    assert verdicts["f1"] == "better" and verdicts["cost_usd"] == "worse"


def test_within_tolerance_is_a_wash():
    spec = _spec()
    r = baseline_compare.run({"f1": 0.9005}, baseline={"f1": 0.9000}, spec=spec, tolerance=0.01)
    assert r.data["deltas"][0].verdict == "wash"
    assert "unchanged" in r.recommendation.lower()
