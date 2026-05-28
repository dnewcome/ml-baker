"""Tests for library mode: measure() as a public primitive and profile().

No MLflow needed — the MLflow logging path is exercised only for its
graceful-degradation behaviour (warn + no-op when mlflow is absent).
"""

from __future__ import annotations

import time

import pytest

import mlprof
from mlprof import (
    Capabilities,
    DatasetSpec,
    EvalMetric,
    FrameworkHints,
    IncompatibleSpecError,
    ModelSpec,
    ProfileReport,
    TargetInstance,
)


# ---- measure() as the public primitive ------------------------------------

def test_measure_times_a_block():
    with mlprof.measure() as m:
        time.sleep(0.05)
    assert m.wall_clock_s >= 0.04
    assert m.peak_rss_mb > 0


# ---- profile(): stages + report -------------------------------------------

def test_profile_captures_overall_and_stages():
    with mlprof.profile(name="job") as p:
        with p.stage("load"):
            time.sleep(0.05)
        with p.stage("train"):
            time.sleep(0.10)

    report = p.report()
    assert isinstance(report, ProfileReport)
    assert report.spec_name == "job"
    assert report.wall_clock_s >= 0.14

    stages = {s.name: s for s in report.stages}
    assert set(stages) == {"load", "train"}
    assert stages["train"].wall_clock_s > stages["load"].wall_clock_s
    # Stages don't account for everything, but shouldn't exceed the whole run.
    assert sum(s.wall_clock_s for s in report.stages) <= report.wall_clock_s + 0.05

    text = report.format()
    assert "Profile: job" in text
    assert "load" in text and "train" in text


def test_profile_report_unavailable_before_exit():
    with mlprof.profile(name="job") as p:
        with pytest.raises(RuntimeError):
            p.report()


# ---- profile(): pre-flight audit gate -------------------------------------

def _gpu_required_cpu_target_spec() -> ModelSpec:
    """A spec with a guaranteed hard blocker: requires_gpu against a CPU-only
    target."""
    return ModelSpec(
        name="needs-gpu",
        train_callable="x:train",
        evaluate_callable="x:evaluate",
        dataset=DatasetSpec(loader="x:load", total_size=1000),
        eval_metrics=[EvalMetric(name="f1", higher_is_better=True, primary=True)],
        framework_hints=FrameworkHints(requires_gpu=True),
        capabilities=Capabilities(),
        targets=[TargetInstance(instance_type="c5.xlarge")],  # CPU-only
    )


def test_profile_raises_on_blocker_when_requested():
    spec = _gpu_required_cpu_target_spec()
    with pytest.raises(IncompatibleSpecError):
        with mlprof.profile(spec=spec, on_blocker="raise"):
            pass  # body should never run


def test_profile_attaches_audit_and_warns_by_default(capsys):
    spec = _gpu_required_cpu_target_spec()
    with mlprof.profile(spec=spec) as p:  # on_blocker defaults to "warn"
        pass
    report = p.report()
    assert report.audit is not None
    assert report.audit.has_blockers
    err = capsys.readouterr().err
    assert "blocker" in err.lower()


# ---- profile(): MLflow path degrades gracefully when mlflow is absent ------

def test_profile_mlflow_run_without_mlflow_warns(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def deny_mlflow(name, *args, **kwargs):
        if name == "mlflow" or name.startswith("mlflow."):
            raise ImportError("mlflow not installed (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", deny_mlflow)

    sentinel = object()  # stand-in for an mlflow Run
    with pytest.warns(UserWarning, match="mlflow"):
        with mlprof.profile(name="job", mlflow_run=sentinel) as p:
            time.sleep(0.01)
    # Report is still produced despite the logging no-op.
    assert p.report().wall_clock_s >= 0.0
