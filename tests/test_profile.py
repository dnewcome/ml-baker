"""Tests for library mode: measure() as a public primitive and profile().

No MLflow needed — the MLflow logging path is exercised only for its
graceful-degradation behaviour (warn + no-op when mlflow is absent).
"""

from __future__ import annotations

import time

import pytest

import mlprobe
from mlprobe import (
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
    with mlprobe.measure() as m:
        time.sleep(0.05)
    assert m.wall_clock_s >= 0.04
    assert m.peak_rss_mb > 0


# ---- profile(): stages + report -------------------------------------------

def test_profile_captures_overall_and_stages():
    with mlprobe.profile(name="job") as p:
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


# ---- stage bottleneck analysis (phase 0 of #13) ---------------------------

def test_bottleneck_identifies_dominant_stage():
    from mlprobe import ProfileReport, StageTiming

    report = ProfileReport(
        spec_name="r", wall_clock_s=100.0, peak_rss_mb=1.0,
        stages=[StageTiming("load", 8.0), StageTiming("cluster", 92.0)],
    )
    b = report.bottleneck()
    assert b is not None
    assert b.name == "cluster"
    assert b.share == 0.92
    assert "← bottleneck" in report.format()


def test_bottleneck_none_without_stages():
    from mlprobe import ProfileReport

    assert ProfileReport(spec_name="r", wall_clock_s=10.0, peak_rss_mb=1.0).bottleneck() is None


def test_analyze_stages_flags_dominant_and_unstaged():
    from mlprobe.profile import analyze_stages
    from mlprobe import StageTiming

    # cluster = 70% of wall-clock → bottleneck; 20% is unstaged → coverage flag.
    findings = analyze_stages(
        [StageTiming("load", 10.0), StageTiming("cluster", 70.0)], wall_clock_s=100.0
    )
    codes = {f.code for f in findings}
    assert "stage_bottleneck" in codes
    assert "unstaged_time" in codes
    bottleneck_msg = next(f.message for f in findings if f.code == "stage_bottleneck")
    assert "cluster" in bottleneck_msg and "70%" in bottleneck_msg


def test_analyze_stages_balanced_no_bottleneck():
    from mlprobe.profile import analyze_stages
    from mlprobe import StageTiming

    # Three even stages fully accounting for the run → no dominant, no unstaged.
    findings = analyze_stages(
        [StageTiming("a", 33.0), StageTiming("b", 33.0), StageTiming("c", 34.0)],
        wall_clock_s=100.0,
    )
    assert findings == []


def test_analyze_stages_empty():
    from mlprobe.profile import analyze_stages

    assert analyze_stages([], wall_clock_s=10.0) == []
    assert analyze_stages([], wall_clock_s=0.0) == []


def test_profile_report_unavailable_before_exit():
    with mlprobe.profile(name="job") as p:
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
        with mlprobe.profile(spec=spec, on_blocker="raise"):
            pass  # body should never run


def test_profile_attaches_audit_and_warns_by_default(capsys):
    spec = _gpu_required_cpu_target_spec()
    with mlprobe.profile(spec=spec) as p:  # on_blocker defaults to "warn"
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
        with mlprobe.profile(name="job", mlflow_run=sentinel) as p:
            time.sleep(0.01)
    # Report is still produced despite the logging no-op.
    assert p.report().wall_clock_s >= 0.0
