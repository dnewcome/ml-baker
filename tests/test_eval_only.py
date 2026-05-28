"""Tests for eval-only mode: mlprof.evaluate_existing (#18).

Evaluates an artifact that already exists on disk, with no training run — the
expensive-training, cheap-re-eval workflow. Uses the synthetic trainable's
artifact format (a small JSON file).
"""

from __future__ import annotations

import json

import mlprof
from mlprof import (
    Capabilities,
    DatasetSpec,
    EvalMetric,
    ModelSpec,
)


def _spec(evaluate_callable: str = "synthetic_trainable:evaluate") -> ModelSpec:
    return ModelSpec(
        name="eval-only",
        train_callable="synthetic_trainable:train",
        evaluate_callable=evaluate_callable,
        dataset=DatasetSpec(loader="synthetic_trainable:load", total_size=10_000),
        eval_metrics=[EvalMetric(name="f1", higher_is_better=True, primary=True)],
        capabilities=Capabilities(),
    )


def _write_artifact(tmp_path, *, n_rows=8000, algorithm="accurate"):
    artifact = tmp_path / "model.json"
    artifact.write_text(json.dumps({"n_rows": n_rows, "algorithm": algorithm}))
    return artifact


def test_evaluate_existing_against_local_artifact(tmp_path):
    spec = _spec()
    artifact = _write_artifact(tmp_path, n_rows=8000, algorithm="accurate")

    result = mlprof.evaluate_existing(spec, artifact)

    assert "f1" in result.metrics
    assert 0.0 < result.metrics["f1"] <= 0.92


def test_evaluate_existing_autoloads_eval_set_and_passes_path_type(tmp_path):
    spec = _spec(evaluate_callable="synthetic_trainable:evaluate_echo")
    artifact = _write_artifact(tmp_path)

    # No eval_set passed → loader is invoked at full size.
    result = mlprof.evaluate_existing(spec, artifact)

    assert result.metrics["artifact_is_str"] == 0.0          # local path → Path
    assert result.metrics["eval_set_rows"] == 10_000          # full-size auto-load


def test_evaluate_existing_passes_uri_through_as_string():
    spec = _spec(evaluate_callable="synthetic_trainable:evaluate_echo")

    # Remote URI must reach the user's evaluate as a plain string (their loader
    # resolves it); eval_set provided so no file is read.
    result = mlprof.evaluate_existing(
        spec, "s3://bucket/path/model.tar.gz", eval_set={"n_rows": 42}
    )

    assert result.metrics["artifact_is_str"] == 1.0
    assert result.metrics["eval_set_rows"] == 42


def test_evaluate_existing_uses_explicit_eval_set(tmp_path):
    spec = _spec()
    # Different n_rows in the artifact vs the (ignored) loader proves we used
    # the artifact's content, and that no training ran.
    artifact = _write_artifact(tmp_path, n_rows=500, algorithm="fast")
    result = mlprof.evaluate_existing(spec, artifact, eval_set={"n_rows": 1})
    # fast ceiling 0.80; at n=500, 0.40 + 0.05*ln(500) ≈ 0.71 < ceiling.
    assert result.metrics["f1"] < 0.80
