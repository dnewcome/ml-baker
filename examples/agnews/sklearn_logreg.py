"""sklearn TF-IDF + LogisticRegression trainable for AG News.

The cheap CPU baseline in the demo. ~92% test accuracy on the full dataset,
trains in tens of seconds on a modest CPU. Use this as the lower-cost,
lower-quality end of the Pareto frontier the demo emits.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from mlprof import EvalResult, RuntimeConfig, TrainResult


def train(config: dict, dataset_subset, output_dir: Path,
          resume_from: Path | None = None,
          runtime: RuntimeConfig = RuntimeConfig()) -> TrainResult:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    texts = [r["text"] for r in dataset_subset]
    labels = [int(r["label"]) for r in dataset_subset]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=int(config.get("max_features", 20000)),
            ngram_range=(1, 2),
            min_df=2,
        )),
        ("clf", LogisticRegression(
            C=float(config.get("C", 1.0)),
            max_iter=int(config.get("max_iter", 200)),
            n_jobs=max(1, runtime.n_cpus),
            random_state=runtime.seed or 0,
        )),
    ])
    pipe.fit(texts, labels)

    artifact = output_dir / "model.pkl"
    with open(artifact, "wb") as f:
        pickle.dump(pipe, f)

    return TrainResult(
        artifact_path=artifact,
        metrics={"train_acc": float(pipe.score(texts, labels))},
    )


def evaluate(artifact_path: Path, eval_set) -> EvalResult:
    from sklearn.metrics import accuracy_score, f1_score

    with open(artifact_path, "rb") as f:
        pipe = pickle.load(f)

    texts = [r["text"] for r in eval_set]
    labels = [int(r["label"]) for r in eval_set]
    preds = pipe.predict(texts)

    return EvalResult(metrics={
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
    })
