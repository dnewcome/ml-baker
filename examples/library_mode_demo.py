"""Library-mode demo: instrument a 'real' run from the inside with profile().

No ML libraries, no probes, no subprocesses — this is what you'd add to your
own training script. Stages are timed individually and rolled up into a
single-run report. Pass an MLflow run to ``profile(mlflow_run=...)`` to also
log the metrics (requires ``mlprof[mlflow]``).

    python examples/library_mode_demo.py
"""

from __future__ import annotations

import time

import mlprof


def load_data():
    time.sleep(0.15)
    return list(range(10_000))


def cluster(data):
    time.sleep(0.45)
    return {"n_clusters": 42}


def main() -> None:
    with mlprof.profile(name="brand-clustering") as p:
        with p.stage("load"):
            data = load_data()
        with p.stage("cluster"):
            _ = cluster(data)

    print(p.report().format())


if __name__ == "__main__":
    main()
