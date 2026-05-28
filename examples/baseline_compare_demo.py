"""Demo: converging on a baseline across a probing session.

baseline_compare treats the baseline as a moving target you carry forward:
the first result *establishes* it, each later candidate is compared as a
tradeoff, and you promote a challenger when it's worth it. No ML libraries —
the "results" here are plain metric dicts (quality + cost), exactly what
to_vector() accepts from real EvalResult / ProfileReport objects.

    python examples/baseline_compare_demo.py
"""

from __future__ import annotations

from mlprof.scenarios import baseline_compare


def main() -> None:
    # Phase 1 — first probe, nothing to compare against yet.
    naive = {"f1": 0.84, "cost_usd": 0.40, "latency_ms": 52.0}
    r = baseline_compare.run(naive)
    print(r.format(), "\n")
    base = r.data["baseline"]

    # Phase 2 — a bigger model: better quality, but pricier and slower-ish.
    bigger = {"f1": 0.91, "cost_usd": 1.20, "latency_ms": 60.0}
    r = baseline_compare.run(bigger, baseline=base)
    print(r.format(), "\n")
    # Decide it's worth it for this phase → promote the challenger.
    base = r.data["candidate"]

    # Phase 3 — a distilled variant: nearly the same quality, much cheaper/faster.
    distilled = {"f1": 0.905, "cost_usd": 0.55, "latency_ms": 28.0}
    r = baseline_compare.run(distilled, baseline=base)
    print(r.format())


if __name__ == "__main__":
    main()
