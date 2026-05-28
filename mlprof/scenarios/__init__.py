"""Scenarios — the question-shaped front end to mlprof.

Each scenario answers one concrete question an engineer actually asks about a
model, planning its own probes and analyzing them with a domain-specific
analyzer. Import the singleton and call ``.run(...)``::

    from mlprof.scenarios import (
        scaling_with_n, parallelization_effect, algorithm_selection,
    )

    print(scaling_with_n.run(spec, target="g5.xlarge").format())

The Cartesian sweep + Pareto frontier (``mlprof.run`` / ``build_report``)
remains available as the lower-level surface; ``scaling_with_n`` is its
scenario-shaped wrapper. See ``PLAN.md`` Decision 4 for the catalog of
scenarios still to come (gpu_vs_cpu, vram_headroom, regression_guard, ...).
"""

from mlprof.scenarios.base import ProbeReq, Scenario, ScenarioResult
from mlprof.scenarios.scaling_with_n import ScalingWithN, scaling_with_n
from mlprof.scenarios.parallelization_effect import (
    ParallelizationEffect,
    parallelization_effect,
)
from mlprof.scenarios.algorithm_selection import (
    AlgorithmSelection,
    algorithm_selection,
)
from mlprof.scenarios.baseline_compare import (
    Baseline,
    BaselineCompare,
    MetricDelta,
    baseline_compare,
    to_vector,
)

__all__ = [
    "Scenario",
    "ScenarioResult",
    "ProbeReq",
    "ScalingWithN",
    "scaling_with_n",
    "ParallelizationEffect",
    "parallelization_effect",
    "AlgorithmSelection",
    "algorithm_selection",
    "Baseline",
    "BaselineCompare",
    "MetricDelta",
    "baseline_compare",
    "to_vector",
]
