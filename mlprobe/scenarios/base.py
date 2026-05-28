"""Scenario framework — the question-shaped front end to mlprobe.

Engineers don't think in Cartesian sweeps; they think in questions about a
specific model: *does this scale linearly?*, *where does parallelization
plateau?*, *is algorithm A or B faster at my data size?*. A ``Scenario``
encapsulates one such question. It knows:

  - the *question* it answers,
  - the *probe plan* needed to answer it (which axis to vary, how),
  - a domain-specific *analyzer* that turns raw probe results into a plain
    answer + recommendation.

Scenarios deliberately reuse the existing probe infrastructure — the same
``ProbeInput``/``run_probe`` contract, the same external measurement, the same
runtime resolution and scaling fits. They differ only in *which axis they
sweep* and *how they interpret the results*. The Cartesian sweep + Pareto
frontier is itself just one scenario (``scaling_with_n``) among many.

This module holds the shared pieces: the ``Scenario`` protocol, the
``ScenarioResult`` shape every scenario returns, and the small helpers each
scenario uses to plan and execute its probes.
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable

from mlprobe.probe import ProbeResult
from mlprobe.protocol import RuntimeConfig
from mlprobe.runner import execute_probes, make_probe_input, write_spec
from mlprobe.spec import ModelSpec
from mlprobe.targets import InstanceSpec, resolve


@dataclass
class ScenarioResult:
    """What every scenario returns: a question, a plain-language answer, a
    recommendation, the raw data behind it, and an optional pass/fail gate.

    ``passed`` is ``None`` for scenarios that have no notion of pass/fail
    (most diagnostic scenarios); it is a bool only when the scenario was given
    an explicit target to gate against (e.g. ``scaling_with_n`` with
    ``max_exponent``)."""

    scenario: str                       # scenario name, e.g. "scaling_with_n"
    question: str
    answer: str
    recommendation: str
    data: dict[str, Any] = field(default_factory=dict)
    passed: bool | None = None

    def format(self) -> str:
        lines = [
            f"=== {self.scenario} ===",
            f"Q: {self.question}",
            f"A: {self.answer}",
            f"→ {self.recommendation}",
        ]
        if self.passed is not None:
            lines.append(f"GATE: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


@runtime_checkable
class Scenario(Protocol):
    """A question-shaped probe pattern.

    Concrete scenarios specialize the signature of ``run`` (e.g.
    ``algorithm_selection`` takes a *list* of specs as its first argument) but
    all return a ``ScenarioResult``. Each scenario is exported as a singleton
    instance, so the user surface is ``scaling_with_n.run(spec, ...)``.
    """

    name: str

    def run(self, spec: ModelSpec, **kwargs: Any) -> ScenarioResult: ...


# ---- Shared probe planning + execution -----------------------------------

@dataclass
class ProbeReq:
    """One probe a scenario wants to run, with the axis values it chose.

    Unlike the sweep matrix (one spec, many configs/subsets), scenarios vary
    arbitrary axes — and ``algorithm_selection`` even varies the *spec* — so a
    request carries everything needed to materialize a single ``ProbeInput``.
    ``label`` is the human-readable axis value for this probe ("n_cpus=4",
    "frac=0.1", the spec name) used in formatted output.
    """

    spec: ModelSpec
    instance: InstanceSpec
    runtime: RuntimeConfig
    subset_fraction: float
    config: dict[str, Any]
    label: str
    seed: int | None = None
    repetition: int = 0


@contextmanager
def _run_dir(run_dir: Path | str | None) -> Iterator[Path]:
    """Yield a working directory for probe I/O. A temp dir (auto-cleaned) when
    the caller didn't ask to persist results; otherwise the given dir."""
    if run_dir is not None:
        path = Path(run_dir)
        path.mkdir(parents=True, exist_ok=True)
        yield path
        return
    tmp = tempfile.mkdtemp(prefix="mlprobe-scenario-")
    try:
        yield Path(tmp)
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


def run_probe_reqs(
    reqs: list[ProbeReq],
    *,
    run_dir: Path | str | None = None,
    launcher: str = "subprocess",
    timeout: int = 1800,
    progress: bool = True,
) -> list[tuple[ProbeReq, ProbeResult]]:
    """Execute a list of scenario probe requests and pair each result back
    with the request (so analyzers keep the axis value alongside the result).

    Distinct specs are each written once; ``ProbeInput``s are built via the
    shared ``make_probe_input`` so scenarios honor the exact same launcher
    contract as the sweep runner.
    """
    with _run_dir(run_dir) as rd:
        spec_paths: dict[int, Path] = {}
        for req in reqs:
            key = id(req.spec)
            if key not in spec_paths:
                spec_paths[key] = write_spec(req.spec, rd, name=f"spec-{len(spec_paths)}")

        probes = [
            make_probe_input(
                spec_path=spec_paths[id(req.spec)],
                config=req.config,
                instance=req.instance,
                runtime=req.runtime,
                subset_fraction=req.subset_fraction,
                idx=idx,
                run_dir=rd,
                repetition=req.repetition,
                seed=req.seed,
            )
            for idx, req in enumerate(reqs)
        ]
        results = execute_probes(
            probes, launcher=launcher, timeout=timeout, progress=progress
        )
    return list(zip(reqs, results))


# ---- Small spec helpers ---------------------------------------------------

def resolve_target(spec: ModelSpec, target: str | None) -> InstanceSpec:
    """Resolve the instance to probe on. Falls back to the spec's first
    declared target when ``target`` is omitted."""
    if target is None:
        if not spec.targets:
            raise ValueError(
                f"scenario needs a target: pass target=... or declare "
                f"targets on spec {spec.name!r}"
            )
        target = spec.targets[0].instance_type
    return resolve(target)


def default_config(spec: ModelSpec) -> dict[str, Any]:
    """The spec's default hyperparameter config — each param pinned at its
    declared default. Scenarios that aren't sweeping hyperparameters probe
    this single config."""
    return {hp.name: hp.default for hp in spec.hyperparameters}


def subset_fraction_for(
    spec: ModelSpec,
    *,
    n: int | None = None,
    subset_fraction: float | None = None,
) -> float:
    """Resolve a probe subset fraction from either an explicit fraction or an
    absolute row count ``n`` (converted via ``dataset.total_size``)."""
    if subset_fraction is not None:
        return subset_fraction
    if n is not None:
        total = spec.dataset.total_size
        if not total:
            raise ValueError(
                f"cannot convert n={n} to a fraction: spec {spec.name!r} has "
                f"no dataset.total_size; pass subset_fraction=... instead"
            )
        return min(1.0, n / total)
    return 1.0


def primary_metric(spec: ModelSpec) -> tuple[str, bool]:
    """Return ``(name, higher_is_better)`` for the spec's primary metric."""
    for m in spec.eval_metrics:
        if m.primary:
            return m.name, m.higher_is_better
    raise ValueError(f"spec {spec.name!r} has no primary eval metric")
