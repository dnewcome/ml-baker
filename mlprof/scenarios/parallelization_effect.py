"""Scenario: ``parallelization_effect`` — where does adding workers plateau?

Probes the model at a fixed data size while varying the worker count
(``n_cpus`` by default, or ``n_gpus``), measures the speedup curve, and fits
Amdahl's law to estimate the parallelizable fraction and the theoretical
ceiling. Crucially it catches the regime where parallelization *overhead*
exceeds the compute it saves — e.g. hundreds of thousands of task dispatches
ending up slower than a single fused kernel.

    parallelization_effect.run(spec, target="c5.4xlarge", n_cpus=[1, 4, 16, 32])
    → "Speedup 1.0/3.1/6.2/6.3x. Plateaus at 16 CPUs (Amdahl P≈0.84,
       ceiling ~6.4x). 32 adds <2% — not worth the instance cost."

Amdahl fit (closed form, numpy-only): with speedup S(p) = T(p_base)/T(p),
Amdahl says S(p) = 1 / ((1-P) + P/p). Rearranging,
    1/S - 1 = P * (1/p - 1),
which is linear through the origin in u = (1/p - 1), so P is a single
least-squares ratio — no nonlinear solver needed.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from mlprof.scenarios.base import (
    ProbeReq,
    ScenarioResult,
    default_config,
    resolve_target,
    run_probe_reqs,
    subset_fraction_for,
)
from mlprof.runtime import resolve_runtime
from mlprof.spec import ModelSpec


# Adding a worker that buys less than this fractional speedup over the
# previous step is treated as "plateaued".
_PLATEAU_MARGINAL_GAIN = 0.10


class ParallelizationEffect:
    name = "parallelization_effect"
    question = "Where does adding parallel workers stop helping?"

    def run(
        self,
        spec: ModelSpec,
        *,
        target: str | None = None,
        n_cpus: list[int] | None = None,
        n_gpus: list[int] | None = None,
        subset_fraction: float | None = None,
        n: int | None = None,
        config: dict[str, Any] | None = None,
        run_dir: str | None = None,
        launcher: str = "subprocess",
        progress: bool = True,
    ) -> ScenarioResult:
        instance = resolve_target(spec, target)
        base_runtime = resolve_runtime(spec.capabilities, instance)
        cfg = config if config is not None else default_config(spec)
        frac = subset_fraction if subset_fraction is not None else _default_frac(spec, n)

        axis, values, make_runtime = self._axis(instance, base_runtime, n_cpus, n_gpus)

        reqs = [
            ProbeReq(spec, instance, make_runtime(k), frac, cfg, label=f"{axis}={k}")
            for k in values
        ]
        paired = run_probe_reqs(
            reqs,
            run_dir=run_dir,
            launcher=launcher,
            timeout=spec.probe.timeout_seconds,
            progress=progress,
        )

        points = [
            {"workers": req.runtime.n_cpus if axis == "n_cpus" else req.runtime.n_gpus,
             "wall_clock_s": res.wall_clock_s}
            for req, res in paired
            if res.error is None and res.wall_clock_s and res.wall_clock_s > 0
        ]
        points.sort(key=lambda p: p["workers"])
        if len(points) < 2:
            return self._inconclusive(spec, paired, instance.instance_type, axis)

        return self._analyze(points, axis, instance.instance_type, cfg, frac, paired)

    def _axis(self, instance, base_runtime, n_cpus, n_gpus):
        if n_gpus is not None:
            values = _clamp(sorted(set(n_gpus)), instance.gpu_count or 1)
            return "n_gpus", values, (lambda k: replace(base_runtime, n_gpus=k))
        cpus = n_cpus or _ladder(instance.vcpus)
        values = _clamp(sorted(set(cpus)), instance.vcpus)
        return "n_cpus", values, (lambda k: replace(base_runtime, n_cpus=k))

    def _analyze(self, points, axis, target, cfg, frac, paired) -> ScenarioResult:
        unit = "CPUs" if axis == "n_cpus" else "GPUs"
        base = points[0]
        base_t = base["wall_clock_s"]
        for p in points:
            p["speedup"] = base_t / p["wall_clock_s"]

        speedups = [p["speedup"] for p in points]
        workers = [p["workers"] for p in points]

        amdahl_p = _fit_amdahl(workers, speedups)
        ceiling = (1.0 / (1.0 - amdahl_p)) if amdahl_p is not None and amdahl_p < 1 else None

        # Overhead regime: the best speedup isn't at the most workers.
        best_idx = max(range(len(speedups)), key=lambda i: speedups[i])
        overhead = best_idx < len(speedups) - 1

        plateau_workers = _plateau_point(workers, speedups)

        curve = " / ".join(f"{s:.1f}x" for s in speedups)
        answer = f"Speedup vs {workers[0]} {unit}: {curve}."
        if amdahl_p is not None:
            answer += f" Amdahl P≈{amdahl_p:.2f}"
            if ceiling is not None:
                answer += f" (theoretical ceiling ~{ceiling:.1f}x)"
            answer += "."

        if overhead:
            slow_at = workers[best_idx + 1]
            recommendation = (
                f"Overhead-bound: past {workers[best_idx]} {unit}, more workers make "
                f"it SLOWER (peaks at {speedups[best_idx]:.1f}x, then regresses by "
                f"{workers[-1]} {unit}). Adding {slow_at}+ {unit} is wasted spend — "
                f"the dispatch/sync cost exceeds the compute saved."
            )
            passed = None
        elif plateau_workers is not None and plateau_workers < workers[-1]:
            recommendation = (
                f"Plateaus around {plateau_workers} {unit}. Beyond that each added "
                f"worker buys <{int(_PLATEAU_MARGINAL_GAIN * 100)}% — pick the "
                f"instance that gives ~{plateau_workers} {unit}, not the biggest one."
            )
            passed = None
        else:
            recommendation = (
                f"Still scaling at {workers[-1]} {unit} — try larger worker counts "
                f"to find the plateau before sizing the instance."
            )
            passed = None

        return ScenarioResult(
            scenario=self.name,
            question=self.question,
            answer=answer,
            recommendation=recommendation,
            passed=passed,
            data={
                "target": target,
                "axis": axis,
                "subset_fraction": frac,
                "config": cfg,
                "points": points,
                "amdahl_parallel_fraction": amdahl_p,
                "theoretical_ceiling": ceiling,
                "plateau_workers": plateau_workers,
                "overhead_bound": overhead,
                "failures": [res.error for _, res in paired if res.error is not None],
            },
        )

    def _inconclusive(self, spec, paired, target, axis) -> ScenarioResult:
        errors = [res.error for _, res in paired if res.error is not None]
        return ScenarioResult(
            scenario=self.name,
            question=self.question,
            answer="Inconclusive — fewer than 2 worker counts produced a timing.",
            recommendation=(
                "Confirm train() honors runtime."
                + ("n_cpus" if axis == "n_cpus" else "n_gpus")
                + " and that the data size is large enough to show speedup. "
                + ("Errors: " + "; ".join(errors[:3]) if errors else "")
            ),
            passed=None,
            data={"target": target, "axis": axis, "failures": errors},
        )


def _default_frac(spec: ModelSpec, n: int | None) -> float:
    if n is not None:
        return subset_fraction_for(spec, n=n)
    # Parallel speedup only shows on a workload big enough to dominate
    # fixed overhead — default to the largest configured probe size.
    if spec.probe.subset_fractions:
        return max(spec.probe.subset_fractions)
    return 1.0


def _ladder(vcpus: int) -> list[int]:
    """Powers of two up to the instance's vCPU count: 1, 2, 4, 8, ..."""
    out, k = [], 1
    while k < max(1, vcpus):
        out.append(k)
        k *= 2
    out.append(max(1, vcpus))
    return sorted(set(out))


def _clamp(values: list[int], ceiling: int) -> list[int]:
    kept = [v for v in values if 1 <= v <= max(1, ceiling)]
    return kept or [1]


def _fit_amdahl(workers: list[int], speedups: list[float]) -> float | None:
    """Closed-form least-squares for the parallelizable fraction P.

    1/S - 1 = P*(1/p - 1)  ⇒  P = Σ(u·y) / Σ(u²),  u = 1/p - 1, y = 1/S - 1.
    The baseline point (p = workers[0], S = 1) has u = y = 0 only when
    workers[0] == 1; otherwise it still contributes consistently.
    """
    num = den = 0.0
    for p, s in zip(workers, speedups):
        if p <= 0 or s <= 0:
            continue
        u = (1.0 / p) - 1.0
        y = (1.0 / s) - 1.0
        num += u * y
        den += u * u
    if den == 0:
        return None
    p_frac = num / den
    return max(0.0, min(1.0, p_frac))


def _plateau_point(workers: list[int], speedups: list[float]) -> int | None:
    """Smallest worker count past which the marginal speedup gain drops below
    the plateau threshold."""
    for i in range(1, len(speedups)):
        prev, cur = speedups[i - 1], speedups[i]
        if prev <= 0:
            continue
        if (cur - prev) / prev < _PLATEAU_MARGINAL_GAIN:
            return workers[i - 1]
    return None


parallelization_effect = ParallelizationEffect()
