"""Scenario: ``scaling_with_n`` — does this scale linearly with data size?

Probes the model at several ``subset_fraction`` values on one target, fits a
time-vs-N scaling law, and reports the asymptotic complexity (the power-law
exponent) plus a recommendation. This is the question the original sweep +
extrapolation machinery already answered — here it becomes a first-class,
single-target scenario with a plain-language answer instead of a frontier.

    scaling_with_n.run(spec, target="g5.xlarge")
    → "Scales as O(N^1.04) (power-law fit, R²=0.99 over 5 sizes). Likely linear."
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mlprobe.scaling import fit_time_scaling
from mlprobe.scenarios.base import (
    ProbeReq,
    ScenarioResult,
    default_config,
    resolve_target,
    run_probe_reqs,
)
from mlprobe.runtime import resolve_runtime
from mlprobe.spec import ModelSpec
from mlprobe.targets import resolve


class ScalingWithN:
    name = "scaling_with_n"
    question = "Does training time scale linearly with data size N?"

    def run(
        self,
        spec: ModelSpec,
        *,
        target: str | None = None,
        subset_fractions: list[float] | None = None,
        config: dict[str, Any] | None = None,
        max_exponent: float | None = None,
        run_dir: str | None = None,
        launcher: str = "subprocess",
        progress: bool = True,
    ) -> ScenarioResult:
        """Fit a time-scaling law from probes at several subset sizes.

        Parameters
        ----------
        target : instance type to probe on (defaults to the spec's first target).
        subset_fractions : sizes to probe (defaults to ``spec.probe.subset_fractions``).
        config : hyperparameter config to pin (defaults to the spec defaults).
        max_exponent : if set, the scenario gates pass/fail on the fitted
            power-law exponent staying at or below this value.
        """
        instance = resolve_target(spec, target)
        runtime = resolve_runtime(spec.capabilities, instance)
        cfg = config if config is not None else default_config(spec)
        fracs = sorted(set(subset_fractions or spec.probe.subset_fractions or [0.1, 0.3, 1.0]))

        reqs = [
            ProbeReq(spec, instance, runtime, f, cfg, label=f"frac={f:g}")
            for f in fracs
        ]
        paired = run_probe_reqs(
            reqs,
            run_dir=run_dir,
            launcher=launcher,
            timeout=spec.probe.timeout_seconds,
            progress=progress,
        )

        ok = [
            (req, res) for req, res in paired
            if res.error is None and res.wall_clock_s is not None
        ]
        if len(ok) < 2:
            return self._inconclusive(spec, paired, instance.instance_type)

        total = spec.dataset.total_size
        xs = [req.subset_fraction * total if total else req.subset_fraction for req, _ in ok]
        ys = [res.wall_clock_s for _, res in ok]

        fit = fit_time_scaling(xs, ys)
        exponent = _power_exponent(xs, ys)

        answer, recommendation, complexity = _describe(fit, exponent, n_points=len(ok))

        # Extrapolate to the full dataset when we know its size.
        extrap_time_s = extrap_cost = None
        if fit is not None and total:
            extrap_time_s = fit.at(float(total))
            extrap_cost = (extrap_time_s / 3600.0) * resolve(
                instance.instance_type
            ).on_demand_usd_per_hour
            answer += (
                f" Extrapolated full run (N={total:,}): "
                f"{_fmt_time(extrap_time_s)}, ~${extrap_cost:,.2f}."
            )

        passed = None
        if max_exponent is not None and exponent is not None:
            passed = exponent <= max_exponent
            recommendation += (
                f" Gate: exponent {exponent:.2f} "
                f"{'≤' if passed else '>'} max_exponent {max_exponent:.2f}."
            )

        return ScenarioResult(
            scenario=self.name,
            question=self.question,
            answer=answer,
            recommendation=recommendation,
            passed=passed,
            data={
                "target": instance.instance_type,
                "config": cfg,
                "points": [
                    {"n": x, "subset_fraction": req.subset_fraction, "wall_clock_s": res.wall_clock_s}
                    for x, (req, res) in zip(xs, ok)
                ],
                "fit_model": fit.model if fit else None,
                "fit_r2": fit.r2 if fit else None,
                "exponent": exponent,
                "complexity": complexity,
                "extrapolated_time_s": extrap_time_s,
                "extrapolated_cost_usd": extrap_cost,
                "failures": [res.error for _, res in paired if res.error is not None],
            },
        )

    def _inconclusive(self, spec, paired, target) -> ScenarioResult:
        errors = [res.error for _, res in paired if res.error is not None]
        return ScenarioResult(
            scenario=self.name,
            question=self.question,
            answer="Inconclusive — fewer than 2 probes succeeded.",
            recommendation=(
                "Check that LoadDatasetFn honors subset_fraction and that "
                "train()/evaluate() run cleanly. Probe errors: "
                + ("; ".join(errors[:3]) if errors else "none reported")
            ),
            passed=None,
            data={"target": target, "failures": errors},
        )


def _describe(fit, exponent, *, n_points) -> tuple[str, str, str]:
    """Plain-language answer + recommendation + a short complexity label."""
    if fit is None:
        return (
            "Could not fit a scaling curve (non-positive or degenerate timings).",
            "Re-run with cleaner timings or more subset sizes.",
            "unknown",
        )

    r2 = fit.r2
    if exponent is not None:
        complexity = f"O(N^{exponent:.2f})"
        head = f"Scales as {complexity}"
    else:
        complexity = {"linear": "O(N)", "loglinear": "O(N·logN)", "log": "O(logN)"}.get(
            fit.model, fit.model
        )
        head = f"Scales as {complexity}"

    answer = f"{head} (best fit: {fit.model}, R²={r2:.2f} over {n_points} sizes)."

    e = exponent if exponent is not None else 1.0
    if e < 1.15:
        rec = "Linear scaling confirmed — full-dataset cost extrapolates predictably."
    elif e < 1.5:
        rec = (
            "Slightly super-linear — cost grows a bit faster than data. Fine for "
            "moderate N; watch it if you scale up an order of magnitude."
        )
    elif e < 2.5:
        rec = (
            "Roughly quadratic — full-dataset cost will balloon. Consider an "
            "algorithmic change, blocking/approximation, or sampling before scaling N."
        )
    else:
        rec = (
            "Steeply super-linear — scaling N is likely infeasible without an "
            "algorithmic change."
        )
    if r2 < 0.9:
        rec += f" (Low fit confidence, R²={r2:.2f} — add more subset sizes.)"
    return answer, rec, complexity


def _power_exponent(xs: list[float], ys: list[float]) -> float | None:
    """Log-log slope = the power-law exponent. Robust headline regardless of
    which model wins the R² contest. Requires strictly positive values."""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if len(x) < 2 or np.any(x <= 0) or np.any(y <= 0):
        return None
    b, _ = np.polyfit(np.log(x), np.log(y), 1)
    return float(b)


def _fmt_time(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{s / 60:.1f}m"
    return f"{s / 3600:.2f}h"


scaling_with_n = ScalingWithN()
