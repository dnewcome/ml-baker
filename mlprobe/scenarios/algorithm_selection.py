"""Scenario: ``algorithm_selection`` — which algorithm wins at my data size?

Given several ModelSpec variants (one per algorithm), probes each at the same
target data size and recommends based on wall-clock and primary quality. The
direct answer to "should I use agglomerative or LSH-blocked at 88k rows?".

    algorithm_selection.run([spec_agg, spec_lsh], target="g5.8xlarge", n=88_000)
    → "agglomerative: 42m, f1=0.998. lsh_blocked: 51m, f1=0.953.
       Recommend agglomerative — faster AND higher quality at this N."

Because the recommendation depends on data size, a variant that fails at this
N (e.g. exceeds the RAM ceiling) is reported as such rather than silently
dropped — that failure *is* the answer ("agglomerative OOMs past ~120k; switch
to lsh_blocked above that").
"""

from __future__ import annotations

from typing import Any

from mlprobe.scenarios.base import (
    ProbeReq,
    ScenarioResult,
    default_config,
    primary_metric,
    resolve_target,
    run_probe_reqs,
    subset_fraction_for,
)
from mlprobe.runtime import resolve_runtime
from mlprobe.spec import ModelSpec


# Quality within this relative band is treated as a tie, so the faster variant
# wins on speed rather than chasing a fractional quality gain.
_QUALITY_TIE_REL = 0.01


class AlgorithmSelection:
    name = "algorithm_selection"
    question = "Which algorithm is the best choice at this data size?"

    def run(
        self,
        specs: list[ModelSpec] | ModelSpec,
        *,
        target: str | None = None,
        n: int | None = None,
        subset_fraction: float | None = None,
        run_dir: str | None = None,
        launcher: str = "subprocess",
        progress: bool = True,
    ) -> ScenarioResult:
        if isinstance(specs, ModelSpec):
            specs = [specs]
        if len(specs) < 2:
            raise ValueError("algorithm_selection needs at least 2 specs to compare")

        reqs: list[ProbeReq] = []
        timeout = 1800
        for spec in specs:
            instance = resolve_target(spec, target)
            runtime = resolve_runtime(spec.capabilities, instance)
            frac = subset_fraction_for(spec, n=n, subset_fraction=subset_fraction)
            reqs.append(ProbeReq(spec, instance, runtime, frac, default_config(spec), label=spec.name))
            timeout = max(timeout, spec.probe.timeout_seconds)

        paired = run_probe_reqs(
            reqs, run_dir=run_dir, launcher=launcher, timeout=timeout, progress=progress
        )

        rows = [self._row(req, res) for req, res in paired]
        return self._analyze(rows, n=n, subset_fraction=subset_fraction)

    def _row(self, req: ProbeReq, res) -> dict[str, Any]:
        metric_name, higher_is_better = primary_metric(req.spec)
        quality = res.eval_metrics.get(metric_name) if res.error is None else None
        return {
            "name": req.spec.name,
            "metric": metric_name,
            "higher_is_better": higher_is_better,
            "wall_clock_s": res.wall_clock_s,
            "quality": quality,
            "cost_usd": res.cost_usd,
            "subset_fraction": req.subset_fraction,
            "error": res.error,
        }

    def _analyze(self, rows: list[dict], *, n, subset_fraction) -> ScenarioResult:
        ok = [r for r in rows if r["error"] is None and r["wall_clock_s"] is not None and r["quality"] is not None]
        failed = [r for r in rows if r not in ok]

        answer_lines = []
        for r in rows:
            if r["error"] is not None:
                answer_lines.append(f"{r['name']}: FAILED ({r['error']})")
            elif r["quality"] is None:
                answer_lines.append(f"{r['name']}: ran but no '{r['metric']}' metric returned")
            else:
                answer_lines.append(
                    f"{r['name']}: {_fmt_time(r['wall_clock_s'])}, "
                    f"{r['metric']}={r['quality']:.3f}, {_fmt_usd(r['cost_usd'])}"
                )
        answer = " | ".join(answer_lines)

        if not ok:
            return ScenarioResult(
                scenario=self.name, question=self.question, answer=answer,
                recommendation="No variant completed successfully at this data size. "
                               "Check the failures above (resource ceiling or code error).",
                passed=None, data={"rows": rows},
            )

        recommendation, winner = self._recommend(ok, failed)
        return ScenarioResult(
            scenario=self.name,
            question=self.question,
            answer=answer,
            recommendation=recommendation,
            passed=None,
            data={"rows": rows, "winner": winner, "n": n, "subset_fraction": subset_fraction},
        )

    def _recommend(self, ok: list[dict], failed: list[dict]) -> tuple[str, str]:
        higher_is_better = ok[0]["higher_is_better"]

        def better_quality(a, b):
            return a["quality"] > b["quality"] if higher_is_better else a["quality"] < b["quality"]

        best_quality = ok[0]
        for r in ok[1:]:
            if better_quality(r, best_quality):
                best_quality = r
        fastest = min(ok, key=lambda r: r["wall_clock_s"])

        failed_note = ""
        if failed:
            names = ", ".join(r["name"] for r in failed)
            failed_note = (
                f" Note: {names} did not complete at this N — likely past a resource "
                f"ceiling, so it's not an option at this size."
            )

        if fastest["name"] == best_quality["name"]:
            r = fastest
            return (
                f"Use {r['name']} — fastest AND best {r['metric']} at this N "
                f"({_fmt_time(r['wall_clock_s'])}, {r['metric']}={r['quality']:.3f}).{failed_note}",
                r["name"],
            )

        # Tradeoff: is the quality edge meaningful, or a tie?
        denom = abs(best_quality["quality"]) or 1.0
        rel_gain = abs(best_quality["quality"] - fastest["quality"]) / denom
        if rel_gain < _QUALITY_TIE_REL:
            return (
                f"Use {fastest['name']} — {_fmt_time(fastest['wall_clock_s'])} vs "
                f"{_fmt_time(best_quality['wall_clock_s'])}, and {fastest['metric']} is "
                f"within {_QUALITY_TIE_REL:.0%} of the best ({fastest['quality']:.3f} vs "
                f"{best_quality['quality']:.3f}) — the speed wins, the quality is a wash.{failed_note}",
                fastest["name"],
            )
        slower_by = best_quality["wall_clock_s"] / fastest["wall_clock_s"]
        return (
            f"Tradeoff: {best_quality['name']} gives the best {best_quality['metric']} "
            f"({best_quality['quality']:.3f}) but is {slower_by:.1f}× slower than "
            f"{fastest['name']} ({fastest['quality']:.3f}). Pick {best_quality['name']} if "
            f"the quality gain matters, {fastest['name']} if speed/cost does.{failed_note}",
            best_quality["name"],
        )


def _fmt_time(s: float | None) -> str:
    if s is None:
        return "?"
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{s / 60:.1f}m"
    return f"{s / 3600:.2f}h"


def _fmt_usd(c: float | None) -> str:
    if c is None:
        return "$?"
    return f"${c:.3f}" if c < 1 else f"${c:.2f}"


algorithm_selection = AlgorithmSelection()
