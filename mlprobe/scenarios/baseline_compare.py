"""Scenario: ``baseline_compare`` — is this candidate a worthwhile tradeoff?

This is the exploration-phase counterpart to a hard regression gate. During
discovery you aren't checking a run against a fixed high-water mark — you're
*converging on a baseline*, judging each candidate against the current best and
deciding whether a move is worth it. The baseline moves as you learn: the first
probe (when you know nothing yet) *establishes* it; a naive model becomes the
thing you reason against; a drifted or retrained model gets compared to what
you had. So the baseline is just the current-best vector you carry forward.

``baseline_compare`` is input-agnostic: every input reduces to one
``{metric: value}`` vector, so it doesn't care whether the numbers came from an
``EvalResult`` (quality), a ``ProfileReport`` (cost/time/memory of a real run),
a ``ProbeResult``, a plain dict, a *list* of those merged, or an artifact path
it evaluates for you. It compares quality and resource metrics *together* —
because the tradeoff is the whole point — and frames worthwhile-ness rather than
declaring pass/fail (that hard gate is a separate, late-phase concern).

    base = baseline_compare.run(first_probe, spec=spec).data["baseline"]  # establish
    r = baseline_compare.run(candidate, baseline=base, spec=spec)         # compare
    print(r.format())
    if i_decide_its_worth_it:
        base = r.data["candidate"]                                        # promote
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlprobe.probe import ProbeResult
from mlprobe.profile import ProfileReport
from mlprobe.protocol import EvalResult
from mlprobe.runner import evaluate_existing
from mlprobe.scenarios.base import ScenarioResult
from mlprobe.spec import ModelSpec


# Metrics whose lower value is better, independent of any spec declaration.
_RESOURCE_LOWER = {
    "cost_usd", "wall_clock_s", "peak_rss_mb", "peak_vram_mb", "peak_ram_mb",
}
# Informational metrics with no inherent "better" direction.
_NEUTRAL = {"gpu_util_avg", "samples_collected"}
# Substrings that, absent a spec declaration, imply lower-is-better.
_LOWER_TOKENS = (
    "loss", "error", "err", "latency", "_ms", "cost", "time", "duration",
    "memory", "rss", "vram",
)


@dataclass
class Baseline:
    """The current-best metric vector you carry forward through probing."""

    vector: dict[str, float]
    label: str = "baseline"

    @classmethod
    def from_result(
        cls,
        result: Any,
        *,
        spec: ModelSpec | None = None,
        eval_set: Any = None,
        label: str = "baseline",
    ) -> "Baseline":
        return cls(vector=to_vector(result, spec=spec, eval_set=eval_set), label=label)


@dataclass(frozen=True)
class MetricDelta:
    metric: str
    baseline: float
    candidate: float
    pct: float | None            # relative change; None when baseline is 0
    direction: str               # "higher" | "lower" | "neutral"
    verdict: str                 # "better" | "worse" | "wash" | "neutral"


# ---- Normalize any result into a flat metric vector ----------------------

def to_vector(
    obj: Any,
    *,
    spec: ModelSpec | None = None,
    eval_set: Any = None,
) -> dict[str, float]:
    """Reduce a result (or a list of results, merged) to ``{metric: value}``.

    Accepts ``EvalResult`` (quality metrics), ``ProfileReport`` /
    ``ProbeResult`` (resource metrics, plus eval metrics for probes), a plain
    dict, a ``Baseline``, or an artifact path/URI (evaluated via
    ``evaluate_existing`` when a ``spec`` is given). Lists are merged left to
    right (later wins on key collisions), so you can pass
    ``[eval_result, profile_report]`` to get quality + cost in one vector.
    """
    if isinstance(obj, Baseline):
        return dict(obj.vector)
    if isinstance(obj, (list, tuple)):
        merged: dict[str, float] = {}
        for item in obj:
            merged.update(to_vector(item, spec=spec, eval_set=eval_set))
        return merged
    if isinstance(obj, dict):
        return {k: float(v) for k, v in obj.items() if _is_number(v)}
    if isinstance(obj, EvalResult):
        return {k: float(v) for k, v in obj.metrics.items() if _is_number(v)}
    if isinstance(obj, ProfileReport):
        return _resource_vector(obj)
    if isinstance(obj, ProbeResult):
        vec = {k: float(v) for k, v in obj.eval_metrics.items() if _is_number(v)}
        vec.update(_resource_vector(obj))
        return vec
    if isinstance(obj, (str, Path)):
        if spec is None:
            raise ValueError(
                "baseline_compare was given an artifact path but no spec; pass "
                "spec=... so it can run evaluate_existing, or pass a result/dict."
            )
        return to_vector(evaluate_existing(spec, obj, eval_set=eval_set), spec=spec)
    raise TypeError(f"don't know how to extract a metric vector from {type(obj).__name__}")


def _resource_vector(obj: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    for attr in ("wall_clock_s", "peak_rss_mb", "peak_vram_mb", "gpu_util_avg", "cost_usd"):
        val = getattr(obj, attr, None)
        if _is_number(val):
            out[attr] = float(val)
    return out


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# ---- Direction + delta ----------------------------------------------------

def direction_of(metric: str, spec: ModelSpec | None) -> str:
    """Is higher or lower better for this metric? A spec's declared
    ``eval_metrics`` are authoritative; otherwise resource metrics are
    lower-is-better, a few are neutral, and the rest default to higher."""
    if spec is not None:
        for m in spec.eval_metrics:
            if m.name == metric:
                return "higher" if m.higher_is_better else "lower"
    if metric in _NEUTRAL:
        return "neutral"
    if metric in _RESOURCE_LOWER:
        return "lower"
    low = metric.lower()
    if any(tok in low for tok in _LOWER_TOKENS):
        return "lower"
    return "higher"


def _delta(metric, base, cand, spec, tolerance) -> MetricDelta:
    direction = direction_of(metric, spec)
    pct = None if base == 0 else (cand - base) / abs(base)
    if direction == "neutral":
        verdict = "neutral"
    elif (pct is not None and abs(pct) < tolerance) or (pct is None and cand == base):
        verdict = "wash"
    else:
        improved = (cand > base) if direction == "higher" else (cand < base)
        verdict = "better" if improved else "worse"
    return MetricDelta(metric, base, cand, pct, direction, verdict)


# ---- The scenario ---------------------------------------------------------

class BaselineCompare:
    name = "baseline_compare"
    question = "Is the candidate a worthwhile tradeoff vs the baseline?"

    def run(
        self,
        candidate: Any,
        baseline: Any = None,
        *,
        spec: ModelSpec | None = None,
        tolerance: float = 0.01,
        eval_set: Any = None,
        label: str = "candidate",
    ) -> ScenarioResult:
        cand_vec = to_vector(candidate, spec=spec, eval_set=eval_set)

        if baseline is None:
            base_obj = Baseline(vector=cand_vec, label="baseline")
            return ScenarioResult(
                scenario=self.name,
                question=self.question,
                answer="No baseline yet — established from this run:\n" + _format_vector(cand_vec),
                recommendation=(
                    "Carry this forward as the baseline (result.data['baseline']) "
                    "and compare future candidates against it."
                ),
                passed=None,
                data={"established": True, "baseline": base_obj,
                      "candidate": base_obj, "deltas": []},
            )

        base_vec = to_vector(baseline, spec=spec, eval_set=eval_set)
        shared = [m for m in base_vec if m in cand_vec]
        deltas = [_delta(m, base_vec[m], cand_vec[m], spec, tolerance) for m in shared]

        summary, recommendation = _summarize(deltas, tolerance)
        answer = summary + "\n" + _format_table(deltas)

        return ScenarioResult(
            scenario=self.name,
            question=self.question,
            answer=answer,
            recommendation=recommendation,
            passed=None,
            data={
                "established": False,
                "baseline": Baseline(base_vec, label="baseline"),
                "candidate": Baseline(cand_vec, label=label),
                "deltas": deltas,
                "only_in_baseline": sorted(set(base_vec) - set(cand_vec)),
                "only_in_candidate": sorted(set(cand_vec) - set(base_vec)),
                "tolerance": tolerance,
            },
        )


def _summarize(deltas: list[MetricDelta], tolerance: float) -> tuple[str, str]:
    better = [d for d in deltas if d.verdict == "better"]
    worse = [d for d in deltas if d.verdict == "worse"]

    parts = [f"{d.metric} {_fmt_pct(d.pct)} ({d.verdict})" for d in deltas if d.verdict != "wash"]
    summary = "; ".join(parts) if parts else f"all metrics within {tolerance:.0%} of baseline"

    if better and not worse:
        names = ", ".join(d.metric for d in better)
        return summary, (
            f"Strict improvement — better on {names}, no regressions. "
            f"Worth adopting as the new baseline."
        )
    if worse and not better:
        names = ", ".join(d.metric for d in worse)
        return summary, (
            f"Regression — worse on {names}, nothing better. Keep the current baseline."
        )
    if better and worse:
        b = ", ".join(d.metric for d in better)
        w = ", ".join(d.metric for d in worse)
        top_b = max(better, key=lambda d: abs(d.pct or 0))
        top_w = max(worse, key=lambda d: abs(d.pct or 0))
        return summary, (
            f"Tradeoff: gaining {b} at the cost of {w} (no axis dominates). "
            f"Worthwhile if {top_b.metric} matters more than {top_w.metric}; "
            f"not if you're constrained on {top_w.metric}."
        )
    return summary, (
        f"Effectively unchanged vs baseline (within {tolerance:.0%}). "
        f"No meaningful tradeoff — keep whichever is simpler/cheaper to run."
    )


# ---- Formatting -----------------------------------------------------------

_MARK = {"better": "▲ better", "worse": "▼ worse", "wash": "≈ wash", "neutral": "· n/a"}


def _format_table(deltas: list[MetricDelta]) -> str:
    if not deltas:
        return "  (no shared metrics to compare)"
    width = max(len(d.metric) for d in deltas)
    header = f"  {'metric':<{width}}  {'baseline':>10}  {'candidate':>10}  {'Δ':>8}"
    rows = [header]
    for d in deltas:
        rows.append(
            f"  {d.metric:<{width}}  {_fmt_num(d.baseline):>10}  "
            f"{_fmt_num(d.candidate):>10}  {_fmt_pct(d.pct):>8}  {_MARK[d.verdict]}"
        )
    return "\n".join(rows)


def _format_vector(vec: dict[str, float]) -> str:
    return "\n".join(f"  {k:<16} {_fmt_num(v)}" for k, v in vec.items()) or "  (empty)"


def _fmt_num(v: float) -> str:
    return f"{v:.4g}"


def _fmt_pct(pct: float | None) -> str:
    if pct is None:
        return "n/a"
    return f"{pct * 100:+.1f}%"


baseline_compare = BaselineCompare()
