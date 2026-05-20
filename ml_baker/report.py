"""Aggregate probe results into the final report.

Workflow:
  1. Group probe results by (config, instance_type).
  2. Per group, fit time-vs-N and quality-vs-N over the probe subset sizes.
  3. Extrapolate both to ``dataset.total_size`` for a full-run prediction.
  4. Compute the Pareto frontier on (cost ↓, quality ↑) — quality direction
     flipped if the primary metric is lower-is-better.

The report also carries the static audit findings so the user gets one
combined artifact: ``what is wrong with this code`` + ``what would it cost
and what quality would you get``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ml_baker.audit import AuditReport
from ml_baker.probe import ProbeResult
from ml_baker.runner import RunResults
from ml_baker.scaling import ScalingFit, fit_quality_scaling, fit_time_scaling
from ml_baker.spec import ModelSpec
from ml_baker.targets import resolve


@dataclass(frozen=True)
class GroupSummary:
    config: dict[str, Any]
    instance_type: str
    n_probes_used: int                      # excludes failures
    time_fit: ScalingFit | None
    quality_fit: ScalingFit | None
    extrapolated_time_s: float | None       # at full dataset
    extrapolated_cost_usd: float | None     # at full dataset
    extrapolated_quality: float | None      # primary metric at full dataset
    primary_metric: str


@dataclass
class Report:
    spec_name: str
    audit: AuditReport
    groups: list[GroupSummary] = field(default_factory=list)
    pareto: list[GroupSummary] = field(default_factory=list)
    failures: list[ProbeResult] = field(default_factory=list)

    def format(self) -> str:
        out = [f"=== Report for {self.spec_name!r} ===", "", "AUDIT:", self.audit.format()]
        if self.failures:
            out += ["", f"PROBE FAILURES ({len(self.failures)}):"]
            for f in self.failures[:10]:
                out.append(f"  - {f.instance_type} subset={f.subset_fraction}: {f.error}")
        out += ["", "EXTRAPOLATED AT FULL DATASET:"]
        for g in self.groups:
            t = _fmt_time(g.extrapolated_time_s)
            c = _fmt_usd(g.extrapolated_cost_usd)
            q = _fmt_metric(g.extrapolated_quality, g.primary_metric)
            cfg = ", ".join(f"{k}={v}" for k, v in g.config.items())
            tf = f"time={g.time_fit.model}@R²{g.time_fit.r2:.2f}" if g.time_fit else "time=?"
            qf = f"qual={g.quality_fit.model}@R²{g.quality_fit.r2:.2f}" if g.quality_fit else "qual=?"
            out.append(f"  {g.instance_type:16s} [{cfg}] → t={t}  cost={c}  {q}  ({tf}, {qf})")
        out += ["", f"PARETO FRONTIER (cost ↓, {self.groups[0].primary_metric if self.groups else 'quality'} optimum):"]
        for g in self.pareto:
            cfg = ", ".join(f"{k}={v}" for k, v in g.config.items())
            out.append(
                f"  {g.instance_type:16s} [{cfg}] → cost={_fmt_usd(g.extrapolated_cost_usd)} "
                f"{_fmt_metric(g.extrapolated_quality, g.primary_metric)}"
            )
        return "\n".join(out)


def build_report(spec: ModelSpec, results: RunResults, audit: AuditReport) -> Report:
    """Aggregate probe results into a single report. ``audit`` is taken as-is
    so the caller can choose whether to include capabilities discovered
    empirically (a future hook)."""
    primary = _primary_metric(spec)
    groups = _build_groups(spec, results, primary)
    pareto = _pareto_frontier(groups, primary_higher_is_better=_higher_is_better(spec, primary))
    return Report(
        spec_name=spec.name,
        audit=audit,
        groups=groups,
        pareto=pareto,
        failures=results.failed,
    )


# ---- Grouping + scaling fits ---------------------------------------------

def _build_groups(spec: ModelSpec, results: RunResults, primary: str) -> list[GroupSummary]:
    grouped: dict[tuple, list[ProbeResult]] = defaultdict(list)
    for r in results.succeeded:
        key = (_config_key(r.config), r.instance_type)
        grouped[key].append(r)

    total_rows = spec.dataset.total_size
    out: list[GroupSummary] = []
    for (_, instance_type), probes in grouped.items():
        config = probes[0].config
        # Use absolute row counts where possible; fall back to fractions when
        # total_size is unknown (scaling shape is preserved either way).
        xs_raw = [p.subset_fraction for p in probes]
        xs = [x * total_rows for x in xs_raw] if total_rows else xs_raw
        full_x = total_rows if total_rows else 1.0

        time_fit = fit_time_scaling(xs, [p.wall_clock_s for p in probes])
        qualities = [p.eval_metrics.get(primary) for p in probes]
        if any(q is None for q in qualities):
            quality_fit = None
        else:
            quality_fit = fit_quality_scaling(xs, [float(q) for q in qualities])

        extrap_t = time_fit.at(full_x) if time_fit else None
        extrap_q = quality_fit.at(full_x) if quality_fit else None
        extrap_cost = (
            (extrap_t / 3600.0) * resolve(instance_type).on_demand_usd_per_hour
            if extrap_t is not None else None
        )

        out.append(GroupSummary(
            config=config,
            instance_type=instance_type,
            n_probes_used=len(probes),
            time_fit=time_fit,
            quality_fit=quality_fit,
            extrapolated_time_s=extrap_t,
            extrapolated_cost_usd=extrap_cost,
            extrapolated_quality=extrap_q,
            primary_metric=primary,
        ))
    return sorted(out, key=lambda g: (g.instance_type, str(g.config)))


def _config_key(config: dict[str, Any]) -> tuple:
    return tuple(sorted(config.items()))


def _primary_metric(spec: ModelSpec) -> str:
    for m in spec.eval_metrics:
        if m.primary:
            return m.name
    raise ValueError(f"spec {spec.name!r} has no primary eval metric")


def _higher_is_better(spec: ModelSpec, name: str) -> bool:
    for m in spec.eval_metrics:
        if m.name == name:
            return m.higher_is_better
    raise KeyError(name)


# ---- Pareto frontier -----------------------------------------------------

def _pareto_frontier(
    groups: list[GroupSummary], primary_higher_is_better: bool
) -> list[GroupSummary]:
    """Non-dominated set on (cost ↓, quality optimum). Groups missing either
    extrapolation are excluded — they cannot be ranked."""
    pts = [
        g for g in groups
        if g.extrapolated_cost_usd is not None and g.extrapolated_quality is not None
    ]
    quality_sign = 1.0 if primary_higher_is_better else -1.0

    def dominates(a: GroupSummary, b: GroupSummary) -> bool:
        a_cost, b_cost = a.extrapolated_cost_usd, b.extrapolated_cost_usd
        a_q = a.extrapolated_quality * quality_sign
        b_q = b.extrapolated_quality * quality_sign
        not_worse = a_cost <= b_cost and a_q >= b_q
        strictly_better = a_cost < b_cost or a_q > b_q
        return not_worse and strictly_better

    return [g for g in pts if not any(dominates(other, g) for other in pts if other is not g)]


# ---- Formatting ---------------------------------------------------------

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
        return "?"
    if c < 1:
        return f"${c:.3f}"
    return f"${c:.2f}"


def _fmt_metric(v: float | None, name: str) -> str:
    if v is None:
        return f"{name}=?"
    return f"{name}={v:.4f}"
