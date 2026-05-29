"""Library mode — instrument a *real* training run from inside it.

Probe mode (``mlprobe.run`` / scenarios) predicts cost/time/memory before you
commit, by running small external probes. That hits a wall for complex
production models: the training code must be importable in mlprobe's
environment. Library mode sidesteps it entirely — you ``import mlprobe`` inside
the training script you're already running, so you're already in the right
environment with the right data on the right hardware.

Two entry points, sharing the existing external-measurement machinery:

``measure()`` (in ``mlprobe.measure``) is the low-level primitive — already
public — for timing one block::

    with mlprobe.measure() as m:
        labels = run_agglomerative(distances, threshold)
    print(m.wall_clock_s, m.peak_rss_mb)

``profile()`` is the unified entry point: it wraps a whole run, optionally
runs the pre-flight ``audit()`` (and can fail fast on blockers), captures
per-stage timings via ``p.stage(...)``, produces a single-run report (the
probe-mode report shape minus the scaling fits, which need multiple sizes),
and — when handed an MLflow run — logs all of it there::

    with mlprobe.profile(spec=spec, mlflow_run=mlflow.active_run()) as p:
        with p.stage("load"):
            data = load_everything()
        with p.stage("cluster"):
            labels = run_agglomerative(data, threshold)

    print(p.report().format())
"""

from __future__ import annotations

import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import psutil

from mlprobe.audit import AuditFinding, AuditReport, audit
from mlprobe.measure import Measurement, _process_tree_rss_mb, measure
from mlprobe.spec import ModelSpec


class IncompatibleSpecError(RuntimeError):
    """Raised by ``profile(on_blocker="raise")`` when the pre-flight audit
    finds a hard incompatibility — so a doomed run fails before it spends
    compute instead of after."""


@dataclass(frozen=True)
class StageTiming:
    """Timing for one ``p.stage(name)`` block. ``rss_mb`` is a process-tree
    RSS snapshot taken at stage end (best-effort) — per-stage peak VRAM/util
    is the job of the stage-profiling pass (#13)."""

    name: str
    wall_clock_s: float
    rss_mb: float | None = None


@dataclass(frozen=True)
class StageBottleneck:
    """The stage that dominates wall-clock. ``share`` is its fraction of the
    run's total wall-clock."""

    name: str
    wall_clock_s: float
    share: float


# A stage taking at least this share of total wall-clock is called out as the
# bottleneck; unstaged time above this share is flagged as a coverage blind spot.
_BOTTLENECK_SHARE = 0.50
_UNSTAGED_SHARE = 0.20


@dataclass
class ProfileReport:
    """Single-run report. Same spirit as the probe-mode report, minus the
    scaling fits and Pareto frontier (those need multiple data sizes; a single
    real run has one)."""

    spec_name: str | None
    wall_clock_s: float
    peak_rss_mb: float
    peak_vram_mb: float | None = None
    gpu_util_avg: float | None = None
    stages: list[StageTiming] = field(default_factory=list)
    audit: AuditReport | None = None
    stage_findings: list[AuditFinding] = field(default_factory=list)

    def bottleneck(self) -> StageBottleneck | None:
        """The stage with the largest share of wall-clock, or ``None`` when no
        stages were recorded."""
        if not self.stages:
            return None
        total = self.wall_clock_s or sum(s.wall_clock_s for s in self.stages) or 1.0
        top = max(self.stages, key=lambda s: s.wall_clock_s)
        return StageBottleneck(top.name, top.wall_clock_s, top.wall_clock_s / total)

    def format(self) -> str:
        title = self.spec_name or "run"
        out = [f"=== Profile: {title} ===", f"  wall_clock: {_fmt_time(self.wall_clock_s)}",
               f"  peak_rss:   {_fmt_mb(self.peak_rss_mb)}"]
        if self.peak_vram_mb is not None:
            out.append(f"  peak_vram:  {_fmt_mb(self.peak_vram_mb)}")
        if self.gpu_util_avg is not None:
            out.append(f"  gpu_util:   {self.gpu_util_avg:.0f}%")

        if self.stages:
            bottleneck = self.bottleneck()
            out.append("\nSTAGES:")
            total = self.wall_clock_s or 1.0
            for s in self.stages:
                pct = 100.0 * s.wall_clock_s / total
                line = f"  {s.name:24s} {_fmt_time(s.wall_clock_s):>8s}  ({pct:4.1f}%)"
                if s.rss_mb is not None:
                    line += f"  rss={_fmt_mb(s.rss_mb)}"
                if bottleneck and s.name == bottleneck.name and bottleneck.share >= _BOTTLENECK_SHARE:
                    line += "  ← bottleneck"
                out.append(line)
            accounted = sum(s.wall_clock_s for s in self.stages)
            if self.wall_clock_s and accounted < self.wall_clock_s:
                unacc = self.wall_clock_s - accounted
                out.append(f"  {'(unstaged)':24s} {_fmt_time(unacc):>8s}  "
                           f"({100.0 * unacc / total:4.1f}%)")

        if self.stage_findings:
            out += ["", "STAGE FINDINGS:"]
            out += [f"  [{f.code}] {f.message}" for f in self.stage_findings]

        if self.audit is not None and self.audit.findings:
            out += ["", "AUDIT:", self.audit.format()]
        return "\n".join(out)


def analyze_stages(stages: list[StageTiming], wall_clock_s: float) -> list[AuditFinding]:
    """Neutral, factual findings about where a run's wall-clock went.

    Phase 0 of stage profiling (#13): surfaces the dominant stage and any large
    unstaged remainder, from the wall-clock data ``p.stage()`` already captures.
    It states *where the time is*, never *what to do about it* (CPU/GPU-bound
    classification needs per-stage resource sampling — a later phase)."""
    findings: list[AuditFinding] = []
    if not stages or wall_clock_s <= 0:
        return findings

    staged = sum(s.wall_clock_s for s in stages)
    top = max(stages, key=lambda s: s.wall_clock_s)
    share = top.wall_clock_s / wall_clock_s

    if share >= _BOTTLENECK_SHARE:
        others = [s for s in stages if s is not top]
        runner_up = max((s.wall_clock_s for s in others), default=0.0)
        factor = f" ({top.wall_clock_s / runner_up:.1f}x the next stage)" if runner_up > 0 else ""
        findings.append(AuditFinding(
            severity="info", code="stage_bottleneck",
            message=(
                f"Stage {top.name!r} dominates: {share:.0%} of wall-clock "
                f"({_fmt_time(top.wall_clock_s)} of {_fmt_time(wall_clock_s)}){factor}. "
                f"Optimization effort is best aimed here."
            ),
        ))

    unstaged = wall_clock_s - staged
    if unstaged / wall_clock_s >= _UNSTAGED_SHARE:
        findings.append(AuditFinding(
            severity="info", code="unstaged_time",
            message=(
                f"{unstaged / wall_clock_s:.0%} of wall-clock "
                f"({_fmt_time(unstaged)}) is outside any stage — wrap it in "
                f"p.stage(...) to attribute it."
            ),
        ))
    return findings


class Profiler:
    """The handle yielded by ``profile()``. Use ``p.stage(name)`` to time
    sub-sections; read ``p.report()`` after the ``with`` block exits."""

    def __init__(self, spec_name: str | None, audit_report: AuditReport | None):
        self.spec_name = spec_name
        self.audit = audit_report
        self._stages: list[StageTiming] = []
        self._report: ProfileReport | None = None

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a named sub-section of the run. Stages may be sequential or
        nested; each is recorded with its own wall-clock and an end-of-stage
        RSS snapshot."""
        proc = psutil.Process()
        start = time.monotonic()
        try:
            yield
        finally:
            dur = time.monotonic() - start
            rss = _safe_rss(proc)
            self._stages.append(StageTiming(name=name, wall_clock_s=dur, rss_mb=rss))

    def report(self) -> ProfileReport:
        if self._report is None:
            raise RuntimeError("report() is only available after the profile() block exits")
        return self._report

    # -- internal --
    def _finalize(self, m: Measurement) -> ProfileReport:
        stages = list(self._stages)
        self._report = ProfileReport(
            spec_name=self.spec_name,
            wall_clock_s=m.wall_clock_s,
            peak_rss_mb=m.peak_rss_mb,
            peak_vram_mb=m.peak_vram_mb,
            gpu_util_avg=m.gpu_util_avg,
            stages=stages,
            audit=self.audit,
            stage_findings=analyze_stages(stages, m.wall_clock_s),
        )
        return self._report


@contextmanager
def profile(
    spec: ModelSpec | None = None,
    *,
    name: str | None = None,
    mlflow_run: Any = None,
    on_blocker: str = "warn",
) -> Iterator[Profiler]:
    """Instrument a real training run.

    Parameters
    ----------
    spec : optional ModelSpec. When given, its pre-flight ``audit()`` runs
        *before* the body and is attached to the report.
    name : label for the report when no spec is given.
    mlflow_run : an MLflow ``Run`` (or anything with ``.info.run_id``). When
        provided, the profile metrics + audit findings are logged to it on
        exit. Requires ``mlprobe[mlflow]``; logging is skipped with a warning if
        mlflow isn't installed.
    on_blocker : what to do if the pre-flight audit finds a hard
        incompatibility — ``"raise"`` (fail fast before spending compute),
        ``"warn"`` (default, print to stderr), or ``"ignore"``.
    """
    audit_report = audit(spec) if spec is not None else None
    if audit_report is not None and audit_report.has_blockers:
        blockers = "; ".join(f"[{f.code}] {f.message}" for f in audit_report.incompatible)
        if on_blocker == "raise":
            raise IncompatibleSpecError(
                f"pre-flight audit found {len(audit_report.incompatible)} blocker(s): {blockers}"
            )
        if on_blocker == "warn":
            print(f"mlprobe: pre-flight audit blockers: {blockers}", file=sys.stderr)

    p = Profiler(spec_name=(spec.name if spec is not None else name), audit_report=audit_report)
    with measure() as m:
        yield p
    report = p._finalize(m)
    if mlflow_run is not None:
        _log_to_mlflow(report, mlflow_run)


# ---- MLflow write side (optional dep) ------------------------------------

def _log_to_mlflow(report: ProfileReport, run: Any) -> None:
    """Log a profile report to an MLflow run. Lazily imports mlflow so it
    stays an optional dependency; no-ops with a warning if it's missing."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        warnings.warn(
            "mlflow_run was passed but mlflow is not installed; skipping logging. "
            "Install with: pip install 'mlprobe[mlflow]'",
            stacklevel=2,
        )
        return

    metrics: dict[str, float] = {
        "mlprobe.wall_clock_s": report.wall_clock_s,
        "mlprobe.peak_rss_mb": report.peak_rss_mb,
    }
    if report.peak_vram_mb is not None:
        metrics["mlprobe.peak_vram_mb"] = report.peak_vram_mb
    if report.gpu_util_avg is not None:
        metrics["mlprobe.gpu_util_avg"] = report.gpu_util_avg
    for s in report.stages:
        metrics[f"mlprobe.stage.{_sanitize_key(s.name)}.wall_clock_s"] = s.wall_clock_s

    tags: dict[str, str] = {}
    if report.audit is not None:
        tags["mlprobe.audit_has_blockers"] = str(report.audit.has_blockers).lower()
        codes = sorted({f.code for f in report.audit.findings})
        if codes:
            tags["mlprobe.audit_findings"] = ",".join(codes)

    run_id = getattr(getattr(run, "info", None), "run_id", None)
    if run_id is not None:
        client = MlflowClient()
        for k, v in metrics.items():
            client.log_metric(run_id, k, v)
        for k, v in tags.items():
            client.set_tag(run_id, k, v)
    else:
        # Fall back to the active-run convenience API.
        mlflow.log_metrics(metrics)
        if tags:
            mlflow.set_tags(tags)


def _sanitize_key(name: str) -> str:
    """MLflow metric keys allow a limited charset; map the rest to '_'."""
    return "".join(c if (c.isalnum() or c in "_-./ ") else "_" for c in name)


def _safe_rss(proc: psutil.Process) -> float | None:
    try:
        return _process_tree_rss_mb(proc)
    except Exception:
        return None


def _fmt_time(s: float) -> str:
    if s < 60:
        return f"{s:.2f}s"
    if s < 3600:
        return f"{s / 60:.1f}m"
    return f"{s / 3600:.2f}h"


def _fmt_mb(mb: float | None) -> str:
    if mb is None:
        return "?"
    if mb < 1024:
        return f"{mb:.0f}MB"
    return f"{mb / 1024:.2f}GB"
