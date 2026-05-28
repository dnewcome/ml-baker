"""Dataset profiling — surface the *shape* of the training data, cheaply.

This is decision-support, not a solver. Per mlprobe's scope boundary, a
``DatasetProfile`` reports *neutral facts* about the data so you (or a separate
AI-assisted chat) can decide which approach to try; it never prescribes an
algorithm from shape. mlprobe stays data-blind: you compute the domain-specific
inputs (block sizes, pairwise similarities, anomaly scores, labels — you own
the blocking key / similarity notion), and mlprobe does the reusable analysis.

Profilers here:

  - ``block_size_profile``    — combinatorial pair-explosion cost of blocked /
                                all-pairs-within-group methods (Σ C(size, 2)).
  - ``class_balance_profile`` — imbalance, entropy, and the small-class floor
                                that limits how far you can subsample.
  - ``stratified_plan``       — per-group draw counts for a representative
                                subsample (the sampling deliverable).
  - ``similarity_profile``    — distribution of pairwise similarities: clean
                                separation vs a blur that floods FP/FN.
  - ``outlier_profile``       — IQR / σ outlier fractions and tail heaviness.

``data_audit(spec)`` is the spec-integrated pre-flight: the explicit parallel
to the static capability ``audit(spec)``, it runs a user-supplied analysis
callable over the (cheaply loaded) data and returns its profiles before any
training is spent.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

from mlprobe.audit import AuditFinding


@dataclass
class DatasetProfile:
    """Computed facts about the training set's shape, plus neutral findings.

    ``findings`` reuse the audit's ``AuditFinding`` shape but are *factual* —
    they surface a cost driver ("largest block drives 78% of pairs"), never a
    prescription ("use a different algorithm"). ``stats`` holds scalar
    summaries; ``data`` holds structured payloads (histograms, per-group plans)."""

    kind: str                                     # what was profiled, e.g. "block_sizes"
    stats: dict[str, float] = field(default_factory=dict)
    findings: list[AuditFinding] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    def format(self) -> str:
        out = [f"=== DatasetProfile: {self.kind} ==="]
        for k, v in self.stats.items():
            out.append(f"  {k:<30} {_fmt_stat(v)}")
        if self.findings:
            out.append("")
            for sev in ("warning", "info"):
                group = [f for f in self.findings if f.severity == sev]
                if not group:
                    continue
                out.append(f"{sev.upper()} ({len(group)}):")
                out += [f"  [{f.code}] {f.message}" for f in group]
        return "\n".join(out)


# ---- Block sizes (pair-explosion cost) ------------------------------------

def block_size_profile(
    block_sizes: Iterable[int] | Mapping[object, int],
    *,
    largest_share_warn: float = 0.25,
    concentration_warn: float = 0.50,
) -> DatasetProfile:
    """Profile the candidate-pair cost implied by a set of block sizes.

    ``block_sizes`` are the sizes of the blocks an all-pairs-within-block method
    compares — a list/iterable of sizes, or a ``{block_key: size}`` mapping
    (e.g. ``df.groupby(key).size().to_dict()``). The cost model is
    ``pairs(block) = C(size, 2)`` summed over blocks, a faithful proxy for that
    family's wall-clock without running it. O(n), no training run.
    """
    sizes = [int(s) for s in (block_sizes.values() if isinstance(block_sizes, Mapping) else block_sizes) if int(s) > 0]
    profile = DatasetProfile(kind="block_sizes")

    n_blocks = len(sizes)
    n_items = sum(sizes)
    profile.stats["n_blocks"] = float(n_blocks)
    profile.stats["n_items"] = float(n_items)

    if n_blocks == 0:
        profile.findings.append(AuditFinding(
            severity="info", code="no_blocks", message="No blocks provided — nothing to profile."))
        return profile

    pairs = [_pairs(s) for s in sizes]
    total_pairs = sum(pairs)
    largest = max(sizes)
    naive_all_pairs = _pairs(n_items)

    profile.stats["largest_block"] = float(largest)
    profile.stats["total_candidate_pairs"] = float(total_pairs)
    profile.stats["naive_all_pairs"] = float(naive_all_pairs)
    profile.findings.append(AuditFinding(
        severity="info", code="block_counts",
        message=f"{n_blocks:,} blocks over {n_items:,} items "
                f"(largest block {largest:,}, mean {n_items / n_blocks:.1f}).",
    ))

    if total_pairs == 0:
        profile.findings.append(AuditFinding(
            severity="info", code="no_within_block_pairs",
            message="All blocks are singletons — no within-block pairs to compare."))
        return profile

    reduction = naive_all_pairs / total_pairs
    profile.stats["blocking_reduction_factor"] = reduction
    profile.findings.append(AuditFinding(
        severity="info", code="blocking_reduction",
        message=f"Blocking yields {total_pairs:,} candidate pairs — "
                f"{reduction:,.0f}x fewer than all-pairs ({naive_all_pairs:,}).",
    ))

    pairs_desc = sorted(pairs, reverse=True)
    largest_share = pairs_desc[0] / total_pairs
    top1pct_k = max(1, math.ceil(0.01 * n_blocks))
    top1pct_share = sum(pairs_desc[:top1pct_k]) / total_pairs
    profile.stats["largest_block_pair_share"] = largest_share
    profile.stats["top1pct_block_pair_share"] = top1pct_share
    profile.stats["blocks_to_90pct_pairs"] = float(_count_to_fraction(pairs_desc, 0.90, total_pairs))

    if largest_share >= largest_share_warn:
        profile.findings.append(AuditFinding(
            severity="warning", code="pair_explosion_largest_block",
            message=f"Largest block ({largest:,} items) generates {pairs_desc[0]:,} pairs "
                    f"= {largest_share:.0%} of all candidate pairs. All-pairs-within-block "
                    f"work is dominated by this one block.",
        ))
    if top1pct_share >= concentration_warn:
        profile.findings.append(AuditFinding(
            severity="warning", code="pair_cost_concentration",
            message=f"Pair cost is concentrated: the top {top1pct_k:,} block(s) "
                    f"({top1pct_k / n_blocks:.1%} of blocks) account for "
                    f"{top1pct_share:.0%} of candidate pairs.",
        ))
    return profile


# ---- Class / group balance ------------------------------------------------

def class_balance_profile(
    labels_or_counts: Iterable[object] | Mapping[object, int],
    *,
    subset_fraction: float | None = None,
    small_floor: int = 30,
    imbalance_warn: float = 10.0,
) -> DatasetProfile:
    """Profile class/group balance.

    Pass per-item ``labels`` (counted for you) or a ``{class: count}`` mapping.
    Reports the imbalance ratio, normalized entropy (1.0 = perfectly balanced),
    and the smallest class — and, when ``subset_fraction`` is given, how many
    items the smallest class would have at that probe size (the floor that
    limits how far you can subsample before a class effectively vanishes).
    """
    counts = _as_counts(labels_or_counts)
    profile = DatasetProfile(kind="class_balance")

    n_classes = len(counts)
    n_items = sum(counts.values())
    profile.stats["n_classes"] = float(n_classes)
    profile.stats["n_items"] = float(n_items)
    profile.data["counts"] = dict(counts)
    if n_classes == 0 or n_items == 0:
        profile.findings.append(AuditFinding(
            severity="info", code="no_classes", message="No labels/counts provided."))
        return profile

    sizes = sorted(counts.values(), reverse=True)
    largest, smallest = sizes[0], sizes[-1]
    imbalance = largest / smallest if smallest > 0 else float("inf")
    props = np.array(sizes, dtype=float) / n_items
    entropy = float(-(props * np.log(props)).sum())
    norm_entropy = entropy / math.log(n_classes) if n_classes > 1 else 1.0
    smallest_label = min(counts, key=counts.get)

    profile.stats["imbalance_ratio"] = imbalance
    profile.stats["normalized_entropy"] = norm_entropy
    profile.stats["smallest_class_size"] = float(smallest)

    profile.findings.append(AuditFinding(
        severity="info", code="class_counts",
        message=f"{n_classes:,} classes over {n_items:,} items; "
                f"normalized entropy {norm_entropy:.2f} (1.0 = balanced).",
    ))
    if imbalance >= imbalance_warn:
        profile.findings.append(AuditFinding(
            severity="warning", code="class_imbalance",
            message=f"Imbalance ratio {imbalance:,.0f}:1 — largest class {largest:,}, "
                    f"smallest class {smallest_label!r} {smallest:,} "
                    f"({smallest / n_items:.1%} of data).",
        ))

    if subset_fraction is not None:
        expected = smallest * subset_fraction
        profile.stats["smallest_class_at_subset"] = expected
        sev = "warning" if expected < small_floor else "info"
        profile.findings.append(AuditFinding(
            severity=sev, code="subsample_floor",
            message=f"At subset_fraction={subset_fraction:g}, the smallest class "
                    f"({smallest_label!r}) would have ~{expected:.0f} items"
                    + (f" (< {small_floor} → too few for a reliable read on it)."
                       if expected < small_floor else "."),
        ))
    return profile


def stratified_plan(
    group_counts: Mapping[object, int],
    *,
    fraction: float | None = None,
    n: int | None = None,
    min_per_group: int = 1,
) -> DatasetProfile:
    """Compute a representative subsample plan: how many items to draw from each
    group to approximately preserve proportions at the target size.

    Give a ``{group: count}`` mapping and either ``fraction`` (of the total) or
    an absolute ``n``. Each group is allocated proportionally but at least
    ``min_per_group`` (never more than it has). mlprobe produces the plan; your
    loader draws it — mlprobe never touches the data."""
    counts = dict(group_counts)
    total = sum(counts.values())
    profile = DatasetProfile(kind="stratified_plan")
    if total == 0:
        profile.findings.append(AuditFinding(
            severity="info", code="empty", message="No groups/counts provided."))
        return profile

    if n is None:
        if fraction is None:
            raise ValueError("stratified_plan needs either fraction= or n=")
        target = round(fraction * total)
    else:
        target = n
    target = max(0, min(target, total))

    plan: dict[object, int] = {}
    raised = 0
    for g, c in counts.items():
        want = round(c / total * target)
        want = min(c, max(want, min(min_per_group, c)))
        if want > round(c / total * target):
            raised += 1
        plan[g] = want

    achieved = sum(plan.values())
    profile.data["plan"] = plan
    profile.stats["target_n"] = float(target)
    profile.stats["achieved_n"] = float(achieved)
    profile.stats["n_groups"] = float(len(counts))
    profile.findings.append(AuditFinding(
        severity="info", code="plan",
        message=f"Plan draws {achieved:,} of {total:,} items across {len(counts):,} groups "
                f"(target {target:,}).",
    ))
    if raised:
        profile.findings.append(AuditFinding(
            severity="info", code="min_per_group_raised",
            message=f"{raised:,} small group(s) were raised to the min_per_group floor "
                    f"({min_per_group}), so the sample slightly over-represents the tail.",
        ))
    return profile


# ---- Pairwise similarity distribution -------------------------------------

def similarity_profile(
    values: Iterable[float],
    *,
    threshold: float | None = None,
    ambiguous_band: float = 0.05,
    bins: int = 50,
) -> DatasetProfile:
    """Profile a sample of pairwise similarity (or distance) values.

    Reports distribution stats and Sarle's bimodality coefficient — a clean
    bimodal distribution means "same" and "different" pairs separate cleanly; a
    unimodal blur means any threshold trades false positives for false
    negatives. If a candidate ``threshold`` is given, also reports the fraction
    of pairs in the ambiguous band around it. (Facts about the distribution —
    not a recommended threshold.)
    """
    x = np.asarray(list(values), dtype=float)
    profile = DatasetProfile(kind="similarity")
    if x.size == 0:
        profile.findings.append(AuditFinding(
            severity="info", code="empty", message="No similarity values provided."))
        return profile

    mean, std, skew, exkurt = _moments(x)
    lo, hi = float(x.min()), float(x.max())
    profile.stats.update({
        "n": float(x.size), "mean": mean, "std": std,
        "min": lo, "max": hi, "median": float(np.median(x)),
    })
    profile.data["quantiles"] = {q: float(np.quantile(x, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}

    bc = _bimodality_coefficient(x.size, skew, exkurt)
    if bc is not None:
        profile.stats["bimodality_coefficient"] = bc
        if bc > 0.555:
            profile.findings.append(AuditFinding(
                severity="info", code="bimodal",
                message=f"Bimodality coefficient {bc:.2f} (>0.55) — two distinct populations; "
                        f"similar vs dissimilar pairs separate fairly cleanly.",
            ))
        else:
            profile.findings.append(AuditFinding(
                severity="warning", code="unimodal_blur",
                message=f"Bimodality coefficient {bc:.2f} (<0.55) — no clear separation between "
                        f"similar and dissimilar pairs; any single threshold will trade false "
                        f"positives for false negatives.",
            ))

    if threshold is not None:
        band = ambiguous_band * (hi - lo)
        frac_ambig = float(np.mean(np.abs(x - threshold) <= band)) if band > 0 else 0.0
        profile.stats["threshold"] = float(threshold)
        profile.stats["ambiguous_fraction"] = frac_ambig
        profile.stats["fraction_above_threshold"] = float(np.mean(x > threshold))
        profile.findings.append(AuditFinding(
            severity="warning" if frac_ambig >= 0.10 else "info",
            code="threshold_ambiguity",
            message=f"{frac_ambig:.0%} of sampled pairs fall within ±{band:.3g} of threshold "
                    f"{threshold:g} — the ambiguous zone where FP/FN concentrate.",
        ))
    return profile


# ---- Outliers / tail heaviness --------------------------------------------

def outlier_profile(
    values: Iterable[float],
    *,
    iqr_k: float = 1.5,
    z_thresh: float = 4.0,
) -> DatasetProfile:
    """Profile outliers in a 1-D array of scores (anomaly scores, distances,
    a numeric feature). Reports the fraction beyond a Tukey IQR fence and beyond
    ``z_thresh`` sigma, plus the most extreme deviation."""
    x = np.asarray(list(values), dtype=float)
    profile = DatasetProfile(kind="outliers")
    if x.size == 0:
        profile.findings.append(AuditFinding(
            severity="info", code="empty", message="No values provided."))
        return profile

    q25, q75 = np.quantile(x, 0.25), np.quantile(x, 0.75)
    iqr = q75 - q25
    mean, std = float(x.mean()), float(x.std())

    iqr_outliers = float(np.mean((x < q25 - iqr_k * iqr) | (x > q75 + iqr_k * iqr))) if iqr > 0 else 0.0
    if std > 0:
        z = np.abs(x - mean) / std
        z_outliers = float(np.mean(z > z_thresh))
        max_z = float(z.max())
    else:
        z_outliers, max_z = 0.0, 0.0

    profile.stats.update({
        "n": float(x.size), "mean": mean, "std": std, "iqr": float(iqr),
        "iqr_outlier_fraction": iqr_outliers,
        "z_outlier_fraction": z_outliers, "max_abs_z": max_z,
    })
    profile.findings.append(AuditFinding(
        severity="info", code="outlier_summary",
        message=f"{iqr_outliers:.1%} beyond {iqr_k:g}×IQR; {z_outliers:.2%} beyond "
                f"{z_thresh:g}σ; most extreme value is {max_z:.1f}σ from the mean.",
    ))
    if max_z >= 2 * z_thresh:
        profile.findings.append(AuditFinding(
            severity="warning", code="heavy_tail",
            message=f"Heavy tail: an extreme value sits {max_z:.0f}σ out — a few points may "
                    f"dominate distance/density-based computations.",
        ))
    return profile


# ---- Token lengths (LLM padding / truncation cost) ------------------------

def token_length_profile(
    lengths: Iterable[int],
    *,
    max_seq_len: int | None = None,
    truncation_warn: float = 0.02,
    waste_warn: float = 1.5,
) -> DatasetProfile:
    """Profile per-example token lengths for an LLM fine-tune.

    ``lengths`` are token counts per example (you tokenize; mlprobe analyzes).
    Surfaces the percentiles, the fraction that would truncate at
    ``max_seq_len``, and the pad-to-max **waste factor** (padded tokens ÷ real
    tokens if every example is padded to ``max_seq_len``) — the quantity that
    drives fine-tune throughput and the seq_len ↔ batch ↔ VRAM tradeoff. Facts
    only; it does not pick a ``max_seq_len`` for you.
    """
    x = np.asarray([int(v) for v in lengths if int(v) >= 0], dtype=float)
    profile = DatasetProfile(kind="token_lengths")
    if x.size == 0:
        profile.findings.append(AuditFinding(
            severity="info", code="empty", message="No token lengths provided."))
        return profile

    q = {p: float(np.quantile(x, p)) for p in (0.5, 0.95, 0.99)}
    longest = int(x.max())
    profile.stats.update({
        "n": float(x.size), "mean": float(x.mean()),
        "p50": q[0.5], "p95": q[0.95], "p99": q[0.99], "max": float(longest),
    })
    profile.findings.append(AuditFinding(
        severity="info", code="length_summary",
        message=f"{x.size:,} examples; tokens p50={q[0.5]:.0f} / p95={q[0.95]:.0f} / "
                f"p99={q[0.99]:.0f} / max={longest:,}.",
    ))

    if max_seq_len is not None:
        truncated = float(np.mean(x > max_seq_len))
        kept = np.minimum(x, max_seq_len)
        waste = (x.size * max_seq_len) / float(kept.sum()) if kept.sum() > 0 else 1.0
        profile.stats["max_seq_len"] = float(max_seq_len)
        profile.stats["truncation_fraction"] = truncated
        profile.stats["pad_to_max_waste_factor"] = waste
        if truncated >= truncation_warn:
            profile.findings.append(AuditFinding(
                severity="warning", code="truncation",
                message=f"{truncated:.1%} of examples exceed max_seq_len={max_seq_len:,} and "
                        f"would be truncated (losing tokens).",
            ))
        if waste >= waste_warn:
            profile.findings.append(AuditFinding(
                severity="warning", code="padding_waste",
                message=f"Padding every example to max_seq_len={max_seq_len:,} processes "
                        f"~{waste:.1f}x the real tokens (p50 is only {q[0.5]:.0f}). "
                        f"Length-bucketing / pad-to-longest-in-batch recovers most of it.",
            ))
    return profile


# ---- Pre-flight data audit (spec-integrated) ------------------------------

def data_audit(
    spec: Any,
    *,
    dataset: Any = None,
    subset_fraction: float = 1.0,
) -> list[DatasetProfile]:
    """Run the spec's optional ``dataset.analyze_callable`` and return its
    ``DatasetProfile``(s) — the explicit, compute-light parallel to the static
    capability ``audit(spec)``, meant to be called pre-flight (before probing).

    The user's analyze callable computes the domain-specific shape (it owns the
    data); mlprobe loads the data cheaply (or you pass ``dataset`` in) and hands
    it over. Returns ``[]`` when no analyze callable is configured.
    """
    from mlprobe.probe import _import_dotted  # local: keep probe stack off the import path

    analyze_ref = getattr(spec.dataset, "analyze_callable", None)
    if not analyze_ref:
        return []

    analyze_fn = _import_dotted(analyze_ref)
    if dataset is None:
        load_fn = _import_dotted(spec.dataset.loader)
        dataset = load_fn(subset_fraction=subset_fraction, split=None, seed=None)

    result = analyze_fn(dataset)
    if isinstance(result, DatasetProfile):
        return [result]
    return list(result)


def format_data_audit(profiles: list[DatasetProfile]) -> str:
    if not profiles:
        return "DATA AUDIT: no analyze callable configured (dataset.analyze_callable)."
    return "\n\n".join(p.format() for p in profiles)


# ---- helpers --------------------------------------------------------------

def block_sizes_from_labels(labels: Iterable[object]) -> Counter:
    """Convenience: turn per-item block labels into a ``{block: size}`` Counter."""
    return Counter(labels)


def _as_counts(labels_or_counts: Iterable[object] | Mapping[object, int]) -> dict[object, int]:
    if isinstance(labels_or_counts, Mapping):
        return {k: int(v) for k, v in labels_or_counts.items() if int(v) > 0}
    return dict(Counter(labels_or_counts))


def _pairs(n: int) -> int:
    return n * (n - 1) // 2


def _count_to_fraction(values_desc: list[int], fraction: float, total: int) -> int:
    target, cum = fraction * total, 0
    for i, p in enumerate(values_desc, start=1):
        cum += p
        if cum >= target:
            return i
    return len(values_desc)


def _moments(x: np.ndarray) -> tuple[float, float, float, float]:
    """Return (mean, std, skewness, excess_kurtosis); skew/kurt are 0 when the
    variance is degenerate."""
    mean = float(x.mean())
    std = float(x.std())
    if std == 0 or x.size < 2:
        return mean, std, 0.0, 0.0
    z = (x - mean) / std
    skew = float(np.mean(z ** 3))
    exkurt = float(np.mean(z ** 4) - 3.0)
    return mean, std, skew, exkurt


def _bimodality_coefficient(n: int, skew: float, exkurt: float) -> float | None:
    """Sarle's bimodality coefficient with finite-sample correction.
    BC = (skew² + 1) / (kurtosis + 3(n-1)²/((n-2)(n-3))). >0.555 ⇒ bimodal-ish."""
    if n < 4:
        return None
    correction = 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    denom = exkurt + correction
    if denom <= 0:
        return None
    return (skew ** 2 + 1.0) / denom


def _fmt_stat(v: float) -> str:
    if v == int(v) and abs(v) < 1e15:
        return f"{int(v):,}"
    if 0 < abs(v) < 1:
        return f"{v:.3f}"
    return f"{v:,.2f}"
