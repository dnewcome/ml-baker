"""Dataset profiling — surface the *shape* of the training data, cheaply.

This is decision-support, not a solver. Per mlprof's scope boundary, a
``DatasetProfile`` reports *neutral facts* about the data so you (or a separate
AI-assisted chat) can decide which approach to try; it never prescribes an
algorithm from shape. mlprof stays data-blind: you compute the domain-specific
inputs (you own the blocking key / similarity notion), and mlprof does the
reusable analysis.

The first signal is **block sizes**, because the dominant training-time
roadblock for blocked / all-pairs-within-group methods (dedup, blocked
clustering, entity resolution) is *combinatorial*: the candidate-pair work is
roughly ``Σ C(block_size, 2)``. A single giant block (a common token, a huge
near-duplicate cluster) makes that sum explode and dominates wall-clock — and
crucially it's computable from an O(n) group-by, with **no training run**. So
the roadblock is predictable up front instead of discovered after a crashed
full run.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from mlprof.audit import AuditFinding


@dataclass
class DatasetProfile:
    """Computed facts about the training set's shape, plus neutral findings.

    ``findings`` reuse the audit's ``AuditFinding`` shape but are *factual* —
    they surface a cost driver ("largest block drives 78% of pairs"), never a
    prescription ("use a different algorithm")."""

    kind: str                                   # what was profiled, e.g. "block_sizes"
    stats: dict[str, float] = field(default_factory=dict)
    findings: list[AuditFinding] = field(default_factory=list)

    def format(self) -> str:
        out = [f"=== DatasetProfile: {self.kind} ==="]
        for k, v in self.stats.items():
            out.append(f"  {k:<28} {_fmt_stat(v)}")
        if self.findings:
            out.append("")
            for sev in ("warning", "info"):
                group = [f for f in self.findings if f.severity == sev]
                if not group:
                    continue
                out.append(f"{sev.upper()} ({len(group)}):")
                out += [f"  [{f.code}] {f.message}" for f in group]
        return "\n".join(out)


def block_size_profile(
    block_sizes: Iterable[int] | Mapping[object, int],
    *,
    largest_share_warn: float = 0.25,
    concentration_warn: float = 0.50,
) -> DatasetProfile:
    """Profile the candidate-pair cost implied by a set of block sizes.

    Parameters
    ----------
    block_sizes :
        The sizes of the blocks an all-pairs-within-block method will compare.
        Accepts a list/iterable of sizes, or a ``{block_key: size}`` mapping
        (e.g. ``df.groupby(key).size().to_dict()`` or
        ``Counter(block_labels)``). You produce these — mlprof never sees the
        data, only the sizes.
    largest_share_warn :
        Flag a roadblock when the single largest block accounts for at least
        this fraction of all candidate pairs.
    concentration_warn :
        Flag concentration when the top 1% of blocks account for at least this
        fraction of all candidate pairs.

    The cost model is ``pairs(block) = C(size, 2) = size*(size-1)/2`` summed
    over blocks — the number of within-block comparisons. This is the work an
    all-pairs / agglomerative-within-block / blocked-similarity method does, so
    it's a faithful proxy for that family's wall-clock without running it.
    """
    sizes = _coerce_sizes(block_sizes)
    profile = DatasetProfile(kind="block_sizes")

    n_blocks = len(sizes)
    n_items = sum(sizes)
    profile.stats["n_blocks"] = float(n_blocks)
    profile.stats["n_items"] = float(n_items)

    if n_blocks == 0:
        profile.findings.append(AuditFinding(
            severity="info", code="no_blocks",
            message="No blocks provided — nothing to profile.",
        ))
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
            message="All blocks are singletons — no within-block pairs to compare.",
        ))
        return profile

    reduction = naive_all_pairs / total_pairs
    profile.stats["blocking_reduction_factor"] = reduction
    profile.findings.append(AuditFinding(
        severity="info", code="blocking_reduction",
        message=f"Blocking yields {total_pairs:,} candidate pairs — "
                f"{reduction:,.0f}x fewer than all-pairs ({naive_all_pairs:,}).",
    ))

    # Concentration: order blocks by their pair contribution (monotonic in size).
    pairs_desc = sorted(pairs, reverse=True)
    largest_share = pairs_desc[0] / total_pairs
    profile.stats["largest_block_pair_share"] = largest_share

    top1pct_k = max(1, math.ceil(0.01 * n_blocks))
    top1pct_share = sum(pairs_desc[:top1pct_k]) / total_pairs
    profile.stats["top1pct_block_pair_share"] = top1pct_share
    profile.stats["blocks_to_90pct_pairs"] = float(_blocks_to_fraction(pairs_desc, 0.90, total_pairs))

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


# ---- helpers --------------------------------------------------------------

def _coerce_sizes(block_sizes: Iterable[int] | Mapping[object, int]) -> list[int]:
    raw = block_sizes.values() if isinstance(block_sizes, Mapping) else block_sizes
    sizes = [int(s) for s in raw if int(s) > 0]
    return sizes


def _pairs(n: int) -> int:
    return n * (n - 1) // 2


def _blocks_to_fraction(pairs_desc: list[int], fraction: float, total: int) -> int:
    target = fraction * total
    cum = 0
    for i, p in enumerate(pairs_desc, start=1):
        cum += p
        if cum >= target:
            return i
    return len(pairs_desc)


def _fmt_stat(v: float) -> str:
    if v == int(v) and abs(v) < 1e15:
        return f"{int(v):,}"
    if 0 < abs(v) < 1:
        return f"{v:.3f}"
    return f"{v:,.2f}"


def block_sizes_from_labels(labels: Iterable[object]) -> Counter:
    """Convenience: turn per-item block labels into a ``{block: size}`` Counter
    you can pass to ``block_size_profile``."""
    return Counter(labels)
