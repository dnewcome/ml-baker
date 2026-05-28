"""Demo: surface a pair-explosion roadblock from block sizes, no training run.

Blocked similarity / dedup / blocked clustering compares every pair *within* a
block, so wall-clock is driven by Σ C(block_size, 2). A single huge block (a
common token, a giant near-duplicate cluster) dominates that sum. This is
computable from an O(n) group-by — so the roadblock is visible up front,
before any training. mlprof surfaces the fact; you decide what to do about it.

    python examples/dataset_profile_demo.py
"""

from __future__ import annotations

from mlprof import block_size_profile


def main() -> None:
    # A realistic skew: one pathological block (a common first-token), a few
    # large ones, and a long tail of small blocks.
    block_sizes = [40_000, 8_000, 6_000] + [50] * 200 + [3] * 5_000

    profile = block_size_profile(block_sizes)
    print(profile.format())


if __name__ == "__main__":
    main()
