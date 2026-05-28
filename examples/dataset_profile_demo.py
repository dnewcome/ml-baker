"""Demo: dataset-shape profiling — surface roadblocks before you train.

All of these are cheap (O(n) or a sample) and *data-blind*: you supply the
domain-specific inputs (block sizes, similarities, labels), mlprof does the
reusable analysis and reports neutral facts. None of it prescribes an
algorithm — it tells you where the cost / risk is so you can decide.

    python examples/dataset_profile_demo.py
"""

from __future__ import annotations

import numpy as np

from mlprof import (
    block_size_profile,
    class_balance_profile,
    outlier_profile,
    similarity_profile,
    stratified_plan,
)


def main() -> None:
    rng = np.random.default_rng(0)

    print("# block sizes — pair-explosion roadblock")
    block_sizes = [40_000, 8_000, 6_000] + [50] * 200 + [3] * 5_000
    print(block_size_profile(block_sizes).format(), "\n")

    print("# class balance — imbalance + how small you can subsample")
    print(class_balance_profile({"common": 47_000, "rare": 2_000, "tiny": 200},
                                subset_fraction=0.02).format(), "\n")

    print("# stratified plan — representative draw at 2% (mlprof computes, your loader draws)")
    print(stratified_plan({"common": 47_000, "rare": 2_000, "tiny": 200},
                          fraction=0.02, min_per_group=20).data["plan"], "\n")

    print("# similarity distribution — clean separation vs blur (FP/FN risk)")
    blurry = rng.normal(0.5, 0.15, 5_000)
    print(similarity_profile(blurry, threshold=0.5).format(), "\n")

    print("# outliers — tail heaviness")
    scores = np.concatenate([rng.normal(0, 1, 5_000), [55.0]])
    print(outlier_profile(scores).format())


if __name__ == "__main__":
    main()
