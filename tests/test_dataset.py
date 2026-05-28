"""Tests for the block-size dataset profiler.

The cost model is Σ C(block_size, 2) — within-block candidate pairs. These
tests pin the math and the roadblock-surfacing behaviour (neutral facts, no
prescriptions).
"""

from __future__ import annotations

from collections import Counter

from mlprof import DatasetProfile, block_size_profile, block_sizes_from_labels


def _codes(profile: DatasetProfile) -> set[str]:
    return {f.code for f in profile.findings}


def test_pair_count_math():
    # sizes [3, 2] → C(3,2)+C(2,2) = 3 + 1 = 4 ; n_items=5 ; naive=C(5,2)=10
    p = block_size_profile([3, 2])
    assert p.stats["n_items"] == 5
    assert p.stats["total_candidate_pairs"] == 4
    assert p.stats["naive_all_pairs"] == 10
    assert p.stats["blocking_reduction_factor"] == 2.5
    assert p.stats["largest_block_pair_share"] == 0.75  # 3 of 4 pairs


def test_giant_block_flags_pair_explosion():
    sizes = [10_000] + [2] * 1_000
    p = block_size_profile(sizes)
    assert "pair_explosion_largest_block" in _codes(p)
    assert "pair_cost_concentration" in _codes(p)
    # The one giant block is essentially all the cost.
    assert p.stats["largest_block_pair_share"] > 0.99


def test_uniform_blocks_no_roadblock():
    p = block_size_profile([10] * 100)
    assert "pair_explosion_largest_block" not in _codes(p)
    assert "pair_cost_concentration" not in _codes(p)
    assert p.stats["largest_block_pair_share"] == 0.01
    assert p.stats["blocking_reduction_factor"] > 100


def test_accepts_mapping_and_counter_and_labels():
    from_list = block_size_profile([3, 2])
    from_dict = block_size_profile({"a": 3, "b": 2})
    from_counter = block_size_profile(Counter({"a": 3, "b": 2}))
    for p in (from_dict, from_counter):
        assert p.stats["total_candidate_pairs"] == from_list.stats["total_candidate_pairs"]

    labels = ["a", "a", "a", "b", "b"]
    assert block_sizes_from_labels(labels) == Counter({"a": 3, "b": 2})
    p = block_size_profile(block_sizes_from_labels(labels))
    assert p.stats["total_candidate_pairs"] == 4


def test_empty_and_all_singletons():
    empty = block_size_profile([])
    assert "no_blocks" in _codes(empty)

    singles = block_size_profile([1, 1, 1])
    assert singles.stats["n_items"] == 3
    assert singles.stats["total_candidate_pairs"] == 0
    assert "no_within_block_pairs" in _codes(singles)


def test_format_is_renderable():
    text = block_size_profile([10_000] + [2] * 50).format()
    assert "DatasetProfile: block_sizes" in text
    assert "pair_explosion_largest_block" in text


def test_findings_are_factual_not_prescriptive():
    # Scope boundary: surface the cost driver, never tell the user which
    # algorithm to use.
    p = block_size_profile([10_000] + [2] * 100)
    blob = " ".join(f.message for f in p.findings).lower()
    for prescription in ("use ", "switch to", "you should", "instead of", "recommend"):
        assert prescription not in blob
