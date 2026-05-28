"""Tests for the block-size dataset profiler.

The cost model is Σ C(block_size, 2) — within-block candidate pairs. These
tests pin the math and the roadblock-surfacing behaviour (neutral facts, no
prescriptions).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from mlprobe import (
    DatasetProfile,
    DatasetSpec,
    EvalMetric,
    ModelSpec,
    block_size_profile,
    block_sizes_from_labels,
    class_balance_profile,
    data_audit,
    outlier_profile,
    similarity_profile,
    stratified_plan,
    token_length_profile,
)


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


# ---- class_balance_profile ------------------------------------------------

def _codes_of(profile):
    return {f.code for f in profile.findings}


def test_class_balance_imbalance_and_floor():
    p = class_balance_profile({"a": 4700, "b": 200, "c": 100}, subset_fraction=0.01)
    assert p.stats["n_classes"] == 3
    assert p.stats["n_items"] == 5000
    assert p.stats["imbalance_ratio"] == 47
    assert p.stats["smallest_class_size"] == 100
    # 100 * 0.01 = 1 item → below the floor → warned.
    assert p.stats["smallest_class_at_subset"] == 1
    assert "class_imbalance" in _codes_of(p)
    assert "subsample_floor" in _codes_of(p)


def test_class_balance_accepts_labels_and_balanced_is_quiet():
    p = class_balance_profile(["a", "a", "b", "b"])
    assert p.data["counts"] == {"a": 2, "b": 2}
    assert p.stats["imbalance_ratio"] == 1
    assert "class_imbalance" not in _codes_of(p)  # balanced → no warning


# ---- stratified_plan ------------------------------------------------------

def test_stratified_plan_preserves_proportions():
    p = stratified_plan({"a": 90, "b": 10}, fraction=0.5)
    assert p.data["plan"] == {"a": 45, "b": 5}
    assert p.stats["achieved_n"] == 50


def test_stratified_plan_min_per_group_protects_tail():
    p = stratified_plan({"a": 1000, "b": 1}, fraction=0.01, min_per_group=1)
    # Proportional would give b ~0; the floor keeps it at 1.
    assert p.data["plan"]["b"] >= 1
    assert "min_per_group_raised" in _codes_of(p)


def test_stratified_plan_requires_a_target():
    with pytest.raises(ValueError):
        stratified_plan({"a": 10, "b": 10})


# ---- similarity_profile ---------------------------------------------------

def test_similarity_bimodal_vs_unimodal():
    rng = np.random.default_rng(0)
    bimodal = np.concatenate([rng.normal(0.2, 0.04, 3000), rng.normal(0.8, 0.04, 3000)])
    unimodal = rng.normal(0.5, 0.15, 6000)

    pb = similarity_profile(bimodal)
    pu = similarity_profile(unimodal)
    assert pb.stats["bimodality_coefficient"] > 0.555
    assert pu.stats["bimodality_coefficient"] < 0.555
    assert "bimodal" in _codes_of(pb)
    assert "unimodal_blur" in _codes_of(pu)


def test_similarity_threshold_ambiguity():
    rng = np.random.default_rng(1)
    vals = rng.normal(0.5, 0.1, 5000)
    p = similarity_profile(vals, threshold=0.5)
    # A threshold in the middle of a unimodal blob has a sizeable ambiguous zone.
    assert p.stats["ambiguous_fraction"] > 0.10
    assert "threshold_ambiguity" in _codes_of(p)


# ---- outlier_profile ------------------------------------------------------

def test_outlier_detects_extreme_value():
    rng = np.random.default_rng(2)
    vals = np.concatenate([rng.normal(0, 1, 2000), [60.0]])
    p = outlier_profile(vals)
    assert p.stats["max_abs_z"] > 10
    assert "heavy_tail" in _codes_of(p)


def test_outlier_clean_data_no_heavy_tail():
    rng = np.random.default_rng(3)
    p = outlier_profile(rng.normal(0, 1, 5000))
    assert "heavy_tail" not in _codes_of(p)


# ---- data_audit (spec-integrated pre-flight) ------------------------------

def _spec(analyze_callable=None):
    return ModelSpec(
        name="da",
        train_callable="synthetic_trainable:train",
        evaluate_callable="synthetic_trainable:evaluate",
        dataset=DatasetSpec(
            loader="synthetic_trainable:load", total_size=10_000,
            analyze_callable=analyze_callable,
        ),
        eval_metrics=[EvalMetric(name="f1", higher_is_better=True, primary=True)],
    )


def test_data_audit_runs_single_profile():
    profiles = data_audit(_spec("synthetic_trainable:analyze"))
    assert len(profiles) == 1
    assert profiles[0].kind == "block_sizes"
    # loader at subset_fraction=1.0 → 10k rows split into two blocks of 5k.
    assert profiles[0].stats["n_items"] == 10_000


def test_data_audit_normalizes_a_list():
    profiles = data_audit(_spec("synthetic_trainable:analyze_multi"))
    kinds = {p.kind for p in profiles}
    assert kinds == {"block_sizes", "class_balance"}


def test_data_audit_empty_when_unconfigured():
    assert data_audit(_spec(None)) == []


# ---- token_length_profile -------------------------------------------------

def test_token_length_percentiles_and_summary():
    lengths = [10] * 90 + [1000] * 10           # mostly short, a few long
    p = token_length_profile(lengths)
    assert p.stats["n"] == 100
    assert p.stats["p50"] == 10
    assert p.stats["max"] == 1000
    assert "length_summary" in _codes_of(p)


def test_token_length_truncation_and_padding_waste():
    lengths = [16] * 95 + [4096] * 5            # p50 tiny, rare very-long
    p = token_length_profile(lengths, max_seq_len=512)
    # 5% exceed 512 → truncation warning.
    assert p.stats["truncation_fraction"] == 0.05
    assert "truncation" in _codes_of(p)
    # Padding all to 512 processes far more than the real tokens → waste warning.
    assert p.stats["pad_to_max_waste_factor"] > 1.5
    assert "padding_waste" in _codes_of(p)


def test_token_length_uniform_no_waste_warning():
    p = token_length_profile([512] * 1000, max_seq_len=512)
    assert p.stats["pad_to_max_waste_factor"] == 1.0
    assert "padding_waste" not in _codes_of(p)
    assert "truncation" not in _codes_of(p)


def test_token_length_empty():
    p = token_length_profile([])
    assert "empty" in _codes_of(p)
