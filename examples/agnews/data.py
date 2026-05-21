"""Shared AG News loader for the mlprof demo.

AG News is a 4-class news topic classification dataset: 120k train / 7.6k test.
Downloaded via HuggingFace ``datasets`` (cached on disk after the first call,
so subsequent subprocess probes reuse the cache and don't re-download).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=4)
def _full_split(split: str) -> Any:
    # Lazy-import so this module can be parsed without `datasets` installed —
    # only the runner subprocess needs the heavy deps.
    from datasets import load_dataset
    return load_dataset("ag_news", split=split)


def load(subset_fraction: float = 1.0, split: str | None = None,
         seed: int | None = None):
    """mlprof LoadDatasetFn. Returns a HF Dataset (consumable by both the
    sklearn and the DistilBERT trainables — they each adapt it differently)."""
    split = split or "train"
    ds = _full_split(split)
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    n = max(1, int(len(ds) * subset_fraction))
    return ds.select(range(n))
