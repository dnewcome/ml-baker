"""Expand HyperParam sweeps into concrete config dicts.

Cartesian-only for now (kept deliberately simple — see project notes; later
expanders may use LLM-guided sampling or hill climbing). Params without a
sweep are pinned at their declared default; params with a sweep contribute
their generated values to the Cartesian product.

The runner caps the expansion at ``ProbeConfig.max_variants`` (if set) with a
deterministic stride so the chosen subset is reproducible.
"""

from __future__ import annotations

from itertools import product
from typing import Any

from mlprobe.spec import (
    CategoricalSweep,
    HyperParam,
    ModelSpec,
    NumericSweep,
)


def expand_sweeps(spec: ModelSpec) -> list[dict[str, Any]]:
    """Return one config dict per Cartesian combination, capped at
    ``spec.probe.max_variants`` if set."""

    per_param: list[tuple[str, list[Any]]] = []
    for hp in spec.hyperparameters:
        per_param.append((hp.name, _values_for(hp)))

    configs: list[dict[str, Any]] = []
    for combo in product(*(vs for _, vs in per_param)):
        configs.append({name: val for (name, _), val in zip(per_param, combo)})

    cap = spec.probe.max_variants
    if cap is not None and len(configs) > cap:
        configs = _stride_sample(configs, cap)
    return configs


def _values_for(hp: HyperParam) -> list[Any]:
    """The list of values this hyperparam contributes. With no sweep, it's
    the single default value (so the Cartesian product still expands cleanly)."""

    if hp.sweep is None:
        return [hp.default]
    if isinstance(hp.sweep, CategoricalSweep):
        return list(hp.sweep.values)
    if isinstance(hp.sweep, NumericSweep):
        return _expand_numeric(hp)
    raise TypeError(f"Unhandled sweep type: {type(hp.sweep)!r}")


def _expand_numeric(hp: HyperParam) -> list[Any]:
    sweep: NumericSweep = hp.sweep  # type: ignore[assignment]
    cast = int if hp.type == "int" else float

    if sweep.step is not None:
        return _arithmetic_range(sweep.min, sweep.max, sweep.step, cast)
    if sweep.log_step is not None:
        return _log_range(sweep.min, sweep.max, sweep.log_step, cast)
    if sweep.count is not None:
        return _linspace(sweep.min, sweep.max, sweep.count, cast)
    raise ValueError(f"{hp.name!r}: NumericSweep has no resolution set")


def _arithmetic_range(lo: float, hi: float, step: float, cast) -> list[Any]:
    out, x = [], lo
    # Tolerance so floating point doesn't drop the endpoint.
    eps = step * 1e-9
    while x <= hi + eps:
        out.append(cast(x))
        x += step
    return out


def _log_range(lo: float, hi: float, factor: float, cast) -> list[Any]:
    if factor <= 1:
        raise ValueError(f"log_step must be > 1 (got {factor})")
    out, x = [], lo
    eps = hi * 1e-9
    while x <= hi + eps:
        out.append(cast(x))
        x *= factor
    return out


def _linspace(lo: float, hi: float, n: int, cast) -> list[Any]:
    if n < 2:
        return [cast(lo)]
    step = (hi - lo) / (n - 1)
    return [cast(lo + i * step) for i in range(n)]


def _stride_sample(items: list, k: int) -> list:
    """Deterministic evenly-spaced subsample. Preserves first and last."""
    if k >= len(items):
        return items
    if k == 1:
        return [items[0]]
    indices = [round(i * (len(items) - 1) / (k - 1)) for i in range(k)]
    return [items[i] for i in indices]
