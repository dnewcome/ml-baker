"""Fit scaling curves (time vs dataset size, quality vs dataset size).

Given probe measurements at several subset_fractions, fit several candidate
models and pick the best by R². The exponent of the power-law fit is the
useful headline for time scaling — it tells you the asymptotic complexity
class (≈1 → linear, ≈2 → quadratic, etc.) the user explicitly asked for.

Stays numpy-only (no scipy) by reducing each candidate to a linear fit in a
transformed space. The true saturating power-law for quality
(``y = c - a*N^-b``) needs a nonlinear solver and is a follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ScalingFit:
    model: str                           # "linear" | "loglinear" | "power" | "log"
    params: dict[str, float]             # model-specific coefficients
    r2: float                            # in original (un-transformed) space
    n_points: int                        # how many data points the fit used
    predict: Callable[[float], float] = field(repr=False)

    def at(self, x: float) -> float:
        return self.predict(x)


def fit_time_scaling(xs: list[float], ys: list[float]) -> ScalingFit | None:
    """Time-vs-size: try linear, O(N log N), power law. Pick best by R²."""
    return _best_fit(xs, ys, models=("linear", "loglinear", "power"))


def fit_quality_scaling(xs: list[float], ys: list[float]) -> ScalingFit | None:
    """Quality-vs-size: try logarithmic growth and power law. Both express
    diminishing returns. (Saturating power law `c - a*N^-b` is better-shaped
    but needs scipy — planned follow-up.)"""
    return _best_fit(xs, ys, models=("log", "power"))


def _best_fit(xs: list[float], ys: list[float], models: tuple[str, ...]) -> ScalingFit | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    candidates: list[ScalingFit] = []
    for name in models:
        fit = _FITTERS[name](xs, ys)
        if fit is not None:
            candidates.append(fit)
    if not candidates:
        return None
    return max(candidates, key=lambda f: f.r2)


# ---- Individual fitters --------------------------------------------------

def _fit_linear(xs: list[float], ys: list[float]) -> ScalingFit | None:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    predict = lambda v, s=slope, b=intercept: float(s * v + b)
    return ScalingFit(
        model="linear",
        params={"slope": float(slope), "intercept": float(intercept)},
        r2=_r2(y, np.array([predict(v) for v in x])),
        n_points=len(xs),
        predict=predict,
    )


def _fit_loglinear(xs: list[float], ys: list[float]) -> ScalingFit | None:
    """y = a * x * log(x) + b — typical for algorithms like comparison sorts."""
    x = np.asarray(xs, dtype=float)
    if np.any(x <= 0):
        return None
    y = np.asarray(ys, dtype=float)
    xlogx = x * np.log(x)
    slope, intercept = np.polyfit(xlogx, y, 1)
    predict = lambda v, s=slope, b=intercept: float(s * v * log(v) + b) if v > 0 else float("nan")
    return ScalingFit(
        model="loglinear",
        params={"a": float(slope), "b": float(intercept)},
        r2=_r2(y, np.array([predict(v) for v in x])),
        n_points=len(xs),
        predict=predict,
    )


def _fit_power(xs: list[float], ys: list[float]) -> ScalingFit | None:
    """y = a * x^b. Fit linearly in log-log space; requires positive y."""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if np.any(x <= 0) or np.any(y <= 0):
        return None
    b, log_a = np.polyfit(np.log(x), np.log(y), 1)
    a = float(np.exp(log_a))
    predict = lambda v, a=a, b=float(b): a * (v ** b)
    return ScalingFit(
        model="power",
        params={"a": a, "exponent": float(b)},
        r2=_r2(y, np.array([predict(v) for v in x])),
        n_points=len(xs),
        predict=predict,
    )


def _fit_log(xs: list[float], ys: list[float]) -> ScalingFit | None:
    """y = a + b * log(x). Common shape for quality-vs-data."""
    x = np.asarray(xs, dtype=float)
    if np.any(x <= 0):
        return None
    y = np.asarray(ys, dtype=float)
    b, a = np.polyfit(np.log(x), y, 1)
    predict = lambda v, a=float(a), b=float(b): a + b * log(v) if v > 0 else float("nan")
    return ScalingFit(
        model="log",
        params={"a": float(a), "b": float(b)},
        r2=_r2(y, np.array([predict(v) for v in x])),
        n_points=len(xs),
        predict=predict,
    )


_FITTERS: dict[str, Callable[[list[float], list[float]], ScalingFit | None]] = {
    "linear":    _fit_linear,
    "loglinear": _fit_loglinear,
    "power":     _fit_power,
    "log":       _fit_log,
}


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination in the original space."""
    if len(y_true) < 2:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot
