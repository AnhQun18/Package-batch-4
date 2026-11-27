"""Numeric and quantile helper utilities."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def safe_divide(
    numerator: pd.Series | np.ndarray | float,
    denominator: pd.Series | np.ndarray | float,
    fill_value: float = 0.0,
) -> pd.Series | np.ndarray:
    """Element-wise division that safely handles zeros and NaNs."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.true_divide(numerator, denominator)
    if isinstance(result, pd.Series):
        return result.replace([np.inf, -np.inf], np.nan).fillna(fill_value)
    result = np.where(np.isfinite(result), result, fill_value)
    return result


def signed_log1p(values: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """Apply sign-preserving log1p transform: sign(x) * log1p(|x|)."""
    return np.sign(values) * np.log1p(np.abs(values))


def fit_quantile_edges(series: pd.Series, n_quantiles: int) -> np.ndarray:
    """Learn strictly increasing quantile bin edges from a 1D series.

    Returns an array of interior cut points (length 0..n_quantiles-1).
    Using ``np.digitize(x, edges) + 1`` then assigns every value to a bin
    in ``[1, len(edges) + 1]`` without ever failing on duplicate values.
    """
    clean = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return np.array([], dtype=float)

    probs = np.linspace(0.0, 1.0, n_quantiles + 1)[1:-1]
    edges = np.quantile(clean, probs)
    unique_edges = np.unique(edges)
    return unique_edges


def apply_quantile_score(
    series: pd.Series,
    edges: np.ndarray,
    n_quantiles: int,
    reverse: bool = False,
) -> pd.Series:
    """Score values into 1..n_quantiles using pre-fitted quantile edges."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if edges.size == 0:
        scores = np.full(values.shape, (n_quantiles + 1) / 2.0, dtype=float)
    else:
        # digitize returns 0..len(edges); shift to 1..(len(edges)+1)
        raw_bins = np.digitize(values, edges, right=True) + 1
        max_bin = edges.size + 1
        if max_bin < n_quantiles:
            # Rescale cleanly into [1, n_quantiles] when duplicates collapsed bins
            scores = 1.0 + (raw_bins - 1.0) * (n_quantiles - 1.0) / max(max_bin - 1, 1)
        else:
            scores = raw_bins.astype(float)

        if reverse:
            scores = (n_quantiles + 1.0) - scores

    scores = np.where(np.isnan(values), np.nan, scores)
    return pd.Series(scores, index=series.index, dtype=float)


def assign_rfm_segment(
    r_score: pd.Series,
    f_score: pd.Series,
    m_score: pd.Series,
    n_quantiles: int = 5,
) -> pd.Series:
    """Map numeric R, F, M scores to standard business segment labels.

    Thresholds scale automatically with ``n_quantiles``.
    """
    r_norm = r_score / float(n_quantiles)
    fm_norm = (f_score + m_score) / (2.0 * float(n_quantiles))

    conditions: List[np.ndarray] = [
        (r_norm >= 0.8) & (fm_norm >= 0.8),
        (r_norm >= 0.6) & (fm_norm >= 0.6),
        (r_norm >= 0.8) & (fm_norm >= 0.4),
        (r_norm >= 0.8) & (fm_norm < 0.4),
        (r_norm >= 0.6) & (fm_norm < 0.6),
        (r_norm >= 0.4) & (fm_norm >= 0.6),
        (r_norm >= 0.4) & (fm_norm < 0.6),
        (r_norm < 0.4) & (fm_norm >= 0.6),
    ]
    choices: Tuple[str, ...] = (
        "Champions",
        "Loyal",
        "Potential Loyalist",
        "New Customers",
        "Promising",
        "Need Attention",
        "About to Sleep",
        "At Risk",
    )
    labels = np.select(conditions, choices, default="Hibernating")
    return pd.Series(labels, index=r_score.index, dtype="object")
