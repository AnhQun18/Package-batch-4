"""Input validation and preprocessing for transactional DataFrames."""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from .config import RFMConfig


def validate_and_prepare(
    df: pd.DataFrame,
    config: RFMConfig,
    reference_date: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, pd.Timestamp, Optional[pd.DataFrame]]:
    """Validate required columns, coerce types, and apply reference_date cutoff.

    Returns
    -------
    clean_df : pd.DataFrame
        Validated transactional DataFrame (non-negative if ``separate`` or ``drop``).
    ref_date : pd.Timestamp
        Resolved reference snapshot timestamp.
    refund_df : pd.DataFrame or None
        Negative-amount rows when ``config.handle_negative_amount == "separate"``,
        otherwise None.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")

    required = [config.customer_col, config.date_col, config.amount_col]
    if config.transaction_col is not None:
        required.append(config.transaction_col)
    if config.category_col is not None:
        required.append(config.category_col)

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s) in DataFrame: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    work = df[required].copy()

    # Drop rows with missing customer ID or date
    work = work.dropna(subset=[config.customer_col, config.date_col])
    if work.empty:
        raise ValueError(
            "No valid rows remain after dropping missing customer_col/date_col."
        )

    # Coerce date column to timezone-naive Timestamp for consistent arithmetic
    dates = pd.to_datetime(work[config.date_col], errors="coerce")
    if dates.isna().all():
        raise ValueError(
            f"Column {config.date_col!r} could not be parsed as datetime."
        )
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert(None)
    work[config.date_col] = dates
    work = work.dropna(subset=[config.date_col])

    # Coerce amount column to numeric
    work[config.amount_col] = pd.to_numeric(work[config.amount_col], errors="coerce")
    work = work.dropna(subset=[config.amount_col])
    if work.empty:
        raise ValueError(
            "No valid rows remain after coercing date_col and amount_col."
        )

    # Resolve reference date
    if reference_date is not None:
        ref_ts = pd.Timestamp(reference_date)
    elif config.reference_date is not None:
        ref_ts = pd.Timestamp(config.reference_date)
    else:
        ref_ts = work[config.date_col].max() + pd.Timedelta(days=1)

    if ref_ts.tzinfo is not None:
        ref_ts = ref_ts.tz_convert(None)

    # Prevent temporal leakage: keep only transactions on or before ref_ts
    work = work.loc[work[config.date_col] <= ref_ts].copy()
    if work.empty:
        raise ValueError(
            f"No transactions found on or before reference_date={ref_ts}."
        )

    # Handle negative amounts
    refund_df: Optional[pd.DataFrame] = None
    if config.handle_negative_amount == "clip_zero":
        work[config.amount_col] = work[config.amount_col].clip(lower=0.0)
    elif config.handle_negative_amount == "drop":
        work = work.loc[work[config.amount_col] >= 0.0].copy()
    elif config.handle_negative_amount == "separate":
        refund_df = work.loc[work[config.amount_col] < 0.0].copy()
        work = work.loc[work[config.amount_col] >= 0.0].copy()

    if work.empty:
        raise ValueError(
            "No valid transactions remain after applying handle_negative_amount="
            f"{config.handle_negative_amount!r}."
        )

    return work, ref_ts, refund_df
