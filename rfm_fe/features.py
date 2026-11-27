"""Core vectorized RFM feature computation functions."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import RFMConfig
from .utils import safe_divide


def compute_base_rfm(
    df: pd.DataFrame,
    config: RFMConfig,
    ref_date: pd.Timestamp,
) -> pd.DataFrame:
    """Compute core Recency, Frequency, Monetary, and Tenure features."""
    cust = config.customer_col
    date = config.date_col
    amt = config.amount_col
    tx = config.transaction_col

    grouped = df.groupby(cust, sort=True)

    agg_dict = {
        date: ["min", "max"],
        amt: ["sum", "mean", "std", "min", "max", "median"],
    }
    base = grouped.agg(agg_dict)
    base.columns = [
        "first_purchase_date",
        "last_purchase_date",
        "monetary_sum",
        "monetary_mean",
        "monetary_std",
        "monetary_min",
        "monetary_max",
        "monetary_median",
    ]

    if tx is not None:
        base["frequency"] = grouped[tx].nunique().astype(float)
    else:
        base["frequency"] = grouped[amt].size().astype(float)

    base["recency_days"] = (
        (ref_date - base["last_purchase_date"]).dt.total_seconds() / 86400.0
    ).clip(lower=0.0)
    base["tenure_days"] = (
        (ref_date - base["first_purchase_date"]).dt.total_seconds() / 86400.0
    ).clip(lower=0.0)
    base["active_span_days"] = (
        (base["last_purchase_date"] - base["first_purchase_date"]).dt.total_seconds()
        / 86400.0
    ).clip(lower=0.0)

    base = base.drop(columns=["first_purchase_date", "last_purchase_date"])
    base["monetary_std"] = base["monetary_std"].fillna(0.0)
    return base


def compute_ratios(base: pd.DataFrame) -> pd.DataFrame:
    """Compute RFM interaction and ratio features."""
    out = pd.DataFrame(index=base.index)
    out["avg_order_value"] = safe_divide(base["monetary_sum"], base["frequency"])
    out["monetary_cv"] = safe_divide(base["monetary_std"], base["monetary_mean"])
    out["monetary_range"] = base["monetary_max"] - base["monetary_min"]
    out["frequency_per_month"] = safe_divide(
        base["frequency"] * 30.0, np.maximum(base["tenure_days"], 1.0)
    )
    out["monetary_per_month"] = safe_divide(
        base["monetary_sum"] * 30.0, np.maximum(base["tenure_days"], 1.0)
    )
    out["recency_to_tenure_ratio"] = safe_divide(
        base["recency_days"], np.maximum(base["tenure_days"], 1.0)
    )
    out["is_repeat_customer"] = (base["frequency"] > 1.0).astype(float)
    return out


def compute_window_features(
    df: pd.DataFrame,
    config: RFMConfig,
    ref_date: pd.Timestamp,
    index: pd.Index,
) -> pd.DataFrame:
    """Compute rolling lookback window RFM features and velocity/trend ratios."""
    cust = config.customer_col
    date = config.date_col
    amt = config.amount_col
    tx = config.transaction_col

    out = pd.DataFrame(index=index)
    days_ago = (ref_date - df[date]).dt.total_seconds() / 86400.0

    for w in config.windows:
        mask = (days_ago >= 0.0) & (days_ago <= float(w))
        sub = df.loc[mask]
        if sub.empty:
            out[f"freq_{w}d"] = 0.0
            out[f"monetary_sum_{w}d"] = 0.0
            out[f"monetary_mean_{w}d"] = 0.0
            continue

        grp = sub.groupby(cust, sort=False)
        if tx is not None:
            freq_w = grp[tx].nunique().astype(float)
        else:
            freq_w = grp[amt].size().astype(float)

        mon_sum_w = grp[amt].sum().astype(float)
        mon_mean_w = grp[amt].mean().astype(float)

        out[f"freq_{w}d"] = freq_w.reindex(index, fill_value=0.0)
        out[f"monetary_sum_{w}d"] = mon_sum_w.reindex(index, fill_value=0.0)
        out[f"monetary_mean_{w}d"] = mon_mean_w.reindex(index, fill_value=0.0)

    # Velocity / trend features comparing shortest vs longest window
    if len(config.windows) >= 2:
        w_short = config.windows[0]
        w_long = config.windows[-1]
        out[f"freq_velocity_{w_short}d_vs_{w_long}d"] = safe_divide(
            out[f"freq_{w_short}d"] * (float(w_long) / float(w_short)),
            out[f"freq_{w_long}d"],
        )
        out[f"monetary_velocity_{w_short}d_vs_{w_long}d"] = safe_divide(
            out[f"monetary_sum_{w_short}d"] * (float(w_long) / float(w_short)),
            out[f"monetary_sum_{w_long}d"],
        )

    return out


def compute_interpurchase_features(
    df: pd.DataFrame,
    config: RFMConfig,
    index: pd.Index,
) -> pd.DataFrame:
    """Compute inter-purchase time (IPT) cadence features per customer."""
    cust = config.customer_col
    date = config.date_col
    tx = config.transaction_col

    if tx is not None:
        order_dates = (
            df[[cust, tx, date]]
            .groupby([cust, tx], as_index=False)[date]
            .min()
        )
    else:
        order_dates = df[[cust, date]].copy()

    order_dates = order_dates.sort_values([cust, date], kind="mergesort")
    order_dates["ipt_days"] = (
        order_dates.groupby(cust, sort=False)[date].diff().dt.total_seconds() / 86400.0
    )

    valid_ipt = order_dates.dropna(subset=["ipt_days"])
    out = pd.DataFrame(index=index)

    if valid_ipt.empty:
        out["ipt_mean_days"] = 0.0
        out["ipt_std_days"] = 0.0
        out["ipt_min_days"] = 0.0
        out["ipt_max_days"] = 0.0
        out["ipt_last_days"] = 0.0
        return out

    grp = valid_ipt.groupby(cust, sort=False)["ipt_days"]
    out["ipt_mean_days"] = grp.mean().reindex(index, fill_value=0.0)
    out["ipt_std_days"] = grp.std().reindex(index, fill_value=0.0).fillna(0.0)
    out["ipt_min_days"] = grp.min().reindex(index, fill_value=0.0)
    out["ipt_max_days"] = grp.max().reindex(index, fill_value=0.0)
    out["ipt_last_days"] = grp.last().reindex(index, fill_value=0.0)
    return out


def compute_temporal_features(
    df: pd.DataFrame,
    config: RFMConfig,
    index: pd.Index,
) -> pd.DataFrame:
    """Compute calendar and purchasing habit features."""
    cust = config.customer_col
    date = config.date_col

    temp = pd.DataFrame(
        {
            cust: df[cust],
            "is_weekend": (df[date].dt.dayofweek >= 5).astype(float),
            "hour": df[date].dt.hour.astype(float),
            "year_month": df[date].dt.to_period("M"),
        }
    )

    grp = temp.groupby(cust, sort=False)
    out = pd.DataFrame(index=index)
    out["weekend_purchase_ratio"] = grp["is_weekend"].mean().reindex(index, fill_value=0.0)
    out["preferred_hour_mean"] = grp["hour"].mean().reindex(index, fill_value=0.0)
    out["active_months_count"] = (
        grp["year_month"].nunique().astype(float).reindex(index, fill_value=0.0)
    )
    return out


def compute_category_features(
    df: pd.DataFrame,
    config: RFMConfig,
    index: pd.Index,
) -> pd.DataFrame:
    """Compute category diversity (unique count, top share, Shannon entropy)."""
    cust = config.customer_col
    cat = config.category_col
    if cat is None:
        return pd.DataFrame(index=index)

    sub = df[[cust, cat]].dropna(subset=[cat])
    out = pd.DataFrame(index=index)
    if sub.empty:
        out["category_nunique"] = 0.0
        out["category_top_share"] = 0.0
        out["category_entropy"] = 0.0
        return out

    counts = sub.groupby([cust, cat], sort=False).size()
    totals = counts.groupby(level=0).transform("sum")
    probs = counts / totals

    nunique = counts.groupby(level=0).size().astype(float)
    top_share = probs.groupby(level=0).max().astype(float)
    entropy = -(probs * np.log2(probs)).groupby(level=0).sum().astype(float)

    out["category_nunique"] = nunique.reindex(index, fill_value=0.0)
    out["category_top_share"] = top_share.reindex(index, fill_value=0.0)
    out["category_entropy"] = entropy.reindex(index, fill_value=0.0)
    return out


def compute_refund_features(
    refund_df: Optional[pd.DataFrame],
    base_df: pd.DataFrame,
    config: RFMConfig,
    index: pd.Index,
) -> pd.DataFrame:
    """Compute refund/return features when handle_negative_amount='separate'."""
    out = pd.DataFrame(index=index)
    if refund_df is None or refund_df.empty:
        out["refund_count"] = 0.0
        out["refund_amount_sum"] = 0.0
        out["refund_rate"] = 0.0
        return out

    cust = config.customer_col
    amt = config.amount_col
    grp = refund_df.groupby(cust, sort=False)[amt]

    ref_count = grp.size().astype(float).reindex(index, fill_value=0.0)
    ref_sum = grp.sum().abs().astype(float).reindex(index, fill_value=0.0)
    total_events = base_df["frequency"] + ref_count

    out["refund_count"] = ref_count
    out["refund_amount_sum"] = ref_sum
    out["refund_rate"] = safe_divide(ref_count, total_events)
    return out
