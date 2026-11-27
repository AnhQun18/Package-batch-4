"""Configuration models for RFM feature engineering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence, Tuple, Union

import pandas as pd

NegativeAmountStrategy = Literal["keep", "clip_zero", "drop", "separate"]
ScalerType = Literal["standard", "minmax", "robust"]


@dataclass
class RFMConfig:
    """Configuration for :class:`rfm_fe.RFMFeatureEngineer`.

    Parameters
    ----------
    customer_col : str
        Column name identifying each customer/entity.
    date_col : str
        Column name containing transaction timestamps/dates.
    amount_col : str
        Column name containing transaction monetary values.
    transaction_col : str, optional
        Column name identifying unique orders/transactions. When provided,
        frequency is computed as the number of unique transactions rather than
        raw row count (useful for line-item datasets).
    category_col : str, optional
        Optional categorical column (e.g., product category or channel) used
        to compute customer diversity features (unique count, concentration,
        Shannon entropy).
    reference_date : str or pd.Timestamp, optional
        Snapshot date used to compute recency, tenure, and rolling windows.
        If None, it is learned during ``fit()`` as ``max(date_col) + 1 day``.
        Transactions strictly after ``reference_date`` are excluded to prevent
        temporal data leakage.
    windows : Sequence[int], default=(30, 90, 180, 365)
        Rolling lookback windows (in days) relative to ``reference_date``.
    n_quantiles : int, default=5
        Number of quantile buckets for R, F, and M scores (e.g. 5 -> scores 1..5).
    rfm_weights : Tuple[float, float, float], default=(1.0, 1.0, 1.0)
        Weights for ``(R, F, M)`` when computing ``rfm_score_weighted``.
    handle_negative_amount : {"keep", "clip_zero", "drop", "separate"}, default="keep"
        How to handle negative monetary values (e.g., refunds/returns):
        - ``"keep"``: keep all rows as-is.
        - ``"clip_zero"``: clip negative values to 0.0.
        - ``"drop"``: drop rows with negative amounts.
        - ``"separate"``: compute main RFM on non-negative rows and add
          dedicated refund features (``refund_count``, ``refund_amount_sum``,
          ``refund_rate``).
    include_windows : bool, default=True
        Whether to generate multi-window RFM and velocity/trend features.
    include_interpurchase : bool, default=True
        Whether to generate inter-purchase time (cadence) features.
    include_temporal : bool, default=True
        Whether to generate calendar/behavioral habit features.
    include_ratios : bool, default=True
        Whether to generate RFM interaction and ratio features.
    include_scores : bool, default=True
        Whether to generate leak-free quantile R, F, M scores (bin edges
        learned in ``fit()`` and reused in ``transform()``).
    include_segment_label : bool, default=False
        Whether to include a human-readable ``rfm_segment`` string column.
        Disabled by default so the output is 100% numeric for any ML model.
    encode_segment : bool, default=False
        Whether to one-hot encode ``rfm_segment`` into numeric binary columns.
    log_transform : bool, default=False
        Whether to apply signed log1p transform ``sign(x) * log1p(|x|)`` to
        continuous non-ratio, non-score features (helpful for linear models,
        distance-based models, and neural networks).
    scaler : {"standard", "minmax", "robust"}, optional
        Optional scikit-learn scaler fitted on training customer features and
        applied in ``transform()``. Leave ``None`` for tree-based models.
    fillna_value : float, default=0.0
        Value used to fill remaining NaN/inf values in numeric features so the
        resulting DataFrame is immediately usable by any model.
    return_customer_col : bool, default=True
        If True, ``customer_col`` is returned as the first column (reset index).
        If False, ``customer_col`` is set as the DataFrame index.
    """

    customer_col: str = "customer_id"
    date_col: str = "order_date"
    amount_col: str = "amount"
    transaction_col: Optional[str] = None
    category_col: Optional[str] = None
    reference_date: Optional[Union[str, pd.Timestamp]] = None
    windows: Sequence[int] = field(default_factory=lambda: (30, 90, 180, 365))
    n_quantiles: int = 5
    rfm_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    handle_negative_amount: NegativeAmountStrategy = "keep"
    include_windows: bool = True
    include_interpurchase: bool = True
    include_temporal: bool = True
    include_ratios: bool = True
    include_scores: bool = True
    include_segment_label: bool = False
    encode_segment: bool = False
    log_transform: bool = False
    scaler: Optional[ScalerType] = None
    fillna_value: float = 0.0
    return_customer_col: bool = True

    def __post_init__(self) -> None:
        self.windows = tuple(sorted({int(w) for w in self.windows if int(w) > 0}))
        if self.n_quantiles < 2:
            raise ValueError("n_quantiles must be at least 2.")
        if len(self.rfm_weights) != 3:
            raise ValueError("rfm_weights must be a 3-tuple of (w_r, w_f, w_m).")
        if self.handle_negative_amount not in ("keep", "clip_zero", "drop", "separate"):
            raise ValueError(
                f"Invalid handle_negative_amount: {self.handle_negative_amount!r}."
            )
        if self.scaler not in (None, "standard", "minmax", "robust"):
            raise ValueError(f"Invalid scaler: {self.scaler!r}.")
