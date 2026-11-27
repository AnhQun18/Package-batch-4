"""Scikit-learn compatible RFMFeatureEngineer and functional API."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .config import NegativeAmountStrategy, RFMConfig, ScalerType
from .features import (
    compute_base_rfm,
    compute_category_features,
    compute_interpurchase_features,
    compute_ratios,
    compute_refund_features,
    compute_temporal_features,
    compute_window_features,
)
from .utils import (
    apply_quantile_score,
    assign_rfm_segment,
    fit_quantile_edges,
    signed_log1p,
)
from .validators import validate_and_prepare


class RFMFeatureEngineer(BaseEstimator, TransformerMixin):
    """Dataset-agnostic and model-agnostic RFM feature engineering transformer.

    Follows the scikit-learn ``fit`` / ``transform`` / ``fit_transform`` API
    so it can be used standalone or inside a ``sklearn.pipeline.Pipeline``
    without temporal or quantile data leakage.
    """

    def __init__(
        self,
        customer_col: str = "customer_id",
        date_col: str = "order_date",
        amount_col: str = "amount",
        transaction_col: Optional[str] = None,
        category_col: Optional[str] = None,
        reference_date: Optional[Union[str, pd.Timestamp]] = None,
        windows: Sequence[int] = (30, 90, 180, 365),
        n_quantiles: int = 5,
        rfm_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        handle_negative_amount: NegativeAmountStrategy = "keep",
        include_windows: bool = True,
        include_interpurchase: bool = True,
        include_temporal: bool = True,
        include_ratios: bool = True,
        include_scores: bool = True,
        include_segment_label: bool = False,
        encode_segment: bool = False,
        log_transform: bool = False,
        scaler: Optional[ScalerType] = None,
        fillna_value: float = 0.0,
        return_customer_col: bool = True,
    ) -> None:
        self.customer_col = customer_col
        self.date_col = date_col
        self.amount_col = amount_col
        self.transaction_col = transaction_col
        self.category_col = category_col
        self.reference_date = reference_date
        self.windows = windows
        self.n_quantiles = n_quantiles
        self.rfm_weights = rfm_weights
        self.handle_negative_amount = handle_negative_amount
        self.include_windows = include_windows
        self.include_interpurchase = include_interpurchase
        self.include_temporal = include_temporal
        self.include_ratios = include_ratios
        self.include_scores = include_scores
        self.include_segment_label = include_segment_label
        self.encode_segment = encode_segment
        self.log_transform = log_transform
        self.scaler = scaler
        self.fillna_value = fillna_value
        self.return_customer_col = return_customer_col

    def _build_config(self) -> RFMConfig:
        return RFMConfig(
            customer_col=self.customer_col,
            date_col=self.date_col,
            amount_col=self.amount_col,
            transaction_col=self.transaction_col,
            category_col=self.category_col,
            reference_date=self.reference_date,
            windows=self.windows,
            n_quantiles=self.n_quantiles,
            rfm_weights=self.rfm_weights,
            handle_negative_amount=self.handle_negative_amount,
            include_windows=self.include_windows,
            include_interpurchase=self.include_interpurchase,
            include_temporal=self.include_temporal,
            include_ratios=self.include_ratios,
            include_scores=self.include_scores,
            include_segment_label=self.include_segment_label,
            encode_segment=self.encode_segment,
            log_transform=self.log_transform,
            scaler=self.scaler,
            fillna_value=self.fillna_value,
            return_customer_col=self.return_customer_col,
        )

    def _build_raw_features(
        self,
        df: pd.DataFrame,
        config: RFMConfig,
        ref_date: pd.Timestamp,
        refund_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        base = compute_base_rfm(df, config, ref_date)
        parts: List[pd.DataFrame] = [base]

        if config.include_ratios:
            parts.append(compute_ratios(base))

        if config.include_windows and len(config.windows) > 0:
            parts.append(compute_window_features(df, config, ref_date, base.index))

        if config.include_interpurchase:
            parts.append(compute_interpurchase_features(df, config, base.index))

        if config.include_temporal:
            parts.append(compute_temporal_features(df, config, base.index))

        if config.category_col is not None:
            parts.append(compute_category_features(df, config, base.index))

        if config.handle_negative_amount == "separate":
            parts.append(compute_refund_features(refund_df, base, config, base.index))

        return pd.concat(parts, axis=1)

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> "RFMFeatureEngineer":
        """Learn reference_date, quantile edges, and optional scaler from df."""
        config = self._build_config()
        clean_df, ref_date, refund_df = validate_and_prepare(df, config)

        self.config_ = config
        self.reference_date_ = ref_date

        features = self._build_raw_features(clean_df, config, ref_date, refund_df)

        # Fit quantile bin edges on training customer distributions
        self.quantile_edges_: Dict[str, np.ndarray] = {}
        if config.include_scores or config.include_segment_label or config.encode_segment:
            self.quantile_edges_["recency_days"] = fit_quantile_edges(
                features["recency_days"], config.n_quantiles
            )
            self.quantile_edges_["frequency"] = fit_quantile_edges(
                features["frequency"], config.n_quantiles
            )
            self.quantile_edges_["monetary_sum"] = fit_quantile_edges(
                features["monetary_sum"], config.n_quantiles
            )

        features = self._append_scores_and_segments(features, config)
        features = self._apply_log_and_fillna(features, config)

        self.numeric_columns_: List[str] = [
            col
            for col in features.columns
            if col != "rfm_segment" and pd.api.types.is_numeric_dtype(features[col])
        ]

        self.scaler_ = None
        if config.scaler is not None and self.numeric_columns_:
            if config.scaler == "standard":
                self.scaler_ = StandardScaler()
            elif config.scaler == "minmax":
                self.scaler_ = MinMaxScaler()
            elif config.scaler == "robust":
                self.scaler_ = RobustScaler()
            self.scaler_.fit(features[self.numeric_columns_].to_numpy(dtype=float))

        self.feature_names_out_: List[str] = list(features.columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform transactional DataFrame into a customer-level feature matrix."""
        if not hasattr(self, "config_"):
            raise RuntimeError(
                "RFMFeatureEngineer must be fitted before calling transform()."
            )

        config = self.config_
        clean_df, ref_date, refund_df = validate_and_prepare(
            df, config, reference_date=self.reference_date_
        )

        features = self._build_raw_features(clean_df, config, ref_date, refund_df)
        features = self._append_scores_and_segments(features, config)
        features = self._apply_log_and_fillna(features, config)

        # Ensure exact column alignment with fit()
        for col in self.feature_names_out_:
            if col not in features.columns:
                features[col] = config.fillna_value if col != "rfm_segment" else "Hibernating"
        features = features[self.feature_names_out_].copy()

        if self.scaler_ is not None and self.numeric_columns_:
            features[self.numeric_columns_] = self.scaler_.transform(
                features[self.numeric_columns_].to_numpy(dtype=float)
            )

        if config.return_customer_col:
            features = features.reset_index()

        return features

    def fit_transform(
        self, df: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit to df and return the transformed customer-level feature DataFrame."""
        return self.fit(df, y=y).transform(df)

    def get_feature_names_out(
        self, input_features: Optional[Sequence[str]] = None
    ) -> np.ndarray:
        """Return output feature names (scikit-learn compatible)."""
        if not hasattr(self, "feature_names_out_"):
            raise RuntimeError("RFMFeatureEngineer is not fitted yet.")
        return np.asarray(self.feature_names_out_, dtype=object)

    def _append_scores_and_segments(
        self, features: pd.DataFrame, config: RFMConfig
    ) -> pd.DataFrame:
        need_scores = (
            config.include_scores or config.include_segment_label or config.encode_segment
        )
        if not need_scores:
            return features

        r_score = apply_quantile_score(
            features["recency_days"],
            self.quantile_edges_["recency_days"],
            config.n_quantiles,
            reverse=True,
        )
        f_score = apply_quantile_score(
            features["frequency"],
            self.quantile_edges_["frequency"],
            config.n_quantiles,
            reverse=False,
        )
        m_score = apply_quantile_score(
            features["monetary_sum"],
            self.quantile_edges_["monetary_sum"],
            config.n_quantiles,
            reverse=False,
        )

        if config.include_scores:
            w_r, w_f, w_m = config.rfm_weights
            features["r_score"] = r_score
            features["f_score"] = f_score
            features["m_score"] = m_score
            features["rfm_score_sum"] = r_score + f_score + m_score
            features["rfm_score_weighted"] = w_r * r_score + w_f * f_score + w_m * m_score

        if config.include_segment_label or config.encode_segment:
            seg = assign_rfm_segment(r_score, f_score, m_score, config.n_quantiles)
            if config.include_segment_label:
                features["rfm_segment"] = seg
            if config.encode_segment:
                dummies = pd.get_dummies(seg, prefix="seg", dtype=float)
                features = pd.concat([features, dummies], axis=1)

        return features

    def _apply_log_and_fillna(
        self, features: pd.DataFrame, config: RFMConfig
    ) -> pd.DataFrame:
        if config.log_transform:
            skip_prefixes = ("seg_",)
            skip_exact = {
                "r_score",
                "f_score",
                "m_score",
                "rfm_score_sum",
                "rfm_score_weighted",
                "is_repeat_customer",
                "recency_to_tenure_ratio",
                "weekend_purchase_ratio",
                "category_top_share",
                "refund_rate",
                "rfm_segment",
            }
            for col in features.columns:
                if col in skip_exact or col.startswith(skip_prefixes):
                    continue
                if pd.api.types.is_numeric_dtype(features[col]):
                    features[col] = signed_log1p(features[col].to_numpy(dtype=float))

        for col in features.columns:
            if col == "rfm_segment":
                continue
            if pd.api.types.is_numeric_dtype(features[col]):
                features[col] = (
                    features[col]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(config.fillna_value)
                )
        return features


def engineer_rfm_features(
    df: pd.DataFrame,
    customer_col: str = "customer_id",
    date_col: str = "order_date",
    amount_col: str = "amount",
    **kwargs,
) -> pd.DataFrame:
    """One-line convenience function to engineer RFM features from a DataFrame."""
    fe = RFMFeatureEngineer(
        customer_col=customer_col,
        date_col=date_col,
        amount_col=amount_col,
        **kwargs,
    )
    return fe.fit_transform(df)
