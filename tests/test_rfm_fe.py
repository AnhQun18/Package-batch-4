"""Comprehensive unit tests for rfm_fe."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from rfm_fe import RFMFeatureEngineer, engineer_rfm_features


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u3", "u4"],
            "tx_id": ["t1", "t2", "t3", "t4", "t4", "t5", "t6"],
            "tx_date": [
                "2026-01-01",
                "2026-02-01",
                "2026-03-01",
                "2026-02-15",
                "2026-02-15",  # same order t4 (2 line items)
                "2026-03-10",
                "2026-03-20",
            ],
            "price": [100.0, 200.0, 300.0, 50.0, 75.0, -20.0, 400.0],
            "cat": ["A", "B", "A", "C", "C", "A", "B"],
        }
    )


def test_one_liner_produces_numeric_model_ready_df(sample_df: pd.DataFrame) -> None:
    feat = engineer_rfm_features(
        sample_df,
        customer_col="user_id",
        date_col="tx_date",
        amount_col="price",
        transaction_col="tx_id",
        category_col="cat",
        reference_date="2026-03-21",
    )
    assert "user_id" in feat.columns
    assert len(feat) == 4

    # u2 has 2 rows but 1 unique tx_id ('t4') -> frequency == 1.0
    u2_freq = feat.loc[feat["user_id"] == "u2", "frequency"].iloc[0]
    assert u2_freq == 1.0

    # All feature columns must be numeric & free of NaN/inf
    feature_cols = [c for c in feat.columns if c != "user_id"]
    for col in feature_cols:
        assert pd.api.types.is_numeric_dtype(feat[col])
        assert not feat[col].isna().any()
        assert np.isfinite(feat[col]).all()


def test_leak_free_fit_transform_consistency(sample_df: pd.DataFrame) -> None:
    train_df = sample_df.iloc[:4].copy()
    test_df = sample_df.iloc[4:].copy()

    fe = RFMFeatureEngineer(
        customer_col="user_id",
        date_col="tx_date",
        amount_col="price",
        reference_date="2026-03-25",
        log_transform=True,
        scaler="standard",
        encode_segment=True,
    )
    X_train = fe.fit_transform(train_df)
    X_test = fe.transform(test_df)

    assert list(X_train.columns) == list(X_test.columns)
    assert fe.get_feature_names_out().tolist() == [
        c for c in X_train.columns if c != "user_id"
    ]


def test_handle_negative_amount_separate(sample_df: pd.DataFrame) -> None:
    # u3 only has a negative transaction (-20.0), so u3 is excluded from main non-negative df
    # Add a positive transaction for u3 to test refund_rate
    df = pd.concat(
        [
            sample_df,
            pd.DataFrame(
                {
                    "user_id": ["u3"],
                    "tx_id": ["t7"],
                    "tx_date": ["2026-03-11"],
                    "price": [80.0],
                    "cat": ["A"],
                }
            ),
        ],
        ignore_index=True,
    )
    feat = engineer_rfm_features(
        df,
        customer_col="user_id",
        date_col="tx_date",
        amount_col="price",
        handle_negative_amount="separate",
        reference_date="2026-03-21",
    )
    u3 = feat.loc[feat["user_id"] == "u3"].iloc[0]
    assert u3["refund_count"] == 1.0
    assert u3["refund_amount_sum"] == 20.0
    assert u3["refund_rate"] == pytest.approx(0.5)


def test_works_with_tree_and_linear_models(sample_df: pd.DataFrame) -> None:
    feat_tree = engineer_rfm_features(
        sample_df,
        customer_col="user_id",
        date_col="tx_date",
        amount_col="price",
        return_customer_col=False,
    )
    y = np.array([1, 0, 1, 0])
    rf = RandomForestClassifier(n_estimators=10, random_state=42).fit(feat_tree, y)
    assert rf.predict(feat_tree).shape == (4,)

    feat_linear = engineer_rfm_features(
        sample_df,
        customer_col="user_id",
        date_col="tx_date",
        amount_col="price",
        log_transform=True,
        scaler="standard",
        return_customer_col=False,
    )
    lr = LogisticRegression(random_state=42).fit(feat_linear, y)
    assert lr.predict(feat_linear).shape == (4,)


def test_out_of_time_transform_and_override(sample_df: pd.DataFrame) -> None:
    train_df = sample_df[sample_df["tx_date"] <= "2026-02-15"].copy()
    future_df = sample_df[sample_df["tx_date"] > "2026-02-15"].copy()

    # Auto snapshot on out-of-time window when reference_date=None
    fe_auto = RFMFeatureEngineer(
        customer_col="user_id",
        date_col="tx_date",
        amount_col="price",
        scaler="standard",
    )
    X_tr = fe_auto.fit_transform(train_df)
    X_te = fe_auto.transform(future_df)
    assert list(X_tr.columns) == list(X_te.columns)
    assert len(X_te) > 0

    # Explicit reference_date override on transform()
    fe_fixed = RFMFeatureEngineer(
        customer_col="user_id",
        date_col="tx_date",
        amount_col="price",
        reference_date="2026-02-16",
    )
    fe_fixed.fit(train_df)
    X_te_override = fe_fixed.transform(future_df, reference_date="2026-03-21")
    assert len(X_te_override) > 0


def test_missing_columns_raises_value_error(sample_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Missing required column"):
        engineer_rfm_features(sample_df, customer_col="non_existent")
