"""Bộ kiểm thử tự động (pytest) cho package `qc` sử dụng dữ liệu giả lập."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # Chạy backend headless khi test
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from data.generate_mock_data import generate_qc_mock_data
from qc import (
    plot_missing,
    plot_duplicate,
    plot_outlier_rate,
    plot_entity_coverage,
)
from qc._common import DatasetError


@pytest.fixture(scope="module")
def mock_df() -> pd.DataFrame:
    """Fixture cung cấp dữ liệu giả lập cho toàn bộ test suite."""
    return generate_qc_mock_data(n_rows=1000, seed=123)


# ======================================================================
# 1. TEST PLOT_MISSING
# ======================================================================
def test_plot_missing_basic(mock_df: pd.DataFrame):
    fig, table = plot_missing(mock_df)
    assert isinstance(fig, plt.Figure)
    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == ["column", "n", "missing", "missing_rate"]
    assert len(table) == len(mock_df.columns)
    # Kiểm tra tính đúng đắn của phép tính
    for _, row in table.iterrows():
        col = row["column"]
        expected_missing = mock_df[col].isna().sum()
        assert row["missing"] == expected_missing
        assert pytest.approx(row["missing_rate"]) == expected_missing / len(mock_df)
    # Thứ tự phải giảm dần
    assert table["missing_rate"].is_monotonic_decreasing
    plt.close(fig)


def test_plot_missing_filter_and_threshold(mock_df: pd.DataFrame):
    fig, table = plot_missing(mock_df, min_missing=0.10, threshold=0.30)
    assert "exceed" in table.columns
    assert (table["missing_rate"] >= 0.10).all()
    assert (table["exceed"] == (table["missing_rate"] > 0.30)).all()
    plt.close(fig)


def test_plot_missing_cols_and_exclude(mock_df: pd.DataFrame):
    cols = ["customer_id", "total_amount", "customer_feedback"]
    fig, table = plot_missing(mock_df, cols=cols, exclude=["customer_id"])
    assert set(table["column"]) == {"total_amount", "customer_feedback"}
    plt.close(fig)


def test_plot_missing_empty_result(mock_df: pd.DataFrame):
    # min_missing quá cao khiến không cột nào thỏa mãn
    fig, table = plot_missing(mock_df, min_missing=1.0)
    # table chỉ có những cột 100% missing (internal_device_token)
    assert (table["missing_rate"] == 1.0).all()
    plt.close(fig)


# ======================================================================
# 2. TEST PLOT_DUPLICATE
# ======================================================================
def test_plot_duplicate_whole_row(mock_df: pd.DataFrame):
    fig, table = plot_duplicate(mock_df, subset=None, keep="first")
    assert list(table.columns) == ["subset", "n_rows", "n_duplicate", "duplicate_rate"]
    assert table.iloc[0]["subset"] == "toàn bộ dòng"
    assert table.iloc[0]["n_rows"] == len(mock_df)
    assert table.iloc[0]["n_duplicate"] > 0
    plt.close(fig)


def test_plot_duplicate_multiple_definitions(mock_df: pd.DataFrame):
    defs = {
        "toàn bộ": None,
        "mã đơn": ["order_id"],
        "khách + ngày": ["customer_id", "order_date"],
    }
    defs_fixed = {
        "toàn bộ": list(mock_df.columns),
        "mã đơn": ["order_id"],
        "khách + ngày": ["customer_id", "order_date"],
    }
    fig, table = plot_duplicate(mock_df, subset=defs_fixed, threshold=0.03)
    assert len(table) == 3
    assert "exceed" in table.columns
    assert (table["exceed"] == (table["duplicate_rate"] > 0.03)).all()
    plt.close(fig)


def test_plot_duplicate_keep_modes(mock_df: pd.DataFrame):
    _, tab_first = plot_duplicate(mock_df, subset=["order_id"], keep="first")
    _, tab_all = plot_duplicate(mock_df, subset=["order_id"], keep=False)
    # keep=False phải có số lượng duplicate lớn hơn hoặc bằng keep="first"
    assert tab_all.iloc[0]["n_duplicate"] >= tab_first.iloc[0]["n_duplicate"]
    plt.close("all")


# ======================================================================
# 3. TEST PLOT_OUTLIER_RATE
# ======================================================================
def test_plot_outlier_rate_methods(mock_df: pd.DataFrame):
    for method, default_k in [("iqr", 1.5), ("zscore", 3.0), ("mad", 3.5)]:
        fig, table = plot_outlier_rate(mock_df, method=method, k=default_k)
        assert list(table.columns) == ["column", "n", "n_outlier", "outlier_rate", "lower", "upper"]
        # Phải bao gồm các cột số hữu hạn
        assert "total_amount" in table["column"].values
        assert "customer_age" in table["column"].values
        # Cột hằng số std/MAD = 0 phải cho lower=NaN, upper=NaN, n_outlier=0
        const_row = table[table["column"] == "constant_fee"].iloc[0]
        if method in ("zscore", "mad"):
            assert np.isnan(const_row["lower"])
            assert np.isnan(const_row["upper"])
            assert const_row["n_outlier"] == 0
        plt.close(fig)


def test_plot_outlier_rate_threshold_and_filter(mock_df: pd.DataFrame):
    fig, table = plot_outlier_rate(mock_df, method="iqr", min_rate=0.01, threshold=0.02)
    assert "exceed" in table.columns
    assert (table["outlier_rate"] >= 0.01).all()
    assert (table["exceed"] == (table["outlier_rate"] > 0.02)).all()
    plt.close(fig)


# ======================================================================
# 4. TEST PLOT_ENTITY_COVERAGE
# ======================================================================
def test_plot_entity_coverage_monthly_and_gaps(mock_df: pd.DataFrame):
    fig, table = plot_entity_coverage(
        mock_df,
        entity="customer_id",
        time="order_date",
        freq="M",
        threshold=0.20,
    )
    assert list(table.columns) == ["period", "n_entity", "coverage", "n_rows", "exceed"]
    assert len(table) > 0
    # Tháng 2024-08 là khoảng gián đoạn cố ý
    gap_rows = table[table["period"].astype(str) == "2024-08"]
    if not gap_rows.empty:
        assert gap_rows.iloc[0]["n_entity"] == 0
        assert gap_rows.iloc[0]["coverage"] == 0.0
        assert gap_rows.iloc[0]["exceed"] == True  # 0.0 < 0.20
    plt.close(fig)


def test_plot_entity_coverage_frequencies(mock_df: pd.DataFrame):
    for f in ["W", "Q"]:
        fig, table = plot_entity_coverage(mock_df, entity="customer_id", time="order_date", freq=f)
        assert len(table) > 0
        plt.close(fig)


# ======================================================================
# 5. TEST INPUT VALIDATION & DATASETERROR
# ======================================================================
def test_dataset_error_handling(mock_df: pd.DataFrame):
    # df không phải DataFrame
    with pytest.raises(DatasetError):
        plot_missing("not a dataframe")

    # DataFrame trùng tên cột
    dup_cols_df = pd.DataFrame([[1, 2]], columns=["col", "col"])
    with pytest.raises(DatasetError):
        plot_missing(dup_cols_df)

    # rate ngoài [0, 1]
    with pytest.raises(DatasetError):
        plot_missing(mock_df, threshold=1.2)
    with pytest.raises(DatasetError):
        plot_duplicate(mock_df, threshold=-0.1)

    # Cột không tồn tại
    with pytest.raises(DatasetError):
        plot_missing(mock_df, cols=["non_existent_column"])

    # plot_outlier: method không hợp lệ hoặc k <= 0
    with pytest.raises(DatasetError):
        plot_outlier_rate(mock_df, method="unknown_method")
    with pytest.raises(DatasetError):
        plot_outlier_rate(mock_df, k=0)

    # plot_entity_coverage: time không phải datetime hoặc entity == time
    with pytest.raises(DatasetError):
        plot_entity_coverage(mock_df, entity="customer_id", time="customer_id")
    with pytest.raises(DatasetError):
        plot_entity_coverage(mock_df, entity="customer_id", time="total_amount")
