"""Test cho các hàm cấp 1: đọc/gộp, schema, trùng lặp, missing cơ bản, text."""

import numpy as np
import pandas as pd
import pytest

import data_processing as dp


# ---------- io ----------

def test_load_table_detects_semicolon_and_vietnamese_encoding(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("tên;giá\nĐà Lat;10\n", encoding="cp1258")
    df = dp.load_table(path)
    assert list(df.columns) == ["tên", "giá"]
    assert df.iloc[0, 0] == "Đà Lat"


def test_load_many_adds_source_column(tmp_path):
    for name in ("a.csv", "b.csv"):
        (tmp_path / name).write_text("x\n1\n", encoding="utf-8")
    df = dp.load_many(tmp_path.glob("*.csv"))
    assert sorted(df["source_file"]) == ["a.csv", "b.csv"]


def test_safe_merge_rejects_duplicate_keys_on_right():
    left = pd.DataFrame({"id": [1, 2]})
    right = pd.DataFrame({"id": [1, 1], "v": [10, 20]})
    with pytest.raises(ValueError, match="trùng khóa"):
        dp.safe_merge(left, right, on="id", relationship="many_to_one")


# ---------- schema ----------

def test_standardize_column_names():
    df = pd.DataFrame(columns=["Ngày Sinh ", "totalPrice($)", "ngay sinh"])
    assert list(dp.standardize_column_names(df).columns) == ["ngay_sinh", "total_price", "ngay_sinh_2"]


def test_to_numeric_handles_symbols_percent_and_accounting():
    df = pd.DataFrame({"a": ["$1,234.5", "15%", "(20)", "abc", None]})
    out = dp.to_numeric(df, "a", percent_to_fraction=True)
    assert out["a"].tolist()[:3] == [1234.5, 0.15, -20.0]
    assert out["a"].iloc[3:].isna().all()


def test_to_numeric_vietnamese_format():
    df = pd.DataFrame({"a": ["1.234,5 ₫"]})
    out = dp.to_numeric(df, "a", thousands_sep=".", decimal_sep=",")
    assert out["a"].iloc[0] == 1234.5


def test_auto_convert_numeric_only_converts_mostly_numeric_columns():
    df = pd.DataFrame({"num": ["1", "2", "3"], "txt": ["a", "b", "1"]})
    out = dp.auto_convert_numeric(df)
    assert out["num"].dtype == "float64"
    assert out["txt"].dtype == object


def test_to_boolean():
    df = pd.DataFrame({"b": ["Yes", "no", "Có", 1, "maybe"]})
    assert dp.to_boolean(df, "b")["b"].tolist() == [True, False, True, True, pd.NA]


def test_drop_useless_columns_respects_exclude():
    df = pd.DataFrame({"empty": [np.nan] * 3, "const": [1, 1, 1], "keep": [1, 1, 1], "x": [1, 2, 3]})
    out = dp.drop_useless_columns(df, exclude="keep")
    assert list(out.columns) == ["keep", "x"]


# ---------- duplicates ----------

def test_drop_exact_duplicates_ignores_columns():
    df = pd.DataFrame({"a": [1, 1], "loaded_at": ["t1", "t2"]})
    assert len(dp.drop_exact_duplicates(df, ignore_columns="loaded_at")) == 1


def test_drop_key_duplicates_strategies():
    df = pd.DataFrame({
        "id": [1, 1, 2],
        "updated": [2, 1, 1],
        "v": [np.nan, 5.0, 7.0],
    })
    latest = dp.drop_key_duplicates(df, "id", strategy="last", order_by="updated")
    assert latest["updated"].tolist() == [2, 1]
    complete = dp.drop_key_duplicates(df, "id", strategy="most_complete")
    assert complete["v"].tolist() == [5.0, 7.0]


# ---------- missing ----------

def test_standardize_missing_tokens_and_sentinels():
    df = pd.DataFrame({"a": [" N/A ", "x", "null"], "b": [-999, 1, 2]})
    out = dp.standardize_missing(df, sentinel_values={"b": [-999]})
    assert out["a"].isna().tolist() == [True, False, True]
    assert out["b"].isna().tolist() == [True, False, False]


def test_drop_high_missing_columns_and_rows():
    df = pd.DataFrame({"a": [1, np.nan, np.nan], "b": [1, 2, np.nan]})
    assert list(dp.drop_high_missing(df, threshold=0.5).columns) == ["b"]
    assert len(dp.drop_high_missing(df, axis="rows", threshold=0.5)) == 2


def test_fill_missing_constant_and_mode_with_category():
    df = pd.DataFrame({"c": pd.Categorical(["a", None, "a"])})
    assert dp.fill_missing(df, "c", strategy="constant", fill_value="Unknown")["c"].tolist() == [
        "a", "Unknown", "a"
    ]
    assert dp.fill_missing(df, "c", strategy="mode")["c"].tolist() == ["a", "a", "a"]


def test_fill_missing_rejects_text_for_median():
    with pytest.raises(TypeError):
        dp.fill_missing(pd.DataFrame({"t": ["a", None]}), "t", strategy="median")


# ---------- text ----------

def test_clean_text():
    df = pd.DataFrame({"t": ["  Hà   Nội\n", "   ", 5]})
    out = dp.clean_text(df, case="lower")
    assert out["t"].iloc[0] == "hà nội"
    assert pd.isna(out["t"].iloc[1])
    assert out["t"].iloc[2] == 5


def test_normalize_categories_ignores_case_and_accents():
    df = pd.DataFrame({"city": ["HN", "ha noi", "Hà  Nội", "Huế", None]})
    out = dp.normalize_categories(
        df, "city", {"Hà Nội": ["HN"]}, unmatched="replace", replacement="Khác"
    )
    assert out["city"].tolist()[:4] == ["Hà Nội", "Hà Nội", "Hà Nội", "Khác"]
    assert pd.isna(out["city"].iloc[4])


def test_functions_do_not_modify_input():
    df = pd.DataFrame({"a": ["N/A", " x "]})
    snapshot = df.copy()
    dp.standardize_missing(df)
    dp.clean_text(df)
    pd.testing.assert_frame_equal(df, snapshot)
