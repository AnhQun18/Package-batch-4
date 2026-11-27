"""Test cho các hàm cấp 2: missing nâng cao, outlier, ngày giờ, ràng buộc, reshape."""

import numpy as np
import pandas as pd
import pytest

import data_processing as dp


# ---------- missing nâng cao ----------

def test_fill_missing_by_group_with_global_fallback():
    df = pd.DataFrame({
        "cat": ["a", "a", "a", "b", "c"],
        "price": [1.0, 3.0, np.nan, 10.0, np.nan],
    })
    out = dp.fill_missing(df, "price", strategy="median", group_by="cat")
    assert out["price"].iloc[2] == 2.0       # median nhóm a
    assert out["price"].iloc[4] == 3.0       # nhóm c rỗng -> median toàn cục


def test_fill_missing_uses_reference_statistics():
    train = pd.DataFrame({"x": [10.0, 10.0]})
    test = pd.DataFrame({"x": [np.nan, 0.0]})
    assert dp.fill_missing(test, "x", strategy="mean", reference=train)["x"].iloc[0] == 10.0


def test_ffill_within_group_and_sorted_by_time_keeps_order():
    df = pd.DataFrame({
        "g": ["a", "a", "b"],
        "t": [2, 1, 1],
        "v": [np.nan, 5.0, np.nan],
    })
    out = dp.fill_missing(df, "v", strategy="ffill", group_by="g", sort_by="t")
    assert out["v"].iloc[0] == 5.0
    assert np.isnan(out["v"].iloc[2])
    assert out["t"].tolist() == [2, 1, 1]


def test_add_missing_flags():
    df = pd.DataFrame({"a": [1, np.nan], "b": [1, 2]})
    out = dp.add_missing_flags(df)
    assert out["a_was_missing"].tolist() == [0, 1]
    assert "b_was_missing" not in out


@pytest.mark.parametrize("method", ["knn", "iterative"])
def test_fill_missing_with_model(method):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, np.nan, 8.0]})
    out = dp.fill_missing_with_model(df, method=method, n_neighbors=2)
    assert out["y"].notna().all()


# ---------- outlier ----------

def test_outlier_bounds_and_actions():
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 100.0]})
    assert dp.handle_outliers(df, action="clip")["v"].max() < 100
    assert len(dp.handle_outliers(df, action="remove")) == 4
    assert dp.handle_outliers(df, action="flag")["v_is_outlier"].tolist() == [0, 0, 0, 0, 1]
    assert dp.handle_outliers(df, method="mad", action="missing")["v"].isna().sum() == 1


def test_outliers_skip_constant_column():
    df = pd.DataFrame({"v": [5.0] * 5})
    assert len(dp.handle_outliers(df, action="remove")) == 5


def test_outliers_reuse_precomputed_bounds():
    df = pd.DataFrame({"v": [0.0, 50.0]})
    out = dp.handle_outliers(df, bounds={"v": (10.0, 20.0)})
    assert out["v"].tolist() == [10.0, 20.0]


def test_enforce_value_rules():
    df = pd.DataFrame({"age": [-1, 30, 200, None], "gender": ["Nam", "Nữ", "x", "Nam"]})
    rules = {"age": {"min": 0, "max": 120}, "gender": {"allowed": ["Nam", "Nữ"]}}
    out = dp.enforce_value_rules(df, rules, action="missing")
    assert out["age"].isna().tolist() == [True, False, True, True]
    assert out["gender"].isna().tolist() == [False, False, True, False]
    clipped = dp.enforce_value_rules(df, {"age": {"min": 0, "max": 120}}, action="clip")
    assert clipped["age"].tolist()[:3] == [0, 30, 120]


# ---------- ngày giờ ----------

def test_parse_datetime_mixed_formats():
    df = pd.DataFrame({"d": ["03/04/2024", "2024-05-06", "not a date", None]})
    out = dp.parse_datetime(df, "d", formats=["%d/%m/%Y", "%Y-%m-%d"])
    assert out["d"].iloc[0] == pd.Timestamp("2024-04-03")
    assert out["d"].iloc[1] == pd.Timestamp("2024-05-06")
    assert out["d"].iloc[2:].isna().all()


def test_parse_datetime_unix_seconds():
    df = pd.DataFrame({"ts": [0, 86400]})
    out = dp.parse_datetime(df, "ts", unit="s")
    assert out["ts"].iloc[1] == pd.Timestamp("1970-01-02")


def test_convert_timezone():
    df = pd.DataFrame({"t": pd.to_datetime(["2024-01-01 07:00"])})
    out = dp.convert_timezone(df, "t", target_tz="UTC", source_tz="Asia/Ho_Chi_Minh", drop_timezone=True)
    assert out["t"].iloc[0] == pd.Timestamp("2024-01-01 00:00")
    with pytest.raises(ValueError):
        dp.convert_timezone(df, "t")


def test_fill_time_gaps_per_group():
    df = pd.DataFrame({
        "store": ["A", "A", "B"],
        "day": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-01"]),
        "sales": [1, 3, 5],
    })
    out = dp.fill_time_gaps(df, "day", "D", group_by="store", indicator_column="is_gap")
    a = out[out["store"] == "A"]
    assert len(a) == 3 and a["is_gap"].tolist() == [0, 1, 0]
    assert list(out.columns) == ["store", "day", "sales", "is_gap"]


# ---------- ràng buộc ----------

def test_enforce_rules_flag_remove_raise():
    df = pd.DataFrame({"start": [1, 5], "end": [2, 3], "qty": [2, 1], "price": [5, 5], "total": [10, 6]})
    rules = {
        "end_after_start": "end >= start",
        "total_ok": lambda d: d.total == d.qty * d.price,
    }
    flagged = dp.enforce_rules(df, rules)
    assert flagged["violates_end_after_start"].tolist() == [0, 1]
    assert flagged["violates_total_ok"].tolist() == [0, 1]
    assert len(dp.enforce_rules(df, rules, action="remove")) == 1
    with pytest.raises(ValueError):
        dp.enforce_rules(df, rules, action="raise")


def test_find_orphan_keys_with_different_key_names():
    orders = pd.DataFrame({"cust": [1, 2, 3]})
    customers = pd.DataFrame({"id": [1, 3]})
    assert dp.find_orphan_keys(orders, customers, "cust", "id")["cust"].tolist() == [2]


# ---------- reshape ----------

def test_wide_long_roundtrip():
    wide = pd.DataFrame({"id": [1, 2], "s2023": [1, 2], "s2024": [3, 4]})
    long = dp.wide_to_long(wide, "id", variable_name="year", value_name="sales")
    assert len(long) == 4
    back = dp.long_to_wide(long, "id", "year", "sales")
    pd.testing.assert_frame_equal(back, wide, check_dtype=False)


def test_aggregate_flat_names():
    df = pd.DataFrame({"g": ["a", "a", None], "v": [1, 3, 5]})
    out = dp.aggregate(df, "g", {"v": ["mean", "max"]})
    assert list(out.columns) == ["g", "v_mean", "v_max"]
    assert len(out) == 2


def test_explode_column_with_separator():
    df = pd.DataFrame({"id": [1, 2], "tags": ["a, b,", None]})
    out = dp.explode_column(df, "tags", separator=",")
    assert out["tags"].tolist()[:2] == ["a", "b"]
    assert len(out) == 3


def test_flatten_json_column():
    df = pd.DataFrame({"info": ['{"addr": {"city": "HN"}, "age": 3}', {"age": 5}, "bad json"]})
    out = dp.flatten_json_column(df, "info")
    assert out["info.addr.city"].iloc[0] == "HN"
    assert out["info.age"].tolist()[:2] == [3, 5]
