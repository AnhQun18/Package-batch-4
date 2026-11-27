"""Xử lý giá trị ngoại lai (outlier) và giá trị vô lý theo quy tắc nghiệp vụ."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from ._utils import ColumnsArg, log_row_change, logger, resolve_columns

# Ngưỡng mặc định thường dùng cho từng phương pháp.
DEFAULT_THRESHOLDS = {"iqr": 1.5, "zscore": 3.0, "mad": 3.5, "percentile": 0.01}
OUTLIER_ACTIONS = ("clip", "remove", "missing", "flag")
RULE_ACTIONS = ("missing", "clip", "remove", "flag")

# Hằng số đưa MAD về cùng thang đo với độ lệch chuẩn khi dữ liệu phân phối chuẩn.
MAD_SCALE = 0.6745

Bounds = tuple[float, float]


def compute_outlier_bounds(
    series: pd.Series,
    method: str = "iqr",
    threshold: Optional[float] = None,
) -> Bounds:
    """Tính ngưỡng (dưới, trên); giá trị nằm ngoài khoảng này là outlier.

    Tham số
    -------
    series : Cột số cần tính ngưỡng.
    method : Phương pháp:
        - "iqr": [Q1 - k*IQR, Q3 + k*IQR]. Bền với dữ liệu lệch, hay dùng nhất.
        - "zscore": [mean - k*std, mean + k*std]. Hợp với dữ liệu gần phân phối chuẩn.
        - "mad": dùng trung vị và MAD (Modified Z-score), bền hơn zscore khi có outlier lớn.
        - "percentile": cắt k phần đuôi mỗi bên, vd k=0.01 -> [P1, P99].
    threshold : Hệ số k của phương pháp. Để None thì dùng mặc định:
        iqr=1.5, zscore=3, mad=3.5, percentile=0.01.
    """
    if method not in DEFAULT_THRESHOLDS:
        raise ValueError(f"method phải là một trong {tuple(DEFAULT_THRESHOLDS)}")
    k = DEFAULT_THRESHOLDS[method] if threshold is None else threshold
    values = series.dropna()

    if method == "percentile":
        return float(values.quantile(k)), float(values.quantile(1 - k))

    if method == "iqr":
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        center_low, center_high, spread = q1, q3, q3 - q1
    elif method == "zscore":
        center_low = center_high = values.mean()
        spread = values.std()
    else:  # mad
        median = values.median()
        center_low = center_high = median
        spread = (values - median).abs().median() / MAD_SCALE

    if not spread or np.isnan(spread):
        logger.warning("Cột %s có độ phân tán = 0, bỏ qua phát hiện outlier.", series.name)
        return -np.inf, np.inf
    return float(center_low - k * spread), float(center_high + k * spread)


def handle_outliers(
    df: pd.DataFrame,
    columns: ColumnsArg = None,
    method: str = "iqr",
    threshold: Optional[float] = None,
    action: str = "clip",
    bounds: Optional[dict[str, Bounds]] = None,
    reference: Optional[pd.DataFrame] = None,
    flag_suffix: str = "_is_outlier",
) -> pd.DataFrame:
    """Phát hiện và xử lý outlier trên các cột số.

    Tham số
    -------
    df : DataFrame đầu vào.
    columns : Cột số cần xử lý. Để None thì dùng mọi cột số.
    method, threshold : Cách tính ngưỡng, xem `compute_outlier_bounds`.
    action : Cách xử lý outlier:
        - "clip": kéo giá trị về đúng ngưỡng gần nhất (winsorize), không mất dòng
        - "remove": xóa cả dòng nếu có outlier ở BẤT KỲ cột nào được chọn
        - "missing": đổi outlier thành NaN để điền lại sau
        - "flag": giữ nguyên giá trị, thêm cột cờ 0/1 (tên cột + flag_suffix)
    bounds : Ngưỡng tính sẵn {tên_cột: (dưới, trên)}, vd tính trên tập train rồi dùng lại
        cho tập test / batch mới. Cột có trong bounds sẽ không tính lại.
    reference : DataFrame dùng để tính ngưỡng (thường là tập train). Để None thì dùng df.
    flag_suffix : Hậu tố tên cột cờ khi action="flag".
    """
    if action not in OUTLIER_ACTIONS:
        raise ValueError(f"action phải là một trong {OUTLIER_ACTIONS}")

    selected = resolve_columns(df, columns, dtype_include=["number"])
    non_numeric = [col for col in selected if not is_numeric_dtype(df[col])]
    if non_numeric:
        raise TypeError(f"Chỉ xử lý outlier cho cột số; không phải số: {non_numeric}")

    ref = df if reference is None else reference
    known_bounds = bounds or {}
    df = df.copy()
    rows_to_remove = pd.Series(False, index=df.index)

    for col in selected:
        lower, upper = known_bounds.get(col) or compute_outlier_bounds(ref[col], method, threshold)
        is_outlier = (df[col] < lower) | (df[col] > upper)
        logger.info("handle_outliers[%s]: %d outlier, ngưỡng [%.4g, %.4g]",
                    col, int(is_outlier.sum()), lower, upper)

        if action == "clip":
            df[col] = df[col].clip(lower=lower, upper=upper)
        elif action == "missing":
            df[col] = df[col].mask(is_outlier, np.nan)
        elif action == "flag":
            df[f"{col}{flag_suffix}"] = is_outlier.astype("int8")
        else:
            rows_to_remove |= is_outlier

    if action == "remove":
        result = df[~rows_to_remove]
        log_row_change("handle_outliers (remove)", len(df), len(result))
        return result
    return df


def enforce_value_rules(
    df: pd.DataFrame,
    rules: dict[str, dict[str, Any]],
    action: str = "missing",
    flag_suffix: str = "_is_invalid",
) -> pd.DataFrame:
    """Xử lý giá trị vô lý theo quy tắc nghiệp vụ cho từng cột.

    Ví dụ rules:
        {
            "age": {"min": 0, "max": 120},
            "price": {"min": 0},
            "order_date": {"max": pd.Timestamp.today()},
            "gender": {"allowed": ["Nam", "Nữ"]},
        }

    Tham số
    -------
    df : DataFrame đầu vào.
    rules : dict {tên_cột: quy tắc}. Mỗi quy tắc có thể gồm:
        - "min": giá trị nhỏ nhất hợp lệ (tính cả biên)
        - "max": giá trị lớn nhất hợp lệ (tính cả biên)
        - "allowed": danh sách giá trị hợp lệ duy nhất
        Ô rỗng không bị coi là vi phạm.
    action : Cách xử lý ô vi phạm:
        - "missing": đổi thành NaN
        - "clip": kéo về min/max (vi phạm "allowed" vẫn đổi thành NaN)
        - "remove": xóa cả dòng
        - "flag": giữ nguyên, thêm cột cờ 0/1 (tên cột + flag_suffix)
    flag_suffix : Hậu tố tên cột cờ khi action="flag".
    """
    if action not in RULE_ACTIONS:
        raise ValueError(f"action phải là một trong {RULE_ACTIONS}")
    resolve_columns(df, list(rules))

    df = df.copy()
    rows_to_remove = pd.Series(False, index=df.index)

    for col, rule in rules.items():
        unknown = set(rule) - {"min", "max", "allowed"}
        if unknown:
            raise ValueError(f"Quy tắc cột {col!r} có khóa không hợp lệ: {sorted(unknown)}")

        series = df[col]
        below = series < rule["min"] if "min" in rule else pd.Series(False, index=df.index)
        above = series > rule["max"] if "max" in rule else pd.Series(False, index=df.index)
        not_allowed = (
            ~series.isin(rule["allowed"]) if "allowed" in rule else pd.Series(False, index=df.index)
        )
        invalid = series.notna() & (below | above | not_allowed)
        logger.info("enforce_value_rules[%s]: %d giá trị vi phạm", col, int(invalid.sum()))

        if action == "clip":
            df[col] = series.clip(lower=rule.get("min"), upper=rule.get("max"))
            df[col] = df[col].mask(series.notna() & not_allowed, np.nan)
        elif action == "missing":
            df[col] = series.mask(invalid, np.nan)
        elif action == "flag":
            df[f"{col}{flag_suffix}"] = invalid.astype("int8")
        else:
            rows_to_remove |= invalid

    if action == "remove":
        result = df[~rows_to_remove]
        log_row_change("enforce_value_rules (remove)", len(df), len(result))
        return result
    return df
