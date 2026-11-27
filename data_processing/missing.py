"""Nhận diện và xử lý giá trị thiếu (missing values)."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from ._utils import ColumnsArg, as_list, log_row_change, logger, resolve_columns

# Các chuỗi thường được dùng thay cho "không có dữ liệu" (so khớp không phân biệt hoa thường).
DEFAULT_MISSING_TOKENS = (
    "", "na", "n/a", "nan", "null", "none", "nil", "-", "--", "?", "#n/a", "missing",
)

STAT_STRATEGIES = ("mean", "median", "mode")
SEQUENTIAL_STRATEGIES = ("ffill", "bfill", "interpolate")
FILL_STRATEGIES = STAT_STRATEGIES + SEQUENTIAL_STRATEGIES + ("constant",)


def standardize_missing(
    df: pd.DataFrame,
    columns: ColumnsArg = None,
    tokens: Sequence[str] = DEFAULT_MISSING_TOKENS,
    sentinel_values: Optional[Union[Sequence[Any], dict[str, Sequence[Any]]]] = None,
) -> pd.DataFrame:
    """Đổi các giá trị "thiếu giả" về NaN để pandas nhận ra là thiếu.

    Tham số
    -------
    df : DataFrame đầu vào.
    columns : Cột cần xử lý. Để None thì xử lý mọi cột.
    tokens : Các chuỗi được coi là thiếu, vd "N/A", "null", "-".
        So khớp sau khi bỏ khoảng trắng hai đầu, không phân biệt hoa thường.
    sentinel_values : Các giá trị (thường là số) được quy ước là thiếu, vd -999 hoặc 9999.
        - list: áp dụng cho mọi cột trong `columns`
        - dict {tên_cột: [giá trị, ...]}: áp dụng riêng cho từng cột
    """
    df = df.copy()
    token_set = {str(token).strip().lower() for token in tokens}
    selected = resolve_columns(df, columns)

    for col in selected:
        is_token = df[col].map(lambda v: isinstance(v, str) and v.strip().lower() in token_set)
        df[col] = df[col].mask(is_token.astype(bool), np.nan)

    for col, values in _sentinels_by_column(sentinel_values, selected).items():
        resolve_columns(df, col)
        df[col] = df[col].mask(df[col].isin(values), np.nan)

    return df


def drop_high_missing(
    df: pd.DataFrame,
    axis: str = "columns",
    threshold: float = 0.5,
    exclude: ColumnsArg = None,
) -> pd.DataFrame:
    """Xóa cột hoặc dòng có tỉ lệ thiếu VƯỢT QUÁ ngưỡng.

    Tham số
    -------
    df : DataFrame đầu vào.
    axis : "columns" để xóa cột, "rows" để xóa dòng.
    threshold : Tỉ lệ thiếu tối đa cho phép, từ 0 đến 1 (0.5 = 50%).
    exclude : Với axis="columns": cột luôn giữ lại.
        Với axis="rows": cột không tính vào tỉ lệ thiếu của dòng.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold phải nằm trong khoảng [0, 1]")
    excluded = set(as_list(exclude))
    resolve_columns(df, list(excluded))

    if axis == "columns":
        ratios = df.isna().mean()
        to_drop = [col for col, ratio in ratios.items() if ratio > threshold and col not in excluded]
        if to_drop:
            logger.info("drop_high_missing: xóa cột %s", to_drop)
        return df.drop(columns=to_drop)

    if axis == "rows":
        considered = [col for col in df.columns if col not in excluded]
        result = df[df[considered].isna().mean(axis=1) <= threshold]
        log_row_change("drop_high_missing (rows)", len(df), len(result))
        return result

    raise ValueError('axis phải là "columns" hoặc "rows"')


def add_missing_flags(
    df: pd.DataFrame,
    columns: ColumnsArg = None,
    suffix: str = "_was_missing",
    only_if_has_missing: bool = True,
) -> pd.DataFrame:
    """Thêm cột đánh dấu 0/1 cho biết ô ban đầu có bị thiếu không.

    Phải gọi TRƯỚC khi điền missing, vì sau khi điền thì không còn biết ô nào từng thiếu.

    Tham số
    -------
    df : DataFrame đầu vào.
    columns : Cột cần tạo cờ. Để None thì xét mọi cột.
    suffix : Hậu tố tên cột cờ, vd "age" -> "age_was_missing".
    only_if_has_missing : True thì chỉ tạo cờ cho cột thực sự có giá trị thiếu.
    """
    df = df.copy()
    for col in resolve_columns(df, columns):
        is_missing = df[col].isna()
        if only_if_has_missing and not is_missing.any():
            continue
        df[f"{col}{suffix}"] = is_missing.astype("int8")
    return df


def fill_missing(
    df: pd.DataFrame,
    columns: ColumnsArg = None,
    strategy: str = "median",
    fill_value: Any = None,
    group_by: ColumnsArg = None,
    sort_by: ColumnsArg = None,
    reference: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Điền giá trị thiếu bằng thống kê, hằng số hoặc theo thứ tự dòng.

    Tham số
    -------
    df : DataFrame đầu vào.
    columns : Cột cần điền. Để None thì tự chọn: cột số với "mean"/"median"/"interpolate",
        mọi cột với các chiến lược còn lại.
    strategy : Cách điền:
        - "mean" / "median": trung bình / trung vị (chỉ cho cột số)
        - "mode": giá trị xuất hiện nhiều nhất (dùng được cho cả cột chữ)
        - "constant": điền bằng `fill_value`
        - "ffill" / "bfill": lấy giá trị dòng trước / dòng sau (dữ liệu theo thời gian)
        - "interpolate": nội suy tuyến tính giữa hai dòng (chỉ cho cột số)
    fill_value : Giá trị điền khi strategy="constant", vd 0 hoặc "Unknown".
    group_by : Tính thống kê / ffill riêng trong từng nhóm, vd median giá theo từng
        "category". Nhóm không có giá trị nào sẽ được điền bằng thống kê toàn cục.
    sort_by : (chỉ ffill/bfill/interpolate) Cột thời gian dùng để sắp thứ tự trước khi điền.
        Kết quả vẫn giữ thứ tự dòng ban đầu.
    reference : (chỉ mean/median/mode) DataFrame dùng để TÍNH thống kê, thường là tập
        train, để tránh rò rỉ dữ liệu khi điền cho tập test. Để None thì dùng chính df.
    """
    if strategy not in FILL_STRATEGIES:
        raise ValueError(f"strategy phải là một trong {FILL_STRATEGIES}")
    if strategy == "constant" and fill_value is None:
        raise ValueError('strategy="constant" cần truyền fill_value')

    numeric_only = strategy in ("mean", "median", "interpolate")
    selected = resolve_columns(df, columns, dtype_include=["number"] if numeric_only else None)
    groups = resolve_columns(df, group_by) if group_by is not None else []
    selected = [col for col in selected if col not in groups]
    if numeric_only:
        _ensure_numeric(df, selected, strategy)

    df = df.copy()
    before = df[selected].isna().sum()

    if strategy == "constant":
        for col in selected:
            df[col] = _fillna_safe(df[col], fill_value)
    elif strategy in STAT_STRATEGIES:
        ref = df if reference is None else reference
        for col in selected:
            df[col] = _fill_with_statistic(df, ref, col, strategy, groups)
    else:
        df = _fill_sequential(df, selected, strategy, groups, as_list(sort_by))

    filled = (before - df[selected].isna().sum())
    logger.info("fill_missing (%s): đã điền %s", strategy, filled[filled > 0].to_dict())
    return df


def fill_missing_with_model(
    df: pd.DataFrame,
    columns: ColumnsArg = None,
    method: str = "knn",
    reference: Optional[pd.DataFrame] = None,
    n_neighbors: int = 5,
    max_iter: int = 10,
    random_state: int = 0,
) -> pd.DataFrame:
    """Điền missing cho cột số bằng mô hình dựa trên các cột số khác.

    Phù hợp khi các cột có tương quan với nhau (vd chiều cao - cân nặng).

    Tham số
    -------
    df : DataFrame đầu vào.
    columns : Cột số tham gia (vừa là cột được điền, vừa là thông tin để điền).
        Để None thì dùng mọi cột số.
    method : "knn" (lấy trung bình K dòng giống nhất) hoặc
        "iterative" (hồi quy lặp từng cột theo các cột khác, kiểu MICE).
    reference : DataFrame dùng để HỌC mô hình (thường là tập train). Để None thì dùng df.
    n_neighbors : (knn) Số dòng láng giềng dùng để tính giá trị điền.
    max_iter : (iterative) Số vòng lặp tối đa.
    random_state : (iterative) Seed để kết quả lặp lại được.
    """
    selected = resolve_columns(df, columns, dtype_include=["number"])
    _ensure_numeric(df, selected, method)
    imputer = _build_imputer(method, n_neighbors, max_iter, random_state)

    ref = df if reference is None else reference
    imputer.fit(ref[selected])

    df = df.copy()
    n_missing = int(df[selected].isna().sum().sum())
    df[selected] = imputer.transform(df[selected])
    logger.info("fill_missing_with_model (%s): đã điền %d ô", method, n_missing)
    return df


def _sentinels_by_column(
    sentinel_values: Optional[Union[Sequence[Any], dict[str, Sequence[Any]]]],
    selected: list[str],
) -> dict[str, list[Any]]:
    if sentinel_values is None:
        return {}
    if isinstance(sentinel_values, dict):
        return {col: list(values) for col, values in sentinel_values.items()}
    return {col: list(sentinel_values) for col in selected}


def _ensure_numeric(df: pd.DataFrame, columns: list[str], strategy: str) -> None:
    non_numeric = [col for col in columns if not is_numeric_dtype(df[col])]
    if non_numeric:
        raise TypeError(f'"{strategy}" chỉ dùng cho cột số; các cột không phải số: {non_numeric}')


def _statistic(series: pd.Series, strategy: str) -> Any:
    if strategy == "mean":
        return series.mean()
    if strategy == "median":
        return series.median()
    modes = series.mode(dropna=True)
    return modes.iloc[0] if len(modes) else np.nan


def _fill_with_statistic(
    df: pd.DataFrame, ref: pd.DataFrame, col: str, strategy: str, groups: list[str]
) -> pd.Series:
    result = df[col]
    if groups:
        grouped = ref.groupby(groups, observed=True)[col]
        group_stats = grouped.agg(strategy) if strategy != "mode" else grouped.agg(
            lambda s: _statistic(s, "mode")
        )
        row_keys = pd.MultiIndex.from_frame(df[groups]) if len(groups) > 1 else pd.Index(df[groups[0]])
        per_row = pd.Series(group_stats.reindex(row_keys).to_numpy(), index=df.index)
        result = _fillna_safe(result, per_row)
    return _fillna_safe(result, _statistic(ref[col], strategy))


def _fill_sequential(
    df: pd.DataFrame, columns: list[str], strategy: str, groups: list[str], sort_by: list[str]
) -> pd.DataFrame:
    ordered = df.sort_values(sort_by, kind="mergesort") if sort_by else df
    target = ordered.groupby(groups, observed=True)[columns] if groups else ordered[columns]

    if strategy == "interpolate":
        if groups:
            filled = target.transform(lambda s: s.interpolate(limit_direction="both"))
        else:
            filled = target.interpolate(limit_direction="both")
    else:
        filled = getattr(target, strategy)()

    df[columns] = filled.reindex(df.index)
    return df


def _fillna_safe(series: pd.Series, value: Any) -> pd.Series:
    """fillna nhưng tự thêm category mới nếu cột là kiểu category."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        new_values = pd.Series(value).dropna().unique() if isinstance(value, pd.Series) else [value]
        missing_categories = [v for v in new_values if pd.notna(v) and v not in series.cat.categories]
        if missing_categories:
            series = series.cat.add_categories(missing_categories)
    return series.fillna(value)


def _build_imputer(method: str, n_neighbors: int, max_iter: int, random_state: int):
    if method == "knn":
        from sklearn.impute import KNNImputer

        return KNNImputer(n_neighbors=n_neighbors, keep_empty_features=True)
    if method == "iterative":
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        return IterativeImputer(
            max_iter=max_iter, random_state=random_state, keep_empty_features=True
        )
    raise ValueError('method phải là "knn" hoặc "iterative"')
