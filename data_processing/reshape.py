"""Biến đổi hình dạng bảng: wide <-> long, gom nhóm, tách list, làm phẳng JSON."""

from __future__ import annotations

import json
from typing import Any, Callable, Optional, Union

import pandas as pd

from ._utils import ColumnsArg, as_list, logger, resolve_columns

AggFunc = Union[str, Callable, list[Union[str, Callable]]]


def wide_to_long(
    df: pd.DataFrame,
    id_columns: ColumnsArg,
    value_columns: ColumnsArg = None,
    variable_name: str = "variable",
    value_name: str = "value",
    drop_missing: bool = False,
) -> pd.DataFrame:
    """Chuyển bảng ngang thành bảng dọc (melt).

    Ví dụ cột [id, sales_2023, sales_2024] -> [id, variable, value].

    Tham số
    -------
    df : DataFrame đầu vào.
    id_columns : Cột giữ nguyên, định danh mỗi dòng (vd "id").
    value_columns : Cột cần "xoay" thành dòng. Để None thì lấy mọi cột còn lại.
    variable_name : Tên cột mới chứa tên cột gốc.
    value_name : Tên cột mới chứa giá trị.
    drop_missing : Bỏ các dòng có giá trị rỗng sau khi xoay.
    """
    ids = resolve_columns(df, id_columns)
    values = resolve_columns(df, value_columns) if value_columns is not None else None
    result = df.melt(id_vars=ids, value_vars=values, var_name=variable_name, value_name=value_name)
    return result.dropna(subset=[value_name]) if drop_missing else result


def long_to_wide(
    df: pd.DataFrame,
    index: ColumnsArg,
    columns: str,
    values: ColumnsArg,
    aggfunc: AggFunc = "first",
    fill_value: Any = None,
    name_separator: str = "_",
) -> pd.DataFrame:
    """Chuyển bảng dọc thành bảng ngang (pivot). Ngược lại với `wide_to_long`.

    Tham số
    -------
    df : DataFrame đầu vào.
    index : Cột định danh mỗi dòng của bảng kết quả.
    columns : Cột mà mỗi giá trị của nó sẽ trở thành một cột mới.
    values : Cột chứa giá trị điền vào bảng.
    aggfunc : Cách gộp khi một ô có nhiều giá trị: "first", "sum", "mean", "count", ...
    fill_value : Giá trị điền cho ô không có dữ liệu. Để None thì giữ NaN.
    name_separator : Ký tự nối tên khi tên cột kết quả có nhiều cấp, vd "sales_2024".
    """
    result = df.pivot_table(
        index=resolve_columns(df, index),
        columns=resolve_columns(df, columns),
        values=resolve_columns(df, values),
        aggfunc=aggfunc,
        fill_value=fill_value,
        observed=True,
    )
    if isinstance(result.columns, pd.MultiIndex) and len(as_list(values)) == 1:
        result.columns = result.columns.droplevel(0)
    result.columns = _flatten_column_names(result.columns, name_separator)
    return result.reset_index()


def aggregate(
    df: pd.DataFrame,
    group_by: ColumnsArg,
    aggregations: dict[str, AggFunc],
    name_separator: str = "_",
    keep_missing_groups: bool = True,
) -> pd.DataFrame:
    """Gom nhóm và tính tổng hợp, trả về bảng phẳng với tên cột dễ đọc.

    Ví dụ aggregations={"price": ["mean", "max"], "order_id": "count"}
    -> cột price_mean, price_max, order_id_count.

    Tham số
    -------
    df : DataFrame đầu vào.
    group_by : Cột dùng để gom nhóm.
    aggregations : dict {tên_cột: hàm hoặc danh sách hàm tổng hợp}.
    name_separator : Ký tự nối tên cột và tên hàm.
    keep_missing_groups : True để giữ nhóm có khóa rỗng (NaN) thay vì bỏ đi.
    """
    groups = resolve_columns(df, group_by)
    resolve_columns(df, list(aggregations))
    normalized = {col: as_list(funcs) if isinstance(funcs, (str, list)) else [funcs]
                  for col, funcs in aggregations.items()}

    result = df.groupby(groups, dropna=not keep_missing_groups, observed=True).agg(normalized)
    result.columns = _flatten_column_names(result.columns, name_separator)
    return result.reset_index()


def explode_column(
    df: pd.DataFrame,
    column: str,
    separator: Optional[str] = None,
    strip: bool = True,
    drop_empty: bool = True,
) -> pd.DataFrame:
    """Tách một ô chứa nhiều giá trị thành nhiều dòng.

    Ví dụ tags = "a, b, c" -> 3 dòng "a", "b", "c" (các cột khác được lặp lại).

    Tham số
    -------
    df : DataFrame đầu vào.
    column : Cột cần tách.
    separator : Ký tự phân tách nếu ô là chuỗi, vd "," hoặc "|".
        Để None nếu ô đã là list/tuple.
    strip : Bỏ khoảng trắng hai đầu mỗi phần tử sau khi tách.
    drop_empty : Bỏ phần tử chuỗi rỗng sau khi tách (vd dấu phẩy thừa ở cuối "a,b,").
        Dòng có ô rỗng ban đầu vẫn được giữ lại.
    """
    resolve_columns(df, column)
    df = df.copy()
    if separator is not None:
        df[column] = df[column].map(
            lambda v: v.split(separator) if isinstance(v, str) else v
        )

    result = df.explode(column)
    if strip:
        result[column] = result[column].map(lambda v: v.strip() if isinstance(v, str) else v)
    if drop_empty:
        result = result[result[column] != ""]

    logger.info("explode_column[%s]: %d -> %d dòng", column, len(df), len(result))
    return result.reset_index(drop=True)


def flatten_json_column(
    df: pd.DataFrame,
    column: str,
    separator: str = ".",
    max_level: Optional[int] = None,
    prefix: Optional[str] = None,
    drop_original: bool = True,
) -> pd.DataFrame:
    """Tách một cột chứa JSON / dict lồng nhau thành nhiều cột phẳng.

    Ví dụ {"address": {"city": "HN"}} -> cột "<prefix>address.city".

    Tham số
    -------
    df : DataFrame đầu vào.
    column : Cột chứa dict hoặc chuỗi JSON. Ô rỗng / không hợp lệ cho ra NaN.
    separator : Ký tự nối tên các cấp lồng nhau.
    max_level : Độ sâu tối đa cần làm phẳng. Để None thì làm phẳng hết.
    prefix : Tiền tố cho tên cột mới. Để None thì dùng "<tên cột><separator>".
    drop_original : Xóa cột JSON gốc sau khi tách.
    """
    resolve_columns(df, column)
    records = [_as_dict(value) for value in df[column]]
    flat = pd.json_normalize(records, sep=separator, max_level=max_level)
    flat.index = df.index
    flat = flat.add_prefix(prefix if prefix is not None else f"{column}{separator}")

    base = df.drop(columns=column) if drop_original else df
    logger.info("flatten_json_column[%s]: tạo %d cột", column, flat.shape[1])
    return pd.concat([base, flat], axis=1)


def _flatten_column_names(columns: pd.Index, separator: str) -> list[str]:
    if not isinstance(columns, pd.MultiIndex):
        return [str(col) for col in columns]
    return [separator.join(str(part) for part in col if str(part) != "") for col in columns]


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}
