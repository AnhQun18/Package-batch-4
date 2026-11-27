"""Phát hiện và loại bỏ dữ liệu trùng lặp."""

from __future__ import annotations

from typing import Union

import pandas as pd

from ._utils import ColumnsArg, as_list, log_row_change, resolve_columns

KEY_DUPLICATE_STRATEGIES = ("first", "last", "most_complete")


def find_duplicates(df: pd.DataFrame, subset: ColumnsArg = None) -> pd.DataFrame:
    """Trả về TẤT CẢ các dòng thuộc nhóm trùng (để xem trước khi xóa).

    Tham số
    -------
    df : DataFrame đầu vào.
    subset : Cột dùng để so trùng. Để None thì so trên toàn bộ cột.
    """
    keys = resolve_columns(df, subset)
    mask = df.duplicated(subset=keys, keep=False)
    return df[mask].sort_values(keys, kind="mergesort")


def drop_exact_duplicates(
    df: pd.DataFrame,
    ignore_columns: ColumnsArg = None,
    keep: str = "first",
) -> pd.DataFrame:
    """Xóa các dòng giống hệt nhau.

    Tham số
    -------
    df : DataFrame đầu vào.
    ignore_columns : Cột bỏ qua khi so sánh (vd cột thời điểm nạp dữ liệu,
        cột tên file nguồn) vì chúng luôn khác nhau giữa các bản sao.
    keep : "first" giữ bản xuất hiện đầu, "last" giữ bản cuối.
    """
    ignored = set(as_list(ignore_columns))
    resolve_columns(df, list(ignored))
    subset = [col for col in df.columns if col not in ignored]

    result = df.drop_duplicates(subset=subset, keep=keep)
    log_row_change("drop_exact_duplicates", len(df), len(result))
    return result


def drop_key_duplicates(
    df: pd.DataFrame,
    keys: Union[str, list[str]],
    strategy: str = "first",
    order_by: ColumnsArg = None,
    ascending: bool = True,
) -> pd.DataFrame:
    """Giữ đúng một dòng cho mỗi giá trị khóa (vd mỗi customer_id một dòng).

    Thứ tự dòng ban đầu được giữ nguyên trong kết quả.

    Tham số
    -------
    df : DataFrame đầu vào.
    keys : Cột (hoặc tổ hợp cột) xác định một bản ghi duy nhất.
    strategy : Cách chọn dòng giữ lại trong mỗi nhóm trùng:
        - "first": dòng đầu tiên (theo `order_by` nếu có, không thì theo thứ tự hiện tại)
        - "last": dòng cuối cùng, ví dụ dòng mới nhất khi order_by là cột ngày cập nhật
        - "most_complete": dòng có ít ô rỗng nhất; hòa thì lấy dòng đầu
    order_by : Cột dùng để sắp xếp trước khi chọn (vd "updated_at").
    ascending : Chiều sắp xếp của `order_by`.
    """
    if strategy not in KEY_DUPLICATE_STRATEGIES:
        raise ValueError(f"strategy phải là một trong {KEY_DUPLICATE_STRATEGIES}")

    key_columns = resolve_columns(df, keys)
    ordered = df
    if order_by is not None:
        ordered = ordered.sort_values(resolve_columns(df, order_by), ascending=ascending, kind="mergesort")

    if strategy == "most_complete":
        completeness = ordered.notna().sum(axis=1)
        ordered = ordered.loc[completeness.sort_values(ascending=False, kind="mergesort").index]
        keep = "first"
    else:
        keep = strategy

    result = ordered.drop_duplicates(subset=key_columns, keep=keep).sort_index()
    log_row_change(f"drop_key_duplicates ({strategy})", len(df), len(result))
    return result
