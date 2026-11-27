"""Hàm tiện ích dùng chung nội bộ cho package data_processing.

Không gọi trực tiếp các hàm ở đây từ notebook; chúng chỉ phục vụ các module khác.
"""

from __future__ import annotations

import logging
import unicodedata
from typing import Optional, Sequence, Union

import pandas as pd

logger = logging.getLogger("data_processing")

# Kiểu chung cho tham số `columns`: một tên cột, danh sách tên cột, hoặc None (= tự chọn).
ColumnsArg = Optional[Union[str, Sequence[str]]]

TEXT_DTYPES = ["object", "string"]


def resolve_columns(
    df: pd.DataFrame,
    columns: ColumnsArg = None,
    dtype_include: Optional[Sequence[str]] = None,
) -> list[str]:
    """Chuẩn hóa tham số `columns` thành list tên cột và kiểm tra cột có tồn tại.

    Tham số
    -------
    df : DataFrame cần lấy cột.
    columns : Tên cột / danh sách tên cột. Nếu None thì lấy tất cả cột
        (hoặc chỉ các cột có kiểu nằm trong `dtype_include`).
    dtype_include : Danh sách kiểu dữ liệu dùng để tự chọn cột khi `columns` là None,
        ví dụ ["number"] hoặc ["object", "string"].
    """
    if columns is None:
        if dtype_include is None:
            return list(df.columns)
        return list(df.select_dtypes(include=list(dtype_include)).columns)

    selected = [columns] if isinstance(columns, str) else list(columns)
    missing = [col for col in selected if col not in df.columns]
    if missing:
        raise KeyError(f"Không tìm thấy cột trong DataFrame: {missing}")
    return selected


def as_list(value: Optional[Union[str, Sequence[str]]]) -> list[str]:
    """Đưa một chuỗi hoặc danh sách về dạng list (None -> list rỗng)."""
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def strip_accents(text: str) -> str:
    """Bỏ dấu tiếng Việt và các ký tự có dấu khác: 'Hà Nội' -> 'Ha Noi'."""
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def log_row_change(step: str, rows_before: int, rows_after: int) -> None:
    """Ghi log số dòng trước/sau một bước xử lý."""
    logger.info(
        "%s: %d -> %d dòng (loại %d)", step, rows_before, rows_after, rows_before - rows_after
    )
