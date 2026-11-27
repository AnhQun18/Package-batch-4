"""Làm sạch dữ liệu chữ và gộp các biến thể của cùng một giá trị category."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Optional

import numpy as np
import pandas as pd

from ._utils import TEXT_DTYPES, ColumnsArg, logger, resolve_columns, strip_accents

CASE_OPTIONS = (None, "lower", "upper", "title")
UNMATCHED_OPTIONS = ("keep", "missing", "replace")


def clean_text(
    df: pd.DataFrame,
    columns: ColumnsArg = None,
    strip: bool = True,
    collapse_whitespace: bool = True,
    case: Optional[str] = None,
    unicode_form: Optional[str] = "NFC",
    remove_pattern: Optional[str] = None,
    empty_to_missing: bool = True,
) -> pd.DataFrame:
    """Làm sạch các ô chữ. Ô không phải chữ (số, NaN, ...) được giữ nguyên.

    Tham số
    -------
    df : DataFrame đầu vào.
    columns : Cột cần làm sạch. Để None thì xử lý mọi cột kiểu chữ.
    strip : Bỏ khoảng trắng ở hai đầu.
    collapse_whitespace : Gộp nhiều khoảng trắng/tab/xuống dòng liên tiếp thành một dấu cách.
    case : Đổi chữ hoa/thường: None (giữ nguyên), "lower", "upper", "title".
    unicode_form : Chuẩn Unicode, nên để "NFC". Tiếng Việt có thể được gõ theo
        hai cách (dựng sẵn / tổ hợp) nhìn giống nhau nhưng so sánh lại khác nhau;
        chuẩn hóa giúp "Hà Nội" == "Hà Nội". Để None để bỏ qua.
    remove_pattern : Biểu thức regex cho phần cần xóa, vd r"[^\\w\\s]" để xóa ký tự đặc biệt.
    empty_to_missing : Đổi chuỗi rỗng (sau khi làm sạch) thành NaN.
    """
    if case not in CASE_OPTIONS:
        raise ValueError(f"case phải là một trong {CASE_OPTIONS}")
    pattern = re.compile(remove_pattern) if remove_pattern else None

    def clean_value(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        if unicode_form:
            value = unicodedata.normalize(unicode_form, value)
        if pattern:
            value = pattern.sub("", value)
        if collapse_whitespace:
            value = re.sub(r"\s+", " ", value)
        if strip:
            value = value.strip()
        if case:
            value = getattr(value, case)()
        if empty_to_missing and value == "":
            return np.nan
        return value

    df = df.copy()
    for col in resolve_columns(df, columns, dtype_include=TEXT_DTYPES):
        df[col] = df[col].map(clean_value)
    return df


def normalize_categories(
    df: pd.DataFrame,
    columns: ColumnsArg,
    mapping: dict[str, list[str]],
    ignore_case: bool = True,
    ignore_accents: bool = True,
    unmatched: str = "keep",
    replacement: Any = "Other",
) -> pd.DataFrame:
    """Gộp các cách viết khác nhau của cùng một giá trị về một tên chuẩn.

    Ví dụ mapping = {"Hà Nội": ["HN", "Ha Noi", "hanoi"], "TP.HCM": ["HCM", "Sài Gòn"]}.
    Tên chuẩn cũng tự được coi là một biến thể của chính nó.

    Tham số
    -------
    df : DataFrame đầu vào.
    columns : Cột cần chuẩn hóa (có thể nhiều cột dùng chung một mapping).
    mapping : dict {tên_chuẩn: [các biến thể]}.
    ignore_case : So khớp không phân biệt hoa thường.
    ignore_accents : So khớp không phân biệt dấu ("Ha Noi" khớp với "Hà Nội").
    unmatched : Xử lý giá trị không khớp biến thể nào:
        - "keep": giữ nguyên
        - "missing": đổi thành NaN
        - "replace": đổi thành `replacement`
    replacement : Giá trị thay thế khi unmatched="replace".
    """
    if unmatched not in UNMATCHED_OPTIONS:
        raise ValueError(f"unmatched phải là một trong {UNMATCHED_OPTIONS}")

    def match_key(value: str) -> str:
        key = re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip()
        if ignore_accents:
            key = strip_accents(key)
        return key.lower() if ignore_case else key

    lookup = {}
    for canonical, variants in mapping.items():
        for variant in [canonical, *variants]:
            lookup[match_key(variant)] = canonical

    def map_value(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        canonical = lookup.get(match_key(value))
        if canonical is not None:
            return canonical
        if unmatched == "missing":
            return np.nan
        if unmatched == "replace":
            return replacement
        return value

    df = df.copy()
    for col in resolve_columns(df, columns):
        original = df[col].astype(object)
        df[col] = original.map(map_value)
        both_missing = original.isna() & df[col].isna()
        changed = int(((original != df[col]) & ~both_missing).sum())
        logger.info("normalize_categories[%s]: đổi %d giá trị", col, changed)
    return df
