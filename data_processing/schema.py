"""Chuẩn hóa tên cột, kiểu dữ liệu và loại bỏ cột vô dụng."""

from __future__ import annotations

import re
from typing import Optional, Sequence

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from ._utils import TEXT_DTYPES, ColumnsArg, as_list, logger, resolve_columns, strip_accents

DEFAULT_CURRENCY_SYMBOLS = ("$", "€", "£", "¥", "₫", "VND", "VNĐ", "USD", "EUR")
DEFAULT_TRUE_VALUES = ("true", "t", "yes", "y", "1", "1.0", "có", "co", "đúng", "dung")
DEFAULT_FALSE_VALUES = ("false", "f", "no", "n", "0", "0.0", "không", "khong", "sai")


def standardize_column_names(df: pd.DataFrame, remove_accents: bool = True) -> pd.DataFrame:
    """Đổi tên tất cả cột về dạng snake_case.

    Ví dụ: "Ngày Sinh " -> "ngay_sinh", "totalPrice($)" -> "total_price".
    Tên trùng sau khi chuẩn hóa sẽ được thêm hậu tố _2, _3, ...

    Tham số
    -------
    df : DataFrame cần đổi tên cột.
    remove_accents : True để bỏ dấu tiếng Việt trong tên cột.
    """
    new_names = _deduplicate_names([_to_snake_case(name, remove_accents) for name in df.columns])
    renamed = {old: new for old, new in zip(df.columns, new_names) if old != new}
    if renamed:
        logger.info("Đổi tên %d cột: %s", len(renamed), renamed)
    return df.set_axis(new_names, axis=1)


def to_numeric(
    df: pd.DataFrame,
    columns: ColumnsArg,
    thousands_sep: Optional[str] = ",",
    decimal_sep: str = ".",
    strip_symbols: Sequence[str] = DEFAULT_CURRENCY_SYMBOLS,
    percent_to_fraction: bool = False,
    accounting_negative: bool = True,
) -> pd.DataFrame:
    """Chuyển cột dạng chữ sang số, dọn ký hiệu tiền tệ, dấu phân cách, %, ...

    Giá trị không chuyển được sẽ thành NaN và số lượng được ghi log.

    Tham số
    -------
    df : DataFrame đầu vào.
    columns : Cột cần chuyển sang số.
    thousands_sep : Dấu phân cách hàng nghìn sẽ bị xóa ("," cho 1,234.5;
        "." cho kiểu Việt Nam 1.234,5). Để None nếu dữ liệu không có.
    decimal_sep : Dấu thập phân của dữ liệu gốc ("." hoặc ",").
    strip_symbols : Các ký hiệu/đơn vị cần xóa trước khi chuyển (tiền tệ, ...).
    percent_to_fraction : True thì "15%" -> 0.15; False thì "15%" -> 15.
    accounting_negative : True thì "(123)" được hiểu là -123 (kiểu kế toán).
    """
    df = df.copy()
    for col in resolve_columns(df, columns):
        if is_numeric_dtype(df[col]) and not is_bool_dtype(df[col]):
            continue
        original = df[col]
        converted = _parse_numeric_series(
            original, thousands_sep, decimal_sep, strip_symbols,
            percent_to_fraction, accounting_negative,
        )
        failed = int((original.notna() & converted.isna()).sum())
        if failed:
            logger.warning("to_numeric[%s]: %d giá trị không chuyển được -> NaN", col, failed)
        df[col] = converted
    return df


def auto_convert_numeric(
    df: pd.DataFrame,
    columns: ColumnsArg = None,
    min_success_ratio: float = 0.95,
    **to_numeric_kwargs,
) -> pd.DataFrame:
    """Tự tìm các cột chữ mà thực chất là số và chuyển chúng sang số.

    Tham số
    -------
    df : DataFrame đầu vào.
    columns : Cột được xét. Để None thì xét mọi cột kiểu chữ.
    min_success_ratio : Một cột chỉ được chuyển nếu ít nhất tỉ lệ này
        (trong số giá trị không rỗng) chuyển được sang số. Ví dụ 0.95 = 95%.
    **to_numeric_kwargs : Truyền tiếp vào `to_numeric` (thousands_sep, decimal_sep, ...).
    """
    df = df.copy()
    for col in resolve_columns(df, columns, dtype_include=TEXT_DTYPES):
        non_missing = df[col].notna().sum()
        if non_missing == 0:
            continue
        parsed = _parse_numeric_series(df[col], **_numeric_parse_options(to_numeric_kwargs))
        success_ratio = parsed.notna().sum() / non_missing
        if success_ratio >= min_success_ratio:
            logger.info("auto_convert_numeric: %s -> số (%.1f%% hợp lệ)", col, success_ratio * 100)
            df[col] = parsed
    return df


def to_boolean(
    df: pd.DataFrame,
    columns: ColumnsArg,
    true_values: Sequence[str] = DEFAULT_TRUE_VALUES,
    false_values: Sequence[str] = DEFAULT_FALSE_VALUES,
) -> pd.DataFrame:
    """Chuyển cột về kiểu boolean (True/False/<NA>).

    So khớp không phân biệt hoa thường và bỏ khoảng trắng hai đầu.
    Giá trị không nằm trong hai danh sách sẽ thành <NA> và được ghi log.

    Tham số
    -------
    df : DataFrame đầu vào.
    columns : Cột cần chuyển.
    true_values : Các giá trị được hiểu là True.
    false_values : Các giá trị được hiểu là False.
    """
    df = df.copy()
    lookup = {str(v).strip().lower(): True for v in true_values}
    lookup.update({str(v).strip().lower(): False for v in false_values})

    for col in resolve_columns(df, columns):
        if is_bool_dtype(df[col]):
            continue
        original = df[col]
        converted = original.map(
            lambda v: lookup.get(str(v).strip().lower(), pd.NA) if pd.notna(v) else pd.NA
        ).astype("boolean")
        unmapped = int((original.notna() & converted.isna()).sum())
        if unmapped:
            logger.warning("to_boolean[%s]: %d giá trị không nhận dạng được -> <NA>", col, unmapped)
        df[col] = converted
    return df


def to_category(
    df: pd.DataFrame,
    columns: ColumnsArg = None,
    max_unique_ratio: float = 0.05,
) -> pd.DataFrame:
    """Chuyển cột chữ có ít giá trị khác nhau sang kiểu category (tiết kiệm bộ nhớ).

    Tham số
    -------
    df : DataFrame đầu vào.
    columns : Cột cần chuyển. Để None thì tự chọn các cột chữ có
        (số giá trị khác nhau / số dòng) <= max_unique_ratio.
    max_unique_ratio : Ngưỡng tỉ lệ giá trị khác nhau, chỉ dùng khi columns=None.
    """
    df = df.copy()
    if columns is None:
        candidates = resolve_columns(df, None, dtype_include=TEXT_DTYPES)
        selected = [
            col for col in candidates
            if len(df) and df[col].nunique(dropna=True) / len(df) <= max_unique_ratio
        ]
    else:
        selected = resolve_columns(df, columns)

    for col in selected:
        df[col] = df[col].astype("category")
    if selected:
        logger.info("to_category: %s", selected)
    return df


def drop_useless_columns(
    df: pd.DataFrame,
    drop_all_missing: bool = True,
    drop_constant: bool = True,
    exclude: ColumnsArg = None,
) -> pd.DataFrame:
    """Xóa cột không mang thông tin.

    Tham số
    -------
    df : DataFrame đầu vào.
    drop_all_missing : Xóa cột mà toàn bộ giá trị đều rỗng.
    drop_constant : Xóa cột chỉ có đúng một giá trị (không tính ô rỗng).
    exclude : Cột luôn giữ lại dù thỏa điều kiện xóa (vd cột ID, cột target).
    """
    protected = set(as_list(exclude))
    to_drop = []
    for col in df.columns:
        if col in protected:
            continue
        n_unique = df[col].nunique(dropna=True)
        if (drop_all_missing and n_unique == 0) or (drop_constant and n_unique == 1):
            to_drop.append(col)

    if to_drop:
        logger.info("drop_useless_columns: xóa %s", to_drop)
    return df.drop(columns=to_drop)


def _to_snake_case(name: object, remove_accents: bool) -> str:
    text = str(name).strip()
    if remove_accents:
        text = strip_accents(text)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)  # camelCase -> camel_Case
    text = re.sub(r"[^\w]+", "_", text.lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unnamed"


def _deduplicate_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    result = []
    for name in names:
        count = seen.get(name, 0) + 1
        seen[name] = count
        result.append(name if count == 1 else f"{name}_{count}")
    return result


def _numeric_parse_options(kwargs: dict) -> dict:
    defaults = {
        "thousands_sep": ",",
        "decimal_sep": ".",
        "strip_symbols": DEFAULT_CURRENCY_SYMBOLS,
        "percent_to_fraction": False,
        "accounting_negative": True,
    }
    unknown = set(kwargs) - set(defaults)
    if unknown:
        raise TypeError(f"Tham số không hợp lệ: {sorted(unknown)}")
    return {**defaults, **kwargs}


def _parse_numeric_series(
    series: pd.Series,
    thousands_sep: Optional[str],
    decimal_sep: str,
    strip_symbols: Sequence[str],
    percent_to_fraction: bool,
    accounting_negative: bool,
) -> pd.Series:
    text = series.astype("string").str.strip()
    for symbol in strip_symbols:
        text = text.str.replace(symbol, "", regex=False)
    text = text.str.replace(r"\s+", "", regex=True)

    if accounting_negative:
        text = text.str.replace(r"^\((.*)\)$", r"-\1", regex=True)

    is_percent = text.str.endswith("%").fillna(False)
    text = text.str.rstrip("%")

    if thousands_sep:
        text = text.str.replace(thousands_sep, "", regex=False)
    if decimal_sep != ".":
        text = text.str.replace(decimal_sep, ".", regex=False)

    numbers = pd.to_numeric(text, errors="coerce").astype("float64")
    if percent_to_fraction:
        numbers = numbers.where(~is_percent.to_numpy(), numbers / 100)
    return numbers
