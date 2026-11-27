"""Chuyển đổi và chuẩn hóa dữ liệu ngày giờ."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from ._utils import ColumnsArg, as_list, logger, resolve_columns


def parse_datetime(
    df: pd.DataFrame,
    columns: ColumnsArg,
    formats: Optional[Sequence[str]] = None,
    dayfirst: bool = True,
    infer_remaining: bool = True,
    utc: bool = False,
    unit: Optional[str] = None,
) -> pd.DataFrame:
    """Chuyển cột sang kiểu datetime, chịu được nhiều định dạng lẫn lộn trong một cột.

    Giá trị không parse được sẽ thành NaT và số lượng được ghi log.

    Tham số
    -------
    df : DataFrame đầu vào.
    columns : Cột cần chuyển.
    formats : Danh sách định dạng thử lần lượt, vd ["%d/%m/%Y", "%Y-%m-%d %H:%M:%S"].
        Giá trị nào khớp định dạng trước thì lấy định dạng đó. Nên khai báo rõ
        để tránh nhầm ngày/tháng (03/04 là 3 tháng 4 hay 4 tháng 3).
    dayfirst : Khi phải tự đoán, coi số đầu tiên là ngày (kiểu Việt Nam / châu Âu).
    infer_remaining : Sau khi thử hết `formats`, để pandas tự đoán các giá trị còn lại.
    utc : True để đưa mọi giá trị về múi giờ UTC. Nên bật khi dữ liệu chứa
        múi giờ khác nhau (vd "+07:00" và "Z").
    unit : Đơn vị khi cột là Unix timestamp dạng số: "s", "ms", "us", "ns".
    """
    df = df.copy()
    for col in resolve_columns(df, columns):
        original = df[col]
        if is_datetime64_any_dtype(original):
            continue

        if unit is not None and is_numeric_dtype(original):
            parsed = pd.to_datetime(original, unit=unit, utc=utc, errors="coerce")
        else:
            parsed = _parse_with_formats(original, formats or [], dayfirst, infer_remaining, utc)

        failed = int((original.notna() & parsed.isna()).sum())
        if failed:
            logger.warning("parse_datetime[%s]: %d giá trị không parse được -> NaT", col, failed)
        df[col] = parsed
    return df


def convert_timezone(
    df: pd.DataFrame,
    columns: ColumnsArg,
    target_tz: str = "UTC",
    source_tz: Optional[str] = None,
    drop_timezone: bool = False,
) -> pd.DataFrame:
    """Đưa cột datetime về một múi giờ chung.

    Tham số
    -------
    df : DataFrame đầu vào.
    columns : Cột datetime cần chuyển.
    target_tz : Múi giờ đích, vd "UTC" hoặc "Asia/Ho_Chi_Minh".
    source_tz : Múi giờ gốc của dữ liệu KHÔNG có thông tin múi giờ, vd "Asia/Ho_Chi_Minh".
        Bắt buộc nếu cột chưa có múi giờ. Giờ không tồn tại / bị lặp do đổi giờ
        mùa hè sẽ được dời tới / đặt NaT.
    drop_timezone : True để bỏ nhãn múi giờ sau khi chuyển (giữ giờ địa phương của target_tz).
    """
    df = df.copy()
    for col in resolve_columns(df, columns):
        series = df[col]
        if not is_datetime64_any_dtype(series):
            raise TypeError(f"Cột {col!r} chưa phải datetime; hãy gọi parse_datetime trước.")
        if series.dt.tz is None:
            if source_tz is None:
                raise ValueError(f"Cột {col!r} chưa có múi giờ; cần truyền source_tz.")
            series = series.dt.tz_localize(source_tz, ambiguous="NaT", nonexistent="shift_forward")
        series = series.dt.tz_convert(target_tz)
        df[col] = series.dt.tz_localize(None) if drop_timezone else series
    return df


def fill_time_gaps(
    df: pd.DataFrame,
    time_column: str,
    freq: str,
    group_by: Optional[Union[str, Sequence[str]]] = None,
    indicator_column: Optional[str] = None,
) -> pd.DataFrame:
    """Thêm các mốc thời gian bị thiếu để chuỗi thời gian liên tục.

    Dòng mới có giá trị NaN ở các cột dữ liệu; có thể điền sau bằng `fill_missing`
    (vd strategy="ffill" hoặc "interpolate").

    Tham số
    -------
    df : DataFrame đầu vào.
    time_column : Cột thời gian (kiểu datetime).
    freq : Bước thời gian mong muốn: "D" (ngày), "h" (giờ), "W" (tuần), "MS" (đầu tháng), ...
    group_by : Lấp khoảng trống riêng cho từng nhóm, vd từng cửa hàng / từng cảm biến.
        Mỗi nhóm chạy từ mốc nhỏ nhất đến lớn nhất của chính nhóm đó.
    indicator_column : Nếu đặt tên, thêm cột 0/1 đánh dấu dòng mới được thêm.
    """
    resolve_columns(df, time_column)
    if not is_datetime64_any_dtype(df[time_column]):
        raise TypeError(f"Cột {time_column!r} chưa phải datetime; hãy gọi parse_datetime trước.")
    groups = as_list(group_by)
    resolve_columns(df, groups)

    def complete(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.set_index(time_column).sort_index()
        if frame.index.has_duplicates:
            raise ValueError(f"Cột {time_column!r} có mốc thời gian trùng; hãy khử trùng trước.")
        full_range = pd.date_range(frame.index.min(), frame.index.max(), freq=freq)
        completed = frame.reindex(frame.index.union(full_range))
        if indicator_column:
            completed[indicator_column] = (~completed.index.isin(frame.index)).astype("int8")
        return completed.rename_axis(time_column).reset_index()

    if not groups:
        result = complete(df)
    else:
        pieces = []
        for key, frame in df.groupby(groups, sort=False, dropna=False):
            piece = complete(frame.drop(columns=groups))
            for col, value in zip(groups, key):
                piece[col] = value
            pieces.append(piece)
        result = pd.concat(pieces, ignore_index=True)[list(df.columns) + (
            [indicator_column] if indicator_column else []
        )]

    logger.info("fill_time_gaps: thêm %d mốc thời gian", len(result) - len(df))
    return result


def _parse_with_formats(
    series: pd.Series, formats: Sequence[str], dayfirst: bool, infer_remaining: bool, utc: bool
) -> pd.Series:
    text = series.where(series.isna(), series.astype(str).str.strip())
    parsed: Optional[pd.Series] = None

    for fmt in formats:
        attempt = pd.to_datetime(text, format=fmt, errors="coerce", utc=utc)
        parsed = attempt if parsed is None else parsed.fillna(attempt)

    if infer_remaining:
        remaining = text if parsed is None else text.where(parsed.isna())
        attempt = pd.to_datetime(remaining, format="mixed", dayfirst=dayfirst, errors="coerce", utc=utc)
        parsed = attempt if parsed is None else parsed.fillna(attempt)

    if parsed is None:
        raise ValueError("Cần truyền formats hoặc để infer_remaining=True.")
    return parsed
