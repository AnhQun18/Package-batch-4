"""Đọc dữ liệu dạng bảng và gộp nhiều nguồn một cách an toàn."""

from __future__ import annotations

import csv
from itertools import islice
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import pandas as pd

from ._utils import as_list, log_row_change, logger

PathLike = Union[str, Path]

FORMAT_BY_EXTENSION = {
    ".csv": "csv",
    ".tsv": "csv",
    ".txt": "csv",
    ".xlsx": "excel",
    ".xls": "excel",
    ".json": "json",
    ".jsonl": "jsonl",
    ".parquet": "parquet",
}

# Thứ tự thử encoding: utf-8-sig đọc được cả UTF-8 có/không BOM,
# cp1258 là encoding tiếng Việt cũ của Windows, latin-1 luôn đọc được nên để cuối.
DEFAULT_ENCODINGS = ("utf-8-sig", "cp1258", "latin-1")
CANDIDATE_DELIMITERS = ",;\t|"
SNIFF_LINES = 50


def load_table(
    path: PathLike,
    file_format: Optional[str] = None,
    encodings: Sequence[str] = DEFAULT_ENCODINGS,
    sep: Optional[str] = None,
    **read_kwargs,
) -> pd.DataFrame:
    """Đọc một file dạng bảng thành DataFrame.

    Tham số
    -------
    path : Đường dẫn tới file.
    file_format : "csv", "excel", "json", "jsonl" hoặc "parquet".
        Để None thì tự đoán theo đuôi file.
    encodings : (chỉ CSV) Danh sách encoding thử lần lượt cho tới khi đọc được.
    sep : (chỉ CSV) Ký tự phân tách cột. Để None thì tự dò trong các ký tự , ; tab |.
    **read_kwargs : Tham số bổ sung truyền thẳng vào hàm đọc của pandas
        (ví dụ sheet_name=... cho Excel, usecols=... cho CSV).
    """
    path = Path(path)
    fmt = file_format or _infer_format(path)

    if fmt == "csv":
        df = _read_csv_with_fallback(path, encodings, sep, read_kwargs)
    elif fmt == "excel":
        df = pd.read_excel(path, **read_kwargs)
    elif fmt == "json":
        df = pd.read_json(path, **read_kwargs)
    elif fmt == "jsonl":
        df = pd.read_json(path, lines=True, **read_kwargs)
    elif fmt == "parquet":
        df = pd.read_parquet(path, **read_kwargs)
    else:
        raise ValueError(f"Định dạng không hỗ trợ: {fmt!r}")

    logger.info("Đọc %s: %d dòng x %d cột", path.name, len(df), df.shape[1])
    return df


def load_many(
    paths: Iterable[PathLike],
    source_column: Optional[str] = "source_file",
    **load_kwargs,
) -> pd.DataFrame:
    """Đọc nhiều file cùng cấu trúc và nối chúng theo chiều dọc.

    Tham số
    -------
    paths : Danh sách đường dẫn file.
    source_column : Tên cột mới ghi lại dòng đến từ file nào. Để None nếu không cần.
    **load_kwargs : Truyền tiếp vào `load_table` cho mỗi file.
    """
    paths = list(paths)
    frames = []
    for path in paths:
        frame = load_table(path, **load_kwargs)
        if source_column:
            frame[source_column] = Path(path).name
        frames.append(frame)

    if not frames:
        raise ValueError("Danh sách file rỗng.")

    reference_columns = set(frames[0].columns)
    for path, frame in zip(paths, frames):
        if set(frame.columns) != reference_columns:
            logger.warning("File %s có cột khác với file đầu tiên; cột thiếu sẽ là NaN.", path)

    return pd.concat(frames, ignore_index=True)


def safe_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: Optional[Union[str, Sequence[str]]] = None,
    left_on: Optional[Union[str, Sequence[str]]] = None,
    right_on: Optional[Union[str, Sequence[str]]] = None,
    how: str = "left",
    relationship: str = "many_to_one",
    indicator_column: Optional[str] = None,
) -> pd.DataFrame:
    """Join hai bảng và kiểm tra quan hệ khóa để tránh nhân dòng ngoài ý muốn.

    Tham số
    -------
    left, right : Hai bảng cần join.
    on : Cột khóa chung (cùng tên ở cả hai bảng).
    left_on, right_on : Dùng thay `on` khi tên cột khóa ở hai bảng khác nhau.
    how : Kiểu join: "left", "right", "inner", "outer".
    relationship : Quan hệ khóa mong đợi: "one_to_one", "one_to_many",
        "many_to_one" hoặc "many_to_many". Nếu dữ liệu vi phạm (vd bảng phải
        có khóa trùng khi khai báo "many_to_one"), hàm báo lỗi thay vì âm thầm nhân dòng.
    indicator_column : Nếu đặt tên, thêm cột cho biết dòng khớp ở "both",
        "left_only" hay "right_only". Hữu ích để tìm dòng không join được.
    """
    try:
        merged = left.merge(
            right,
            on=on,
            left_on=left_on,
            right_on=right_on,
            how=how,
            validate=relationship,
            indicator=indicator_column or False,
        )
    except pd.errors.MergeError as error:
        right_keys = as_list(right_on or on)
        duplicated = int(right.duplicated(subset=right_keys or None).sum())
        raise ValueError(
            f"Quan hệ khóa không đúng {relationship!r}: {error}. "
            f"Bảng phải có {duplicated} dòng trùng khóa; hãy khử trùng trước khi join."
        ) from error

    log_row_change(f"safe_merge ({how})", len(left), len(merged))
    return merged


def _infer_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in FORMAT_BY_EXTENSION:
        raise ValueError(
            f"Không đoán được định dạng từ đuôi {suffix!r}; hãy truyền file_format."
        )
    return FORMAT_BY_EXTENSION[suffix]


def _read_csv_with_fallback(
    path: Path, encodings: Sequence[str], sep: Optional[str], read_kwargs: dict
) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    for encoding in encodings:
        try:
            delimiter = sep or _sniff_delimiter(path, encoding)
            df = pd.read_csv(path, encoding=encoding, sep=delimiter, **read_kwargs)
            logger.info("CSV %s: encoding=%s, sep=%r", path.name, encoding, delimiter)
            return df
        except UnicodeDecodeError as error:
            last_error = error
    raise ValueError(f"Không đọc được {path} với các encoding {list(encodings)}") from last_error


def _sniff_delimiter(path: Path, encoding: str) -> str:
    with path.open(encoding=encoding) as handle:
        sample = "".join(islice(handle, SNIFF_LINES))
    try:
        return csv.Sniffer().sniff(sample, delimiters=CANDIDATE_DELIMITERS).delimiter
    except csv.Error:
        return ","
