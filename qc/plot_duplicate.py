"""Data Quality: tỷ lệ dòng trùng."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

import pandas as pd

from ._common import (DatasetError, _as_list, _bar_size, _check_df, _check_rate, _finish, _make_ax,
                      _rate_bar)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["plot_duplicate"]


def plot_duplicate(df: pd.DataFrame, subset: Sequence[str] | Mapping[str, Sequence[str] | str] | None = None, *,
                   keep: str | bool = "first", threshold: float | None = None,
                   ax: Axes | None = None, figsize: tuple[float, float] | None = None,
                   title: str | None = None, **kwargs: Any) -> tuple[Figure, pd.DataFrame]:
    """Tỷ lệ dòng trùng theo một hoặc nhiều định nghĩa "trùng".

    subset: None = trùng cả dòng; list cột = một định nghĩa (trùng trên các cột đó);
        dict {tên: cột hoặc list cột} = so nhiều định nghĩa trong cùng một chart.
    keep: "first" | "last" | False, như DataFrame.duplicated (False = tính cả dòng gốc là trùng).
    threshold: định nghĩa có duplicate_rate > ngưỡng (0-1) tô đỏ; None = không tô.
    title: None = tự động, "" = không tiêu đề. **kwargs chuyển cho seaborn.barplot (vd color).
    Trả về (Figure, bảng: subset, n_rows, n_duplicate, duplicate_rate), giữ thứ tự truyền vào;
    có thêm cột bool `exceed` khi truyền threshold.
    """
    _check_df(df)
    _check_rate("threshold", threshold)
    if keep not in ("first", "last", False):
        raise DatasetError(f"keep phải là 'first', 'last' hoặc False, nhận {keep!r}.")

    if subset is None:
        defs = {"toàn bộ dòng": list(df.columns)}
    elif isinstance(subset, Mapping):
        if not subset:
            raise DatasetError("subset rỗng.")
        defs = {str(k): _as_list(v) for k, v in subset.items()}
    else:
        names = _as_list(subset)
        if not names:
            raise DatasetError("subset rỗng. Dùng subset=None để so cả dòng.")
        defs = {" + ".join(map(str, names)): names}
    for name, cols in defs.items():
        absent = [c for c in cols if c not in df.columns]
        if absent:
            raise DatasetError(f"subset '{name}' có cột không tồn tại: {absent}.")
        if not cols:
            raise DatasetError(f"subset '{name}' rỗng (DataFrame không có cột?).")

    n = len(df)
    rows = []
    for name, cols in defs.items():
        n_dup = int(df.duplicated(subset=cols, keep=keep).sum())
        rows.append({"subset": name, "n_rows": n, "n_duplicate": n_dup, "duplicate_rate": n_dup / n if n else 0.0})
    table = pd.DataFrame(rows, columns=["subset", "n_rows", "n_duplicate", "duplicate_rate"])
    if threshold is not None:
        table["exceed"] = table["duplicate_rate"] > threshold

    fig, axis = _make_ax(ax, figsize, _bar_size(len(table)))
    _rate_bar(axis, table["subset"], table["duplicate_rate"], threshold, "tỷ lệ trùng", **kwargs)
    _finish(fig, axis, title, f"Duplicate · {n:,} dòng · keep={keep!r}", created=ax is None)
    return fig, table
