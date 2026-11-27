"""Data Quality: tỷ lệ thiếu theo cột."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import pandas as pd

from ._common import (Columns, _bar_size, _check_rate, _finish, _make_ax, _no_data, _rate_bar,
                      _select_columns)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["plot_missing"]


def plot_missing(df: pd.DataFrame, cols: Columns = None, *, exclude: Sequence[str] | None = None,
                 min_missing: float | None = None, threshold: float | None = None,
                 ax: Axes | None = None, figsize: tuple[float, float] | None = None,
                 title: str | None = None, **kwargs: Any) -> tuple[Figure, pd.DataFrame]:
    """Tỷ lệ giá trị thiếu của từng cột.

    cols/exclude: chọn cột (None = mọi cột).
    min_missing: chỉ hiển thị (cả bảng lẫn chart) cột có missing_rate >= mức này (0-1); None = hiện hết.
    threshold: cột có missing_rate > ngưỡng (0-1) tô đỏ và vẽ vạch ngưỡng; None = không tô.
    title: None = tự động, "" = không tiêu đề. **kwargs chuyển cho seaborn.barplot (vd color).
    Trả về (Figure, bảng: column, n, missing, missing_rate), sắp theo missing_rate giảm dần;
    có thêm cột bool `exceed` khi truyền threshold. n = tổng số dòng.
    """
    _check_rate("min_missing", min_missing)
    _check_rate("threshold", threshold)
    selected = _select_columns(df, cols, exclude)

    n = len(df)
    missing = df[selected].isna().sum()
    table = pd.DataFrame({"column": selected, "n": n, "missing": missing.to_numpy()})
    table["missing_rate"] = table["missing"] / n if n else 0.0
    if threshold is not None:
        table["exceed"] = table["missing_rate"] > threshold
    if min_missing is not None:
        table = table[table["missing_rate"] >= min_missing]
    table = table.sort_values("missing_rate", ascending=False, kind="stable").reset_index(drop=True)

    fig, axis = _make_ax(ax, figsize, _bar_size(len(table)))
    if table.empty:
        _no_data(axis, "không có cột nào thỏa min_missing")
    else:
        _rate_bar(axis, table["column"], table["missing_rate"], threshold, "tỷ lệ thiếu", **kwargs)
    _finish(fig, axis, title, f"Missing theo cột · {n:,} dòng", created=ax is None)
    return fig, table
