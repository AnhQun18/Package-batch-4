"""Hỗ trợ dùng chung cho các hàm vẽ EDA: chọn cột, kiểm tra kind, lưới Axes, nhãn.

Nội bộ của package `eda`; người dùng import từ `ds_kit.data_inspector.eda`.
Interface document: docs/interfaces/eda.md
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..dataset import DatasetError

if TYPE_CHECKING:  # chỉ cho type hint; không bắt buộc cài matplotlib để import module
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

Columns = str | Sequence[str] | None


def _check_kinds(kind: str | Sequence[str], registry: Mapping[str, Any]) -> list[str]:
    kinds = list(kind) if isinstance(kind, (list, tuple)) else [kind]
    if not kinds:
        raise DatasetError(f"kind rỗng. Chọn trong {list(registry)}.")
    bad = [k for k in kinds if k not in registry]
    if bad:
        raise DatasetError(f"kind không hỗ trợ: {bad}. Chọn trong {list(registry)}.")
    if len(set(kinds)) != len(kinds):
        raise DatasetError(f"kind bị trùng: {kinds}.")
    return kinds


def _select_columns(df: Any, cols: Columns, exclude: Sequence[str] | None,
                    auto: Callable[[pd.Series], bool], explicit: Callable[[pd.Series], bool],
                    kind: str = "hợp lệ") -> list[str]:
    """Chọn cột: cols=None -> các cột thỏa `auto`; cols tường minh -> kiểm tra tồn tại và thỏa `explicit`;
    rồi bỏ `exclude`. Raise DatasetError nếu df sai, cột không tồn tại/không hợp lệ, hoặc danh sách rỗng."""
    if not isinstance(df, pd.DataFrame):
        raise DatasetError(f"Cần pandas DataFrame, nhận {type(df).__name__}.")
    if df.columns.has_duplicates:
        raise DatasetError(f"DataFrame có tên cột trùng: {df.columns[df.columns.duplicated()].tolist()}. Đổi tên trước.")

    if cols is None:
        selected = [c for c in df.columns if auto(df[c])]
    else:
        names = list(cols) if isinstance(cols, (list, tuple, pd.Index)) else [cols]
        if not names:
            raise DatasetError("cols rỗng. Dùng cols=None để chọn mọi cột hợp lệ.")
        dup = sorted({str(c) for c in names if names.count(c) > 1})
        if dup:
            raise DatasetError(f"cols bị trùng: {dup}.")
        absent = [c for c in names if c not in df.columns]
        if absent:
            raise DatasetError(f"Cột không tồn tại: {absent}. Cột hiện có: {list(df.columns)}.")
        bad = [f"{c} ({df[c].dtype})" for c in names if not explicit(df[c])]
        if bad:
            raise DatasetError(f"Cột không phải kiểu {kind}: {bad}.")
        selected = names

    if exclude is not None:
        ex = list(exclude) if isinstance(exclude, (list, tuple, pd.Index)) else [exclude]
        absent = [c for c in ex if c not in df.columns]
        if absent:
            raise DatasetError(f"exclude có cột không tồn tại: {absent}.")
        selected = [c for c in selected if c not in ex]

    if not selected:
        raise DatasetError(f"Không có cột kiểu {kind} nào để vẽ. Kiểm tra dtype hoặc truyền cols/exclude khác.")
    return selected


def _require_plotting() -> tuple[Any, Any]:
    """Import (matplotlib.pyplot, seaborn); thiếu -> ImportError hướng dẫn `pip install ds-kit[viz]`."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as e:
        raise ImportError("Hàm vẽ cần matplotlib và seaborn. Cài bằng: pip install -e \".[viz]\" "
                          "(hoặc pip install matplotlib seaborn).") from e
    return plt, sns


def _make_grid(n: int, ncols: int, ax: Axes | Sequence[Axes] | None,
               figsize: tuple[float, float] | None, subplot_kws: Mapping[str, Any] | None) -> tuple[Figure, list[Axes]]:
    """Dùng Axes được truyền vào (kiểm tra đủ n) hoặc tạo lưới ceil(n/ncols) x ncols; ẩn ô thừa."""
    plt, _ = _require_plotting()
    from matplotlib.axes import Axes as _Axes

    if ax is not None:
        if figsize is not None or subplot_kws is not None:
            raise DatasetError("figsize/subplot_kws chỉ dùng khi ax=None (tạo lưới mới); với ax có sẵn hãy chỉnh figure của bạn.")
        axes = [ax] if isinstance(ax, _Axes) else list(np.ravel(np.asarray(ax, dtype=object)))
        if not all(isinstance(a, _Axes) for a in axes):
            raise DatasetError("ax phải là matplotlib Axes hoặc dãy Axes.")
        if len(axes) != n:
            raise DatasetError(f"Cần {n} Axes cho {n} ô, nhận {len(axes)}. Truyền đúng số Axes hoặc để ax=None.")
        return axes[0].figure, axes

    if isinstance(ncols, bool) or not isinstance(ncols, int) or ncols < 1:
        raise DatasetError(f"ncols phải là số nguyên >= 1, nhận {ncols!r}.")
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    if figsize is None:
        figsize = (4.5 * ncols, 3.2 * nrows)
    fig, grid = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, **dict(subplot_kws or {}))
    flat = list(grid.ravel())
    for extra in flat[n:]:
        extra.set_visible(False)
    return fig, flat[:n]


def _no_data(ax: Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)


def _apply_labels(fig: Figure, ax: Axes, *, title: str | None, col_title: str | None,
                  info: Mapping[str, Any], auto_title: str) -> None:
    """Tiêu đề figure và từng ô theo quy tắc: None = tự động/không, "" = bỏ, chuỗi = dùng (format với info)."""
    if col_title is None:
        ax.set_title(auto_title, fontsize="medium")
    elif col_title:
        try:
            ax.set_title(col_title.format(**info), fontsize="medium")
        except (KeyError, IndexError, ValueError) as e:
            raise DatasetError(f"col_title {col_title!r} không format được ({e}). Khóa có sẵn: {list(info)}.") from None
    else:
        ax.set_title("")
    if title:
        fig.suptitle(title)
