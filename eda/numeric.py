"""EDA cho cột số: plot_numeric (hist, box, kde, violin, ecdf).

CẤU TRÚC:
    1. Vẽ (public)      plot_numeric
    2. Dữ liệu vẽ       _numeric_data   (test được không cần matplotlib)
    3. Hỗ trợ           _is_numeric, _numeric_plotters

Quy ước chung (cols/exclude/ncols/ax/figsize/subplot_kws/title/col_title/**kwargs, giá trị trả về) xem
docstring package `ds_kit.data_inspector.eda` và docs/interfaces/eda.md.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from ._common import (DatasetError, Columns, _apply_labels, _make_grid, _no_data, _require_plotting, _select_columns)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["plot_numeric"]


# ======================================================================
# 1. VẼ
# ======================================================================
def plot_numeric(df: pd.DataFrame, cols: Columns = None, *, exclude: Sequence[str] | None = None,
                 kind: str | Sequence[str] = "hist", bins: int | str | None = None, ncols: int | None = None,
                 ax: Axes | Sequence[Axes] | None = None, figsize: tuple[float, float] | None = None,
                 subplot_kws: Mapping[str, Any] | None = None, title: str | None = None,
                 col_title: str | None = None, **kwargs: Any) -> tuple[Figure, pd.DataFrame]:
    """Một hàm cho mọi chart phân phối cột số: vẽ đúng cột và loại chart người dùng truyền vào.

    Cột hợp lệ: dtype số, không phải bool. cols=None -> mọi cột như vậy.
    kind: một hoặc nhiều tên chart trong registry `_numeric_plotters()`
        ("hist", "box", "kde", "violin", "ecdf"); thêm loại mới = thêm một entry, không sửa hàm.
        Mỗi cặp (cột, kind) là một ô. Nhiều kind -> mỗi cột một hàng, mỗi kind một ô trong hàng.
    bins: số bin hoặc quy tắc numpy ("auto", "sturges", ...); None = "auto". Chỉ dùng khi có "hist".
    ncols: số ô mỗi hàng; None = 3 nếu một kind, = số kind nếu nhiều kind.
    Trả về: (Figure, long) với long gồm cột `column, value` (các giá trị hữu hạn đã vẽ).
    Raises DatasetError: df không hợp lệ, cột không tồn tại/không phải số, không còn cột nào để vẽ,
        kind rỗng/trùng/không có trong registry, bins dùng khi không có "hist",
        số Axes truyền vào không khớp số ô.
    """
    plotters = _numeric_plotters()
    kinds = list(kind) if isinstance(kind, (list, tuple)) else [kind]
    if bins is not None and "hist" not in kinds:
        raise DatasetError(f"bins chỉ dùng với kind 'hist', nhận kind={kinds}.")
    if ncols is None:
        ncols = 3 if len(kinds) == 1 else len(kinds)

    long, info = _numeric_data(df, cols, exclude=exclude)
    cells = [(row, k) for row in info.to_dict("records") for k in kinds]
    fig, axes = _make_grid(len(cells), ncols, ax, figsize, subplot_kws)
    values_by_col = {c: g["value"] for c, g in long.groupby("column", sort=False)}

    for axis, (row, k) in zip(axes, cells):
        col = row["column"]
        values = values_by_col.get(col, pd.Series(dtype="float64")).rename(col)
        if len(values):
            plotters[k](values, axis, bins, **kwargs)
        else:
            _no_data(axis, "không có giá trị hữu hạn")
        auto = f"{col}" + (f" · {k}" if len(kinds) > 1 else "") + f"\nn={row['n']:,} · missing={row['missing']:,}"
        if row["infinite"]:
            auto += f" · inf={row['infinite']:,}"
        _apply_labels(fig, axis, title=title, col_title=col_title, info={**row, "kind": k}, auto_title=auto)

    if ax is None:
        fig.tight_layout()
    return fig, long


# ======================================================================
# 2. DỮ LIỆU VẼ (NỘI BỘ)
# ======================================================================
def _numeric_data(df: pd.DataFrame, cols: Columns = None, *,
                  exclude: Sequence[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trả (long: column, value - giá trị hữu hạn ; info: column, n, missing, infinite).

    n = số giá trị hữu hạn được vẽ; missing = NaN/None; infinite = ±inf. Thứ tự cột theo lựa chọn.
    """
    selected = _select_columns(df, cols, exclude, auto=_is_numeric, explicit=_is_numeric, kind="số (không phải bool)")
    longs, infos = [], []
    for col in selected:
        s = df[col]
        missing = int(s.isna().sum())
        arr = s.dropna().to_numpy(dtype="float64")
        finite = arr[np.isfinite(arr)]
        infos.append({"column": col, "n": len(finite), "missing": missing, "infinite": len(arr) - len(finite)})
        longs.append(pd.DataFrame({"column": pd.Series([col] * len(finite), dtype=object), "value": finite}))
    long = pd.concat(longs, ignore_index=True)
    info = pd.DataFrame(infos, columns=["column", "n", "missing", "infinite"])
    return long, info


# ======================================================================
# 3. HỖ TRỢ (NỘI BỘ)
# ======================================================================
def _is_numeric(s: pd.Series) -> bool:
    return pdt.is_numeric_dtype(s) and not pdt.is_bool_dtype(s)


def _numeric_plotters() -> Mapping[str, Callable[..., Any]]:
    """Registry {kind: hàm vẽ(values, ax, bins, **kwargs)}: hist→histplot, box→boxplot,
    kde→kdeplot, violin→violinplot, ecdf→ecdfplot."""
    _, sns = _require_plotting()
    return {
        "hist": lambda v, ax, bins, **kw: sns.histplot(x=v, ax=ax, bins="auto" if bins is None else bins, **kw),
        "box": lambda v, ax, bins, **kw: sns.boxplot(x=v, ax=ax, **kw),
        "kde": lambda v, ax, bins, **kw: sns.kdeplot(x=v, ax=ax, **kw),
        "violin": lambda v, ax, bins, **kw: sns.violinplot(x=v, ax=ax, **kw),
        "ecdf": lambda v, ax, bins, **kw: sns.ecdfplot(x=v, ax=ax, **kw),
    }
