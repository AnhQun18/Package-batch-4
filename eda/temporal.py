"""EDA cho cột thời gian: plot_temporal (line, bar, cumulative, weekday, month, hour, heatmap).

CẤU TRÚC:
    1. Vẽ (public)      plot_temporal
    2. Dữ liệu vẽ       _temporal_data, _temporal_frame   (test được không cần matplotlib)
    3. Hỗ trợ           _is_datetime_or_int, _check_freq, _temporal_plotters

Quy ước chung xem docstring package `ds_kit.data_inspector.eda` và docs/interfaces/eda.md.
"""
from __future__ import annotations

import calendar
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from ..dataset import DatasetError
from ._common import (Columns, _apply_labels, _make_grid, _no_data, _require_plotting, _select_columns)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["plot_temporal"]

_DATETIME_ONLY_KINDS = ("weekday", "month", "hour", "heatmap")

# ======================================================================
# 1. VẼ
# ======================================================================
def plot_temporal(df: pd.DataFrame, cols: Columns = None, *, exclude: Sequence[str] | None = None,
                  kind: str | Sequence[str] = "line", freq: str = "D", fill_gaps: bool = True,
                  max_xticks: int | None = 12, ncols: int | None = None,
                  ax: Axes | Sequence[Axes] | None = None, figsize: tuple[float, float] | None = None,
                  subplot_kws: Mapping[str, Any] | None = None, title: str | None = None,
                  col_title: str | None = None, **kwargs: Any) -> tuple[Figure, pd.DataFrame]:
    """Số record theo thời gian hoặc theo chu kỳ; một hàm cho mọi loại chart thời gian.

    Cột hợp lệ khi cols=None: dtype datetime. Truyền cols tường minh thì nhận thêm số nguyên (vd năm).
    kind: một hoặc nhiều tên chart trong registry `_temporal_plotters()`:
        theo dòng thời gian (cột datetime hoặc số nguyên): "line", "bar", "cumulative";
        theo chu kỳ (chỉ cột datetime): "weekday", "month", "hour", "heatmap" (thứ trong tuần x giờ).
        Mỗi cặp (cột, kind) là một ô. Nhiều kind -> mỗi cột một hàng.
    freq: độ mịn gom của line/bar/cumulative với cột datetime, mã tần suất pandas ("D", "W", "M", "Q", "Y", "h"...);
        cột số nguyên: mỗi giá trị là một điểm, bỏ qua freq.
    fill_gaps: True = period datetime không có record hiện count = 0 (không áp dụng cho cột số nguyên).
    max_xticks: số nhãn tối đa trên trục x của kind "bar"; None = không giới hạn.
    ncols: số ô mỗi hàng; None = 1 nếu một kind, = số kind nếu nhiều kind.
    Trả về: (Figure, long) với long gồm `column, kind, x, y, count`:
        line/bar: x = period (pandas Period hoặc số nguyên); cumulative: count là số tích lũy;
        weekday: x = 0..6 (thứ Hai = 0); month: x = 1..12; hour: x = 0..23; heatmap: x = thứ, y = giờ.
        y chỉ có giá trị với heatmap.
    Raises DatasetError: như plot_numeric; thêm dtype không hợp lệ, freq sai, kind theo chu kỳ dùng với cột số nguyên.
    """
    plotters = _temporal_plotters()
    kinds = list(kind) if isinstance(kind, (list, tuple)) else [kind]
    _check_freq(freq)
    if max_xticks is not None and (isinstance(max_xticks, bool) or not isinstance(max_xticks, int) or max_xticks < 1):
        raise DatasetError(f"max_xticks phải là số nguyên >= 1 hoặc None, nhận {max_xticks!r}.")
    if ncols is None:
        ncols = 1 if len(kinds) == 1 else len(kinds)

    long, info = _temporal_data(df, cols, exclude=exclude, kinds=kinds, freq=freq, fill_gaps=fill_gaps)
    cells = [(row, k) for row in info.to_dict("records") for k in kinds]
    fig, axes = _make_grid(len(cells), ncols, ax, figsize, subplot_kws)
    groups = {key: g for key, g in long.groupby(["column", "kind"], sort=False)}

    for axis, (row, k) in zip(axes, cells):
        col = row["column"]
        frame = groups.get((col, k))
        if frame is None or frame.empty or row["n"] == 0:
            _no_data(axis, "không có giá trị")
        else:
            plotters[k](frame, axis, max_xticks, **kwargs)
        auto = f"{col}" + (f" · {k}" if len(kinds) > 1 else "")
        if k in ("line", "bar", "cumulative") and row["is_datetime"]:
            auto += f" · freq={freq}"
        auto += f"\nn={row['n']:,} · missing={row['missing']:,}"
        _apply_labels(fig, axis, title=title, col_title=col_title, info={**row, "kind": k}, auto_title=auto)

    if ax is None:
        fig.tight_layout()
    return fig, long


# ======================================================================
# 2. DỮ LIỆU VẼ (NỘI BỘ)
# ======================================================================
def _temporal_data(df: pd.DataFrame, cols: Columns = None, *, exclude: Sequence[str] | None = None,
                   kinds: Sequence[str] = ("line",), freq: str = "D",
                   fill_gaps: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trả (long: column, kind, x, y, count ; info: column, n, missing, is_datetime).

    n = số giá trị không thiếu. Xem plot_temporal cho ý nghĩa của x, y theo từng kind.
    """
    selected = _select_columns(df, cols, exclude, auto=pdt.is_datetime64_any_dtype, explicit=_is_datetime_or_int,
                               kind="thời gian (datetime hoặc số nguyên)")
    longs, infos = [], []
    for col in selected:
        s = df[col]
        is_dt = bool(pdt.is_datetime64_any_dtype(s))
        bad = [k for k in kinds if k in _DATETIME_ONLY_KINDS] if not is_dt else []
        if bad:
            raise DatasetError(f"Cột '{col}' (dtype {s.dtype}) là số nguyên, không dùng được kind {bad}: "
                               "các kind theo chu kỳ cần cột datetime.")
        v = s.dropna()
        if is_dt and v.dt.tz is not None:
            v = v.dt.tz_localize(None)
        infos.append({"column": col, "n": len(v), "missing": int(s.isna().sum()), "is_datetime": is_dt})
        if v.empty:
            continue
        for k in kinds:
            longs.append(_temporal_frame(v, k, is_dt, freq, fill_gaps).assign(column=col, kind=k))
    cols_out = ["column", "kind", "x", "y", "count"]
    long = pd.concat(longs, ignore_index=True)[cols_out] if longs else pd.DataFrame(
        {"column": pd.Series(dtype=object), "kind": pd.Series(dtype=object), "x": pd.Series(dtype=object),
         "y": pd.Series(dtype=object), "count": pd.Series(dtype="int64")})
    info = pd.DataFrame(infos, columns=["column", "n", "missing", "is_datetime"])
    return long, info


def _temporal_frame(v: pd.Series, kind: str, is_dt: bool, freq: str, fill_gaps: bool) -> pd.DataFrame:
    """Bảng x, y, count của một cột (đã bỏ thiếu, đã bỏ timezone) cho một kind."""
    def frame(x: Sequence[Any], count: Sequence[int], y: Sequence[Any] | None = None) -> pd.DataFrame:
        return pd.DataFrame({"x": pd.Series(list(x), dtype=object),
                             "y": pd.Series(list(y) if y is not None else [None] * len(x), dtype=object),
                             "count": np.asarray(count, dtype="int64")})

    if kind in ("line", "bar", "cumulative"):
        if is_dt:
            vc = v.dt.to_period(freq).value_counts().sort_index()
            if fill_gaps:
                vc = vc.reindex(pd.period_range(vc.index.min(), vc.index.max(), freq=freq), fill_value=0)
        else:
            vc = v.value_counts().sort_index()
        count = vc.cumsum() if kind == "cumulative" else vc
        return frame(vc.index, count.to_numpy())
    if kind == "weekday":
        vc = v.dt.dayofweek.value_counts().reindex(range(7), fill_value=0)
        return frame(vc.index, vc.to_numpy())
    if kind == "month":
        vc = v.dt.month.value_counts().reindex(range(1, 13), fill_value=0)
        return frame(vc.index, vc.to_numpy())
    if kind == "hour":
        vc = v.dt.hour.value_counts().reindex(range(24), fill_value=0)
        return frame(vc.index, vc.to_numpy())
    if kind == "heatmap":
        grid = pd.MultiIndex.from_product([range(7), range(24)])
        vc = pd.Series(1, index=pd.MultiIndex.from_arrays([v.dt.dayofweek, v.dt.hour])).groupby(level=[0, 1]).sum()
        vc = vc.reindex(grid, fill_value=0)
        return frame(vc.index.get_level_values(0), vc.to_numpy(), y=vc.index.get_level_values(1))
    raise DatasetError(f"kind {kind!r} không hỗ trợ.")  # không xảy ra: kind đã được kiểm tra


# ======================================================================
# 3. HỖ TRỢ (NỘI BỘ)
# ======================================================================
def _is_datetime_or_int(s: pd.Series) -> bool:
    return bool(pdt.is_datetime64_any_dtype(s) or (pdt.is_integer_dtype(s) and not pdt.is_bool_dtype(s)))


def _check_freq(freq: Any) -> None:
    try:
        pd.Period("2000-01-01", freq=freq)
    except Exception:
        raise DatasetError(f"freq {freq!r} không hợp lệ. Dùng mã tần suất pandas, vd 'D', 'W', 'M', 'Q', 'Y', 'h'.") from None


def _temporal_plotters() -> Mapping[str, Callable[..., Any]]:
    """Registry {kind: hàm vẽ(frame, ax, max_xticks, **kwargs)} với frame = các dòng x, y, count của kind đó:
    line/cumulative→lineplot, bar→barplot, weekday/month/hour→barplot nhãn lịch, heatmap→heatmap."""
    from matplotlib.ticker import MaxNLocator
    _, sns = _require_plotting()

    def line(frame: pd.DataFrame, ax: Axes, max_xticks: int | None, **kw: Any) -> None:
        x = frame["x"].tolist()
        xs = pd.PeriodIndex(x).to_timestamp() if isinstance(x[0], pd.Period) else np.asarray(x)
        sns.lineplot(x=xs, y=frame["count"].to_numpy(), ax=ax, **kw)

    def bar(frame: pd.DataFrame, ax: Axes, max_xticks: int | None, **kw: Any) -> None:
        labels = [str(x) for x in frame["x"]]
        sns.barplot(x=labels, y=frame["count"].to_numpy(), order=labels, ax=ax, **kw)
        if max_xticks is not None:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=max_xticks, integer=True))

    def cyclic(names: Callable[[int], str]) -> Callable[..., None]:
        def draw(frame: pd.DataFrame, ax: Axes, max_xticks: int | None, **kw: Any) -> None:
            labels = [names(int(x)) for x in frame["x"]]
            sns.barplot(x=labels, y=frame["count"].to_numpy(), order=labels, ax=ax, **kw)
        return draw

    def heatmap(frame: pd.DataFrame, ax: Axes, max_xticks: int | None, **kw: Any) -> None:
        grid = frame.assign(y=frame["y"].astype(int), x=frame["x"].astype(int)).pivot(index="y", columns="x", values="count")
        kw.setdefault("xticklabels", [calendar.day_abbr[int(i)] for i in grid.columns])
        kw.setdefault("yticklabels", [str(int(i)) for i in grid.index])
        sns.heatmap(grid, ax=ax, **kw)
        ax.invert_yaxis()

    return {
        "line": line, "bar": bar, "cumulative": line,
        "weekday": cyclic(lambda i: calendar.day_abbr[i]),
        "month": cyclic(lambda i: calendar.month_abbr[i]),
        "hour": cyclic(str),
        "heatmap": heatmap,
    }
