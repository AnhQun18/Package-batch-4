"""EDA cho cột phân loại: plot_categorical (bar chart tần suất).

CẤU TRÚC:
    1. Vẽ (public)      plot_categorical
    2. Dữ liệu vẽ       _categorical_data   (test được không cần matplotlib)
    3. Hỗ trợ           _is_categorical_like, _rank

Quy ước chung xem docstring package `ds_kit.data_inspector.eda` và docs/interfaces/eda.md.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from ..dataset import DatasetError
from ._common import (Columns, _apply_labels, _make_grid, _no_data, _require_plotting, _select_columns)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["plot_categorical"]


# ======================================================================
# 1. VẼ
# ======================================================================
def plot_categorical(df: pd.DataFrame, cols: Columns = None, *, exclude: Sequence[str] | None = None,
                     kind: Literal["bar", "pie"] = "bar",
                     top_n: int | None = 20, normalize: bool = False, show_other: bool = True,
                     other_label: str = "(other)", sort: Literal["count", "value"] | None = "count",
                     orient: Literal["v", "h"] = "v", label_rotation: float | None = None, ncols: int = 3,
                     ax: Axes | Sequence[Axes] | None = None, figsize: tuple[float, float] | None = None,
                     subplot_kws: Mapping[str, Any] | None = None, title: str | None = None,
                     col_title: str | None = None, **kwargs: Any) -> tuple[Figure, pd.DataFrame]:
    """Chart tần suất, mỗi cột một ô: bar (seaborn.barplot) hoặc pie (matplotlib).

    Cột hợp lệ khi cols=None: dtype category, object, string, bool. Truyền cols tường minh thì nhận
        mọi cột có giá trị hash được (vd cột số nguyên mã hóa nhóm).
    kind: "bar" hoặc "pie". Pie: nhãn ghi tỷ lệ %, **kwargs chuyển cho Axes.pie (colors, startangle,
        wedgeprops...); normalize, orient, label_rotation không có tác dụng. Pie nên dùng top_n nhỏ (vd 5-8).
    top_n: số category mỗi ô, chọn theo tần suất lớn nhất; None = tất cả.
    normalize: False = trục là count, True = proportion (trên giá trị không thiếu).
    show_other / other_label: gộp phần ngoài top_n thành một cột/lát với nhãn `other_label` (pie tô xám).
    sort: thứ tự hiển thị: "count" giảm dần theo tần suất, "value" theo giá trị, None = thứ tự xuất hiện.
        Với pie luôn là "count" (bỏ qua giá trị truyền vào).
    orient: "v" bar đứng, "h" bar ngang. label_rotation: góc xoay nhãn category; None = mặc định.
    Trả về: (Figure, long) với long gồm `column, value, count, proportion` đúng như đã vẽ.
    Raises DatasetError: như plot_numeric; thêm top_n < 1, sort/orient sai, giá trị không hash được,
        other_label trùng một category có thật, nhãn category trùng nhau sau khi chuyển thành chuỗi.
    """
    if kind not in ("bar", "pie"):
        raise DatasetError(f"kind phải là 'bar' hoặc 'pie', nhận {kind!r}.")
    if orient not in ("v", "h"):
        raise DatasetError(f"orient phải là 'v' hoặc 'h', nhận {orient!r}.")
    _, sns = _require_plotting()
    if kind == "pie":
        sort = "count"  # pie luôn sắp theo tần suất giảm dần, (other) ở cuối
    long, info = _categorical_data(df, cols, exclude=exclude, top_n=top_n, show_other=show_other,
                                   other_label=other_label, sort=sort)
    measure = "proportion" if normalize else "count"
    fig, axes = _make_grid(len(info), ncols, ax, figsize, subplot_kws)
    by_col = {c: g for c, g in long.groupby("column", sort=False)}

    for axis, row in zip(axes, info.to_dict("records")):
        col = row["column"]
        sub = by_col.get(col)
        if sub is None or sub.empty:
            _no_data(axis, "không có giá trị")
        else:
            labels = sub["value"].map(str).tolist()
            if len(set(labels)) != len(labels):
                raise DatasetError(f"Cột '{col}': các category trùng nhãn sau khi chuyển thành chuỗi (vd 1 và '1'). "
                                   "Đồng nhất kiểu dữ liệu của cột trước.")
            if kind == "pie":
                kw = dict(kwargs)
                kw.setdefault("autopct", "%1.1f%%")
                kw.setdefault("startangle", 90)
                kw.setdefault("counterclock", False)
                wedges = axis.pie(sub["count"].to_numpy(), labels=labels, **kw)[0]
                if show_other and "colors" not in kw:
                    for w, v in zip(wedges, sub["value"]):
                        if v == other_label:
                            w.set_facecolor("lightgray")
                axis.set_aspect("equal")
            elif orient == "v":
                sns.barplot(x=labels, y=sub[measure].to_numpy(), order=labels, ax=axis, **kwargs)
                axis.set_ylabel(measure)
            else:
                sns.barplot(y=labels, x=sub[measure].to_numpy(), order=labels, orient="h", ax=axis, **kwargs)
                axis.set_xlabel(measure)
            if kind == "bar" and label_rotation is not None:
                axis.tick_params(axis="x" if orient == "v" else "y", labelrotation=label_rotation)
        auto = f"{col}\nn={row['n']:,} · missing={row['missing']:,} · categories={row['cardinality']:,}"
        _apply_labels(fig, axis, title=title, col_title=col_title, info=row, auto_title=auto)

    if ax is None:
        fig.tight_layout()
    return fig, long


# ======================================================================
# 2. DỮ LIỆU VẼ (NỘI BỘ)
# ======================================================================
def _categorical_data(df: pd.DataFrame, cols: Columns = None, *, exclude: Sequence[str] | None = None,
                      top_n: int | None = 20, show_other: bool = True, other_label: str = "(other)",
                      sort: Literal["count", "value"] | None = "count") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trả (long: column, value, count, proportion ; info: column, n, missing, cardinality).

    top_n chọn các category có tần suất lớn nhất; phần còn lại gộp thành một dòng `other_label` (nếu show_other);
    `sort` chỉ quyết định thứ tự hiển thị (dòng other luôn ở cuối). proportion = count / số giá trị không thiếu.
    n = số giá trị không thiếu; cardinality = số category quan sát được.
    """
    if top_n is not None and (isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1):
        raise DatasetError(f"top_n phải là số nguyên >= 1 hoặc None, nhận {top_n!r}.")
    if sort not in ("count", "value", None):
        raise DatasetError(f"sort phải là 'count', 'value' hoặc None, nhận {sort!r}.")
    selected = _select_columns(df, cols, exclude, auto=_is_categorical_like, explicit=lambda s: True,
                               kind="phân loại (category/object/string/bool)")
    longs, infos = [], []
    for col in selected:
        s = df[col]
        missing = int(s.isna().sum())
        v = s.dropna()
        if isinstance(v.dtype, pd.CategoricalDtype):
            v = v.astype(object)  # tránh value_counts tự thêm category chưa xuất hiện
        try:
            counts = v.value_counts(sort=False)
        except TypeError:
            raise DatasetError(f"Cột '{col}' chứa giá trị không hash được (list, dict, ...), không đếm được tần suất.") from None
        n, cardinality = int(counts.sum()), len(counts)
        infos.append({"column": col, "n": n, "missing": missing, "cardinality": cardinality})

        ranked = _rank(counts)
        rest = 0
        if top_n is not None and len(ranked) > top_n:
            rest = int(ranked.iloc[top_n:].sum())
            ranked = ranked.iloc[:top_n]
            if show_other and other_label in counts.index:
                raise DatasetError(f"Cột '{col}' đã có category tên {other_label!r}. Đặt other_label khác.")
        if sort == "value":
            try:
                ranked = ranked.sort_index(kind="stable")
            except TypeError:
                ranked = ranked.iloc[np.argsort([str(i) for i in ranked.index], kind="stable")]
        elif sort is None:
            keep = set(ranked.index)
            ranked = counts[[i for i in counts.index if i in keep]]
        values, c = list(ranked.index), ranked.to_numpy(dtype="int64").tolist()
        if show_other and rest:
            values.append(other_label)
            c.append(rest)
        c = np.asarray(c, dtype="int64")
        longs.append(pd.DataFrame({
            "column": pd.Series([col] * len(c), dtype=object), "value": pd.Series(values, dtype=object),
            "count": c, "proportion": c / n if n else np.full(len(c), np.nan)}))
    long = pd.concat(longs, ignore_index=True)
    info = pd.DataFrame(infos, columns=["column", "n", "missing", "cardinality"])
    return long, info


# ======================================================================
# 3. HỖ TRỢ (NỘI BỘ)
# ======================================================================
def _is_categorical_like(s: pd.Series) -> bool:
    return (isinstance(s.dtype, pd.CategoricalDtype) or pdt.is_bool_dtype(s)
            or pdt.is_object_dtype(s) or pdt.is_string_dtype(s))


def _rank(counts: pd.Series) -> pd.Series:
    """Sắp count giảm dần, đồng hạng theo giá trị tăng dần (không so sánh được thì theo chuỗi)."""
    try:
        counts = counts.sort_index(kind="stable")
    except TypeError:
        counts = counts.iloc[np.argsort([str(i) for i in counts.index], kind="stable")]
    return counts.sort_values(ascending=False, kind="stable")
