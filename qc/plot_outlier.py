"""Data Quality: tỷ lệ outlier theo cột số."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from ._common import (Columns, DatasetError, _bar_size, _check_rate, _finish, _make_ax, _no_data,
                      _rate_bar, _select_columns)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["plot_outlier_rate"]


# Mỗi phương pháp: (hàm (series, k) -> (lower, upper), k mặc định). Thêm phương pháp = thêm một dòng.
# Độ rộng bằng 0 (std/MAD = 0) -> trả (nan, nan): không có outlier.
def _iqr(v: pd.Series, k: float) -> tuple[float, float]:
    q1, q3 = v.quantile(0.25), v.quantile(0.75)
    return q1 - k * (q3 - q1), q3 + k * (q3 - q1)


def _zscore(v: pd.Series, k: float) -> tuple[float, float]:
    std = v.std(ddof=0)
    return (np.nan, np.nan) if std == 0 else (v.mean() - k * std, v.mean() + k * std)


def _mad(v: pd.Series, k: float) -> tuple[float, float]:  # modified z-score: 0.6745 * (x - median) / MAD
    med = v.median()
    mad = (v - med).abs().median()
    return (np.nan, np.nan) if mad == 0 else (med - k * mad / 0.6745, med + k * mad / 0.6745)


_METHODS: dict[str, tuple[Callable[[pd.Series, float], tuple[float, float]], float]] = {
    "iqr": (_iqr, 1.5),
    "zscore": (_zscore, 3.0),
    "mad": (_mad, 3.5),
}


def plot_outlier_rate(df: pd.DataFrame, cols: Columns = None, *, exclude: Sequence[str] | None = None,
                      method: str = "iqr", k: float | None = None, min_rate: float | None = None,
                      threshold: float | None = None, ax: Axes | None = None,
                      figsize: tuple[float, float] | None = None, title: str | None = None,
                      **kwargs: Any) -> tuple[Figure, pd.DataFrame]:
    """Tỷ lệ outlier của từng cột số.

    cols/exclude: chọn cột (None = mọi cột số, không tính bool).
    method: "iqr" (ngoài [Q1-k*IQR, Q3+k*IQR]), "zscore" (ngoài mean ± k*std),
        "mad" (modified z-score, ngoài median ± k*MAD/0.6745).
    k: hệ số của phương pháp; None = mặc định theo method (iqr 1.5, zscore 3, mad 3.5).
    min_rate: chỉ hiển thị cột có outlier_rate >= mức này (0-1); None = hiện hết.
    threshold: cột có outlier_rate > ngưỡng (0-1) tô đỏ; None = không tô.
    title: None = tự động, "" = không tiêu đề. **kwargs chuyển cho seaborn.barplot (vd color).
    Giá trị thiếu và ±inf bị loại khỏi tính toán. Cột có std/MAD = 0 (zscore/mad): lower, upper = NaN, 0 outlier.
    Trả về (Figure, bảng: column, n, n_outlier, outlier_rate, lower, upper) sắp theo outlier_rate giảm dần;
    có thêm cột bool `exceed` khi truyền threshold. n = số giá trị hữu hạn.
    """
    if method not in _METHODS:
        raise DatasetError(f"method {method!r} không hỗ trợ. Chọn trong {list(_METHODS)}.")
    func, default_k = _METHODS[method]
    if k is None:
        k = default_k
    elif isinstance(k, bool) or not isinstance(k, (int, float)) or k <= 0:
        raise DatasetError(f"k phải là số > 0, nhận {k!r}.")
    _check_rate("min_rate", min_rate)
    _check_rate("threshold", threshold)
    selected = _select_columns(df, cols, exclude,
                               auto=lambda s: pdt.is_numeric_dtype(s) and not pdt.is_bool_dtype(s)
                               and not pdt.is_complex_dtype(s), kind="số")

    rows = []
    for col in selected:
        v = pd.to_numeric(df[col], errors="coerce").astype(float)
        v = v[np.isfinite(v)]
        if v.empty:
            rows.append({"column": col, "n": 0, "n_outlier": 0, "outlier_rate": 0.0, "lower": np.nan, "upper": np.nan})
            continue
        lower, upper = func(v, k)
        n_out = int(((v < lower) | (v > upper)).sum())  # so sánh với NaN là False
        rows.append({"column": col, "n": len(v), "n_outlier": n_out, "outlier_rate": n_out / len(v),
                     "lower": lower, "upper": upper})
    table = pd.DataFrame(rows, columns=["column", "n", "n_outlier", "outlier_rate", "lower", "upper"])
    if threshold is not None:
        table["exceed"] = table["outlier_rate"] > threshold
    if min_rate is not None:
        table = table[table["outlier_rate"] >= min_rate]
    table = table.sort_values("outlier_rate", ascending=False, kind="stable").reset_index(drop=True)

    fig, axis = _make_ax(ax, figsize, _bar_size(len(table)))
    if table.empty:
        _no_data(axis, "không có cột nào thỏa min_rate")
    else:
        _rate_bar(axis, table["column"], table["outlier_rate"], threshold, "tỷ lệ outlier", **kwargs)
    _finish(fig, axis, title, f"Outlier theo cột · {method}, k={k:g}", created=ax is None)
    return fig, table
