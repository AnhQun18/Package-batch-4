"""Hỗ trợ dùng chung cho các hàm qc: chọn cột, kiểm tra tham số, import matplotlib, vẽ bar tỷ lệ.

Nội bộ của package `qc`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

import pandas as pd

try:  # khi nằm cạnh eda trong ds_kit dùng chung lỗi của dataset
    from ..dataset import DatasetError
except ImportError:  # chạy độc lập
    class DatasetError(ValueError):
        """Đầu vào không hợp lệ."""

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

Columns = str | Sequence[str] | None

RED = "tab:red"


def _check_df(df: Any) -> None:
    if not isinstance(df, pd.DataFrame):
        raise DatasetError(f"Cần pandas DataFrame, nhận {type(df).__name__}.")
    if df.columns.has_duplicates:
        raise DatasetError(f"DataFrame có tên cột trùng: {df.columns[df.columns.duplicated()].tolist()}. Đổi tên trước.")


def _as_list(x: Any) -> list:
    return list(x) if isinstance(x, (list, tuple, pd.Index)) else [x]


def _select_columns(df: Any, cols: Columns, exclude: Sequence[str] | None,
                    auto: Callable[[pd.Series], bool] = lambda s: True, kind: str = "hợp lệ") -> list[str]:
    """cols=None -> mọi cột thỏa `auto`; cols tường minh -> kiểm tra tồn tại và thỏa `auto`; rồi bỏ `exclude`."""
    _check_df(df)
    if cols is None:
        selected = [c for c in df.columns if auto(df[c])]
    else:
        names = _as_list(cols)
        if not names:
            raise DatasetError("cols rỗng. Dùng cols=None để chọn mọi cột hợp lệ.")
        dup = sorted({str(c) for c in names if names.count(c) > 1})
        if dup:
            raise DatasetError(f"cols bị trùng: {dup}.")
        absent = [c for c in names if c not in df.columns]
        if absent:
            raise DatasetError(f"Cột không tồn tại: {absent}. Cột hiện có: {list(df.columns)}.")
        bad = [f"{c} ({df[c].dtype})" for c in names if not auto(df[c])]
        if bad:
            raise DatasetError(f"Cột không phải kiểu {kind}: {bad}.")
        selected = names
    if exclude is not None:
        ex = _as_list(exclude)
        absent = [c for c in ex if c not in df.columns]
        if absent:
            raise DatasetError(f"exclude có cột không tồn tại: {absent}.")
        selected = [c for c in selected if c not in ex]
    if not selected:
        raise DatasetError(f"Không có cột kiểu {kind} nào. Kiểm tra dtype hoặc truyền cols/exclude khác.")
    return selected


def _check_rate(name: str, value: Any) -> None:
    """Tham số tỷ lệ: None hoặc số trong [0, 1]."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise DatasetError(f"{name} phải là số trong [0, 1] (tỷ lệ) hoặc None, nhận {value!r}.")


def _require_plotting() -> tuple[Any, Any]:
    """Import (matplotlib.pyplot, seaborn); thiếu -> ImportError hướng dẫn cài."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as e:
        raise ImportError("Hàm vẽ cần matplotlib và seaborn. Cài bằng: pip install -e \".[viz]\" "
                          "(hoặc pip install matplotlib seaborn).") from e
    return plt, sns


def _make_ax(ax: Axes | None, figsize: tuple[float, float] | None,
             default_size: tuple[float, float]) -> tuple[Figure, Axes]:
    """Dùng Axes được truyền vào hoặc tạo figure một ô."""
    plt, _ = _require_plotting()
    from matplotlib.axes import Axes as _Axes
    if ax is not None:
        if figsize is not None:
            raise DatasetError("figsize chỉ dùng khi ax=None; với ax có sẵn hãy chỉnh figure của bạn.")
        if not isinstance(ax, _Axes):
            raise DatasetError("ax phải là một matplotlib Axes.")
        return ax.figure, ax
    fig, new = plt.subplots(figsize=figsize or default_size)
    return fig, new


def _no_data(ax: Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _finish(fig: Figure, ax: Axes, title: str | None, auto_title: str, created: bool) -> None:
    """title: None = tự động, "" = không tiêu đề, chuỗi = dùng chuỗi đó."""
    ax.set_title(auto_title if title is None else title)
    if created:
        fig.tight_layout()


def _bar_size(n: int) -> tuple[float, float]:
    return (7.0, max(2.5, 0.35 * n + 1.5))


def _rate_bar(ax: Axes, labels: Sequence[str], rates: Sequence[float], threshold: float | None,
              xlabel: str, **kwargs: Any) -> None:
    """Bar ngang theo tỷ lệ; thanh có rate > threshold tô đỏ (threshold=None: không tô)."""
    _, sns = _require_plotting()
    from matplotlib.ticker import PercentFormatter
    labels = [str(x) for x in labels]
    rates = [float(r) for r in rates]
    sns.barplot(x=rates, y=labels, order=labels, orient="h", ax=ax, **kwargs)
    if threshold is not None:
        for patch, r in zip(ax.containers[0], rates):
            if r > threshold:
                patch.set_facecolor(RED)
        ax.axvline(threshold, color=RED, linestyle="--", linewidth=1)
    ax.bar_label(ax.containers[0], labels=[f"{r:.1%}" for r in rates], padding=3)
    top = max(rates + ([threshold] if threshold is not None else []) + [0.01])
    ax.set_xlim(0, min(1.0, top * 1.15) if top <= 0.87 else 1.0)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
