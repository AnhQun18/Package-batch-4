"""Data Quality: độ phủ entity (customer...) theo thời gian."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from pandas.api import types as pdt

from ._common import RED, DatasetError, _check_df, _check_rate, _finish, _make_ax, _no_data, _require_plotting

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["plot_entity_coverage"]


def plot_entity_coverage(df: pd.DataFrame, entity: str, time: str, *, freq: str = "M",
                         threshold: float | None = None, ax: Axes | None = None,
                         figsize: tuple[float, float] | None = None, title: str | None = None,
                         **kwargs: Any) -> tuple[Figure, pd.DataFrame]:
    """Số entity (vd customer) xuất hiện theo từng kỳ.

    entity: cột định danh. time: cột datetime. freq: mã tần suất pandas ("D", "W", "M", "Q", "Y"...).
    coverage = số entity có mặt trong kỳ / tổng số entity khác nhau của cả bảng.
    Kỳ trống giữa kỳ nhỏ nhất và lớn nhất vẫn có mặt với 0 (fill_gaps như eda).
    threshold: kỳ có coverage < ngưỡng (0-1) tô đỏ, vẽ vạch ngưỡng; None = không tô.
    title: None = tự động, "" = không tiêu đề. **kwargs chuyển cho seaborn.lineplot (vd color).
    Dòng thiếu entity hoặc time bị loại. Timezone bị bỏ.
    Trả về (Figure, bảng: period, n_entity, coverage, n_rows) theo thứ tự thời gian;
    có thêm cột bool `exceed` (coverage < threshold) khi truyền threshold.
    """
    _check_df(df)
    _check_rate("threshold", threshold)
    for name in (entity, time):
        if name not in df.columns:
            raise DatasetError(f"Cột không tồn tại: {name!r}. Cột hiện có: {list(df.columns)}.")
    if entity == time:
        raise DatasetError("entity và time phải là hai cột khác nhau.")
    if not pdt.is_datetime64_any_dtype(df[time]):
        raise DatasetError(f"Cột time '{time}' phải là datetime, nhận {df[time].dtype}.")
    try:
        pd.Period("2000-01-01", freq=freq)
    except Exception:
        raise DatasetError(f"freq {freq!r} không hợp lệ. Dùng mã tần suất pandas, vd 'D', 'W', 'M', 'Q', 'Y'.") from None

    data = df[[entity, time]].dropna()
    t = data[time].dt.tz_localize(None) if data[time].dt.tz is not None else data[time]
    total = int(data[entity].nunique())
    columns = ["period", "n_entity", "coverage", "n_rows"]

    if data.empty:
        table = pd.DataFrame({"period": pd.Series(dtype=object), "n_entity": pd.Series(dtype="int64"),
                              "coverage": pd.Series(dtype=float), "n_rows": pd.Series(dtype="int64")})
        if threshold is not None:
            table["exceed"] = pd.Series(dtype=bool)
    else:
        period = t.dt.to_period(freq)
        grouped = data.groupby(period.to_numpy(), sort=True)
        full = pd.period_range(period.min(), period.max(), freq=freq)
        n_entity = grouped[entity].nunique().reindex(full, fill_value=0)
        n_rows = grouped[entity].size().reindex(full, fill_value=0)
        table = pd.DataFrame({"period": full, "n_entity": n_entity.to_numpy(),
                              "coverage": n_entity.to_numpy() / total, "n_rows": n_rows.to_numpy()})[columns]
        if threshold is not None:
            table["exceed"] = table["coverage"] < threshold

    fig, axis = _make_ax(ax, figsize, (9.0, 4.0))
    if table.empty:
        _no_data(axis, "không có dữ liệu")
    else:
        _, sns = _require_plotting()
        x = table["period"].dt.to_timestamp()
        sns.lineplot(x=x, y=table["n_entity"], marker="o", ax=axis, **kwargs)
        if threshold is not None:
            axis.axhline(threshold * total, color=RED, linestyle="--", linewidth=1)
            low = table[table["exceed"]]
            if not low.empty:
                axis.scatter(low["period"].dt.to_timestamp(), low["n_entity"], color=RED, zorder=3)
        axis.set_ylim(bottom=0)
        axis.set_xlabel(f"kỳ ({freq})")
        axis.set_ylabel("số entity")
    _finish(fig, axis, title, f"Entity coverage · {entity} theo {time} · tổng {total:,} entity", created=ax is None)
    return fig, table
