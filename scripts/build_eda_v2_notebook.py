"""Sinh notebook kiểm thử package `eda_v2` trên data/mock_transactions.csv.

Chạy: python scripts/build_eda_v2_notebook.py  ->  notebooks/04_test_eda_v2.ipynb
"""
from pathlib import Path

from build_notebooks import build_notebook, create_cell

SETUP = """import sys
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # không phụ thuộc backend của kernel; figure được hiện bằng show(fig)
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, display

repo_root = Path.cwd().resolve()
if not (repo_root / "eda_v2").exists():  # đang chạy từ notebooks/
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from eda_v2 import (plot_category_distribution, plot_numeric_comparison, plot_numeric_distribution,
                    plot_entity_coverage, plot_temporal_distribution)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


def show(fig):
    # Hiện figure dưới dạng ảnh PNG (chạy đúng với mọi backend), rồi đóng figure
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    display(Image(buf.getvalue()))
"""


def md(text):
    return create_cell("markdown", text)


def code(text):
    return create_cell("code", text)


def generate_notebook(out_path: Path):
    cells = [
        md("""# Kiểm thử package `eda_v2` trên `data/mock_transactions.csv`

Năm hàm, mỗi hàm trả về `(figure, bảng số liệu)`; notebook xem figure bằng `show(fig)` và in bảng ra để đối chiếu.

| Hàm | Dùng cho |
|---|---|
| `plot_category_distribution` | cột phân loại: top K, nhóm Other, count, percentage |
| `plot_numeric_comparison` | so sánh phân phối cột số giữa các nhóm, tuỳ chọn ẩn outlier |
| `plot_numeric_distribution` | histogram, mean, median, percentile |
| `plot_entity_coverage` | hoạt động của entity theo thời gian |
| `plot_temporal_distribution` | aggregate, resample theo thời gian |
"""),
        code(SETUP),
        code("""df = pd.read_csv(repo_root / "data" / "mock_transactions.csv", parse_dates=["order_date"])
print(f"{df.shape[0]:,} dòng x {df.shape[1]} cột")
display(df.head())
display(df.dtypes.to_frame("dtype").T)
"""),
        md("""---
## 1. `plot_category_distribution`

Mặc định `top_k=10`, các nhóm ít gặp hơn được gộp vào `Other`.
"""),
        code("""fig, table = plot_category_distribution(df, "payment_method")
show(fig)
display(table)
"""),
        md("Ép `top_k=2` để thấy nhóm `Other` xuất hiện; tổng `percentage` luôn bằng 100."),
        code("""fig, table = plot_category_distribution(df, "shipping_region", top_k=2, other_label="Khác", figsize=(10, 4))
show(fig)
display(table)
assert abs(table["percentage"].sum() - 100) < 1e-9
assert table["count"].sum() == df["shipping_region"].notna().sum()
"""),
        md("`top_k=None` giữ mọi nhóm, không gộp."),
        code("""fig, table = plot_category_distribution(df, "order_status", top_k=None)
show(fig)
assert "Other" not in table["category"].tolist()
"""),
        md("""---
## 2. `plot_numeric_comparison`

So sánh `total_amount` giữa các phương thức thanh toán; bảng là `describe()` theo nhóm.
"""),
        code("""fig, table = plot_numeric_comparison(df, "total_amount", "payment_method")
show(fig)
display(table)
"""),
        md("`hide_outliers=True` bỏ các điểm outlier khỏi chart (mặc định `False` = hiện); bảng `describe()` vẫn tính trên toàn bộ dữ liệu."),
        code("""fig, table = plot_numeric_comparison(df, "total_amount", "payment_method", hide_outliers=True)
show(fig)
"""),
        code("""fig, table = plot_numeric_comparison(df, "delivery_days", "shipping_region", top_k=2)
show(fig)
display(table)
assert len(table) <= 2
"""),
        md("""---
## 3. `plot_numeric_distribution`

Histogram kèm mean (đỏ), median (xanh), percentile (xám). `total_amount` lệch phải, nên mean > median.
"""),
        code("""fig, stats = plot_numeric_distribution(df, "total_amount")
show(fig)
display(stats)
"""),
        code("""fig, stats = plot_numeric_distribution(df, "customer_age", bins=15, percentiles=(10, 50, 90))
show(fig)
display(stats)
"""),
        md("`metric_with_inf` chứa ±inf: hàm phải bỏ chúng trước khi tính, nếu không mean/median sẽ là inf/nan."),
        code("""fig, stats = plot_numeric_distribution(df, "metric_with_inf", percentiles=())
show(fig)
display(stats)
assert stats.notna().all() and stats.abs().lt(float("inf")).all()
"""),
        md("""---
## 4. `plot_entity_coverage`

Heatmap số dòng của mỗi `customer_id` theo kỳ. Hàng = entity (sắp theo mức hoạt động), cột = kỳ.
"""),
        code("""fig, table = plot_entity_coverage(df, "customer_id", "order_date", freq="M", top_k=30)
show(fig)
display(table.iloc[:5, :6])
assert table.shape[0] <= 30
"""),
        md("Kỳ theo tuần và chỉ top 10 entity."),
        code("""fig, table = plot_entity_coverage(df, "customer_id", "order_date", freq="W", top_k=10, figsize=(16, 5))
show(fig)
print(table.shape)
"""),
        md("""---
## 5. `plot_temporal_distribution`

Không truyền `value_column` thì đếm số dòng; có `value_column` thì aggregate theo `agg`. Lưu ý resample theo tháng dùng `freq="ME"` (pandas >= 2.2).
"""),
        code("""fig, series = plot_temporal_distribution(df, "order_date", freq="W")
show(fig)
display(series.head())
assert series.sum() == df["order_date"].notna().sum()
"""),
        code("""fig, series = plot_temporal_distribution(df, "order_date", "total_amount", freq="ME", agg="sum")
show(fig)
display(series.head())
"""),
        code("""fig, series = plot_temporal_distribution(df, "order_date", "total_amount", freq="ME", agg="mean")
show(fig)
display(series.head())
"""),
        md("""---
## 6. Không sửa dữ liệu đầu vào

Sau mọi lần gọi ở trên, `df` phải giữ nguyên.
"""),
        code("""df_after = pd.read_csv(repo_root / "data" / "mock_transactions.csv", parse_dates=["order_date"])
pd.testing.assert_frame_equal(df, df_after)
print("df không bị sửa.")
"""),
    ]
    build_notebook(cells, out_path)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    generate_notebook(root / "notebooks" / "04_test_eda_v2.ipynb")
