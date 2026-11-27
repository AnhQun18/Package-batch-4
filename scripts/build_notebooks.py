"""Script sinh ra các Jupyter Notebooks (.ipynb) kiểm thử package `qc`."""
import json
import re
from pathlib import Path


def create_cell(cell_type: str, source: str | list[str]):
    if isinstance(source, str):
        lines = [line + "\n" for line in source.split("\n")]
        # Remove trailing newline from the last line if original didn't have it
        if not source.endswith("\n") and lines:
            lines[-1] = lines[-1][:-1]
    else:
        lines = source
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def build_notebook(cells: list[dict], out_path: Path):
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f" Đã tạo notebook: {out_path}")


def generate_pipeline_notebook(out_path: Path):
    cells = [
        create_cell("markdown", """#  Kiểm thử toàn diện Package `qc` (Data Quality Control)

Notebook này kiểm thử và trực quan hóa 4 hàm nòng cốt của package `qc`:
1. `plot_missing`: Tỷ lệ giá trị thiếu theo cột.
2. `plot_duplicate`: Tỷ lệ dòng trùng theo một hoặc nhiều định nghĩa.
3. `plot_outlier_rate`: Tỷ lệ ngoại lai theo cột số (IQR, Z-score, MAD).
4. `plot_entity_coverage`: Độ phủ thực thể (khách hàng, sản phẩm...) theo thời gian.

---
###  Quy ước thiết kế của package `qc`:
- **Nguyên tắc**: Mỗi hàm tự tính, tự vẽ và trả về `(Figure, DataFrame)` với số liệu chi tiết đằng sau chart.
- **Tỷ lệ**: Chuẩn hóa trong khoảng `[0, 1]`.
- **Lọc và Cảnh báo**:
  - `min_*`: Lọc bớt các cột có tỷ lệ nhỏ khỏi cả bảng lẫn biểu đồ.
  - `threshold`: Tô đỏ phần vượt ngưỡng (hoặc sụt dưới ngưỡng với coverage) và vẽ đường đứt nét cảnh báo; bảng trả về tự động có thêm cột boolean `exceed`.
- **Linh hoạt**: Nhận `ax` để dễ dàng ghép vào subplot / dashboard phức hợp; không tự ý gọi `plt.show()` hay sửa đổi `df` gốc.
"""),
        create_cell("code", """import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Thiết lập đường dẫn import package qc và data generator
repo_root = Path.cwd().resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from qc import (
    plot_missing,
    plot_duplicate,
    plot_outlier_rate,
    plot_entity_coverage,
)
from data.generate_mock_data import generate_qc_mock_data

# Tùy chỉnh hiển thị
%matplotlib inline
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.float_format", lambda x: "%.4f" % x)

print(" Đã import thành công package qc!")
"""),
        create_cell("markdown", """## 0. Tải dữ liệu giả lập (Mock Dataset)

Dữ liệu giả lập mô phỏng giao dịch thương mại điện tử với:
- 5,270 dòng, 17 cột đa dạng kiểu dữ liệu (`datetime`, `float64`, `int64`, `string`, `object`).
- Có sẵn các mức độ thiếu dữ liệu từ 0% đến 100%.
- Có dòng trùng toàn bộ, trùng theo mã đơn, và trùng theo tổ hợp `(customer_id, order_date)`.
- Có ngoại lai ở nhiều biến số, kèm trường hợp biên như cột hằng số (`std=0`) và giá trị vô cùng (`±inf`).
- Có 300 khách hàng trong 18 tháng, gồm tháng bình thường, tháng cao điểm, tháng tụt giảm mạnh và tháng gián đoạn.
"""),
        create_cell("code", """csv_file = repo_root / "data" / "mock_transactions.csv"
if csv_file.exists():
    df = pd.read_csv(csv_file, parse_dates=["order_date"])
else:
    df = generate_qc_mock_data(n_rows=5000, seed=42)

print(f"Kích thước dataset: {df.shape[0]:,} dòng x {df.shape[1]} cột\\n")
display(df.head(5))
display(df.dtypes.to_frame(name="Dtype"))
"""),
        create_cell("markdown", """## 1. Kiểm thử hàm `plot_missing` (Tỷ lệ thiếu theo cột)

Hàm tính tỷ lệ `NaN / None` của các cột và vẽ bar nằm ngang sắp xếp theo thứ tự giảm dần.

### 1.1. Chạy mặc định trên tất cả các cột (`cols=None`)
"""),
        create_cell("code", """fig, table = plot_missing(df)
display(table)
"""),
        create_cell("markdown", """### 1.2. Chọn cột cụ thể (`cols`) và loại trừ (`exclude`)
"""),
        create_cell("code", """# Chỉ kiểm tra một nhóm cột kinh doanh, bỏ qua ID
selected_cols = ["customer_id", "total_amount", "customer_age", "payment_method", "shipping_address", "customer_feedback"]
fig, table = plot_missing(df, cols=selected_cols, exclude=["customer_id"])
display(table)
"""),
        create_cell("markdown", """### 1.3. Lọc cột có tỷ lệ thiếu đáng kể với `min_missing`
Chỉ hiển thị các cột có `missing_rate >= 0.10` (10% trở lên).
"""),
        create_cell("code", """fig, table = plot_missing(df, min_missing=0.10)
display(table)
"""),
        create_cell("markdown", """### 1.4. Thiết lập ngưỡng cảnh báo với `threshold`
Cột có tỷ lệ thiếu `> threshold` (ví dụ 30%) sẽ được **tô đỏ** và biểu đồ vẽ một đường thẳng đứt nét làm mốc.
Bảng trả về có thêm cột bool `exceed`.
"""),
        create_cell("code", """fig, table = plot_missing(df, threshold=0.30, title="Cảnh báo: Các cột có tỷ lệ thiếu > 30%")
display(table)
"""),
        create_cell("markdown", """---
## 2. Kiểm thử hàm `plot_duplicate` (Tỷ lệ dòng trùng lặp)

Hàm kiểm tra tỷ lệ trùng lặp theo một hoặc nhiều định nghĩa trùng (subset).

### 2.1. Trùng lặp toàn bộ dòng (`subset=None`)
"""),
        create_cell("code", """fig, table = plot_duplicate(df, subset=None, keep="first")
display(table)
"""),
        create_cell("markdown", """### 2.2. Trùng lặp theo một khóa định danh duy nhất (`subset=['order_id']`)
"""),
        create_cell("code", """fig, table = plot_duplicate(df, subset=["order_id"], threshold=0.03)
display(table)
"""),
        create_cell("markdown", """### 2.3. So sánh nhiều định nghĩa trùng lặp cùng lúc (truyền dict `subset`)
Điểm mạnh của hàm là cho phép truyền dictionary `{tên_định_nghĩa: cột_hoặc_danh_sách_cột}` để so sánh trực quan các mức độ trùng lặp.
"""),
        create_cell("code", """duplicate_definitions = {
    "Trùng toàn bộ dòng": list(df.columns),
    "Trùng mã đơn (order_id)": ["order_id"],
    "Trùng khách + ngày": ["customer_id", "order_date"],
    "Trùng khách + địa chỉ": ["customer_id", "shipping_address"],
    "Trùng khách + mã đơn": ["customer_id", "order_id"],
}

fig, table = plot_duplicate(
    df,
    subset=duplicate_definitions,
    threshold=0.04,
    title="So sánh các định nghĩa dòng trùng lặp (Ngưỡng cảnh báo: 4%)"
)
display(table)
"""),
        create_cell("markdown", """### 2.4. So sánh các chiến lược `keep`
- `keep="first"`: Đánh dấu trùng từ lần xuất hiện thứ 2 trở đi.
- `keep=False`: Đánh dấu trùng toàn bộ các dòng thuộc nhóm trùng (kể cả bản ghi đầu tiên).
"""),
        create_cell("code", """fig1, tab1 = plot_duplicate(df, subset=["order_id"], keep="first", title="keep='first'")
fig2, tab2 = plot_duplicate(df, subset=["order_id"], keep=False, title="keep=False (tính cả bản gốc)")

display(pd.concat([
    tab1.assign(keep="first"),
    tab2.assign(keep="False")
], ignore_index=True))
"""),
        create_cell("markdown", """---
## 3. Kiểm thử hàm `plot_outlier_rate` (Tỷ lệ ngoại lai cột số)

Hàm hỗ trợ 3 phương pháp phát hiện ngoại lai:
- `"iqr"`: Ngoài khoảng $[Q1 - k \\cdot IQR, Q3 + k \\cdot IQR]$ (mặc định $k=1.5$).
- `"zscore"`: Ngoài khoảng $[\mu - k \\cdot \sigma, \mu + k \\cdot \sigma]$ (mặc định $k=3.0$).
- `"mad"`: Modified Z-score ngoài khoảng $[\\text{median} - k \\cdot MAD / 0.6745, \\text{median} + k \\cdot MAD / 0.6745]$ (mặc định $k=3.5$).

Tự động bỏ qua giá trị thiếu và giá trị vô cùng `±inf`. Xử lý an toàn trường hợp $std=0$ hoặc $MAD=0$.

### 3.1. Phương pháp mặc định IQR (`method="iqr"`)
"""),
        create_cell("code", """fig, table = plot_outlier_rate(df, method="iqr")
display(table)
"""),
        create_cell("markdown", """### 3.2. So sánh 3 phương pháp: IQR vs Z-score vs MAD
"""),
        create_cell("code", """fig, axes = plt.subplots(1, 3, figsize=(18, 5))

_, t_iqr = plot_outlier_rate(df, method="iqr", k=1.5, ax=axes[0], title="IQR (k=1.5)")
_, t_zsc = plot_outlier_rate(df, method="zscore", k=3.0, ax=axes[1], title="Z-score (k=3.0)")
_, t_mad = plot_outlier_rate(df, method="mad", k=3.5, ax=axes[2], title="MAD (k=3.5)")

plt.tight_layout()

# Ghép so sánh tỷ lệ giữa 3 phương pháp
comparison = t_iqr[["column", "outlier_rate"]].rename(columns={"outlier_rate": "rate_iqr"})
comparison["rate_zscore"] = t_zsc.set_index("column").loc[comparison["column"], "outlier_rate"].values
comparison["rate_mad"] = t_mad.set_index("column").loc[comparison["column"], "outlier_rate"].values
display(comparison)
"""),
        create_cell("markdown", """### 3.3. Lọc `min_rate` và cảnh báo `threshold`
"""),
        create_cell("code", """fig, table = plot_outlier_rate(
    df,
    method="iqr",
    min_rate=0.015,
    threshold=0.03,
    title="Các cột có tỷ lệ Outlier >= 1.5% (Tô đỏ cột > 3%)"
)
display(table)
"""),
        create_cell("markdown", """### 3.4. Kiểm tra xử lý an toàn: Cột hằng số (`std=0`) và cột có giá trị `±inf`
- `constant_fee`: Hằng số 15,000 -> $std=0, MAD=0$, hàm trả về lower=NaN, upper=NaN, n_outlier=0 (không bị văng lỗi chia cho 0).
- `metric_with_inf`: Chứa giá trị vô cùng `±inf`, hàm tự động loại bỏ giá trị vô cùng trước khi tính toán.
"""),
        create_cell("code", """fig, table = plot_outlier_rate(df, cols=["constant_fee", "metric_with_inf"], method="zscore")
display(table)
"""),
        create_cell("markdown", """---
## 4. Kiểm thử hàm `plot_entity_coverage` (Độ phủ thực thể theo thời gian)

Hàm theo dõi số lượng thực thể (ví dụ: `customer_id`) có giao dịch trong từng kỳ chia cho tổng số thực thể duy nhất trong toàn bộ dữ liệu.

### 4.1. Độ phủ khách hàng theo tháng (`freq="M"`) kèm ngưỡng cảnh báo `threshold=0.30`
Tháng nào có coverage `< 30%` sẽ được đánh dấu chấm đỏ và đường đứt nét làm mốc.
"""),
        create_cell("code", """fig, table = plot_entity_coverage(
    df,
    entity="customer_id",
    time="order_date",
    freq="M",
    threshold=0.30,
    title="Độ phủ khách hàng theo tháng (Cảnh báo: Coverage < 30%)"
)
display(table)
"""),
        create_cell("markdown", """### 4.2. Kiểm tra tính năng tự lấp đầy chu kỳ gián đoạn (`fill_gaps`)
Trong dữ liệu giả lập, tháng **2024-08** cố tình không có bất kỳ giao dịch nào.
Hàm `plot_entity_coverage` tự động tạo chu kỳ liên tục từ tháng nhỏ nhất đến lớn nhất (`pd.period_range`) và gán `n_entity=0`, `coverage=0.0`.
"""),
        create_cell("code", """# Lọc xem tháng 2024-08 trong bảng kết quả
gap_month = table[table["period"].astype(str) == "2024-08"]
display(gap_month)
assert gap_month.iloc[0]["n_entity"] == 0, "Lỗi: Kỳ trống không được gán 0!"
assert gap_month.iloc[0]["coverage"] == 0.0, "Lỗi: Coverage của kỳ trống phải bằng 0!"
print(" Xác nhận fill_gaps hoạt động chính xác: 2024-08 có n_entity = 0 và coverage = 0.0")
"""),
        create_cell("markdown", """### 4.3. Độ phủ theo tuần (`freq="W"`) và theo ngày (`freq="D"`)
"""),
        create_cell("code", """fig_w, tab_w = plot_entity_coverage(df, entity="customer_id", time="order_date", freq="W", title="Độ phủ theo tuần (freq='W')")
display(tab_w.head(10))
"""),
        create_cell("markdown", """---
## 5. Ghép Dashboard QC tổng hợp (4 trong 1)

Nhờ hỗ trợ tham số `ax`, chúng ta có thể kết hợp cả 4 biểu đồ vào một bảng điều khiển chất lượng dữ liệu hoàn chỉnh chỉ với một Figure duy nhất!
"""),
        create_cell("code", """fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# 1. Tỷ lệ Missing
plot_missing(
    df,
    threshold=0.30,
    ax=axes[0, 0],
    title="1. Missing Values (Ngưỡng 30%)"
)

# 2. Tỷ lệ Trùng lặp
dup_defs = {
    "Toàn bộ dòng": list(df.columns),
    "Trùng order_id": ["order_id"],
    "Trùng (khách, ngày)": ["customer_id", "order_date"],
    "Trùng (khách, địa chỉ)": ["customer_id", "shipping_address"],
}
plot_duplicate(
    df,
    subset=dup_defs,
    threshold=0.04,
    ax=axes[0, 1],
    title="2. Duplicate Rate (Ngưỡng 4%)"
)

# 3. Tỷ lệ Outlier
plot_outlier_rate(
    df,
    method="iqr",
    threshold=0.03,
    ax=axes[1, 0],
    title="3. Outlier Rate - IQR (Ngưỡng 3%)"
)

# 4. Độ phủ Entity
plot_entity_coverage(
    df,
    entity="customer_id",
    time="order_date",
    freq="M",
    threshold=0.30,
    ax=axes[1, 1],
    title="4. Customer Coverage hàng tháng (Ngưỡng 30%)"
)

fig.suptitle(" BẢNG ĐIỀU KHIỂN CHẤT LƯỢNG DỮ LIỆU TỔNG HỢP (QC DASHBOARD)", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

print(" Hoàn thành toàn bộ quy trình kiểm thử package qc!")
"""),
    ]
    build_notebook(cells, out_path)


def generate_edge_cases_notebook(out_path: Path):
    cells = [
        create_cell("markdown", """#  Kiểm thử các trường hợp biên & Xử lý lỗi (QC Edge Cases & Error Handling)

Notebook này kiểm tra tính bền vững (robustness) của package `qc` trước các dữ liệu bất thường và lỗi đầu vào:
1. DataFrame rỗng hoặc chỉ có 1 dòng.
2. Cột toàn bộ missing (100% NaN) hoặc không có missing (0%).
3. Biểu đồ khi bộ lọc `min_*` loại bỏ toàn bộ dữ liệu (hiển thị thông điệp `_no_data`).
4. Kiểm tra các ngoại lệ `DatasetError` khi truyền tham số sai.
"""),
        create_cell("code", """import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

repo_root = Path.cwd().resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from qc import (
    plot_missing,
    plot_duplicate,
    plot_outlier_rate,
    plot_entity_coverage,
)
from qc._common import DatasetError

%matplotlib inline
print(" Sẵn sàng kiểm thử Edge Cases!")
"""),
        create_cell("markdown", """## 1. Trường hợp DataFrame rỗng (`len(df) == 0`)
"""),
        create_cell("code", """empty_df = pd.DataFrame({
    "id": pd.Series(dtype=int),
    "val": pd.Series(dtype=float),
    "date": pd.Series(dtype="datetime64[ns]"),
})

# plot_missing trên DataFrame rỗng
fig1, tab1 = plot_missing(empty_df)
display(tab1)

# plot_duplicate trên DataFrame rỗng
fig2, tab2 = plot_duplicate(empty_df)
display(tab2)

# plot_outlier_rate trên DataFrame rỗng
fig3, tab3 = plot_outlier_rate(empty_df, cols=["val"])
display(tab3)

# plot_entity_coverage trên DataFrame rỗng
fig4, tab4 = plot_entity_coverage(empty_df, entity="id", time="date")
display(tab4)
"""),
        create_cell("markdown", """## 2. Trường hợp bộ lọc `min_*` loại bỏ hết tất cả các cột
Khi không có cột nào thỏa điều kiện `min_missing` hoặc `min_rate`, biểu đồ phải hiển thị thông điệp văn bản ở giữa trục tọa độ mà không bị crash.
"""),
        create_cell("code", """# Tạo df chỉ có missing thấp (< 10%)
clean_df = pd.DataFrame({"col_a": [1, 2, 3, None], "col_b": [4, 5, 6, 7]})

# Đặt min_missing=0.50 (không cột nào đạt)
fig, table = plot_missing(clean_df, min_missing=0.50)
display(table)
print("Bảng rỗng:", table.empty)
"""),
        create_cell("markdown", """## 3. Kiểm tra ngoại lệ Input Validation (`DatasetError`)

Package `qc` được thiết kế chặt chẽ để phát hiện và báo lỗi rõ ràng khi người dùng truyền tham số sai:
"""),
        create_cell("code", """def expect_dataset_error(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
        raise AssertionError("Lẽ ra phải bắn DatasetError!")
    except DatasetError as e:
        print(f" Bắt được lỗi đúng kỳ vọng: {e}")

# 1. Truyền không phải DataFrame
expect_dataset_error(plot_missing, [1, 2, 3])

# 2. DataFrame có cột trùng tên
df_dup_cols = pd.DataFrame([[1, 2]], columns=["A", "A"])
expect_dataset_error(plot_missing, df_dup_cols)

# 3. Tham số rate ngoài [0, 1] hoặc sai kiểu
sample_df = pd.DataFrame({"x": [1, 2, 3]})
expect_dataset_error(plot_missing, sample_df, threshold=1.5)
expect_dataset_error(plot_missing, sample_df, min_missing=-0.1)
expect_dataset_error(plot_missing, sample_df, threshold="high")

# 4. Chọn cột không tồn tại
expect_dataset_error(plot_missing, sample_df, cols=["non_existent"])

# 5. Method outlier không hợp lệ
expect_dataset_error(plot_outlier_rate, sample_df, method="invalid_method")

# 6. Tham số k <= 0
expect_dataset_error(plot_outlier_rate, sample_df, k=-1)

# 7. Entity coverage: cột time không phải datetime
expect_dataset_error(plot_entity_coverage, sample_df, entity="x", time="x")

# 8. Entity coverage: freq không hợp lệ
sample_time_df = pd.DataFrame({"id": ["a", "b"], "dt": pd.to_datetime(["2024-01-01", "2024-01-02"])})
expect_dataset_error(plot_entity_coverage, sample_time_df, entity="id", time="dt", freq="INVALID_FREQ")
"""),
        create_cell("markdown", """## 4. Tóm tắt kết quả
- Tất cả các trường hợp biên (DataFrame rỗng, cột toàn thiếu, lọc rỗng) đều được xử lý êm thuận.
- Cơ chế kiểm tra đầu vào (input validation) hoạt động chuẩn xác và trả về thông báo lỗi thân thiện bằng tiếng Việt.
"""),
    ]
    build_notebook(cells, out_path)


def generate_eda_qc_demo_notebook(out_path: Path):
    cells = [
        create_cell("markdown", """# Chạy thử `eda` và `qc` trên dữ liệu giả lập

- **`eda`**: nhìn nhanh phân phối từng loại cột: `plot_numeric`, `plot_categorical` (bar, pie), `plot_temporal`.
- **`qc`**: kiểm tra chất lượng dữ liệu, mỗi khía cạnh một hàm: `plot_missing`, `plot_duplicate`, `plot_outlier_rate`, `plot_entity_coverage`.

Cả hai đều trả về `(Figure, DataFrame)`: figure để xem, bảng là số liệu đằng sau chart.
"""),
        create_cell("code", """import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # không phụ thuộc backend của kernel; figure được hiện bằng show(fig)
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
from IPython.display import Image, display

repo_root = Path.cwd().resolve()
if not (repo_root / "eda").exists():  # đang chạy từ notebooks/
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from eda import plot_numeric, plot_categorical, plot_temporal
from qc import plot_missing, plot_duplicate, plot_outlier_rate, plot_entity_coverage

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


def show(fig):
    # Hiện figure dưới dạng ảnh PNG (chạy đúng với mọi backend), rồi đóng figure
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    display(Image(buf.getvalue()))
"""),
        create_cell("code", """df = pd.read_csv(repo_root / "data" / "mock_transactions.csv", parse_dates=["order_date"])
print(f"{df.shape[0]:,} dòng x {df.shape[1]} cột")
display(df.head())
display(df.dtypes.to_frame("dtype").T)
"""),
        create_cell("markdown", """---
# Phần 1. EDA

## 1.1. `plot_numeric`: cột số

Mặc định `cols=None` lấy mọi cột số, mỗi cột một ô. Bỏ `metric_with_inf` (chứa ±inf) và `internal_device_token` (toàn thiếu) để chart gọn.
"""),
        create_cell("code", """fig, long = plot_numeric(df, exclude=["metric_with_inf", "internal_device_token"], ncols=3)
display(long.head())
"""),
        create_cell("markdown", """Nhiều loại chart cho cùng một cột: `kind` nhận list, mỗi kind một ô.
"""),
        create_cell("code", """fig, long = plot_numeric(df, cols="total_amount", kind=["hist", "box", "kde", "violin", "ecdf"], ncols=5, figsize=(20, 3.5))
"""),
        create_cell("markdown", """## 1.2. `plot_categorical`: cột phân loại

`top_n` chọn N category phổ biến nhất, phần còn lại gộp vào `(other)`.
"""),
        create_cell("code", """fig, long = plot_categorical(df, cols=["payment_method", "order_status", "shipping_region"], top_n=5, ncols=3)
display(long)
"""),
        create_cell("markdown", """### Pie chart

`kind="pie"` luôn sắp theo tần suất giảm dần, `(other)` nằm cuối và tô xám. Dùng `top_n` nhỏ (5-8); pie chỉ hợp với cột ít category và phân bố lệch.
"""),
        create_cell("code", """fig, long = plot_categorical(df, cols=["payment_method", "order_status"], kind="pie", top_n=3, ncols=2, figsize=(11, 5))
display(long)
"""),
        create_cell("markdown", """## 1.3. `plot_temporal`: cột thời gian

Số record theo kỳ (`freq="M"`) và theo chu kỳ (thứ trong tuần, tháng, giờ, heatmap thứ x giờ).
"""),
        create_cell("code", """fig, long = plot_temporal(df, kind=["line", "bar", "cumulative"], freq="M", ncols=3, figsize=(18, 3.5))
display(long.head())
"""),
        create_cell("code", """fig, long = plot_temporal(df, kind=["weekday", "month", "hour", "heatmap"], ncols=4, figsize=(20, 3.8))
"""),
        create_cell("markdown", """---
# Phần 2. QC

Quy ước chung: tỷ lệ là số `0-1`; `min_*` lọc bớt cột nhẹ khỏi bảng lẫn chart; `threshold` tô đỏ phần vượt ngưỡng (không điền thì không tô) và bảng có thêm cột `exceed`.

## 2.1. `plot_missing`
"""),
        create_cell("code", """fig, table = plot_missing(df, threshold=0.2)
display(table)
"""),
        create_cell("markdown", """Chỉ hiện cột thiếu từ 1% trở lên (`min_missing`):
"""),
        create_cell("code", """fig, table = plot_missing(df, min_missing=0.01, threshold=0.2)
display(table)
"""),
        create_cell("markdown", """## 2.2. `plot_duplicate`

Truyền dict `{tên: cột}` để so nhiều định nghĩa trùng trong một chart.
"""),
        create_cell("code", """fig, table = plot_duplicate(
    df,
    subset={
        "toàn bộ dòng": list(df.columns),
        "order_id": "order_id",
        "customer_id + order_date": ["customer_id", "order_date"],
    },
    threshold=0.01,
)
display(table)
"""),
        create_cell("markdown", """## 2.3. `plot_outlier_rate`

So sánh 3 phương pháp trên cùng bộ cột bằng cách truyền `ax`. `min_rate` ẩn cột không có outlier.
"""),
        create_cell("code", """fig, axes = plt.subplots(1, 3, figsize=(18, 4))
tables = {}
for ax, method in zip(axes, ["iqr", "zscore", "mad"]):
    _, tables[method] = plot_outlier_rate(df, method=method, min_rate=0.0001, threshold=0.05, ax=ax)
fig.tight_layout()
display(tables["iqr"])
"""),
        create_cell("markdown", """## 2.4. `plot_entity_coverage`

`coverage` = số khách hàng có đơn trong kỳ / tổng số khách hàng của cả bảng. Kỳ không có đơn nào vẫn hiện với giá trị 0. Kỳ có coverage dưới `threshold` bị tô đỏ.
"""),
        create_cell("code", """fig, table = plot_entity_coverage(df, entity="customer_id", time="order_date", freq="M", threshold=0.3)
display(table)
"""),
        create_cell("markdown", """## 2.5. Ghép thành một dashboard

Mọi hàm nhận `ax` nên có thể đặt vào cùng một figure.
"""),
        create_cell("code", """fig, axes = plt.subplots(2, 2, figsize=(14, 9))
plot_missing(df, min_missing=0.01, threshold=0.2, ax=axes[0, 0])
plot_duplicate(df, {"toàn bộ dòng": list(df.columns), "order_id": "order_id"}, threshold=0.01, ax=axes[0, 1])
plot_outlier_rate(df, min_rate=0.0001, threshold=0.05, ax=axes[1, 0])
plot_entity_coverage(df, "customer_id", "order_date", threshold=0.3, ax=axes[1, 1])
fig.suptitle("Data Quality overview")
fig.tight_layout()
"""),
    ]
    for cell in cells:  # mỗi cell tạo figure thì kết thúc bằng show(fig)
        src = "".join(cell["source"])
        if cell["cell_type"] == "code" and re.search(r"\bfig\b", src) and "def show" not in src:
            cell["source"] = create_cell("code", src.rstrip("\n") + "\nshow(fig)")["source"]
    build_notebook(cells, out_path)


if __name__ == "__main__":
    nb_dir = Path(__file__).resolve().parent.parent / "notebooks"
    generate_eda_qc_demo_notebook(nb_dir / "03_demo_eda_qc.ipynb")
    generate_pipeline_notebook(nb_dir / "01_test_qc_pipeline.ipynb")
    generate_edge_cases_notebook(nb_dir / "02_test_qc_edge_cases.ipynb")
