# data_processing (Kiên) – Cấp 1 & 2

Bộ hàm làm sạch dữ liệu dạng bảng, **dùng được cho mọi dataset**.

## Quy ước chung

- Mỗi hàm nhận `DataFrame` và trả về `DataFrame` **mới**, không sửa dữ liệu gốc.
- Không hard-code: tên cột, ngưỡng và quy tắc đều truyền qua tham số.
- Tham số `columns=None` nghĩa là hàm tự chọn các cột phù hợp (vd mọi cột số).
- Các hàm có `reference=` hoặc `bounds=` cho phép **tính trên tập train rồi áp dụng cho test / batch mới**, để không rò rỉ dữ liệu.
- Mỗi bước ghi log những gì đã thay đổi. Bật log bằng `logging.basicConfig(level=logging.INFO)`.
- Docstring của từng hàm giải thích đầy đủ các tham số: dùng `help(dp.fill_missing)`.

## Danh sách hàm

| Module | Hàm | Việc làm |
|---|---|---|
| `io` | `load_table` | Đọc CSV/Excel/JSON/Parquet, tự dò encoding và dấu phân cách |
| | `load_many` | Đọc nhiều file rồi nối lại, ghi tên file nguồn |
| | `safe_merge` | Join có kiểm tra quan hệ khóa, tránh nhân dòng |
| `schema` | `standardize_column_names` | Tên cột về `snake_case`, bỏ dấu |
| | `to_numeric` / `auto_convert_numeric` | Chữ → số (`$`, `,`, `%`, `(50)`, kiểu `1.234,5`) |
| | `to_boolean`, `to_category` | Chuẩn hóa kiểu bool / category |
| | `drop_useless_columns` | Xóa cột rỗng hoàn toàn / chỉ có một giá trị |
| `duplicates` | `find_duplicates` | Xem các dòng trùng trước khi xóa |
| | `drop_exact_duplicates` | Xóa dòng giống hệt (có thể bỏ qua vài cột) |
| | `drop_key_duplicates` | Mỗi khóa một dòng: `first` / `last` / `most_complete` |
| `missing` | `standardize_missing` | `"N/A"`, `"null"`, `-999`… → NaN |
| | `drop_high_missing` | Xóa cột/dòng thiếu quá ngưỡng |
| | `add_missing_flags` | Cột cờ `*_was_missing` (gọi trước khi điền) |
| | `fill_missing` | mean/median/mode/constant/ffill/bfill/interpolate, theo nhóm |
| | `fill_missing_with_model` | KNN / Iterative (MICE) |
| `text` | `clean_text` | Khoảng trắng, hoa/thường, Unicode NFC, regex |
| | `normalize_categories` | Gộp biến thể (`HN`, `Ha Noi` → `Hà Nội`) |
| `outliers` | `compute_outlier_bounds` | Ngưỡng IQR / Z-score / MAD / percentile |
| | `handle_outliers` | clip / remove / missing / flag |
| | `enforce_value_rules` | Quy tắc min / max / allowed theo nghiệp vụ |
| `datetime_utils` | `parse_datetime` | Nhiều định dạng lẫn lộn, Unix timestamp |
| | `convert_timezone` | Đưa về một múi giờ |
| | `fill_time_gaps` | Thêm mốc thời gian bị thiếu (theo nhóm) |
| `constraints` | `enforce_rules` / `find_rule_violations` | Ràng buộc giữa các cột (`end >= start`, …) |
| | `find_orphan_keys` | Khóa ngoại không có ở bảng cha |
| `reshape` | `wide_to_long` / `long_to_wide` | melt / pivot, tên cột phẳng |
| | `aggregate` | groupby + agg, tên cột kiểu `price_mean` |
| | `explode_column` | `"a, b, c"` → 3 dòng |
| | `flatten_json_column` | JSON lồng nhau → nhiều cột |

## Cách dùng

```python
import data_processing as dp

clean = (
    raw
    .pipe(dp.standardize_column_names)
    .pipe(dp.standardize_missing, sentinel_values={"price": [-999]})
    .pipe(dp.clean_text)
    .pipe(dp.to_numeric, ["price"])
    .pipe(dp.parse_datetime, ["order_date"], formats=["%d/%m/%Y"])
    .pipe(dp.drop_key_duplicates, "customer_id", strategy="last", order_by="order_date")
    .pipe(dp.handle_outliers, ["price"], action="clip")
    .pipe(dp.fill_missing, ["price"], strategy="median", group_by="category")
)
```

Ví dụ chạy được đầy đủ: [examples/demo_pipeline.py](../examples/demo_pipeline.py).

### Thứ tự nên làm

1. Chuẩn hóa tên cột → missing giả → làm sạch text
2. Ép kiểu (số, ngày giờ, bool) → gộp biến thể category
3. Khử trùng (sau khi đã chuẩn hóa, để các bản ghi trùng thực sự giống nhau)
4. Quy tắc nghiệp vụ → outlier
5. `add_missing_flags` → điền missing

### Tập train / test

```python
bounds = {c: dp.compute_outlier_bounds(train[c]) for c in ["price"]}
train = dp.handle_outliers(train, ["price"], bounds=bounds)
test  = dp.handle_outliers(test,  ["price"], bounds=bounds)
test  = dp.fill_missing(test, ["price"], strategy="median", reference=train)
```

## Chạy test

```bash
pip install -r requirements.txt
python -m pytest -q
```
