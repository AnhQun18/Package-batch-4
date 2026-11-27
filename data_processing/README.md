# data_processing (Kiên): Cấp 1 & 2

Bộ hàm làm sạch dữ liệu dạng bảng, **dùng được cho mọi dataset**: không hàm nào hard-code tên cột hay ngưỡng.

```python
import data_processing as dp
```

## Mục lục

- [Quy ước chung](#quy-ước-chung)
- [Tham số dùng chung](#tham-số-dùng-chung)
- [1. Đọc & gộp dữ liệu (`io`)](#1-đọc--gộp-dữ-liệu-io)
- [2. Chuẩn hóa schema (`schema`)](#2-chuẩn-hóa-schema-schema)
- [3. Trùng lặp (`duplicates`)](#3-trùng-lặp-duplicates)
- [4. Missing values (`missing`)](#4-missing-values-missing)
- [5. Text & category (`text`)](#5-text--category-text)
- [6. Outlier & giá trị vô lý (`outliers`)](#6-outlier--giá-trị-vô-lý-outliers)
- [7. Ngày giờ (`datetime_utils`)](#7-ngày-giờ-datetime_utils)
- [8. Ràng buộc chéo (`constraints`)](#8-ràng-buộc-chéo-constraints)
- [9. Reshape (`reshape`)](#9-reshape-reshape)
- [Pipeline mẫu & thứ tự nên làm](#pipeline-mẫu--thứ-tự-nên-làm)
- [Chạy test](#chạy-test)

---

## Quy ước chung

- Mỗi hàm nhận `DataFrame` và trả về `DataFrame` **mới**; dữ liệu gốc không bị sửa.
- Các hàm nối được với nhau bằng `.pipe(...)`.
- Mỗi bước ghi log những gì đã thay đổi. Bật log bằng:
  ```python
  import logging
  logging.basicConfig(level=logging.INFO)
  ```
- Truyền tên cột không tồn tại → hàm báo `KeyError` ngay, không âm thầm bỏ qua.
- Tham số chọn từ một danh sách cố định (vd `strategy`, `action`) mà truyền sai → báo `ValueError` kèm danh sách giá trị hợp lệ.

## Tham số dùng chung

Các tham số dưới đây xuất hiện ở nhiều hàm, luôn mang cùng một ý nghĩa. Bảng của từng hàm bên dưới không lặp lại `df`.

| Tham số | Giải thích |
|---|---|
| `df` | DataFrame đầu vào. Luôn là tham số đầu tiên. |
| `columns` | Cột cần xử lý. Nhận **một tên cột** (`"price"`) hoặc **danh sách tên cột** (`["price", "qty"]`). Để `None` thì hàm tự chọn các cột phù hợp với nó (vd mọi cột số cho outlier, mọi cột chữ cho text). |
| `exclude` | Cột **không** bị tác động, dù thỏa điều kiện. Thường dùng cho cột ID hoặc cột target. |
| `group_by` | Cột dùng để chia nhóm; phép tính được làm riêng trong từng nhóm. Một tên hoặc danh sách tên. |
| `reference` | DataFrame dùng để **tính thống kê / học tham số** (thường là tập **train**). Kết quả được áp dụng lên `df`. Để `None` thì dùng chính `df`. Mục đích là tránh rò rỉ dữ liệu từ test vào train. |
| `bounds` | Ngưỡng đã tính sẵn, dạng `{tên_cột: (dưới, trên)}`, để dùng lại cho test hoặc batch mới. |
| `action` | Cách xử lý dòng/ô vi phạm. Mỗi hàm có danh sách riêng, nhưng nghĩa của từng giá trị giống nhau: `"remove"` xóa cả dòng, `"missing"` đổi ô thành NaN, `"clip"` kéo về ngưỡng, `"flag"` giữ nguyên và thêm cột cờ 0/1, `"raise"` báo lỗi. |

---

## 1. Đọc & gộp dữ liệu (`io`)

### `load_table(path, file_format=None, encodings=(...), sep=None, **read_kwargs)`

Đọc một file dạng bảng thành DataFrame.

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `path` | **bắt buộc** | Đường dẫn tới file. |
| `file_format` | `None` | `"csv"`, `"excel"`, `"json"`, `"jsonl"` hoặc `"parquet"`. `None` = tự đoán theo đuôi file (`.csv .tsv .txt .xlsx .xls .json .jsonl .parquet`). |
| `encodings` | `("utf-8-sig", "cp1258", "latin-1")` | *(chỉ CSV)* Danh sách encoding thử lần lượt tới khi đọc được. `utf-8-sig` đọc được cả UTF-8 có và không có BOM; `cp1258` là encoding tiếng Việt cũ của Windows; `latin-1` luôn đọc được nên đặt cuối. |
| `sep` | `None` | *(chỉ CSV)* Ký tự phân tách cột. `None` = tự dò trong `,` `;` tab `\|`. |
| `**read_kwargs` | | Tham số khác truyền thẳng cho hàm đọc của pandas, vd `sheet_name="Sheet2"` (Excel), `usecols=[...]`, `nrows=1000`. |

```python
df = dp.load_table("data/raw.csv")
df = dp.load_table("data/report.xlsx", sheet_name="2024")
```

### `load_many(paths, source_column="source_file", **load_kwargs)`

Đọc nhiều file cùng cấu trúc rồi nối theo chiều dọc. Nếu các file có cột khác nhau, hàm ghi cảnh báo và để NaN ở cột thiếu.

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `paths` | **bắt buộc** | Danh sách đường dẫn, hoặc kết quả `Path(...).glob("*.csv")`. |
| `source_column` | `"source_file"` | Tên cột mới ghi lại dòng đến từ file nào. `None` = không thêm. |
| `**load_kwargs` | | Truyền tiếp vào `load_table` cho từng file (vd `sep=";"`). |

```python
from pathlib import Path
df = dp.load_many(Path("data/batches").glob("*.csv"))
```

### `safe_merge(left, right, on=None, left_on=None, right_on=None, how="left", relationship="many_to_one", indicator_column=None)`

Join hai bảng và **kiểm tra quan hệ khóa**, để không âm thầm nhân dòng khi bảng phải có khóa trùng.

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `left`, `right` | **bắt buộc** | Bảng trái và bảng phải. |
| `on` | `None` | Cột khóa có **cùng tên** ở hai bảng. |
| `left_on`, `right_on` | `None` | Dùng thay `on` khi tên cột khóa ở hai bảng **khác nhau**. |
| `how` | `"left"` | Kiểu join: `"left"` giữ mọi dòng bảng trái, `"right"` giữ mọi dòng bảng phải, `"inner"` chỉ giữ dòng khớp, `"outer"` giữ tất cả. |
| `relationship` | `"many_to_one"` | Quan hệ khóa mong đợi: `"one_to_one"`, `"one_to_many"`, `"many_to_one"`, `"many_to_many"`. Ví dụ `"many_to_one"` = bảng trái được phép trùng khóa (nhiều đơn hàng), bảng phải mỗi khóa chỉ một dòng (một khách hàng). Dữ liệu vi phạm → báo `ValueError` kèm số dòng trùng. |
| `indicator_column` | `None` | Nếu đặt tên, thêm cột cho biết mỗi dòng là `"both"` (khớp), `"left_only"` hay `"right_only"`. Hữu ích để tìm dòng không join được. |

```python
orders = dp.safe_merge(orders, customers, on="customer_id", indicator_column="_match")
```

---

## 2. Chuẩn hóa schema (`schema`)

### `standardize_column_names(df, remove_accents=True)`

Đổi tên mọi cột về `snake_case`: `"Ngày Sinh "` → `ngay_sinh`, `"totalPrice($)"` → `total_price`. Nếu hai cột trùng tên sau khi đổi, cột sau được thêm hậu tố `_2`, `_3`, …

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `remove_accents` | `True` | `True` = bỏ dấu tiếng Việt trong tên cột (`đ` → `d`). `False` = giữ dấu. |

### `to_numeric(df, columns, thousands_sep=",", decimal_sep=".", strip_symbols=(...), percent_to_fraction=False, accounting_negative=True)`

Chuyển cột chữ sang số. Giá trị không chuyển được thành NaN, và số lượng được ghi log. Cột đã là số thì bỏ qua.

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `columns` | **bắt buộc** | Cột cần chuyển. |
| `thousands_sep` | `","` | Dấu phân cách hàng nghìn, sẽ bị xóa. `","` cho kiểu `1,234.5`; `"."` cho kiểu Việt Nam `1.234,5`. `None` nếu dữ liệu không có. |
| `decimal_sep` | `"."` | Dấu thập phân trong dữ liệu gốc: `"."` hoặc `","`. |
| `strip_symbols` | `("$", "€", "£", "¥", "₫", "VND", "VNĐ", "USD", "EUR")` | Ký hiệu tiền tệ hoặc đơn vị cần xóa trước khi chuyển. Có thể truyền danh sách riêng, vd `("kg", "cm")`. |
| `percent_to_fraction` | `False` | `True`: `"15%"` → `0.15`. `False`: `"15%"` → `15`. |
| `accounting_negative` | `True` | `True`: `"(123)"` hiểu là `-123` (cách ghi số âm trong kế toán). |

```python
df = dp.to_numeric(df, ["price", "discount"], percent_to_fraction=True)
df = dp.to_numeric(df, "gia", thousands_sep=".", decimal_sep=",")   # "1.234,5 ₫" -> 1234.5
```

### `auto_convert_numeric(df, columns=None, min_success_ratio=0.95, **to_numeric_kwargs)`

Tự tìm cột chữ mà thực chất chứa số rồi chuyển sang số. Hữu ích khi chưa biết trước dataset.

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `columns` | `None` | Cột được xét. `None` = mọi cột kiểu chữ. |
| `min_success_ratio` | `0.95` | Chỉ chuyển cột khi ít nhất 95% giá trị không rỗng chuyển được thành số. Tăng ngưỡng để thận trọng hơn. |
| `**to_numeric_kwargs` | | Truyền tiếp `thousands_sep`, `decimal_sep`, `strip_symbols`, `percent_to_fraction`, `accounting_negative` (xem `to_numeric`). |

### `to_boolean(df, columns, true_values=(...), false_values=(...))`

Chuyển cột về kiểu boolean (`True` / `False` / `<NA>`). So khớp không phân biệt hoa thường, bỏ khoảng trắng hai đầu. Giá trị không nhận ra được thành `<NA>`, và số lượng được ghi log.

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `columns` | **bắt buộc** | Cột cần chuyển. |
| `true_values` | `true, t, yes, y, 1, có, co, đúng, dung` | Các giá trị được hiểu là `True`. |
| `false_values` | `false, f, no, n, 0, không, khong, sai` | Các giá trị được hiểu là `False`. |

### `to_category(df, columns=None, max_unique_ratio=0.05)`

Chuyển cột chữ có ít giá trị khác nhau sang kiểu `category` để tiết kiệm bộ nhớ.

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `columns` | `None` | Cột cần chuyển. `None` = tự chọn theo `max_unique_ratio`. |
| `max_unique_ratio` | `0.05` | *(chỉ khi `columns=None`)* Cột được chọn nếu số giá trị khác nhau chia cho số dòng ≤ 5%. |

### `drop_useless_columns(df, drop_all_missing=True, drop_constant=True, exclude=None)`

Xóa cột không mang thông tin.

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `drop_all_missing` | `True` | Xóa cột mà mọi ô đều rỗng. |
| `drop_constant` | `True` | Xóa cột chỉ có đúng một giá trị (không tính ô rỗng). |
| `exclude` | `None` | Cột luôn giữ lại. |

---

## 3. Trùng lặp (`duplicates`)

### `find_duplicates(df, subset=None)`

Trả về **mọi** dòng thuộc các nhóm trùng, đã sắp xếp cạnh nhau, để xem trước khi xóa.

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `subset` | `None` | Cột dùng để so trùng. `None` = so trên toàn bộ cột. |

### `drop_exact_duplicates(df, ignore_columns=None, keep="first")`

Xóa các dòng giống hệt nhau.

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `ignore_columns` | `None` | Cột bỏ qua khi so sánh. Dùng cho cột luôn khác nhau giữa các bản sao, như thời điểm nạp dữ liệu hay tên file nguồn. |
| `keep` | `"first"` | `"first"` giữ bản xuất hiện đầu tiên, `"last"` giữ bản cuối cùng. |

### `drop_key_duplicates(df, keys, strategy="first", order_by=None, ascending=True)`

Giữ đúng **một dòng cho mỗi giá trị khóa** (vd mỗi `customer_id` một dòng). Thứ tự dòng ban đầu được giữ nguyên.

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `keys` | **bắt buộc** | Cột (hoặc tổ hợp cột) xác định một bản ghi duy nhất. |
| `strategy` | `"first"` | Cách chọn dòng giữ lại trong mỗi nhóm trùng: **`"first"`** giữ dòng đầu, **`"last"`** giữ dòng cuối, **`"most_complete"`** giữ dòng có ít ô rỗng nhất (hòa thì lấy dòng đầu). |
| `order_by` | `None` | Cột dùng để sắp xếp **trước khi** chọn. Ví dụ `order_by="updated_at"` kết hợp `strategy="last"` sẽ giữ bản ghi mới nhất. |
| `ascending` | `True` | Chiều sắp xếp của `order_by`: `True` tăng dần, `False` giảm dần. |

```python
df = dp.drop_key_duplicates(df, "customer_id", strategy="last", order_by="updated_at")
```

---

## 4. Missing values (`missing`)

### `standardize_missing(df, columns=None, tokens=(...), sentinel_values=None)`

Đổi các giá trị **"thiếu giả"** về NaN để pandas nhận ra đó là ô thiếu.

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `columns` | `None` | Cột cần xử lý. `None` = mọi cột. |
| `tokens` | `"", "na", "n/a", "nan", "null", "none", "nil", "-", "--", "?", "#n/a", "missing"` | Các chuỗi được coi là thiếu. So khớp sau khi bỏ khoảng trắng hai đầu, không phân biệt hoa thường (`" N/A "` vẫn khớp). |
| `sentinel_values` | `None` | Giá trị **quy ước** là thiếu, thường là số, vd `-999`, `9999`. **list**: áp dụng cho mọi cột trong `columns`. **dict** `{tên_cột: [giá trị...]}`: áp dụng riêng từng cột (nên dùng dạng này để không xóa nhầm). |

```python
df = dp.standardize_missing(df, sentinel_values={"age": [-1, 999], "income": [-999]})
```

### `drop_high_missing(df, axis="columns", threshold=0.5, exclude=None)`

Xóa cột hoặc dòng có tỉ lệ thiếu **vượt quá** ngưỡng.

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `axis` | `"columns"` | `"columns"` = xóa cột, `"rows"` = xóa dòng. |
| `threshold` | `0.5` | Tỉ lệ thiếu tối đa cho phép, từ 0 đến 1 (`0.5` = 50%). Cột/dòng thiếu **hơn** mức này bị xóa. |
| `exclude` | `None` | Với `axis="columns"`: cột luôn giữ. Với `axis="rows"`: cột không được tính vào tỉ lệ thiếu của dòng. |

### `add_missing_flags(df, columns=None, suffix="_was_missing", only_if_has_missing=True)`

Thêm cột 0/1 đánh dấu ô ban đầu có bị thiếu hay không. Việc bị thiếu đôi khi cũng mang thông tin hữu ích cho mô hình.
⚠️ Phải gọi **trước** khi điền missing; sau khi điền thì không còn biết ô nào từng thiếu.

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `columns` | `None` | Cột cần tạo cờ. `None` = mọi cột. |
| `suffix` | `"_was_missing"` | Hậu tố tên cột cờ: `age` → `age_was_missing`. |
| `only_if_has_missing` | `True` | `True` = chỉ tạo cờ cho cột thực sự có ô thiếu. |

### `fill_missing(df, columns=None, strategy="median", fill_value=None, group_by=None, sort_by=None, reference=None)`

Điền giá trị thiếu.

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `columns` | `None` | Cột cần điền. `None` = tự chọn: cột số với `mean` / `median` / `interpolate`, mọi cột với các chiến lược còn lại. |
| `strategy` | `"median"` | Cách điền: **`"mean"`** trung bình (chỉ cột số), **`"median"`** trung vị (chỉ cột số, ít bị outlier kéo lệch), **`"mode"`** giá trị xuất hiện nhiều nhất (dùng được cho cả cột chữ), **`"constant"`** điền bằng `fill_value`, **`"ffill"`** lấy giá trị dòng trước, **`"bfill"`** lấy giá trị dòng sau, **`"interpolate"`** nội suy tuyến tính giữa hai dòng (chỉ cột số). |
| `fill_value` | `None` | Giá trị điền, **bắt buộc** khi `strategy="constant"`. Ví dụ `0` hoặc `"Unknown"`. |
| `group_by` | `None` | Tính thống kê hoặc ffill **trong từng nhóm**, vd median giá theo từng `category`. Nhóm không có giá trị nào sẽ được điền bằng thống kê toàn bảng. |
| `sort_by` | `None` | *(chỉ `ffill` / `bfill` / `interpolate`)* Cột thời gian dùng để sắp thứ tự trước khi điền. Kết quả vẫn giữ thứ tự dòng ban đầu. |
| `reference` | `None` | *(chỉ `mean` / `median` / `mode`)* Bảng dùng để **tính** thống kê, thường là train. |

```python
df = dp.fill_missing(df, ["price"], strategy="median", group_by="category")
df = dp.fill_missing(df, ["city"], strategy="constant", fill_value="Unknown")
df = dp.fill_missing(df, ["temp"], strategy="interpolate", group_by="sensor_id", sort_by="timestamp")
test = dp.fill_missing(test, ["price"], strategy="median", reference=train)
```

### `fill_missing_with_model(df, columns=None, method="knn", reference=None, n_neighbors=5, max_iter=10, random_state=0)`

Điền missing cho **cột số** bằng mô hình học từ các cột số khác. Phù hợp khi các cột tương quan với nhau (vd chiều cao và cân nặng).

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `columns` | `None` | Các cột số tham gia, vừa là cột được điền vừa là thông tin để điền. `None` = mọi cột số. |
| `method` | `"knn"` | **`"knn"`**: lấy trung bình của K dòng giống nhất. **`"iterative"`**: hồi quy lặp từng cột theo các cột khác (kiểu MICE), chậm hơn nhưng thường chính xác hơn. |
| `reference` | `None` | Bảng dùng để **học** mô hình (thường là train). |
| `n_neighbors` | `5` | *(knn)* Số dòng láng giềng dùng để tính giá trị điền. |
| `max_iter` | `10` | *(iterative)* Số vòng lặp tối đa. |
| `random_state` | `0` | *(iterative)* Seed để kết quả lặp lại được giữa các lần chạy. |

---

## 5. Text & category (`text`)

### `clean_text(df, columns=None, strip=True, collapse_whitespace=True, case=None, unicode_form="NFC", remove_pattern=None, empty_to_missing=True)`

Làm sạch ô chữ. Ô không phải chữ (số, NaN, …) được giữ nguyên.

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `columns` | `None` | Cột cần xử lý. `None` = mọi cột chữ. |
| `strip` | `True` | Bỏ khoảng trắng ở hai đầu. |
| `collapse_whitespace` | `True` | Gộp nhiều khoảng trắng, tab hoặc xuống dòng liên tiếp thành một dấu cách. |
| `case` | `None` | Đổi chữ hoa/thường: `None` giữ nguyên, `"lower"` chữ thường, `"upper"` chữ hoa, `"title"` viết hoa chữ cái đầu mỗi từ. |
| `unicode_form` | `"NFC"` | Chuẩn Unicode. Tiếng Việt có thể được gõ theo 2 cách (dựng sẵn / tổ hợp): nhìn giống hệt nhau nhưng máy so sánh lại khác nhau. Chuẩn hóa về `"NFC"` sửa được việc này. `None` = bỏ qua. |
| `remove_pattern` | `None` | Regex cho phần cần xóa, vd `r"[^\w\s]"` để xóa ký tự đặc biệt, `r"<[^>]+>"` để xóa thẻ HTML. |
| `empty_to_missing` | `True` | Chuỗi rỗng sau khi làm sạch thì đổi thành NaN. |

### `normalize_categories(df, columns, mapping, ignore_case=True, ignore_accents=True, unmatched="keep", replacement="Other")`

Gộp các cách viết khác nhau của cùng một giá trị về **một tên chuẩn**.

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `columns` | **bắt buộc** | Cột cần chuẩn hóa. Nhiều cột có thể dùng chung một `mapping`. |
| `mapping` | **bắt buộc** | dict `{tên_chuẩn: [các biến thể]}`. Tên chuẩn cũng tự được coi là biến thể của chính nó. |
| `ignore_case` | `True` | So khớp không phân biệt hoa thường. |
| `ignore_accents` | `True` | So khớp không phân biệt dấu: `"Ha Noi"` khớp với `"Hà Nội"`. |
| `unmatched` | `"keep"` | Xử lý giá trị không khớp biến thể nào: `"keep"` giữ nguyên, `"missing"` đổi thành NaN, `"replace"` đổi thành `replacement`. |
| `replacement` | `"Other"` | Giá trị thay thế khi `unmatched="replace"`. |

```python
mapping = {"Hà Nội": ["HN", "Hanoi"], "TP.HCM": ["HCM", "Sài Gòn", "Saigon"]}
df = dp.normalize_categories(df, "city", mapping)
```

---

## 6. Outlier & giá trị vô lý (`outliers`)

### `compute_outlier_bounds(series, method="iqr", threshold=None)`

Tính ngưỡng `(dưới, trên)`; giá trị nằm ngoài khoảng này là outlier. Nếu cột không có độ phân tán (vd mọi giá trị bằng nhau), hàm trả `(-inf, inf)` và ghi cảnh báo.

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `series` | **bắt buộc** | Một cột số, vd `train["price"]`. |
| `method` | `"iqr"` | **`"iqr"`**: `[Q1 − k·IQR, Q3 + k·IQR]`, bền với dữ liệu lệch, nên dùng mặc định. **`"zscore"`**: `[mean − k·std, mean + k·std]`, hợp với dữ liệu gần phân phối chuẩn. **`"mad"`**: dùng trung vị và MAD (Modified Z-score), bền hơn `zscore` khi có outlier rất lớn. **`"percentile"`**: cắt `k` phần đuôi mỗi bên. |
| `threshold` | `None` | Hệ số `k`. `None` = mặc định theo phương pháp: `iqr=1.5`, `zscore=3`, `mad=3.5`, `percentile=0.01` (tức giữ khoảng P1–P99). Tăng `k` thì ít giá trị bị coi là outlier hơn. |

### `handle_outliers(df, columns=None, method="iqr", threshold=None, action="clip", bounds=None, reference=None, flag_suffix="_is_outlier")`

Phát hiện và xử lý outlier trên cột số.

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `columns` | `None` | Cột số cần xử lý. `None` = mọi cột số. |
| `method`, `threshold` | `"iqr"`, `None` | Cách tính ngưỡng, xem `compute_outlier_bounds`. |
| `action` | `"clip"` | **`"clip"`**: kéo giá trị về ngưỡng gần nhất (winsorize), không mất dòng. **`"remove"`**: xóa cả dòng nếu có outlier ở **bất kỳ** cột nào được chọn. **`"missing"`**: đổi outlier thành NaN để điền lại sau. **`"flag"`**: giữ nguyên giá trị, thêm cột cờ 0/1. |
| `bounds` | `None` | Ngưỡng tính sẵn `{tên_cột: (dưới, trên)}`. Cột có trong `bounds` thì không tính lại. |
| `reference` | `None` | Bảng dùng để tính ngưỡng (thường là train). |
| `flag_suffix` | `"_is_outlier"` | Hậu tố tên cột cờ khi `action="flag"`: `price` → `price_is_outlier`. |

```python
bounds = {c: dp.compute_outlier_bounds(train[c]) for c in ["price", "qty"]}
train = dp.handle_outliers(train, ["price", "qty"], bounds=bounds)
test = dp.handle_outliers(test, ["price", "qty"], bounds=bounds)
```

### `enforce_value_rules(df, rules, action="missing", flag_suffix="_is_invalid")`

Xử lý giá trị **vô lý theo nghiệp vụ**: tuổi âm, giá âm, ngày ở tương lai, giá trị không thuộc danh sách cho phép. Ô rỗng không bị coi là vi phạm.

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `rules` | **bắt buộc** | dict `{tên_cột: quy_tắc}`. Mỗi quy tắc là dict có thể gồm: `"min"` (giá trị nhỏ nhất hợp lệ, tính cả biên), `"max"` (giá trị lớn nhất hợp lệ, tính cả biên), `"allowed"` (danh sách các giá trị hợp lệ duy nhất). |
| `action` | `"missing"` | `"missing"` đổi ô vi phạm thành NaN, `"clip"` kéo về `min`/`max` (vi phạm `allowed` vẫn thành NaN), `"remove"` xóa dòng, `"flag"` thêm cột cờ 0/1. |
| `flag_suffix` | `"_is_invalid"` | Hậu tố tên cột cờ khi `action="flag"`. |

```python
rules = {
    "age": {"min": 0, "max": 120},
    "price": {"min": 0},
    "order_date": {"max": pd.Timestamp.today()},
    "gender": {"allowed": ["Nam", "Nữ"]},
}
df = dp.enforce_value_rules(df, rules, action="missing")
```

---

## 7. Ngày giờ (`datetime_utils`)

### `parse_datetime(df, columns, formats=None, dayfirst=True, infer_remaining=True, utc=False, unit=None)`

Chuyển cột sang kiểu datetime, chịu được **nhiều định dạng lẫn lộn trong một cột**. Giá trị không parse được thành `NaT`, và số lượng được ghi log.

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `columns` | **bắt buộc** | Cột cần chuyển. |
| `formats` | `None` | Danh sách định dạng thử **lần lượt**, vd `["%d/%m/%Y", "%Y-%m-%d %H:%M:%S"]`. Giá trị khớp định dạng nào trước thì lấy định dạng đó. Nên khai báo rõ để tránh nhầm ngày với tháng (`03/04` là 3/4 hay 4/3). Ký hiệu: `%d` ngày, `%m` tháng, `%Y` năm 4 số, `%H:%M:%S` giờ:phút:giây. |
| `dayfirst` | `True` | Khi phải tự đoán, coi số đầu tiên là **ngày** (kiểu Việt Nam / châu Âu). Đặt `False` cho dữ liệu kiểu Mỹ `mm/dd/yyyy`. |
| `infer_remaining` | `True` | Sau khi thử hết `formats`, để pandas tự đoán các giá trị còn lại. Đặt `False` nếu chỉ muốn chấp nhận đúng các định dạng đã khai báo. |
| `utc` | `False` | `True` = đưa mọi giá trị về múi giờ UTC. Nên bật khi dữ liệu chứa nhiều múi giờ (vd `+07:00` và `Z`). |
| `unit` | `None` | Dùng khi cột là **Unix timestamp dạng số**: `"s"` giây, `"ms"` mili giây, `"us"` micro giây, `"ns"` nano giây. |

```python
df = dp.parse_datetime(df, ["order_date"], formats=["%d/%m/%Y", "%Y-%m-%d"])
df = dp.parse_datetime(df, "created_ts", unit="s")
```

### `convert_timezone(df, columns, target_tz="UTC", source_tz=None, drop_timezone=False)`

Đưa cột datetime về một múi giờ chung. Cột phải là datetime (gọi `parse_datetime` trước).

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `columns` | **bắt buộc** | Cột datetime cần chuyển. |
| `target_tz` | `"UTC"` | Múi giờ đích, vd `"UTC"`, `"Asia/Ho_Chi_Minh"`. |
| `source_tz` | `None` | Múi giờ gốc của dữ liệu **chưa có** thông tin múi giờ. **Bắt buộc** nếu cột chưa có múi giờ. Giờ không tồn tại do đổi giờ mùa hè được dời tới; giờ bị lặp được đặt thành `NaT`. |
| `drop_timezone` | `False` | `True` = bỏ nhãn múi giờ sau khi chuyển, chỉ giữ giờ địa phương của `target_tz`. |

### `fill_time_gaps(df, time_column, freq, group_by=None, indicator_column=None)`

Thêm các mốc thời gian bị thiếu để chuỗi thời gian liên tục. Dòng mới có NaN ở các cột dữ liệu; điền sau bằng `fill_missing` (vd `strategy="ffill"`). Nếu có mốc thời gian trùng, hàm báo lỗi (cần khử trùng trước).

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `time_column` | **bắt buộc** | Cột thời gian (kiểu datetime). |
| `freq` | **bắt buộc** | Bước thời gian: `"D"` ngày, `"h"` giờ, `"min"` phút, `"W"` tuần, `"MS"` đầu tháng, `"ME"` cuối tháng. |
| `group_by` | `None` | Lấp khoảng trống riêng cho từng nhóm (vd từng cửa hàng, từng cảm biến). Mỗi nhóm chạy từ mốc nhỏ nhất đến mốc lớn nhất của chính nhóm đó. |
| `indicator_column` | `None` | Nếu đặt tên, thêm cột 0/1 đánh dấu dòng vừa được thêm. |

```python
df = dp.fill_time_gaps(df, "date", "D", group_by="store_id", indicator_column="is_gap")
df = dp.fill_missing(df, ["sales"], strategy="ffill", group_by="store_id", sort_by="date")
```

---

## 8. Ràng buộc chéo (`constraints`)

Một **quy tắc** là điều kiện mà dòng **hợp lệ** phải thỏa. Có 2 cách viết:
- **Chuỗi biểu thức**: `"end_date >= start_date"`, `"qty > 0"`.
- **Hàm** nhận DataFrame và trả về Series True/False: `lambda d: (d.total - d.qty * d.price).abs() < 0.01`.

⚠️ So sánh với NaN luôn cho False, tức là bị tính là vi phạm. Muốn bỏ qua ô rỗng thì viết rõ trong điều kiện: `"end_date.isna() | (end_date >= start_date)"`.

### `find_rule_violations(df, rules)`

Trả về bảng True/False: mỗi cột là một quy tắc, `True` nghĩa là dòng đó **vi phạm**. Dùng để xem trước.

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `rules` | **bắt buộc** | dict `{tên_quy_tắc: điều_kiện_hợp_lệ}`. |

### `enforce_rules(df, rules, action="flag", flag_prefix="violates_")`

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `rules` | **bắt buộc** | Như trên. |
| `action` | `"flag"` | `"flag"` thêm một cột 0/1 cho mỗi quy tắc; `"remove"` xóa dòng vi phạm **bất kỳ** quy tắc nào; `"raise"` báo lỗi nếu có dòng vi phạm (dùng như một bước kiểm tra bắt buộc trong pipeline). |
| `flag_prefix` | `"violates_"` | Tiền tố tên cột cờ: quy tắc `end_after_start` → cột `violates_end_after_start`. |

```python
rules = {
    "end_after_start": "end_date >= start_date",
    "total_matches": lambda d: (d.total - d.qty * d.price).abs() < 0.01,
}
df = dp.enforce_rules(df, rules, action="flag")
```

### `find_orphan_keys(child, parent, child_key, parent_key=None)`

Tìm các dòng ở bảng con có khóa **không tồn tại** ở bảng cha, vd đơn hàng có `customer_id` không có trong bảng khách hàng.

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `child` | **bắt buộc** | Bảng chứa khóa ngoại (vd `orders`). |
| `parent` | **bắt buộc** | Bảng chứa khóa chính (vd `customers`). |
| `child_key` | **bắt buộc** | Cột khóa ở bảng con: một cột hoặc danh sách cột. |
| `parent_key` | `None` | Cột khóa tương ứng ở bảng cha. `None` = trùng tên với `child_key`. |

---

## 9. Reshape (`reshape`)

### `wide_to_long(df, id_columns, value_columns=None, variable_name="variable", value_name="value", drop_missing=False)`

Bảng ngang → bảng dọc (melt). Ví dụ `[id, sales_2023, sales_2024]` → `[id, variable, value]`.

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `id_columns` | **bắt buộc** | Cột giữ nguyên, định danh mỗi dòng. |
| `value_columns` | `None` | Cột cần "xoay" thành dòng. `None` = mọi cột còn lại. |
| `variable_name` | `"variable"` | Tên cột mới chứa **tên cột gốc**. |
| `value_name` | `"value"` | Tên cột mới chứa **giá trị**. |
| `drop_missing` | `False` | Bỏ các dòng có giá trị rỗng sau khi xoay. |

### `long_to_wide(df, index, columns, values, aggfunc="first", fill_value=None, name_separator="_")`

Bảng dọc → bảng ngang (pivot), ngược với `wide_to_long`.

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `index` | **bắt buộc** | Cột định danh mỗi dòng của bảng kết quả. |
| `columns` | **bắt buộc** | Cột mà **mỗi giá trị** của nó trở thành một cột mới. |
| `values` | **bắt buộc** | Cột chứa giá trị điền vào bảng. |
| `aggfunc` | `"first"` | Cách gộp khi một ô có nhiều giá trị: `"first"`, `"sum"`, `"mean"`, `"count"`, `"max"`, … |
| `fill_value` | `None` | Giá trị cho ô không có dữ liệu. `None` = giữ NaN. |
| `name_separator` | `"_"` | Ký tự nối tên khi tên cột có nhiều cấp, vd `sales_2024`. |

### `aggregate(df, group_by, aggregations, name_separator="_", keep_missing_groups=True)`

Gom nhóm và tính tổng hợp, trả về bảng phẳng có tên cột dễ đọc.

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `group_by` | **bắt buộc** | Cột dùng để gom nhóm. |
| `aggregations` | **bắt buộc** | dict `{tên_cột: hàm hoặc danh sách hàm}`, vd `{"price": ["mean", "max"], "order_id": "count"}` → cột `price_mean`, `price_max`, `order_id_count`. |
| `name_separator` | `"_"` | Ký tự nối tên cột với tên hàm. |
| `keep_missing_groups` | `True` | `True` = giữ nhóm có khóa rỗng (NaN) thay vì bỏ đi. |

### `explode_column(df, column, separator=None, strip=True, drop_empty=True)`

Tách một ô chứa nhiều giá trị thành nhiều dòng: `tags = "a, b, c"` → 3 dòng, các cột khác được lặp lại.

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `column` | **bắt buộc** | Cột cần tách. |
| `separator` | `None` | Ký tự phân tách nếu ô là chuỗi, vd `","` hoặc `"\|"`. `None` nếu ô đã là list/tuple. |
| `strip` | `True` | Bỏ khoảng trắng hai đầu của từng phần tử sau khi tách. |
| `drop_empty` | `True` | Bỏ phần tử chuỗi rỗng sinh ra khi tách (vd dấu phẩy thừa ở cuối `"a,b,"`). Dòng có ô rỗng ngay từ đầu vẫn được giữ lại. |

### `flatten_json_column(df, column, separator=".", max_level=None, prefix=None, drop_original=True)`

Tách một cột chứa JSON / dict lồng nhau thành nhiều cột phẳng: `{"address": {"city": "HN"}}` → cột `info.address.city`.

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `column` | **bắt buộc** | Cột chứa dict hoặc chuỗi JSON. Ô rỗng hoặc JSON lỗi → NaN. |
| `separator` | `"."` | Ký tự nối tên các cấp lồng nhau. |
| `max_level` | `None` | Độ sâu tối đa cần làm phẳng. `None` = làm phẳng hết. |
| `prefix` | `None` | Tiền tố cho tên cột mới. `None` = `"<tên cột><separator>"`. |
| `drop_original` | `True` | Xóa cột JSON gốc sau khi tách. |

---

## Pipeline mẫu & thứ tự nên làm

1. Chuẩn hóa tên cột → đổi missing giả về NaN → làm sạch text
2. Ép kiểu (số, ngày giờ, bool) → gộp biến thể category
3. Khử trùng. Làm **sau** bước 1–2, vì `" HN"` và `"Ha Noi"` chỉ trùng nhau sau khi chuẩn hóa.
4. Quy tắc nghiệp vụ → outlier
5. `add_missing_flags` → điền missing

```python
clean = (
    raw
    .pipe(dp.standardize_column_names)
    .pipe(dp.standardize_missing, sentinel_values={"price": [-999]})
    .pipe(dp.clean_text)
    .pipe(dp.to_numeric, ["price"])
    .pipe(dp.parse_datetime, ["order_date"], formats=["%d/%m/%Y"])
    .pipe(dp.normalize_categories, "city", {"Hà Nội": ["HN"]})
    .pipe(dp.drop_key_duplicates, "customer_id", strategy="last", order_by="order_date")
    .pipe(dp.enforce_value_rules, {"price": {"min": 0}})
    .pipe(dp.handle_outliers, ["price"], action="clip")
    .pipe(dp.add_missing_flags, ["price"])
    .pipe(dp.fill_missing, ["price"], strategy="median", group_by="category")
)
```

Ví dụ chạy được đầy đủ: [examples/demo_pipeline.py](../examples/demo_pipeline.py).

## Chạy test

```bash
pip install -r requirements.txt
python -m pytest -q
```
