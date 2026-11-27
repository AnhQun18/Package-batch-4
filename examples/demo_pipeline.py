"""Ví dụ nối các bước cấp 1 + 2 thành một pipeline trên dữ liệu bẩn tự tạo.

Chạy từ thư mục gốc repo:  python examples/demo_pipeline.py
Mọi tên cột / ngưỡng / quy tắc đều nằm ở phần CONFIG; đổi dataset chỉ cần sửa CONFIG.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_processing as dp  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

raw = pd.DataFrame({
    "Mã KH": [1, 1, 2, 3, 4, 5],
    "Thành Phố": [" HN", "Ha Noi", "Sài Gòn", "hcm", "N/A", "Huế"],
    "Giá ($)": ["1,200", "1,200", "(50)", "300", "-999", "999,999"],
    "Ngày Đặt": ["03/04/2024", "03/04/2024", "2024-05-06", "07/08/2024", "abc", "01/01/2099"],
    "Ghi chú": [None] * 6,
})

# ---------------- CONFIG (chỉ phần này phụ thuộc dataset) ----------------
KEY = "ma_kh"
NUMERIC_COLUMNS = ["gia"]
DATE_COLUMNS = ["ngay_dat"]
DATE_FORMATS = ["%d/%m/%Y", "%Y-%m-%d"]
CITY_MAPPING = {"Hà Nội": ["HN"], "TP.HCM": ["HCM", "Sài Gòn"]}
VALUE_RULES = {"gia": {"min": 0}, "ngay_dat": {"max": pd.Timestamp.today()}}
# -------------------------------------------------------------------------

clean = (
    raw
    .pipe(dp.standardize_column_names)
    .pipe(dp.drop_useless_columns, exclude=KEY)
    .pipe(dp.standardize_missing, sentinel_values={"gia": ["-999"]})
    .pipe(dp.clean_text)
    .pipe(dp.to_numeric, NUMERIC_COLUMNS)
    .pipe(dp.parse_datetime, DATE_COLUMNS, formats=DATE_FORMATS)
    .pipe(dp.normalize_categories, "thanh_pho", CITY_MAPPING)
    # Khử trùng SAU khi chuẩn hóa: " HN" và "Ha Noi" chỉ trùng nhau sau bước trên.
    .pipe(dp.drop_key_duplicates, KEY, strategy="most_complete")
    .pipe(dp.enforce_value_rules, VALUE_RULES, action="missing")
    .pipe(dp.handle_outliers, NUMERIC_COLUMNS, method="iqr", action="flag")
    .pipe(dp.add_missing_flags, NUMERIC_COLUMNS)
    .pipe(dp.fill_missing, NUMERIC_COLUMNS, strategy="median")
    .pipe(dp.fill_missing, "thanh_pho", strategy="constant", fill_value="Unknown")
)

print(clean.to_string())
