"""data_processing: bộ hàm làm sạch dữ liệu dạng bảng dùng chung cho mọi dataset.

Quy ước cho mọi hàm:
- Nhận DataFrame, trả về DataFrame MỚI (không sửa dữ liệu gốc).
- Không hard-code tên cột: mọi cột / ngưỡng / quy tắc đều truyền qua tham số.
- Mỗi bước ghi log những gì đã thay đổi (logger "data_processing").

Các hàm nối được với nhau bằng DataFrame.pipe, xem README trong thư mục này.
"""

from .constraints import enforce_rules, find_orphan_keys, find_rule_violations
from .datetime_utils import convert_timezone, fill_time_gaps, parse_datetime
from .duplicates import drop_exact_duplicates, drop_key_duplicates, find_duplicates
from .io import load_many, load_table, safe_merge
from .missing import (
    add_missing_flags,
    drop_high_missing,
    fill_missing,
    fill_missing_with_model,
    standardize_missing,
)
from .outliers import compute_outlier_bounds, enforce_value_rules, handle_outliers
from .reshape import aggregate, explode_column, flatten_json_column, long_to_wide, wide_to_long
from .schema import (
    auto_convert_numeric,
    drop_useless_columns,
    standardize_column_names,
    to_boolean,
    to_category,
    to_numeric,
)
from .text import clean_text, normalize_categories

__all__ = [
    # Cấp 1 - đọc & gộp
    "load_table", "load_many", "safe_merge",
    # Cấp 1 - schema
    "standardize_column_names", "to_numeric", "auto_convert_numeric",
    "to_boolean", "to_category", "drop_useless_columns",
    # Cấp 1 - trùng lặp
    "find_duplicates", "drop_exact_duplicates", "drop_key_duplicates",
    # Cấp 1 + 2 - missing
    "standardize_missing", "drop_high_missing", "add_missing_flags",
    "fill_missing", "fill_missing_with_model",
    # Cấp 1 - text
    "clean_text", "normalize_categories",
    # Cấp 2 - outlier & giá trị vô lý
    "compute_outlier_bounds", "handle_outliers", "enforce_value_rules",
    # Cấp 2 - ngày giờ
    "parse_datetime", "convert_timezone", "fill_time_gaps",
    # Cấp 2 - ràng buộc chéo
    "find_rule_violations", "enforce_rules", "find_orphan_keys",
    # Cấp 2 - reshape
    "wide_to_long", "long_to_wide", "aggregate", "explode_column", "flatten_json_column",
]
