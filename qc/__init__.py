"""QC: kiểm tra chất lượng dữ liệu bảng, mỗi khía cạnh một hàm.

CẤU TRÚC PACKAGE:

    plot_missing.py    plot_missing          tỷ lệ thiếu theo cột
    plot_duplicate.py  plot_duplicate        tỷ lệ dòng trùng (một hoặc nhiều định nghĩa trùng)
    plot_outlier.py    plot_outlier_rate     tỷ lệ outlier theo cột số (iqr, zscore, mad)
    plot_coverage.py   plot_entity_coverage  số entity (vd customer) xuất hiện theo kỳ
    _common.py         hỗ trợ dùng chung: chọn cột, kiểm tra tham số, Axes, bar tỷ lệ

Nguyên tắc:
    - Mỗi hàm tự tính, tự vẽ và trả về (Figure, DataFrame). Không class, không engine.
    - Tỷ lệ là số 0-1. `min_*` lọc bớt cột nhẹ khỏi bảng và chart; `threshold` tô đỏ phần vượt
      ngưỡng (None = không tô); khi truyền threshold, bảng có thêm cột bool `exceed`.
    - Quy ước tham số giống eda: cols/exclude, ax, figsize, title (None = tự động, "" = không),
      **kwargs chuyển cho seaborn.
    - Không sửa df, không plt.show(), không lưu file.
    - matplotlib/seaborn chỉ import khi vẽ (optional dependency `viz`).
"""
from .plot_coverage import plot_entity_coverage
from .plot_duplicate import plot_duplicate
from .plot_missing import plot_missing
from .plot_outlier import plot_outlier_rate

__all__ = ["plot_missing", "plot_duplicate", "plot_outlier_rate", "plot_entity_coverage"]
