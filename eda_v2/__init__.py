"""EDA v2: các hàm visualize nhanh dữ liệu bảng, mỗi file một hàm.

CẤU TRÚC PACKAGE:

    plot_category_distribution.py  top K, nhóm Other, count, percentage
    plot_numeric_comparison.py     so sánh phân phối giữa các nhóm (boxplot), tuỳ chọn ẩn outlier
    plot_numeric_distribution.py   histogram + mean, median, percentile
    plot_entity_coverage.py        tổng hợp hoạt động entity theo thời gian (heatmap)
    plot_temporal_distribution.py  aggregate, resample theo thời gian (line)

Nguyên tắc:
    - Mỗi hàm nhận df + tên cột, tự vẽ và trả về (figure, bảng số liệu đằng sau chart).
    - Kích thước figure qua **params (figsize), có default để gọi nhanh.
    - Không plt.show(), không lưu file, không sửa df.
"""
from .plot_category_distribution import plot_category_distribution
from .plot_entity_coverage import plot_entity_coverage
from .plot_numeric_comparison import plot_numeric_comparison
from .plot_numeric_distribution import plot_numeric_distribution
from .plot_temporal_distribution import plot_temporal_distribution

__all__ = ["plot_category_distribution", "plot_numeric_comparison", "plot_numeric_distribution",
           "plot_entity_coverage", "plot_temporal_distribution"]
