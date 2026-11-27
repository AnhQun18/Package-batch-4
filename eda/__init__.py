"""EDA: hàm visualize nhanh dữ liệu bảng (matplotlib + seaborn).

CẤU TRÚC PACKAGE:

    numeric.py      plot_numeric      cột số (hist, box, kde, violin, ecdf)
    categorical.py  plot_categorical  cột phân loại (bar tần suất)
    temporal.py     plot_temporal     cột thời gian (line, bar, cumulative, weekday, month, hour, heatmap)
    _common.py      hỗ trợ dùng chung: chọn cột, lưới Axes, nhãn, import matplotlib/seaborn

Nguyên tắc:
    - Một cột hay cả bảng là tùy người dùng: `cols=None` = mọi cột hợp lệ theo dtype,
      `cols="a"` = một cột, `cols=[...]` = tập cột; `exclude=[...]` để bỏ bớt.
    - Không hard-code: loại chart, bố cục lưới, nhãn, tiêu đề, kích thước... đều là tham số;
      default chỉ để gọi nhanh. Style/màu chuyển qua **kwargs cho seaborn.
    - Mỗi hàm vẽ = bước chuẩn bị dữ liệu (nội bộ, test được không cần matplotlib) + render.
    - Mỗi hàm vẽ trả về (Figure, DataFrame): figure và bảng số liệu đằng sau chart.
      Không plt.show(), không lưu file. Nhận `ax=` để vẽ vào Axes có sẵn.
    - Không sửa df; giá trị thiếu (và ±inf với numeric) bị loại, số lượng ghi ở tiêu đề mỗi ô.
    - Không phụ thuộc tên cột/dataset cụ thể, không phụ thuộc profiling.py.
    - matplotlib/seaborn là optional dependency (`pip install ds-kit[viz]`), chỉ import khi vẽ.

Tham số chung cho cả 3 hàm vẽ:
    cols      None = mọi cột hợp lệ theo dtype; str = một cột; list = tập cột (giữ thứ tự truyền vào).
    exclude   cột bỏ ra khỏi lựa chọn (vd cột ID); None = không bỏ.
    ncols     số ô trên một hàng của lưới; số hàng tự tính.
    ax        Axes hoặc dãy Axes (đủ số ô được vẽ) để vẽ vào; None = tạo lưới mới.
    figsize   kích thước figure khi tạo mới; None = tự tính theo số ô.
    subplot_kws  dict chuyển cho plt.subplots (sharex, sharey, ...); chỉ dùng khi ax=None.
    title     tiêu đề figure: None hoặc "" = không; chuỗi = tự đặt.
    col_title tiêu đề từng ô: None = tự động (tên cột + n, missing...); "" = bỏ; chuỗi format
    với các khóa {column}, {n}, {missing}, ... (numeric/temporal thêm {kind}) = tự đặt.
    **kwargs  chuyển nguyên cho hàm seaborn tương ứng (color, palette, ...).
    Trả về: (Figure, DataFrame bảng long là số liệu đằng sau chart).

Interface document: docs/interfaces/eda.md
"""
from .categorical import plot_categorical
from .numeric import plot_numeric
from .temporal import plot_temporal

__all__ = ["plot_numeric", "plot_categorical", "plot_temporal"]
