# `rfm-fe` — Dataset-Agnostic & Model-Agnostic RFM Feature Engineering

`rfm-fe` là một Python package giúp biến đổi **bất kỳ DataFrame giao dịch (`df`) nào** thành bảng đặc trưng (feature matrix) cấp khách hàng xoay quanh **RFM (Recency, Frequency, Monetary)**, sẵn sàng đưa trực tiếp vào **bất kỳ mô hình Machine Learning nào** (Tree-based: XGBoost, LightGBM, Random Forest; hoặc Linear/Neural/Clustering: Logistic Regression, K-Means, MLP...).

---

## 1. Cài đặt

```bash
pip install -e .
```

Yêu cầu: Python >= 3.9, `numpy`, `pandas`, `scikit-learn`.

---

## 2. Cách sử dụng nhanh (Quick Start)

Giả sử bạn đã có sẵn một DataFrame tên là `df`:

### Cách 1: Dùng hàm 1 dòng `engineer_rfm_features(df, ...)`
Phù hợp cho khám phá dữ liệu (EDA), baseline nhanh, hoặc các bài toán phân cụm (Clustering):

```python
from rfm_fe import engineer_rfm_features

features_df = engineer_rfm_features(
    df,
    customer_col="customer_id",   # tên cột ID khách hàng trong df của bạn
    date_col="order_date",        # tên cột ngày/giờ giao dịch
    amount_col="amount",          # tên cột giá trị tiền
    transaction_col="order_id",   # (tùy chọn) dùng nếu 1 đơn hàng có nhiều dòng sản phẩm
    category_col="category",      # (tùy chọn) dùng để tính độ đa dạng danh mục
)
```

### Cách 2: Dùng `RFMFeatureEngineer` chuẩn `scikit-learn` (Khuyên dùng cho Train/Test)
Đảm bảo **không bị rò rỉ dữ liệu (zero data leakage)** khi chia tập train/test:

```python
from rfm_fe import RFMFeatureEngineer

fe = RFMFeatureEngineer(
    customer_col="user_id",
    date_col="tx_time",
    amount_col="total_price",
    reference_date="2026-03-31",  # mốc thời gian chụp (snapshot)
    windows=(30, 90, 180, 365),   # các cửa sổ thời gian (ngày)
    log_transform=False,          # True nếu dùng Linear / Neural / K-Means
    scaler=None,                  # "standard" | "minmax" | "robust" nếu cần scale
)

X_train = fe.fit_transform(train_df)
X_test = fe.transform(test_df)
```

---

## 3. Cách dùng theo từng loại mô hình

| Loại mô hình | Cấu hình gợi ý | Lý do |
| :--- | :--- | :--- |
| **Tree-based** (XGBoost, LightGBM, CatBoost, RandomForest) | `log_transform=False`, `scaler=None` | Cây quyết định bất biến với phép biến đổi đơn điệu và không cần scale. |
| **Linear / Logistic / SVM / Neural Net / K-Means** | `log_transform=True`, `scaler="standard"` (hoặc `"robust"`) | Phân phối tiền tệ & tần suất thường lệch phải (right-skewed); `signed_log1p` + `StandardScaler` giúp mô hình hội tụ nhanh và ổn định. |
| **Phân tích nghiệp vụ / Dashboard** | `include_segment_label=True` | Sinh thêm cột `rfm_segment` (Champions, Loyal, At Risk, Hibernating...). |

---

## 4. Kiến trúc code (Cách hiểu codebase)

Package được chia thành 5 module rõ ràng trong `rfm_fe/`:

1. `rfm_fe/config.py` (`RFMConfig`):
   - Lưu trữ toàn bộ cấu hình và kiểm tra tính hợp lệ của tham số (`__post_init__`).
2. `rfm_fe/validators.py` (`validate_and_prepare`):
   - Kiểm tra các cột bắt buộc có tồn tại trong `df` hay không.
   - Ép kiểu `date_col` về `Timestamp` (bỏ timezone để tính toán đồng nhất) và `amount_col` về số.
   - Cắt dữ liệu theo `reference_date` (`date_col <= reference_date`) để chống rò rỉ tương lai.
   - Xử lý giá trị âm (hoàn tiền/trả hàng) theo 4 chiến lược: `"keep"`, `"clip_zero"`, `"drop"`, `"separate"`.
3. `rfm_fe/features.py`:
   - Chứa các hàm tính toán vectorized bằng `pandas` & `numpy`, chia theo từng nhóm đặc trưng độc lập:
     - `compute_base_rfm`: `recency_days`, `frequency`, `monetary_sum/mean/std/min/max/median`, `tenure_days`, `active_span_days`.
     - `compute_ratios`: `avg_order_value`, `monetary_cv`, `monetary_range`, `frequency_per_month`, `monetary_per_month`, `recency_to_tenure_ratio`, `is_repeat_customer`.
     - `compute_window_features`: `freq_{w}d`, `monetary_sum_{w}d`, `monetary_mean_{w}d` cho từng cửa sổ `w` trong `windows`, kèm vận tốc tăng trưởng (`freq_velocity_*`, `monetary_velocity_*`).
     - `compute_interpurchase_features`: khoảng cách giữa các lần mua (`ipt_mean_days`, `ipt_std_days`, `ipt_min_days`, `ipt_max_days`, `ipt_last_days`).
     - `compute_temporal_features`: `weekend_purchase_ratio`, `preferred_hour_mean`, `active_months_count`.
     - `compute_category_features`: `category_nunique`, `category_top_share`, `category_entropy` (Shannon entropy).
     - `compute_refund_features`: `refund_count`, `refund_amount_sum`, `refund_rate` (khi `handle_negative_amount="separate"`).
4. `rfm_fe/utils.py`:
   - `safe_divide`: chia an toàn, tự động xử lý chia cho 0 / `inf` / `NaN`.
   - `signed_log1p`: $\text{sign}(x) \cdot \ln(1 + |x|)$ giúp xử lý được cả giá trị âm.
   - `fit_quantile_edges` & `apply_quantile_score`: học ngưỡng phân vị trên tập `fit` và áp dụng lại trên `transform` (xử lý an toàn khi dữ liệu có nhiều giá trị trùng lặp).
   - `assign_rfm_segment`: gán nhãn phân khúc khách hàng từ điểm R, F, M.
5. `rfm_fe/engineer.py` (`RFMFeatureEngineer`, `engineer_rfm_features`):
   - Kế thừa `BaseEstimator` và `TransformerMixin` của `scikit-learn`.
   - Điều phối toàn bộ pipeline: validate -> tính raw features -> tính điểm RFM -> `log_transform` & `fillna` -> `scaler` -> căn chỉnh đúng thứ tự cột giữa `fit` và `transform`.

---

## 5. Chạy kiểm thử (Tests)

```bash
pytest -v
```
