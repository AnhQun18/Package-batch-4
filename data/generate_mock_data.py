"""Script tạo dữ liệu giả lập (mock data) đa chiều để kiểm thử toàn diện package `qc`.

Bao gồm:
- Tỷ lệ thiếu (missing values) từ 0% đến 100% với nhiều cột.
- Dòng trùng (duplicates) ở nhiều cấp độ: toàn bộ dòng, trùng khóa đơn, trùng khóa tổ hợp.
- Dữ liệu ngoại lai (outliers) trên các phân phối khác nhau, kèm trường hợp biên (std=0, inf, all-nan).
- Độ phủ thực thể theo thời gian (entity coverage): khách hàng theo tháng/ngày, có tháng sụt giảm và tháng gián đoạn (gap).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def generate_qc_mock_data(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Tạo DataFrame giả lập giao dịch thương mại điện tử phục vụ kiểm thử QC."""
    rng = np.random.default_rng(seed)

    # 1. Thực thể và Thời gian (Entity Coverage)
    # 300 khách hàng hoạt động trong vòng 18 tháng (2024-01 đến 2025-06)
    n_customers = 300
    customer_ids = [f"CUST_{i:04d}" for i in range(1, n_customers + 1)]

    # Tạo timeline có các tháng bình thường, 1 tháng sụt giảm mạnh (2024-02), và 1 tháng gián đoạn hoàn toàn (2024-08)
    months = pd.date_range("2024-01-01", "2025-06-30", freq="MS")
    records: list[dict] = []

    # Phân bổ số giao dịch cho từng tháng
    for m in months:
        m_str = m.strftime("%Y-%m")
        if m_str == "2024-08":
            # Tháng gián đoạn hoàn toàn: 0 giao dịch (để test fill_gaps)
            continue
        elif m_str == "2024-02":
            # Tháng Tết sụt giảm mạnh: ít khách hàng tham gia (chỉ ~35 khách ~ 11.6% coverage)
            n_trans = 50
            active_pool = rng.choice(customer_ids, size=35, replace=False)
        elif m_str in ("2024-11", "2024-12"):
            # Mùa cao điểm: hầu hết khách hàng đều tham gia
            n_trans = int(n_rows * 0.1)
            active_pool = rng.choice(customer_ids, size=int(n_customers * 0.85), replace=False)
        else:
            # Tháng bình thường
            n_trans = int(n_rows * 0.055)
            active_pool = rng.choice(customer_ids, size=int(n_customers * 0.60), replace=False)

        cust_choices = rng.choice(active_pool, size=n_trans)
        # Random ngày trong tháng
        days_in_month = pd.Period(m_str).days_in_month
        random_days = rng.integers(1, days_in_month + 1, size=n_trans)
        random_hours = rng.integers(0, 24, size=n_trans)
        random_minutes = rng.integers(0, 60, size=n_trans)

        for c_id, day, h, minute in zip(cust_choices, random_days, random_hours, random_minutes):
            records.append({
                "customer_id": c_id,
                "order_date": pd.Timestamp(year=m.year, month=m.month, day=int(day), hour=int(h), minute=int(minute)),
            })

    base_df = pd.DataFrame(records)
    # Lấy đúng số dòng hoặc điều chỉnh cho vừa n_rows
    if len(base_df) > n_rows:
        base_df = base_df.iloc[:n_rows].copy()
    n = len(base_df)

    # 2. Khóa chính
    base_df["order_id"] = [f"ORD_{100000 + i}" for i in range(n)]

    # 3. Các cột số và ngoại lai (Outliers)
    # - total_amount: log-normal phân phối chuẩn (200k - 2 triệu), có outlier tiền khủng (>50 triệu) và số âm (lỗi nhập)
    normal_amounts = np.exp(rng.normal(loc=13.2, scale=0.6, size=n))  # ~ 200k - 2M
    outlier_idx_amt = rng.choice(n, size=int(n * 0.035), replace=False)
    normal_amounts[outlier_idx_amt[:len(outlier_idx_amt)//2]] = rng.uniform(40_000_000, 150_000_000, size=len(outlier_idx_amt)//2)
    normal_amounts[outlier_idx_amt[len(outlier_idx_amt)//2:]] = rng.uniform(-100_000, -10_000, size=len(outlier_idx_amt) - len(outlier_idx_amt)//2)
    base_df["total_amount"] = np.round(normal_amounts, -3)

    # - customer_age: phân phối chuẩn (mean=32, std=7), có outlier như 120, 150, -5
    ages = rng.normal(loc=32, scale=7, size=n)
    outlier_idx_age = rng.choice(n, size=int(n * 0.025), replace=False)
    ages[outlier_idx_age[:len(outlier_idx_age)//2]] = rng.choice([115, 128, 145, 999], size=len(outlier_idx_age)//2)
    ages[outlier_idx_age[len(outlier_idx_age)//2:]] = rng.choice([-5, -1, 0, 3], size=len(outlier_idx_age) - len(outlier_idx_age)//2)
    base_df["customer_age"] = np.round(ages).astype(int)

    # - delivery_days: phân phối poisson (lambda=3), outlier 60-120 ngày
    deliv = rng.poisson(lam=3.2, size=n)
    outlier_idx_deliv = rng.choice(n, size=int(n * 0.02), replace=False)
    deliv[outlier_idx_deliv] = rng.integers(45, 120, size=len(outlier_idx_deliv))
    base_df["delivery_days"] = deliv

    # - discount_rate: 0.0 - 0.3, outlier > 1.0 (150%, 400%)
    discounts = rng.uniform(0.0, 0.25, size=n)
    outlier_idx_disc = rng.choice(n, size=int(n * 0.015), replace=False)
    discounts[outlier_idx_disc] = rng.uniform(1.5, 5.0, size=len(outlier_idx_disc))
    base_df["discount_rate"] = np.round(discounts, 2)

    # - constant_fee: hằng số 15000 (std=0, MAD=0) để kiểm tra xử lý chia cho 0 trong plot_outlier_rate
    base_df["constant_fee"] = 15000.0

    # - metric_with_inf: chứa np.inf, -np.inf để kiểm tra việc lọc bỏ giá trị không hữu hạn
    inf_series = rng.normal(50, 10, size=n)
    inf_idx = rng.choice(n, size=int(n * 0.02), replace=False)
    inf_series[inf_idx[:len(inf_idx)//2]] = np.inf
    inf_series[inf_idx[len(inf_idx)//2:]] = -np.inf
    base_df["metric_with_inf"] = inf_series

    # 4. Các cột phân loại (Categorical)
    base_df["payment_method"] = rng.choice(["credit_card", "momo", "cod", "bank_transfer", "vnpay"], size=n, p=[0.35, 0.25, 0.20, 0.15, 0.05])
    base_df["order_status"] = rng.choice(["COMPLETED", "DELIVERING", "PENDING", "CANCELLED", "REFUNDED"], size=n, p=[0.70, 0.12, 0.08, 0.06, 0.04])
    base_df["shipping_region"] = rng.choice(["Miền Bắc", "Miền Nam", "Miền Trung"], size=n, p=[0.45, 0.40, 0.15])
    base_df["shipping_address"] = [f"Số {rng.integers(1, 999)} Đường ABC, Quận {rng.integers(1, 12)}" for _ in range(n)]

    # 5. Các cột bổ sung phục vụ tỷ lệ thiếu (Missing rates đa dạng)
    base_df["voucher_code"] = [f"SALE_{rng.integers(10, 99)}" for _ in range(n)]
    base_df["customer_feedback"] = ["Sản phẩm tốt, giao nhanh!" for _ in range(n)]
    base_df["cancellation_reason"] = rng.choice(["Đổi ý", "Tìm thấy giá rẻ hơn", "Thời gian giao quá lâu", "Sai thông tin đặt hàng"], size=n)
    base_df["internal_device_token"] = [None] * n  # 100% missing

    # Bơm giá trị thiếu theo tỷ lệ cụ thể
    # - total_amount: 1.5% missing
    base_df.loc[rng.choice(n, size=int(n * 0.015), replace=False), "total_amount"] = np.nan
    # - payment_method: 4% missing
    base_df.loc[rng.choice(n, size=int(n * 0.04), replace=False), "payment_method"] = None
    # - shipping_address: 10% missing
    base_df.loc[rng.choice(n, size=int(n * 0.10), replace=False), "shipping_address"] = None
    # - voucher_code: 35% missing (vượt ngưỡng trung bình)
    base_df.loc[rng.choice(n, size=int(n * 0.35), replace=False), "voucher_code"] = None
    # - customer_feedback: 65% missing (vượt ngưỡng cao)
    base_df.loc[rng.choice(n, size=int(n * 0.65), replace=False), "customer_feedback"] = None
    # - cancellation_reason: 88% missing (rất cao)
    base_df.loc[rng.choice(n, size=int(n * 0.88), replace=False), "cancellation_reason"] = None

    # 6. Bơm dòng trùng (Duplicates)
    # - Trùng toàn bộ dòng (exact rows): copy 120 dòng (khoảng 2.4%)
    n_exact_dup = 120
    exact_dup_indices = rng.choice(n, size=n_exact_dup, replace=False)
    exact_duplicates = base_df.iloc[exact_dup_indices].copy()

    # - Trùng theo order_id (khóa đơn bị nhân bản): 100 dòng
    n_key_dup = 100
    key_dup_source = base_df.iloc[rng.choice(n, size=n_key_dup, replace=False)].copy()
    key_dup_source["total_amount"] = key_dup_source["total_amount"] + 1000  # Khác amount nhưng trùng order_id

    # - Trùng tổ hợp (customer_id, order_date): 150 dòng
    n_combo_dup = 150
    combo_dup_source = base_df.iloc[rng.choice(n, size=n_combo_dup, replace=False)].copy()
    combo_dup_source["order_id"] = [f"ORD_RETRY_{i}" for i in range(n_combo_dup)]  # Khác order_id nhưng cùng khách và ngày

    # Nối tất cả các dòng trùng vào DataFrame
    df_all = pd.concat([base_df, exact_duplicates, key_dup_source, combo_dup_source], ignore_index=True)

    # Xáo trộn lại dữ liệu
    df_all = df_all.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df_all


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    out_file = out_dir / "mock_transactions.csv"
    df = generate_qc_mock_data(n_rows=5000, seed=42)
    df.to_csv(out_file, index=False)
    print(f" Đã tạo thành công dữ liệu giả lập {len(df):,} dòng tại: {out_file}")
    print("\nThông tin sơ bộ:")
    print(df.info())
