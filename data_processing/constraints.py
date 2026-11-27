"""Kiểm tra ràng buộc giữa các cột và toàn vẹn khóa giữa các bảng."""

from __future__ import annotations

from typing import Callable, Optional, Union

import pandas as pd

from ._utils import as_list, log_row_change, logger, resolve_columns

# Một quy tắc là điều kiện mà dòng HỢP LỆ phải thỏa:
# - chuỗi biểu thức kiểu pandas.eval, vd "end_date >= start_date"
# - hoặc hàm nhận DataFrame, trả về Series True/False, vd lambda d: d.total == d.qty * d.price
Rule = Union[str, Callable[[pd.DataFrame], pd.Series]]
RULE_ACTIONS = ("flag", "remove", "raise")


def find_rule_violations(df: pd.DataFrame, rules: dict[str, Rule]) -> pd.DataFrame:
    """Trả về bảng True/False: mỗi cột là một quy tắc, True = dòng đó VI PHẠM.

    Tham số
    -------
    df : DataFrame cần kiểm tra.
    rules : dict {tên_quy_tắc: điều_kiện_hợp_lệ}. Ví dụ:
        {
            "end_after_start": "end_date >= start_date",
            "total_matches": lambda d: (d.total - d.qty * d.price).abs() < 0.01,
        }
        Lưu ý: so sánh với NaN luôn cho False (= vi phạm). Nếu muốn bỏ qua ô rỗng,
        viết rõ trong điều kiện: "end_date.isna() | (end_date >= start_date)".
    """
    violations = {}
    for name, rule in rules.items():
        valid = df.eval(rule, engine="python") if isinstance(rule, str) else rule(df)
        valid = pd.Series(valid, index=df.index).fillna(False).astype(bool)
        violations[name] = ~valid
    return pd.DataFrame(violations, index=df.index)


def enforce_rules(
    df: pd.DataFrame,
    rules: dict[str, Rule],
    action: str = "flag",
    flag_prefix: str = "violates_",
) -> pd.DataFrame:
    """Kiểm tra quy tắc giữa các cột và xử lý dòng vi phạm.

    Tham số
    -------
    df : DataFrame đầu vào.
    rules : Quy tắc hợp lệ, xem `find_rule_violations`.
    action : Cách xử lý dòng vi phạm:
        - "flag": thêm một cột 0/1 cho mỗi quy tắc (tên = flag_prefix + tên quy tắc)
        - "remove": xóa dòng vi phạm bất kỳ quy tắc nào
        - "raise": báo lỗi nếu có dòng vi phạm (dùng như một bước kiểm tra bắt buộc)
    flag_prefix : Tiền tố tên cột cờ khi action="flag".
    """
    if action not in RULE_ACTIONS:
        raise ValueError(f"action phải là một trong {RULE_ACTIONS}")

    violations = find_rule_violations(df, rules)
    counts = violations.sum()
    logger.info("enforce_rules: số dòng vi phạm %s", counts.to_dict())

    if action == "raise":
        failed = counts[counts > 0]
        if not failed.empty:
            raise ValueError(f"Dữ liệu vi phạm quy tắc: {failed.to_dict()}")
        return df

    if action == "remove":
        result = df[~violations.any(axis=1)]
        log_row_change("enforce_rules (remove)", len(df), len(result))
        return result

    df = df.copy()
    for name in violations.columns:
        df[f"{flag_prefix}{name}"] = violations[name].astype("int8")
    return df


def find_orphan_keys(
    child: pd.DataFrame,
    parent: pd.DataFrame,
    child_key: Union[str, list[str]],
    parent_key: Optional[Union[str, list[str]]] = None,
) -> pd.DataFrame:
    """Tìm dòng ở bảng con có khóa KHÔNG tồn tại ở bảng cha (khóa ngoại "mồ côi").

    Ví dụ: đơn hàng có customer_id không có trong bảng khách hàng.

    Tham số
    -------
    child : Bảng chứa khóa ngoại (vd bảng orders).
    parent : Bảng chứa khóa chính (vd bảng customers).
    child_key : Cột khóa ở bảng con (một cột hoặc danh sách cột).
    parent_key : Cột khóa tương ứng ở bảng cha. Để None nếu trùng tên với child_key.
    """
    child_cols = resolve_columns(child, child_key)
    parent_cols = resolve_columns(parent, parent_key if parent_key is not None else child_key)
    if len(child_cols) != len(parent_cols):
        raise ValueError("child_key và parent_key phải có cùng số cột.")

    parent_keys = parent[parent_cols].drop_duplicates().set_axis(child_cols, axis=1)
    matched = child[child_cols].merge(parent_keys, how="left", indicator=True)["_merge"]
    is_orphan = (matched == "left_only").to_numpy()

    orphans = child[is_orphan]
    logger.info("find_orphan_keys: %d dòng mồ côi trên %d", len(orphans), len(child))
    return orphans
