import io
import streamlit as st
import pandas as pd

# =========================
# Config
# =========================
DATA_PATH = "obcoupondata.csv"  # 默认数据源文件名
NEW_CUST_CUTOFF = pd.Timestamp("2025-12-31")  # 纯新客：Acquisition Date 晚于该日期
FACE_VALUES = [100, 200, 500, 1000]  # 可申请面额
REQUIRED_COLS = {"advertiser_id", "date", "billed_rev", "acquisition_date"}

st.set_page_config(page_title="Coupon Eligibility Checker", layout="wide")
st.title("Coupon 申请规则校验（批量CID）")
st.caption(
    "规则：申請日T-2回滚30天统计BilledRev；BilledRev ≥ 面额×3；"
    "且满足：纯新客(Acquisition Date > 2025/12/31) 或 召回(此前180天无消耗)。"
)

# =========================
# Helpers: load & normalize
# =========================
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def _validate_and_cast(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_columns(df)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV缺少必要列：{sorted(list(missing))}；必须包含：{sorted(list(REQUIRED_COLS))}")

    df["advertiser_id"] = df["advertiser_id"].astype(str).str.strip()

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["acquisition_date"] = pd.to_datetime(df["acquisition_date"], errors="coerce").dt.normalize()

    df["billed_rev"] = pd.to_numeric(df["billed_rev"], errors="coerce").fillna(0.0)

    # 去掉关键字段缺失的行
    df = df.dropna(subset=["advertiser_id", "date"]).copy()

    # 聚合：同一CID同一天多行时合并
    df = (
        df.groupby(["advertiser_id", "date"], as_index=False)
          .agg(
              billed_rev=("billed_rev", "sum"),
              acquisition_date=("acquisition_date", "min")
          )
    )

    return df.sort_values(["advertiser_id", "date"])

@st.cache_data
def load_from_path(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _validate_and_cast(df)

@st.cache_data
def load_from_upload(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    return _validate_and_cast(df)

# =========================
# Rule functions
# =========================
def window_30d_t_minus_2(apply_dt: pd.Timestamp):
    """
    申请日T-2为窗口截止日（含），回滚30天（含）：
    end = apply_dt - 2 days
    start = end - 29 days
    """
    end = apply_dt - pd.Timedelta(days=2)
    start = end - pd.Timedelta(days=29)
    return start.normalize(), end.normalize()

def sum_billed_rev(df: pd.DataFrame, adv_id: str, start: pd.Timestamp, end: pd.Timestamp) -> float:
    m = (df["advertiser_id"] == adv_id) & (df["date"] >= start) & (df["date"] <= end)
    return float(df.loc[m, "billed_rev"].sum())

def get_acq_date(df: pd.DataFrame, adv_id: str):
    s = df.loc[df["advertiser_id"] == adv_id, "acquisition_date"].dropna()
    if s.empty:
        return None
    return pd.Timestamp(s.min()).normalize()

def reactivation_check(df: pd.DataFrame, adv_id: str, win_start: pd.Timestamp):
    """
    召回（停投>=180天）判定：
    在30天窗口开始日之前，往前连续180天（win_start-180 ~ win_start-1）无消耗(>0) 视为召回。
    """
    lb_start = win_start - pd.Timedelta(days=180)
    lb_end = win_start - pd.Timedelta(days=1)

    spent = sum_billed_rev(df, adv_id, lb_start, lb_end)
    has_spend = spent > 0
    return {
        "lookback_start": lb_start.date().isoformat(),
        "lookback_end": lb_end.date().isoformat(),
        "lookback_spend": round(spent, 2),
        "is_reactivated": (not has_spend),
        "detail": "停投≥180天后再活跃（召回老客）" if (not has_spend) else "活跃客户（180天内有消耗）"
    }

def evaluate_one(df: pd.DataFrame, adv_id: str, apply_dt: pd.Timestamp, face_value: int):
    apply_dt = pd.Timestamp(apply_dt).normalize()

    w_start, w_end = window_30d_t_minus_2(apply_dt)
    rev_30d = sum_billed_rev(df, adv_id, w_start, w_end)

    need_rev = float(face_value) * 3.0
    revenue_ok = rev_30d >= need_rev

    acq = get_acq_date(df, adv_id)
    if acq is None:
        cust_type = "Unknown"
        cust_ok = False
        cust_detail = "缺少Acquisition Date，无法判定纯新客/召回"
        react_info = None
    else:
        # 纯新客
        if acq > NEW_CUST_CUTOFF:
            cust_type = "New"
            cust_ok = True
            cust_detail = f"纯新客（Acquisition Date {acq.date().isoformat()} > {NEW_CUST_CUTOFF.date().isoformat()}）"
            react_info = None
        else:
            # 召回（停投>=180天）
            r = reactivation_check(df, adv_id, w_start)
            cust_type = "Reactivated" if r["is_reactivated"] else "Active"
            cust_ok = r["is_reactivated"]
            cust_detail = r["detail"]
            react_info = r

    final_ok = bool(revenue_ok and cust_ok)

    out = {
        "advertiser_id": adv_id,
        "apply_date": apply_dt.date().isoformat(),
        "face_value": int(face_value),
        "window_start": w_start.date().isoformat(),
        "window_end": w_end.date().isoformat(),
        "billedrev_30d": round(rev_30d, 2),
        "need_billedrev_face_x3": round(need_rev, 2),
        "ok_revenue_rule": revenue_ok,
        "acquisition_date": None if acq is None else acq.date().isoformat(),
        "customer_type": cust_type,
        "customer_detail": cust_detail,
        "final_eligible": final_ok,
    }

    if react_info:
        out.update({
            "react_lookback_start": react_info["lookback_start"],
            "react_lookback_end": react_info["lookback_end"],
            "react_lookback_spend": react_info["lookback_spend"],
            "react_is_reactivated": react_info["is_reactivated"],
        })
    else:
        out.update({
            "react_lookback_start": None,
            "react_lookback_end": None,
            "react_lookback_spend": None,
            "react_is_reactivated": None,
        })

    return out

# =========================
# Load data (upload or local)
# =========================
with st.sidebar:
    st.header("数据源")
    up = st.file_uploader("上传 obcoupondata.csv（可选）", type=["csv"])
    st.caption("必须包含列：advertiser_id, date, billed_rev, acquisition_date")

try:
    if up is not None:
        df = load_from_upload(up.getvalue())
        st.sidebar.success("已使用上传的CSV数据")
    else:
        df = load_from_path(DATA_PATH)
        st.sidebar.info(f"未上传文件，读取本地：{DATA_PATH}")
except Exception as e:
    st.error(f"加载数据失败：{e}")
    st.stop()

adv_list = sorted(df["advertiser_id"].unique().tolist())

# =========================
# Mode selection
# =========================
st.subheader("批量验证模式")
mode = st.radio(
    "选择输入方式",
    ["同一申请日 + 同一面额（批量CID）", "每个CID自带申请日/面额（上传申请清单）"],
    horizontal=True
)

# =========================
# Mode A: same apply date & face value
# =========================
if mode == "同一申请日 + 同一面额（批量CID）":
    c1, c2, c3 = st.columns(3)
    face_value = c1.selectbox("申请 coupon 面额", FACE_VALUES, index=0)
    apply_date = c2.date_input("申请日期", value=pd.Timestamp.today().date())
    run_all = c3.checkbox("验证全部CID", value=False)

    apply_dt = pd.Timestamp(apply_date).normalize()

    if run_all:
        selected = adv_list
        st.info(f"将验证全部CID：{len(selected)} 个")
    else:
        default_sel = adv_list[: min(20, len(adv_list))]
        selected = st.multiselect("选择需要验证的CID（可多选）", options=adv_list, default=default_sel)

    if not selected:
        st.warning("请至少选择一个CID。")
        st.stop()

    if st.button("开始批量验证", type="primary"):
        rows = [evaluate_one(df, adv_id, apply_dt, face_value) for adv_id in selected]
        res = pd.DataFrame(rows)

        total = len(res)
        ok_cnt = int(res["final_eligible"].sum())
        st.success(f"验证完成：满足申请要求 {ok_cnt}/{total}")

        show_cols = [
            "advertiser_id", "final_eligible",
            "billedrev_30d", "need_billedrev_face_x3", "ok_revenue_rule",
            "acquisition_date", "customer_type", "customer_detail",
            "react_lookback_start", "react_lookback_end", "react_lookback_spend", "react_is_reactivated",
            "window_start", "window_end", "apply_date", "face_value"
        ]
        st.dataframe(res[show_cols].sort_values(["final_eligible", "advertiser_id"], ascending=[True, True]),
                     use_container_width=True)

        bad = res[res["final_eligible"] == False]
        with st.expander("仅查看不满足的CID及原因", expanded=False):
            st.dataframe(bad[show_cols], use_container_width=True)

        csv_bytes = res.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "下载结果CSV",
            data=csv_bytes,
            file_name=f"coupon_validation_{apply_dt.date().isoformat()}_face{face_value}.csv",
            mime="text/csv"
        )

# =========================
# Mode B: upload application list
# =========================
else:
    st.info("上传申请清单CSV：必须包含 advertiser_id, apply_date, face_value 三列（apply_date为日期，face_value为100/200/500/1000）")
    app_up = st.file_uploader("上传申请清单（CSV）", type=["csv"], key="app_list")
    st.caption("示例行：advertiser_id,apply_date,face_value  ->  12345,2026-02-02,200")

    if app_up is None:
        st.stop()

    try:
        app_df = pd.read_csv(app_up)
        app_df.columns = [c.strip().lower() for c in app_df.columns]
        need = {"advertiser_id", "apply_date", "face_value"}
        miss = need - set(app_df.columns)
        if miss:
            raise ValueError(f"申请清单缺少列：{sorted(list(miss))}")

        app_df["advertiser_id"] = app_df["advertiser_id"].astype(str).str.strip()
        app_df["apply_date"] = pd.to_datetime(app_df["apply_date"], errors="coerce").dt.normalize()
        app_df["face_value"] = pd.to_numeric(app_df["face_value"], errors="coerce").astype("Int64")

        app_df = app_df.dropna(subset=["advertiser_id", "apply_date", "face_value"])
        app_df = app_df[app_df["face_value"].isin(FACE_VALUES)].copy()

        if app_df.empty:
            st.warning("申请清单有效行为空（检查日期/面额格式）。")
            st.stop()

        if st.button("开始批量验证（按清单）", type="primary"):
            rows = []
            for _, r in app_df.iterrows():
                rows.append(evaluate_one(df, r["advertiser_id"], r["apply_date"], int(r["face_value"])))
            res = pd.DataFrame(rows)

            total = len(res)
            ok_cnt = int(res["final_eligible"].sum())
            st.success(f"验证完成：满足申请要求 {ok_cnt}/{total}")

            show_cols = [
                "advertiser_id", "final_eligible",
                "billedrev_30d", "need_billedrev_face_x3", "ok_revenue_rule",
                "acquisition_date", "customer_type", "customer_detail",
                "react_lookback_start", "react_lookback_end", "react_lookback_spend", "react_is_reactivated",
                "window_start", "window_end", "apply_date", "face_value"
            ]
            st.dataframe(res[show_cols].sort_values(["final_eligible", "advertiser_id"], ascending=[True, True]),
                         use_container_width=True)

            csv_bytes = res.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                "下载结果CSV",
                data=csv_bytes,
                file_name="coupon_validation_by_app_list.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"解析申请清单失败：{e}")
        st.stop()
``
