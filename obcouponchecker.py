import io
import streamlit as st
import pandas as pd

# =========================
# Config
# =========================
DATA_PATH = "obcoupondata.csv"  # 默认数据源文件名（与脚本同目录）
NEW_CUST_CUTOFF = pd.Timestamp("2025-12-31")  # 纯新客：AcquisitionDate > 2025-12-31
FACE_VALUES = [100, 200, 500, 1000]
FRAUD_OK_VALUE = "Not-Fraud"

REQUIRED_COLS = ["Date", "AdvertiserId", "FraudStatus", "AcquisitionDate", "BilledRev"]

st.set_page_config(page_title="OB Coupon Checker", layout="wide")
st.title("OB Coupon 申请校验")
st.caption(
    "规则：申请日T-2回滚30天统计BilledRev；BilledRev ≥ 面额×3；
     FraudStatus=Not-Fraud；"
    "仅自动判定纯新客，非纯新客返回“请人工校验是否为再活跃老客”。"
)

# =========================
# Helpers
# =========================
def _last_nonempty(series: pd.Series):
    s = series.dropna().astype(str).str.strip()
    s = s[s != ""]
    return s.iloc[-1] if len(s) else None

def validate_and_cast(df: pd.DataFrame) -> pd.DataFrame:
    # Trim column names
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV缺少列：{missing}；需要包含：{REQUIRED_COLS}")

    # Cast types
    df["AdvertiserId"] = df["AdvertiserId"].astype(str).str.strip()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df["AcquisitionDate"] = pd.to_datetime(df["AcquisitionDate"], errors="coerce").dt.normalize()

    # BilledRev: handle thousand separators like "4,453"
    billed = df["BilledRev"].astype(str).str.replace(",", "", regex=False).str.strip()
    df["BilledRev"] = pd.to_numeric(billed, errors="coerce").fillna(0.0)

    df["FraudStatus"] = df["FraudStatus"].astype(str).str.strip()

    # Drop rows missing key fields
    df = df.dropna(subset=["AdvertiserId", "Date"]).copy()

    # Aggregate same AdvertiserId + Date (if multiple rows per day)
    df = (
        df.groupby(["AdvertiserId", "Date"], as_index=False)
          .agg({
              "BilledRev": "sum",
              "AcquisitionDate": "min",
              "FraudStatus": _last_nonempty
          })
          .sort_values(["AdvertiserId", "Date"])
          .reset_index(drop=True)
    )
    return df

@st.cache_data
def load_from_path(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    return validate_and_cast(raw)

@st.cache_data
def load_from_upload(file_bytes: bytes) -> pd.DataFrame:
    raw = pd.read_csv(io.BytesIO(file_bytes))
    return validate_and_cast(raw)

def window_30d_t_minus_2(apply_dt: pd.Timestamp):
    """WindowEnd = ApplyDate - 2 days; WindowStart = WindowEnd - 29 days (inclusive)"""
    apply_dt = pd.Timestamp(apply_dt).normalize()
    window_end = (apply_dt - pd.Timedelta(days=2)).normalize()
    window_start = (window_end - pd.Timedelta(days=29)).normalize()
    return window_start, window_end

def sum_billedrev(df: pd.DataFrame, adv_id: str, start: pd.Timestamp, end: pd.Timestamp) -> float:
    m = (df["AdvertiserId"] == adv_id) & (df["Date"] >= start) & (df["Date"] <= end)
    return float(df.loc[m, "BilledRev"].sum())

def get_acquisition_date(df: pd.DataFrame, adv_id: str):
    s = df.loc[df["AdvertiserId"] == adv_id, "AcquisitionDate"].dropna()
    return pd.Timestamp(s.min()).normalize() if len(s) else None

def get_fraud_status_asof(df: pd.DataFrame, adv_id: str, asof_dt: pd.Timestamp):
    sub = df.loc[(df["AdvertiserId"] == adv_id) & (df["Date"] <= asof_dt), ["Date", "FraudStatus"]].dropna()
    if sub.empty:
        return None
    sub = sub.sort_values("Date")
    return str(sub.iloc[-1]["FraudStatus"]).strip()

def fraud_ok(status: str) -> bool:
    if status is None:
        return False
    return str(status).strip().lower() == FRAUD_OK_VALUE.lower()

def evaluate_one(df: pd.DataFrame, adv_id: str, apply_dt: pd.Timestamp, face_value: int) -> dict:
    apply_dt = pd.Timestamp(apply_dt).normalize()
    w_start, w_end = window_30d_t_minus_2(apply_dt)

    # Revenue rule
    billed_30d = sum_billedrev(df, adv_id, w_start, w_end)
    need_rev = float(face_value) * 3.0
    ok_revenue = billed_30d >= need_rev

    # Fraud rule (as-of WindowEnd)
    f_status = get_fraud_status_asof(df, adv_id, w_end)
    ok_fraud = fraud_ok(f_status)
    fraud_detail = "通过" if ok_fraud else "不通过"

    # New customer rule (manual for non-new)
    acq = get_acquisition_date(df, adv_id)
    if acq is None:
        cust_type = "Unknown"
        cust_detail = "缺少AcquisitionDate，无法判断是否纯新客"
        ok_customer = False
    elif acq > NEW_CUST_CUTOFF:
        cust_type = "New"
        cust_detail = f"纯新客（AcquisitionDate={acq.date().isoformat()}）"
        ok_customer = True
    else:
        cust_type = "Non-New"
        cust_detail = "请手动校验是否为再活跃老客"
        ok_customer = False

    final_ok = bool(ok_revenue and ok_fraud and ok_customer)

    return {
        "AdvertiserId": adv_id,
        "ApplyDate": apply_dt.date().isoformat(),
        "FaceValue": int(face_value),
        "WindowStart": w_start.date().isoformat(),
        "WindowEnd": w_end.date().isoformat(),
        "BilledRev_30d": round(billed_30d, 2),
        "Need_BilledRev_FaceX3": round(need_rev, 2),
        "OK_RevenueRule": ok_revenue,
        "FraudStatus_AsOf_WindowEnd": f_status,
        "OK_FraudRule": ok_fraud,
        "FraudDetail": fraud_detail,
        "AcquisitionDate": None if acq is None else acq.date().isoformat(),
        "CustomerType": cust_type,
        "CustomerDetail": cust_detail,
        "FinalEligible": final_ok
    }

# =========================
# Load data (upload or local)
# =========================
st.sidebar.header("数据源")
up = st.sidebar.file_uploader("上传CSV（可选）", type=["csv"])
st.sidebar.caption(f"需列：{REQUIRED_COLS}")

try:
    if up is not None:
        df = load_from_upload(up.getvalue())
        st.sidebar.success("已使用上传CSV")
    else:
        df = load_from_path(DATA_PATH)
        st.sidebar.info(f"未上传文件，读取本地：{DATA_PATH}")
except Exception as e:
    st.error(f"加载数据失败：{e}")
    st.stop()

adv_list = sorted(df["AdvertiserId"].unique().tolist())

# =========================
# Batch UI
# =========================
st.subheader("批量验证（同一申请日 + 同一面额）")
c1, c2, c3 = st.columns(3)
face_value = c1.selectbox("申请Coupon面额", FACE_VALUES, index=0)
apply_date = c2.date_input("申请日期", value=pd.Timestamp.today().date())
run_all = c3.checkbox("验证全部CID", value=False)

apply_dt = pd.Timestamp(apply_date).normalize()

default_sel = adv_list[: min(20, len(adv_list))]
selected = adv_list if run_all else st.multiselect("选择验证CID（可多选）", options=adv_list, default=default_sel)

if not selected:
    st.warning("请选择至少一个CID")
    st.stop()

if st.button("开始批量验证", type="primary"):
    rows = [evaluate_one(df, adv_id, apply_dt, int(face_value)) for adv_id in selected]
    res = pd.DataFrame(rows)

    total = len(res)
    ok_cnt = int(res["FinalEligible"].sum())

    st.success(f"验证完成：满足申请要求 {ok_cnt} / {total}")

    show_cols = [
        "AdvertiserId", "FinalEligible",
        "BilledRev_30d", "Need_BilledRev_FaceX3", "OK_RevenueRule",
        "FraudStatus_AsOf_WindowEnd", "OK_FraudRule", "FraudDetail",
        "AcquisitionDate", "CustomerType", "CustomerDetail",
        "WindowStart", "WindowEnd", "ApplyDate", "FaceValue"
    ]
    st.dataframe(
        res[show_cols].sort_values(["FinalEligible", "AdvertiserId"], ascending=[False, True]),
        use_container_width=True
    )

    bad = res[res["FinalEligible"] == False]
    with st.expander("不满足CID原因查看", expanded=True):
        st.dataframe(bad[show_cols], use_container_width=True)

    csv_bytes = res.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "下载结果CSV",
        data=csv_bytes,
        file_name=f"coupon_validation_{apply_dt.date().isoformat()}_face{face_value}.csv",
        mime="text/csv"
    )

