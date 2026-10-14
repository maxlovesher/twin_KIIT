"""GlucoTwin clinician dashboard.

    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from glucotwin import config as C  # noqa: E402
from glucotwin import features as F  # noqa: E402
from glucotwin import modeling as M  # noqa: E402
from glucotwin import twin as T  # noqa: E402
from glucotwin.interop import fhir_bundle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import streamlit.components.v1 as components  # noqa: E402
from patient_view import render_phone  # noqa: E402
from body_view import build_organs, render_body  # noqa: E402

from glucotwin import physiology as ph  # noqa: E402

st.set_page_config(page_title="GlucoTwin · Clinician Console", page_icon="🩺", layout="wide")
st.markdown("""
<style>
  .block-container { padding-top: 3.2rem; max-width: 1560px; }
  [data-testid="stSidebar"] { background: #f4f3ef; }
  [data-testid="stMetric"] { background: #ffffff; border: 1px solid #e6e4dd; border-radius: 14px;
                             padding: 12px 16px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
  [data-testid="stMetricLabel"] p { font-size: .82rem; color: #52514e; }
  [data-testid="stMetricValue"] { font-size: 1.65rem; }
  .stTabs [data-baseweb="tab-list"] { gap: 2px; border-bottom: 1px solid #e1e0d9; }
  .stTabs [data-baseweb="tab"] { padding: 8px 14px; border-radius: 10px 10px 0 0; font-weight: 600; }
  .stTabs [aria-selected="true"] { background: #eaf2fc; }
  [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; }
  .gt-hero { background: linear-gradient(100deg, #0d366b 0%, #1c5cab 60%, #2a78d6 100%); color: #fff;
             border-radius: 16px; padding: 14px 22px; display: flex; justify-content: space-between;
             align-items: center; gap: 12px; margin-bottom: 8px; flex-wrap: wrap; }
  .gt-hero b { font-size: 1.2rem; letter-spacing: .2px; }
  .gt-hero .sub { opacity: .85; font-size: .88rem; }
  .gt-chip { display: inline-block; background: rgba(255,255,255,.16); padding: 5px 12px; border-radius: 99px;
             margin-left: 6px; font-size: .86rem; white-space: nowrap; }
</style>
""", unsafe_allow_html=True)

# Reference palette
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK2, MUTED, GRID, CRITICAL = "#52514e", "#898781", "#e1e0d9", "#d03b3b"
RANGE_FILL = "rgba(42,120,214,0.08)"

MEAL_PRESETS = {
    "Custom": (60, 0.6),
    "White rice (2 cups) + sambar": (80, 0.72),
    "Chapati ×3 + dal + sabzi": (60, 0.55),
    "Masala dosa + chutney": (65, 0.68),
    "Idli ×4 + sambar": (50, 0.70),
    "Chicken biryani (1 plate)": (90, 0.65),
    "Ragi mudde + saaru (low-GI)": (55, 0.40),
    "Poha + chai with sugar": (55, 0.75),
    "Millet khichdi + curd (low-GI)": (50, 0.38),
}
CARB_TYPES = {"Refined / high-GI": 0.75, "Mixed meal": 0.6, "Whole-grain / low-GI": 0.4}


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading digital twins…")
def load():
    ehr = pd.read_csv(C.RAW_DIR / "ehr.csv")
    wear = pd.read_parquet(C.RAW_DIR / "wearables.parquet")
    pred = pd.read_parquet(C.PROCESSED_DIR / "predictions.parquet")
    params = pd.read_csv(C.MODELS_DIR / "twin_params.csv")
    bundle: M.Bundle = joblib.load(C.MODELS_DIR / "glucotwin_bundle.joblib")
    metrics = json.loads((C.REPORTS_DIR / "metrics.json").read_text())
    split = pd.read_csv(C.PROCESSED_DIR / "splits.csv").set_index("patient_id")["split"]
    return ehr, wear, pred, params, bundle, metrics, split


@st.cache_data(show_spinner="Running the patient's twin…", max_entries=16)
def patient_timeline(pid: str):
    ehr, wear, _, params, *_ = load()
    df = wear[wear.patient_id == pid].sort_values("timestamp").reset_index(drop=True)
    erow = ehr.set_index("patient_id", drop=False).loc[pid]
    calib = T.PatientTwin.from_row(params.set_index("patient_id", drop=False).loc[pid])
    prior = T.PatientTwin(pid, T.prior_params(erow))
    k = C.CALIBRATION_DAYS * C.STEPS_PER_DAY
    frame = F.build_patient_frame(df, F.static_features(ehr).loc[pid], calib, with_labels=False,
                                  onboarding_twin=prior, switch_step=k)
    st_a = prior.track(df.iloc[:k])
    states = np.vstack([st_a, calib.track(df.iloc[k:], s0=st_a[-1])])
    return df, frame, states


def twin_at(pid: str, idx: int) -> T.PatientTwin:
    ehr, _, _, params, *_ = load()
    if idx < C.CALIBRATION_DAYS * C.STEPS_PER_DAY:
        return T.PatientTwin(pid, T.prior_params(ehr.set_index("patient_id", drop=False).loc[pid]))
    return T.PatientTwin.from_row(params.set_index("patient_id", drop=False).loc[pid])


ehr, wear, pred, params, bundle, metrics, split = load()
EHR = ehr.set_index("patient_id")
T0 = pd.Timestamp(C.SIM_START)
CLOCK_MIN = T0 + timedelta(days=C.FEATURE_START_DAY + 1)
CLOCK_MAX = T0 + timedelta(days=C.N_DAYS) - timedelta(hours=2, minutes=5)


def _clamp(t: pd.Timestamp):
    return min(max(t, CLOCK_MIN), CLOCK_MAX).to_pydatetime()


def shift_clock(minutes: int) -> None:
    st.session_state.clock = _clamp(pd.Timestamp(st.session_state.clock) + timedelta(minutes=minutes))


# Defaults: open on a held-out patient just before a correctly predicted spike
qp = st.query_params
if "clock" not in st.session_state and "pid" in qp and "t" in qp and qp["pid"] in EHR.index:
    st.session_state.pid = qp["pid"]                  # deep link, e.g. ?pid=GT0014&t=2026-08-12T11:30
    st.session_state.clock = _clamp(pd.Timestamp(qp["t"]))
if "clock" not in st.session_state:
    test = pred[(pred.split == "test") & (pred.timestamp >= T0 + timedelta(days=9))]
    hit = test[test.alert & (test.y == 1)].sort_values("timestamp")
    first = hit.iloc[0] if len(hit) else test.iloc[0]
    st.session_state.pid = first.patient_id
    st.session_state.clock = _clamp(first.timestamp)
# Changes requested on the previous run (must be applied before widgets are created)
if st.session_state.pop("_advance", False):
    shift_clock(15)
if "_pending_pid" in st.session_state:
    st.session_state.pid = st.session_state.pop("_pending_pid")

# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🩺 GlucoTwin")
    st.caption("Hybrid digital twin for Type 2 Diabetes: predicts glucose spikes up to 2 h ahead "
               "from EHR + CGM + wearables.")
    st.markdown("**Simulation clock**")
    st.slider("Clock", min_value=CLOCK_MIN.to_pydatetime(), max_value=CLOCK_MAX.to_pydatetime(),
              step=timedelta(minutes=5), format="ddd DD MMM, HH:mm", key="clock", label_visibility="collapsed")
    c1, c2 = st.columns(2)
    c1.button("◀ −1 h", on_click=shift_clock, args=(-60,), use_container_width=True)
    c2.button("+1 h ▶", on_click=shift_clock, args=(60,), use_container_width=True)
    c1.button("◀ −5 min", on_click=shift_clock, args=(-5,), use_container_width=True)
    c2.button("+5 min ▶", on_click=shift_clock, args=(5,), use_container_width=True)
    autoplay = st.toggle("▶ Live replay (+15 min every 2 s)", value=False)

    def label(p: str) -> str:
        r = EHR.loc[p]
        return f"{p} · {r.age}{r.sex} · HbA1c {r.hba1c_pct}% · {split[p]}"

    st.selectbox("Patient", EHR.index.tolist(), key="pid", format_func=label)
    st.divider()
    st.caption("⚠️ Research prototype on **synthetic** data (no real patient data). "
               "Not a medical device; decision support only.")

now = pd.Timestamp(st.session_state.clock)
pid = st.session_state.pid

_snap = pred[pred.timestamp == now]
st.markdown(
    f"<div class='gt-hero'><div><b>🩺 GlucoTwin · Clinician console</b><br>"
    f"<span class='sub'>Hybrid digital twin for Type 2 Diabetes · decision support, not a medical device</span></div>"
    f"<div><span class='gt-chip'>🕒 {now:%a %d %b, %H:%M}</span>"
    f"<span class='gt-chip'>👥 {len(_snap)} patients</span>"
    f"<span class='gt-chip'>⚠️ {int(_snap.alert.sum())} spike alerts</span></div></div>",
    unsafe_allow_html=True)

tab_ward, tab_body, tab_pt, tab_app, tab_whatif, tab_perf, tab_how = st.tabs(
    ["🏥 Ward overview", "🧍 Twin overview", "🧬 Virtual patient", "📱 Patient app", "🔮 What-if simulator",
     "📊 Model performance", "ℹ️ How it works"])


def arrow(roc: float) -> str:
    if pd.isna(roc):
        return "·"
    return ("⇈" if roc > 2 else "↑" if roc > 1 else "↗" if roc > 0.5 else
            "⇊" if roc < -2 else "↓" if roc < -1 else "↘" if roc < -0.5 else "→")


# ======================================================================================
# Ward overview
# ======================================================================================
def ward_table(now: pd.Timestamp) -> pd.DataFrame:
    snap = pred[pred.timestamp == now].set_index("patient_id")
    w7 = wear[(wear.timestamp > now - timedelta(days=7)) & (wear.timestamp <= now)]
    tir = w7.assign(ok=w7.cgm.between(70, 180)).groupby("patient_id").ok.mean() * 100
    prev = wear[wear.timestamp == now - timedelta(minutes=15)].set_index("patient_id").cgm
    tbl = pd.DataFrame({
        "Patient": snap.index,
        "Age/Sex": [f"{EHR.loc[p, 'age']}{EHR.loc[p, 'sex']}" for p in snap.index],
        "HbA1c %": EHR.loc[snap.index, "hba1c_pct"].to_numpy(),
        "Glucose": snap.cgm.to_numpy(),
        "Trend": ((snap.cgm - prev.reindex(snap.index)) / 15).map(arrow).to_numpy(),
        "Spike risk (2 h)": (snap.p_spike * 100).to_numpy(),
        "Alert": np.where(snap.alert, "⚠️ Spike likely", ""),
        "TIR 7d %": tir.reindex(snap.index).to_numpy(),
        "Cohort": [split[p] for p in snap.index],
    })
    return tbl.sort_values("Spike risk (2 h)", ascending=False).reset_index(drop=True)


with tab_ward:
    tbl = ward_table(now)
    st.markdown(f"### Ward overview · {now:%a %d %b %Y, %H:%M}")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Patients monitored", len(tbl))
    k2.metric("Active spike alerts", int((tbl.Alert != "").sum()))
    k3.metric("Currently above 180 mg/dL", int((tbl.Glucose > 180).sum()))
    k4.metric("Median 7-day time in range", f"{tbl['TIR 7d %'].median():.0f}%")
    show_test = st.checkbox("Show only held-out test patients (never seen by the model)", value=False)
    view = tbl[tbl.Cohort == "test"].reset_index(drop=True) if show_test else tbl
    ev = st.dataframe(
        view, hide_index=True, use_container_width=True, height=520, on_select="rerun",
        selection_mode="single-row", key="ward_table",
        column_config={
            "Spike risk (2 h)": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
            "Glucose": st.column_config.NumberColumn(format="%d mg/dL"),
            "TIR 7d %": st.column_config.NumberColumn(format="%.0f%%"),
        })
    if ev.selection.rows:
        chosen = view.iloc[ev.selection.rows[0]].Patient
        if chosen != pid:
            st.session_state._pending_pid = chosen
            st.rerun()
    st.caption("Select a row, then open the **Virtual patient** tab. Risk = calibrated probability that CGM "
               "glucose exceeds 180 mg/dL within the next 2 hours.")


# ======================================================================================
# Virtual patient
# ======================================================================================
df, frame, states = patient_timeline(pid)
idx = min(int(np.searchsorted(df.timestamp.to_numpy(), np.datetime64(now))), len(df) - 1)
row = frame.iloc[idx]
X = frame.iloc[[idx]][F.ALL_FEATURES]
p_now = float(bundle.risk.predict_proba(X)[0])
thr = bundle.risk.threshold
fc = bundle.forecast(X).iloc[0]
twin_now = twin_at(pid, idx)
hour_now = now.hour + now.minute / 60
twin_traj = twin_now.what_if(states[idx], hour_now, horizon_min=120, dt=5.0)
cgm_now = row.cgm_now
alert_now = bool(p_now >= thr and not pd.isna(cgm_now) and cgm_now <= C.HYPER_THRESHOLD)
r_ehr = EHR.loc[pid]


@st.cache_resource
def hypo_model():
    """Low-glucose model trained on real patients (scripts/hypo_real_data.py); optional."""
    path = C.MODELS_DIR / "hypo_model.joblib"
    return joblib.load(path) if path.exists() else None


HYPO = hypo_model()
p_low = float(HYPO["clf"].predict_proba(X[HYPO["cols"]])[:, 1][0]) if HYPO else None
low_alert = bool(HYPO and p_low >= HYPO["threshold"] and not pd.isna(cgm_now) and cgm_now >= 70)


def main_chart() -> go.Figure:
    lo = max(0, idx - 24 * 12)
    h = df.iloc[lo: idx + 1]
    pr = pred[(pred.patient_id == pid) & (pred.timestamp > h.timestamp.iloc[0]) & (pred.timestamp <= now)]
    fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.42, 0.13, 0.11, 0.11, 0.11, 0.12],
                        subplot_titles=("Glucose (mg/dL)", "Spike risk, next 2 h", "Steps / 5 min",
                                        "Heart rate (bpm)", "HRV RMSSD (ms)", "Sleep stage"))
    fig.add_hrect(y0=70, y1=180, fillcolor=RANGE_FILL, line_width=0, row=1, col=1)
    fig.add_hline(y=180, line=dict(color=MUTED, width=1, dash="dash"), row=1, col=1)
    fig.add_trace(go.Scatter(x=h.timestamp, y=h.cgm, name="CGM", line=dict(color=BLUE, width=2),
                             hovertemplate="%{y:.0f} mg/dL<extra>CGM</extra>"), row=1, col=1)
    tt = [now + timedelta(minutes=5 * (k + 1)) for k in range(len(twin_traj))]
    fig.add_trace(go.Scatter(x=[now] + tt, y=[cgm_now] + list(twin_traj), name="Twin physiology (no new meals)",
                             line=dict(color=VIOLET, width=2, dash="dot"),
                             hovertemplate="%{y:.0f} mg/dL<extra>Twin</extra>"), row=1, col=1)
    hs = list(M.FORECAST_HORIZONS)
    fx = [now] + [now + timedelta(minutes=h_) for h_ in hs]
    fy = [cgm_now] + [fc[f"fc{h_}"] for h_ in hs]
    lo_b = [cgm_now] + [fc[f"fc{h_}"] + bundle.resid_q[h_][0] for h_ in hs]
    hi_b = [cgm_now] + [fc[f"fc{h_}"] + bundle.resid_q[h_][1] for h_ in hs]
    fig.add_trace(go.Scatter(x=fx + fx[::-1], y=hi_b + lo_b[::-1], fill="toself", fillcolor="rgba(235,104,52,0.15)",
                             line=dict(width=0), hoverinfo="skip", name="80% forecast band"), row=1, col=1)
    fig.add_trace(go.Scatter(x=fx, y=fy, name="GlucoTwin forecast", mode="lines+markers",
                             line=dict(color=ORANGE, width=2), marker=dict(size=8, line=dict(color="white", width=2)),
                             hovertemplate="%{y:.0f} mg/dL<extra>Hybrid forecast</extra>"), row=1, col=1)
    meals = h[h.carbs_logged > 0]
    fig.add_trace(go.Scatter(x=meals.timestamp, y=[55] * len(meals), mode="markers+text", name="Logged meal",
                             marker=dict(symbol="triangle-up", size=11, color=GREEN),
                             text=[f"{c:.0f} g" for c in meals.carbs_logged], textposition="top center",
                             textfont=dict(size=10, color=INK2),
                             hovertemplate="%{text} carbs<extra>Meal log</extra>"), row=1, col=1)
    al = pr[pr.alert]
    fig.add_trace(go.Scatter(x=al.timestamp, y=al.cgm, mode="markers", name="Spike alert",
                             marker=dict(size=8, color=CRITICAL, line=dict(color="white", width=1.5)),
                             hovertemplate="alert issued<extra></extra>"), row=1, col=1)
    fig.add_vline(x=now, line=dict(color=INK2, width=1))
    fig.add_trace(go.Scatter(x=pr.timestamp, y=pr.p_spike, line=dict(color=ORANGE, width=2), showlegend=False,
                             fill="tozeroy", fillcolor="rgba(235,104,52,0.12)",
                             hovertemplate="%{y:.0%}<extra>Risk</extra>"), row=2, col=1)
    fig.add_hline(y=thr, line=dict(color=CRITICAL, width=1, dash="dash"), row=2, col=1)
    fig.add_trace(go.Bar(x=h.timestamp, y=h.steps, marker_color=AQUA, showlegend=False,
                         hovertemplate="%{y} steps<extra></extra>"), row=3, col=1)
    fig.add_trace(go.Scatter(x=h.timestamp, y=h.hr, line=dict(color=RED, width=1.5), showlegend=False,
                             hovertemplate="%{y:.0f} bpm<extra></extra>"), row=4, col=1)
    fig.add_trace(go.Scatter(x=h.timestamp, y=h.hrv_rmssd, mode="markers", marker=dict(size=4, color=MAGENTA),
                             showlegend=False, hovertemplate="%{y:.0f} ms<extra>HRV</extra>"), row=5, col=1)
    fig.add_trace(go.Scatter(x=h.timestamp, y=h.sleep_stage.where(h.in_sleep == 1),
                             line=dict(color=VIOLET, width=1.5, shape="hv"), showlegend=False,
                             hovertemplate="stage %{y}<extra></extra>"), row=6, col=1)
    ymax = max(320.0, float(np.nanmax(h.cgm.to_numpy())) + 20)
    fig.update_yaxes(range=[40, ymax], row=1, col=1)
    fig.update_yaxes(range=[0, 1], tickformat=".0%", row=2, col=1)
    fig.update_yaxes(tickvals=[0, 1, 2, 3], ticktext=["Awake", "Light", "Deep", "REM"], range=[-0.3, 3.3], row=6, col=1)
    fig.update_xaxes(range=[h.timestamp.iloc[0], now + timedelta(hours=2)], gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    fig.update_layout(height=880, margin=dict(l=10, r=10, t=90, b=10), plot_bgcolor="white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.035, x=0), hovermode="x unified",
                      bargap=0.1)
    fig.update_annotations(font=dict(size=12, color=INK2), x=0, xanchor="left")
    return fig


def agp_chart() -> go.Figure:
    hist = df[(df.timestamp > now - timedelta(days=14)) & (df.timestamp <= now)].copy()
    hist["tod"] = hist.timestamp.dt.hour + (hist.timestamp.dt.minute // 30) * 0.5
    q = hist.groupby("tod").cgm.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack()
    fig = go.Figure()
    fig.add_hrect(y0=70, y1=180, fillcolor=RANGE_FILL, line_width=0)
    fig.add_trace(go.Scatter(x=q.index, y=q[0.95], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=q.index, y=q[0.05], fill="tonexty", fillcolor="rgba(42,120,214,0.12)",
                             line=dict(width=0), name="5–95%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=q.index, y=q[0.75], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=q.index, y=q[0.25], fill="tonexty", fillcolor="rgba(42,120,214,0.28)",
                             line=dict(width=0), name="25–75%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=q.index, y=q[0.5], line=dict(color=BLUE, width=2), name="Median",
                             hovertemplate="%{x:.1f} h · median %{y:.0f} mg/dL<extra></extra>"))
    fig.add_hline(y=180, line=dict(color=MUTED, width=1, dash="dash"))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                      legend=dict(orientation="h", y=1.12, x=0),
                      xaxis=dict(title="Hour of day", tickvals=list(range(0, 25, 3)), gridcolor=GRID),
                      yaxis=dict(title="mg/dL", gridcolor=GRID))
    return fig


with tab_pt:
    st.markdown(f"### {pid} · {r_ehr.age} y {('male' if r_ehr.sex == 'M' else 'female')} · {r_ehr.city} · "
                f"T2D for {r_ehr.diabetes_duration_y:.0f} y")
    cohort_note = {"test": ":blue-badge[held-out test patient]", "val": ":gray-badge[validation patient]",
                   "train": ":gray-badge[training patient]"}[split[pid]]
    twin_note = (":violet-badge[twin: calibrated]" if idx >= C.CALIBRATION_DAYS * C.STEPS_PER_DAY
                 else ":orange-badge[twin: onboarding week (EHR priors)]")
    st.markdown(f"{cohort_note} {twin_note} &nbsp; {now:%a %d %b, %H:%M}")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Glucose now", "—" if pd.isna(cgm_now) else f"{cgm_now:.0f} mg/dL",
              None if pd.isna(row.cgm_roc) else f"{row.cgm_roc:+.1f} mg/dL/min {arrow(row.cgm_roc)}",
              delta_color="inverse")
    m2.metric("Spike risk (next 2 h)", f"{p_now:.0%}", "⚠ ALERT" if alert_now else "below alert threshold",
              delta_color="inverse" if alert_now else "off")
    m3.metric("Forecast in 60 min", f"{fc['fc60']:.0f} mg/dL",
              f"{bundle.resid_q[60][0] + fc['fc60']:.0f}–{bundle.resid_q[60][1] + fc['fc60']:.0f} (80% range)",
              delta_color="off")
    w7 = df[(df.timestamp > now - timedelta(days=7)) & (df.timestamp <= now)].cgm
    m4.metric("Time in range (7 d)", f"{w7.between(70, 180).mean():.0%}",
              f"{(w7 > 180).mean():.0%} above range", delta_color="off")
    m5.metric("GMI (7 d)", f"{3.31 + 0.02392 * w7.mean():.1f}%", f"lab HbA1c {r_ehr.hba1c_pct}%", delta_color="off")

    left, right = st.columns([2.2, 1])
    with left:
        st.plotly_chart(main_chart(), use_container_width=True, config={"displaylogo": False})
    with right:
        with st.container(border=True):
            if alert_now:
                st.markdown(f"#### 🔴 Spike likely within 2 h\n**{p_now:.0%}** probability that glucose exceeds "
                            f"180 mg/dL (alert threshold {thr:.0%}).")
            elif not pd.isna(cgm_now) and cgm_now > 180:
                st.markdown("#### 🟠 Currently above range\nGlucose is already above 180 mg/dL.")
            else:
                st.markdown(f"#### 🟢 No spike expected\n{p_now:.0%} probability of exceeding 180 mg/dL in 2 h.")
            if p_low is not None:
                st.caption(f"{'🟠 ' if low_alert else ''}Low-glucose risk (<70 mg/dL in 2 h): **{p_low:.0%}** · "
                           "model trained on real patients (ShanghaiT2DM)")
            contrib = M.explain_rows(bundle, X).iloc[0]
            st.markdown("**Why: top drivers**")
            for g, c, text in M.narrate(row, contrib, top=4):
                st.markdown(f"{'▲' if c > 0 else '▼'} **{g}** ({'raises' if c > 0 else 'lowers'} risk)  \n"
                            f"<span style='color:{INK2}'>{text}</span>", unsafe_allow_html=True)
            if not pd.isna(cgm_now) and df.in_sleep.iloc[idx] == 0:
                base = twin_now.what_if(states[idx], hour_now, horizon_min=120)
                walked = twin_now.what_if(states[idx], hour_now, horizon_min=120, walk_min=15,
                                          walk_delay_min=0, walk_intensity=0.75)
                if base.max() - walked.max() >= 5:
                    st.markdown("**Twin-tested suggestion**")
                    st.info(f"🚶 A 15-minute brisk walk now is predicted to lower the 2-h peak from "
                            f"**{base.max():.0f}** to **{walked.max():.0f} mg/dL** "
                            f"(−{base.max() - walked.max():.0f}).")
        with st.container(border=True):
            st.markdown("**Twin phenotype** (vs. cohort)")
            phe = twin_now.phenotype(params)
            st.progress(phe["insulin_sensitivity_pct"] / 100,
                        text=f"Insulin sensitivity: {phe['insulin_sensitivity_pct']:.0f}th percentile")
            st.progress(phe["beta_cell_response_pct"] / 100,
                        text=f"β-cell response: {phe['beta_cell_response_pct']:.0f}th percentile")
            st.progress(phe["basal_glucose_pct"] / 100,
                        text=f"Basal glucose {phe['basal_glucose']:.0f} mg/dL: {phe['basal_glucose_pct']:.0f}th percentile")
            fit = params.set_index("patient_id").loc[pid]
            st.caption(f"Calibrated on the onboarding week · 30-min fit error {fit.fit_rmse_30:.0f} mg/dL")
        with st.container(border=True):
            st.markdown("**Share with the hospital record** (FHIR R4 / ABDM-ready)")
            assessment = {
                "timestamp": now.isoformat(), "cgm": None if pd.isna(cgm_now) else float(cgm_now),
                "spike_probability_2h": p_now, "alert": alert_now,
                "drivers": [{"group": g, "text": t} for g, _, t in M.narrate(row, contrib, top=4)],
            }
            bundle_json = json.dumps(fhir_bundle({"patient_id": pid, **r_ehr.to_dict()}, assessment), indent=2,
                                     default=lambda o: o.item() if hasattr(o, "item") else str(o))
            st.download_button("⬇ Download FHIR bundle", bundle_json, file_name=f"{pid}_{now:%Y%m%dT%H%M}_fhir.json",
                               mime="application/fhir+json", use_container_width=True)
            st.caption("Patient, conditions (SNOMED), labs (LOINC), medications (ATC), latest CGM and a "
                       "RiskAssessment for the 2-hour spike prediction.")

    c1, c2 = st.columns([1, 1.3])
    with c1:
        with st.container(border=True):
            st.markdown("**Electronic health record**")
            meds = [m.replace("med_", "").replace("_", " ") for m in EHR.columns
                    if m.startswith("med_") and r_ehr[m] == 1]
            comorb = [c.replace("_", " ") for c in ["hypertension", "dyslipidemia", "ckd", "neuropathy",
                                                    "coronary_artery_disease"] if r_ehr[c] == 1]
            st.markdown(
                f"- **Labs ({r_ehr.lab_date})**: HbA1c {r_ehr.hba1c_pct}% · FPG {r_ehr.fpg_mg_dl:.0f} · "
                f"LDL {r_ehr.ldl_mg_dl:.0f} · TG {r_ehr.tg_mg_dl:.0f} · eGFR {r_ehr.egfr:.0f}\n"
                f"- **Vitals**: BMI {r_ehr.bmi} · waist {r_ehr.waist_cm:.0f} cm · BP {r_ehr.sbp_mmhg:.0f}/{r_ehr.dbp_mmhg:.0f}\n"
                f"- **Comorbidities**: {', '.join(comorb) or 'none recorded'}\n"
                f"- **Medications**: {', '.join(meds) or 'none'}\n"
                f"- **Genetics**: TCF7L2 rs7903146 risk alleles {r_ehr.tcf7l2_risk_alleles} · "
                f"KCNQ1 {r_ehr.kcnq1_risk_alleles} · PRS z {r_ehr.polygenic_risk_z:+.1f} · "
                f"family history {'yes' if r_ehr.family_history_t2d else 'no'}\n"
                f"- **Diet**: {r_ehr.diet_staple}-staple, {'vegetarian' if r_ehr.vegetarian else 'non-vegetarian'}")
    with c2:
        with st.container(border=True):
            st.markdown("**Ambulatory glucose profile** (last 14 days)")
            st.plotly_chart(agp_chart(), use_container_width=True, config={"displaylogo": False})


# ======================================================================================
# Twin overview (human body map)
# ======================================================================================
def _f(x):
    return None if x is None or pd.isna(x) else float(x)


with tab_body:
    phe_b = twin_now.phenotype(params)
    w7b = df[(df.timestamp > now - timedelta(days=7)) & (df.timestamp <= now)].cgm
    s_now = states[idx]
    body_vals = dict(
        cgm=_f(cgm_now), roc=_f(row.cgm_roc), tir7=float(w7b.between(70, 180).mean()) if w7b.notna().any() else None,
        beta_pct=phe_b["beta_cell_response_pct"], insulin_now=float(s_now[ph.S_I]),
        secretagogue=bool(r_ehr.med_sulfonylurea or r_ehr.med_dpp4i),
        si_pct=phe_b["insulin_sensitivity_pct"], steps_1h=_f(row.steps_60),
        ex_boost=float(s_now[ph.S_EXM] * ph.EX_SI_GAIN * 100),
        gb=float(twin_now.params[ph.P_GB]), dawn=float(twin_now.params[ph.P_DAWN] * 60),
        metformin=bool(r_ehr.med_metformin), gut_carbs=_f(row.twin_gut_carbs), min_since_meal=_f(row.min_since_meal),
        egfr=_f(r_ehr.egfr), sglt2=bool(r_ehr.med_sglt2i), hr=_f(df.hr.iloc[idx]), hrv_rel=_f(row.hrv_rel_day),
        hr_still_elev=_f(row.hr_still_elev), sleep_h=_f(row.last_sleep_h), sleep_vs_usual=_f(row.sleep_vs_usual_h),
        neuropathy=bool(r_ehr.neuropathy),
    )
    if low_alert:
        risk_text, risk_status = f"Next 2 h: low glucose likely ({p_low:.0%})", "critical"
    elif alert_now:
        risk_text, risk_status = f"Next 2 h: spike likely ({p_now:.0%})", "serious"
    elif not pd.isna(cgm_now) and cgm_now > 180:
        risk_text, risk_status = "Above range now", "warning"
    else:
        risk_text, risk_status = f"Next 2 h: stable ({p_now:.0%} spike risk)", "good"
    calibrated = idx >= C.CALIBRATION_DAYS * C.STEPS_PER_DAY
    body_header = {
        "who": f"{pid} · {r_ehr.age} y {'male' if r_ehr.sex == 'M' else 'female'} · T2D for {r_ehr.diabetes_duration_y:.0f} y",
        "sub": f"{now:%a %d %b, %H:%M} · HbA1c {r_ehr.hba1c_pct}% · twin "
               f"{'calibrated on onboarding week' if calibrated else 'running on EHR priors (onboarding)'}",
        "risk_text": risk_text, "risk_status": risk_status, "hr": body_vals["hr"],
    }
    components.html(render_body(build_organs(body_vals), body_header), height=780, scrolling=False)
    st.caption("Each organ is a part of the physiology GlucoTwin models for this patient: the pancreas is β-cell "
               "insulin output, muscles are insulin sensitivity, the liver is background glucose output, the gut is "
               "carbohydrate still being absorbed. Move the clock to watch the twin change through the day.")


# ======================================================================================
# Patient app
# ======================================================================================
with tab_app:
    st.markdown("### What the patient sees on their phone")
    lang = st.radio("Language", ["English", "हिंदी (Hindi)"], horizontal=True, label_visibility="collapsed")
    suggestion = None
    if not pd.isna(cgm_now) and df.in_sleep.iloc[idx] == 0:
        b_ = twin_now.what_if(states[idx], hour_now, horizon_min=120)
        w_ = twin_now.what_if(states[idx], hour_now, horizon_min=120, walk_min=15, walk_delay_min=0,
                              walk_intensity=0.75)
        if b_.max() - w_.max() >= 5:
            suggestion = {"peak_without": float(b_.max()), "peak_with": float(w_.max())}
    today = df[(df.timestamp.dt.normalize() == now.normalize()) & (df.timestamp <= now)]
    phone_state = {
        "time": now.strftime("%a %H:%M"), "cgm": None if pd.isna(cgm_now) else float(cgm_now),
        "roc": None if pd.isna(row.cgm_roc) else float(row.cgm_roc),
        "p_spike": p_now, "alert": alert_now, "p_low": p_low or 0.0, "low_alert": low_alert,
        "drivers": [(g, c) for g, c, _ in M.narrate(row, contrib, top=4)], "suggestion": suggestion,
        "history": df.cgm.iloc[max(0, idx - 35): idx + 1].astype(float).tolist(),
        "forecast": [(h_, float(fc[f"fc{h_}"])) for h_ in M.FORECAST_HORIZONS],
        "tir_today": float(today.cgm.between(70, 180).mean()) if today.cgm.notna().any() else 0.0,
        "steps_today": float(today.steps.sum()), "sleep_h": float(row.last_sleep_h),
    }
    a1, a2 = st.columns([1, 1.15])
    with a1:
        components.html(render_phone(phone_state, "hi" if lang.startswith("हिंदी") else "en"), height=900)
    with a2:
        with st.container(border=True):
            st.markdown(
                "**Design principle: one number, one prediction, one reason, one action.**\n\n"
                "- The doctor sees the full console; the patient sees only what they can act on in the next 2 hours.\n"
                "- Same engine: the alert, the reasons and the walking suggestion come from the patient's own twin "
                "and the models the clinician sees.\n"
                "- Plain language in **English or Hindi**.\n"
                "- Low-glucose warnings (from the model trained on real patients) take priority over high-glucose ones.\n"
                "- *I'll do it / Remind me* responses would flow back to the care team as adherence data "
                "(conceptual in this prototype).")
        st.caption("Use the simulation clock in the sidebar to watch the screen change before, during and after a meal.")


# ======================================================================================
# What-if
# ======================================================================================
with tab_whatif:
    st.markdown(f"### What-if simulator · {pid} · starting {now:%a %H:%M} at "
                f"{'—' if pd.isna(cgm_now) else f'{cgm_now:.0f} mg/dL'}")
    st.caption("Runs the patient's personalised physiological twin forward from its current state. "
               "Explore how meal choices, post-meal walks and sleep change the glucose response.")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        preset = st.selectbox("Meal", list(MEAL_PRESETS), index=1)
        carbs0, gi0 = MEAL_PRESETS[preset]
        carbs = st.slider("Carbohydrates (g)", 0, 150, int(carbs0), step=5)
        meal_in = st.slider("Meal starts in (min)", 0, 180, 15, step=5)
    with cc2:
        gi_vals = list(CARB_TYPES.values())
        ctype = st.radio("Carbohydrate quality", list(CARB_TYPES),
                         index=int(np.argmin([abs(v - gi0) for v in gi_vals])))
        walk = st.slider("Walk after the meal (min)", 0, 60, 15, step=5)
        walk_delay = st.slider("Walk starts … min after eating", 0, 90, 20, step=5)
    with cc3:
        sleep_q = st.radio("Last night's sleep", ["Normal (7 h)", "Short (< 6 h)", "Very short (< 5 h)"])
        si = {"Normal (7 h)": 1.0, "Short (< 6 h)": 0.86, "Very short (< 5 h)": 0.76}[sleep_q]
        horizon = st.slider("Horizon (h)", 2, 8, 5)

    gut = float(row.twin_gut_carbs)
    if gut > 10:
        st.info(f"The twin is still absorbing ~{gut:.0f} g of carbohydrate from a recent meal; the dotted line shows "
                "that meal alone, and the scenario is added on top. Move the clock to before a meal for a clean comparison.")
    H_ = horizon * 60
    kw = dict(meal_carbs=carbs, meal_in_min=meal_in, fast_frac=CARB_TYPES[ctype], si_mult=si)
    s_idx = states[idx]
    no_meal = twin_now.what_if(s_idx, hour_now, horizon_min=H_, si_mult=si)
    meal_only = twin_now.what_if(s_idx, hour_now, horizon_min=H_, **kw)
    meal_walk = twin_now.what_if(s_idx, hour_now, horizon_min=H_, walk_min=walk, walk_delay_min=walk_delay, **kw)
    tx = [now + timedelta(minutes=k + 1) for k in range(H_)]
    fig = go.Figure()
    fig.add_hrect(y0=70, y1=180, fillcolor=RANGE_FILL, line_width=0)
    fig.add_hline(y=180, line=dict(color=MUTED, width=1, dash="dash"))
    for y, name, color, dash in ((no_meal, "No additional meal", MUTED, "dot"),
                                 (meal_only, f"{preset} ({carbs} g)", ORANGE, "solid"),
                                 (meal_walk, f"… + {walk}-min walk", BLUE, "solid")):
        fig.add_trace(go.Scatter(x=tx, y=y, name=name, line=dict(color=color, width=2.5, dash=dash),
                                 hovertemplate="%{y:.0f} mg/dL<extra>" + name + "</extra>"))
    fig.update_layout(height=420, plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.08, x=0), hovermode="x unified",
                      yaxis=dict(title="Glucose (mg/dL)", gridcolor=GRID), xaxis=dict(gridcolor=GRID))
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    def above(y) -> int:
        return int((y > 180).sum())

    s1, s2, s3 = st.columns(3)
    s1.metric("Peak with meal", f"{meal_only.max():.0f} mg/dL", f"{above(meal_only)} min above 180", delta_color="off")
    s2.metric("Peak with meal + walk", f"{meal_walk.max():.0f} mg/dL",
              f"{meal_walk.max() - meal_only.max():+.0f} mg/dL · {above(meal_walk) - above(meal_only):+d} min above 180",
              delta_color="inverse")
    swap = twin_now.what_if(s_idx, hour_now, horizon_min=H_, walk_min=walk, walk_delay_min=walk_delay,
                            **{**kw, "fast_frac": CARB_TYPES["Whole-grain / low-GI"]})
    s3.metric("…and swapping to low-GI carbs", f"{swap.max():.0f} mg/dL",
              f"{swap.max() - meal_only.max():+.0f} mg/dL vs meal alone", delta_color="inverse")
    st.caption("Twin-simulated responses are personalised estimates for counselling and exploration, "
               "not dosing advice.")


# ======================================================================================
# Model performance
# ======================================================================================
with tab_perf:
    sizes = metrics["split_sizes"]
    st.markdown(f"### How well does GlucoTwin work? ({sizes['test_patients']} held-out patients, "
                "never seen during training)")
    ab = pd.DataFrame(metrics["ablations"]).T
    ab = ab[["auroc", "auprc", "event_sensitivity", "median_lead_min", "alert_precision", "false_alerts_per_day"]]
    ab.columns = ["AUROC", "AUPRC", "Spikes caught", "Median warning (min)", "Alert precision", "False alerts / day"]
    st.dataframe(ab.sort_values("AUROC", ascending=False).astype(float), use_container_width=True,
                 column_config={"Spikes caught": st.column_config.NumberColumn(format="percent"),
                                "Alert precision": st.column_config.NumberColumn(format="percent"),
                                "AUROC": st.column_config.NumberColumn(format="%.3f"),
                                "AUPRC": st.column_config.NumberColumn(format="%.3f"),
                                "False alerts / day": st.column_config.NumberColumn(format="%.2f"),
                                "Median warning (min)": st.column_config.NumberColumn(format="%.0f")})
    figs = C.REPORTS_DIR / "figures"
    for name, cap in [("real_data.png", "Validation on real patients: ShanghaiT2DM (96 people with T2D, 105 recordings, "
                                        "grouped 5-fold cross-validation). Synthetic-only model shown as zero-shot."),
                      ("hypo_real_data.png", "Low-glucose (<70 mg/dL) prediction on real patients, with the alarm "
                                             "trade-off against the 'urgent low soon' rule."),
                      ("by_history.png", "Performance by how much monitoring history the patient has."),
                      ("ablation.png", "Ablation: which data streams are needed?"),
                      ("calibration_forecast.png", "Probability calibration and glucose-forecast error."),
                      ("lead_time.png", "Warning time before each caught spike."),
                      ("group_importance.png", "Permutation importance by data stream."),
                      ("twin_recovery.png", "The twin recovers hidden physiology from one week of data."),
                      ("cohort_validation.png", "Synthetic cohort sanity checks.")]:
        if (figs / name).exists():
            st.image(str(figs / name), caption=cap, use_container_width=True)


# ======================================================================================
# How it works
# ======================================================================================
with tab_how:
    how = ROOT / "docs" / "how_it_works.md"
    st.markdown(how.read_text(encoding="utf-8") if how.exists() else "See README.md")


# --------------------------------------------------------------------------------------
if autoplay:
    time.sleep(2)
    st.session_state._advance = True
    st.rerun()
