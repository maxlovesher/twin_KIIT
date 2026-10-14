"""Causal feature engineering: fuses static EHR, dynamic wearable streams and twin outputs.

Every dynamic feature at row t uses only rows <= t (rolling windows look back, "routine"
features look at *yesterday's* future, which is already in the past). The same code runs
offline (training) and online (``engine.py``) so there is no train/serve skew.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

H = C.HORIZON_STEPS                 # 24 steps = 120 min
DAY = C.STEPS_PER_DAY               # 288 steps
ENGINE_BUFFER_STEPS = 6 * DAY       # history the online engine keeps to reproduce every feature

STATIC_CLINICAL = [
    "age", "sex_male", "bmi", "waist_cm", "diabetes_duration_y", "hba1c_pct", "fpg_mg_dl",
    "ldl_mg_dl", "hdl_mg_dl", "tg_mg_dl", "egfr", "sbp_mmhg", "dbp_mmhg", "hypertension",
    "dyslipidemia", "ckd", "neuropathy", "coronary_artery_disease", "staple_rice",
    "med_metformin", "med_sulfonylurea", "med_dpp4i", "med_sglt2i", "med_basal_insulin",
    "med_statin", "med_betablocker",
]
STATIC_GENETIC = ["tcf7l2_risk_alleles", "kcnq1_risk_alleles", "polygenic_risk_z", "family_history_t2d"]

FEATURE_GROUPS: dict[str, list[str]] = {
    "Glucose trend (CGM)": [
        "cgm_now", "cgm_d5", "cgm_d15", "cgm_d30", "cgm_d60", "cgm_roc", "cgm_accel",
        "cgm_mean_1h", "cgm_std_1h", "cgm_min_3h", "cgm_max_3h", "cgm_mean_24h", "cgm_cv_24h",
        "tar_24h", "tar_72h", "cgm_gap_1h"],
    "Personal routine": [
        "routine_max2h_d1", "routine_max2h_3d", "routine_spike_3d", "tod_mean_3d", "cgm_same_time_d1"],
    "Meals (logged)": [
        "carbs_30", "carbs_60", "carbs_120", "carbs_240", "min_since_meal", "carbs_today", "meals_today"],
    "Activity": [
        "steps_15", "steps_30", "steps_60", "steps_120", "steps_today", "active_min_3h", "sedentary_min_2h"],
    "Heart rate & HRV": [
        "hr_30", "hr_vs_rest", "hr_still_elev", "hrv_3h", "hrv_rel_day", "hrv_night_rel"],
    "Sleep": [
        "asleep_now", "last_sleep_h", "last_sleep_deep_frac", "last_sleep_rem_frac",
        "last_sleep_awakenings", "last_sleep_eff", "sleep_debt_h", "sleep_vs_usual_h", "min_since_wake"],
    "Time of day": ["tod_sin", "tod_cos", "is_weekend"],
    "Digital twin": [
        "twin_g30", "twin_g60", "twin_g120", "twin_max120", "twin_rise120", "twin_resid_30",
        "twin_resid_60", "twin_insulin", "twin_gut_carbs", "twin_ex_memory",
        "twin_SI", "twin_beta", "twin_Gb"],
    "Clinical (EHR)": STATIC_CLINICAL,
    "Genetics": STATIC_GENETIC,
}
ALL_FEATURES = [f for fs in FEATURE_GROUPS.values() for f in fs]
FEATURE_TO_GROUP = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}


# --------------------------------------------------------------------------------------
def static_features(ehr: pd.DataFrame) -> pd.DataFrame:
    """Model-ready static features. Columns a source does not provide (e.g. genetics from Synthea) become NaN."""
    s = ehr.set_index("patient_id").copy()
    s["sex_male"] = (s["sex"] == "M").astype(float) if "sex" in s else np.nan
    staple = s["diet_staple"] if "diet_staple" in s else pd.Series(np.nan, index=s.index)
    known = staple.notna() & (staple != "unknown")
    s["staple_rice"] = np.where(known, (staple == "rice").astype(float), np.nan)
    return s.reindex(columns=STATIC_CLINICAL + STATIC_GENETIC).apply(pd.to_numeric, errors="coerce").astype(float)


def _roll(x: pd.Series, w: int, how: str, minp: int | None = None) -> pd.Series:
    r = x.rolling(w, min_periods=minp if minp is not None else max(1, w // 3))
    return getattr(r, how)()


def _future_max(s: pd.Series, h: int = H) -> pd.Series:
    """max over (t, t+h] — only ever used shifted by >= 1 day for features."""
    return s[::-1].rolling(h, min_periods=max(1, h // 4)).max()[::-1].shift(-1)


def is_spike(s: pd.Series, fmax: pd.Series) -> pd.Series:
    """Hyperglycaemic excursion in (t, t+2h]: above 180 mg/dL AND >= 40 mg/dL above the level at t."""
    return (fmax > C.HYPER_THRESHOLD) & (fmax >= s + C.SPIKE_RISE)


def _since(mask: pd.Series, cap: float) -> pd.Series:
    idx = pd.Series(np.where(mask.to_numpy(), np.arange(len(mask)), np.nan), index=mask.index).ffill()
    out = (np.arange(len(mask)) - idx) * C.STEP_MIN
    return out.fillna(cap).clip(upper=cap)


def _sleep_features(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    sl = df["in_sleep"].to_numpy()
    stage = df["sleep_stage"].to_numpy()
    hrv = df["hrv_rmssd"].to_numpy(np.float64)
    cols = ["last_sleep_h", "last_sleep_deep_frac", "last_sleep_rem_frac", "last_sleep_awakenings",
            "last_sleep_eff", "sleep_vs_usual_h", "hrv_night_rel", "wake_idx"]
    out = np.full((n, len(cols)), np.nan)
    # episodes = runs of in_sleep==1 lasting >= 2 h that start and end inside the frame
    # (a night already in progress when the data begins is truncated, so it is skipped)
    change = np.flatnonzero(np.diff(np.r_[0, sl, 0]))
    starts, ends = change[0::2], change[1::2]  # ends exclusive
    history = []  # (asleep_h, hrv_mean)
    for a, b in zip(starts, ends):
        if b - a < 24 or b >= n or a == 0:
            continue
        st = stage[a:b]
        asleep_h = (st > 0).sum() * C.STEP_MIN / 60
        awak = int(np.sum((st[1:] == 0) & (st[:-1] > 0)))
        hv = np.nanmean(hrv[a:b]) if np.any(~np.isnan(hrv[a:b])) else np.nan
        prev = history[-3:]
        usual = np.mean([h[0] for h in prev]) if prev else np.nan
        prev_hrv = [h[1] for h in prev if not np.isnan(h[1])]
        out[b] = [asleep_h, (st == 2).mean(), (st == 3).mean(), awak, asleep_h / ((b - a) * C.STEP_MIN / 60),
                  asleep_h - usual, hv / np.mean(prev_hrv) if prev_hrv else np.nan, b]
        history.append((asleep_h, hv))
    f = pd.DataFrame(out, columns=cols, index=df.index).ffill()
    f["sleep_debt_h"] = 7.0 - f["last_sleep_h"]
    f["min_since_wake"] = ((np.arange(n) - f["wake_idx"]) * C.STEP_MIN).clip(upper=1440)
    f.loc[sl == 1, "min_since_wake"] = 0.0
    f["asleep_now"] = sl.astype(float)
    return f.drop(columns="wake_idx")


def dynamic_features(df: pd.DataFrame, twin_cols: pd.DataFrame | None = None) -> pd.DataFrame:
    """Features for every row of one patient's time-ordered 5-min stream."""
    df = df.reset_index(drop=True)
    ts = pd.DatetimeIndex(df["timestamp"])
    s = df["cgm"].astype(float)
    cg = s.ffill(limit=3)
    f = pd.DataFrame(index=df.index)

    # --- glucose trend
    f["cgm_now"] = cg
    for k in (1, 3, 6, 12):
        f[f"cgm_d{k * 5}"] = cg - cg.shift(k)
    f["cgm_roc"] = f["cgm_d15"] / 15.0
    f["cgm_accel"] = f["cgm_d15"] - (cg.shift(3) - cg.shift(6))
    f["cgm_mean_1h"] = _roll(s, 12, "mean")
    f["cgm_std_1h"] = _roll(s, 12, "std")
    f["cgm_min_3h"] = _roll(s, 36, "min")
    f["cgm_max_3h"] = _roll(s, 36, "max")
    f["cgm_mean_24h"] = _roll(s, DAY, "mean")
    f["cgm_cv_24h"] = _roll(s, DAY, "std") / f["cgm_mean_24h"]
    above = (s > C.HYPER_THRESHOLD).astype(float).where(s.notna())
    f["tar_24h"] = _roll(above, DAY, "mean")
    f["tar_72h"] = _roll(above, 3 * DAY, "mean")
    f["cgm_gap_1h"] = _roll(s.isna().astype(float), 12, "sum")

    # --- personal routine (what happened at this time on previous days)
    fmax = _future_max(s)
    spike = is_spike(s, fmax).astype(float).where(fmax.notna())
    smooth = s.rolling(7, center=True, min_periods=1).mean()
    f["routine_max2h_d1"] = fmax.shift(DAY)
    f["routine_max2h_3d"] = pd.concat([fmax.shift(k * DAY) for k in (1, 2, 3)], axis=1).mean(axis=1)
    f["routine_spike_3d"] = pd.concat([spike.shift(k * DAY) for k in (1, 2, 3)], axis=1).mean(axis=1)
    f["tod_mean_3d"] = pd.concat([smooth.shift(k * DAY) for k in (1, 2, 3)], axis=1).mean(axis=1)
    f["cgm_same_time_d1"] = smooth.shift(DAY)

    # --- meals
    c = df["carbs_logged"].astype(float)
    day = ts.normalize()
    for w, name in ((6, "30"), (12, "60"), (24, "120"), (48, "240")):
        f[f"carbs_{name}"] = c.rolling(w, min_periods=1).sum()
    f["min_since_meal"] = _since(c > 0, 720.0)
    f["carbs_today"] = c.groupby(day).cumsum().to_numpy()
    f["meals_today"] = (c > 0).astype(int).groupby(day).cumsum().to_numpy()

    # --- activity
    st = df["steps"].astype(float)
    for w, name in ((3, "15"), (6, "30"), (12, "60"), (24, "120")):
        f[f"steps_{name}"] = st.rolling(w, min_periods=1).sum()
    f["steps_today"] = st.groupby(day).cumsum().to_numpy()
    f["active_min_3h"] = (st > 250).astype(float).rolling(36, min_periods=1).sum() * C.STEP_MIN
    awake = df["in_sleep"] == 0
    f["sedentary_min_2h"] = ((st < 25) & awake).astype(float).rolling(24, min_periods=1).sum() * C.STEP_MIN

    # --- heart
    hr = df["hr"].astype(float)
    still_awake = (st < 50) & awake
    f["hr_30"] = _roll(hr, 6, "mean")
    rest = _roll(hr.where(df["in_sleep"] == 1), 3 * DAY, "mean", minp=24)
    f["hr_vs_rest"] = f["hr_30"] - rest
    f["hr_still_elev"] = (_roll(hr.where(still_awake), 6, "mean", minp=1)
                          - _roll(hr.where(still_awake), 3 * DAY, "mean", minp=24))
    hrv_day = df["hrv_rmssd"].astype(float).where(awake)
    f["hrv_3h"] = _roll(hrv_day, 36, "mean", minp=1)
    f["hrv_rel_day"] = f["hrv_3h"] / _roll(hrv_day, 3 * DAY, "mean", minp=12)

    # --- sleep
    f = pd.concat([f, _sleep_features(df)], axis=1)

    # --- time
    hod = ts.hour + ts.minute / 60.0
    f["tod_sin"] = np.sin(2 * np.pi * hod / 24)
    f["tod_cos"] = np.cos(2 * np.pi * hod / 24)
    f["is_weekend"] = (ts.dayofweek >= 5).astype(float)

    # --- digital twin
    if twin_cols is not None:
        tc = twin_cols.reset_index(drop=True)
        for col in ("twin_g30", "twin_g60", "twin_g120", "twin_max120", "twin_insulin",
                    "twin_gut_carbs", "twin_ex_memory"):
            f[col] = tc[col]
        f["twin_rise120"] = tc["twin_max120"] - cg
        f["twin_resid_30"] = cg - tc["twin_g30"].shift(6)
        f["twin_resid_60"] = cg - tc["twin_g60"].shift(12)
    return f


def labels(df: pd.DataFrame) -> pd.DataFrame:
    s = df["cgm"].astype(float).reset_index(drop=True)
    fmax = _future_max(s)
    n_future = s.notna()[::-1].astype(float).rolling(H, min_periods=1).sum()[::-1].shift(-1)
    eligible = s.notna() & (s <= C.HYPER_THRESHOLD) & (n_future >= H // 2)
    y = is_spike(s, fmax).astype(float).where(eligible)
    return pd.DataFrame({"y": y, "g_t30": s.shift(-6), "g_t60": s.shift(-12), "g_t120": s.shift(-24),
                         "future_max": fmax})


def add_twin_static(f: pd.DataFrame, twin) -> pd.DataFrame:
    f["twin_SI"] = twin.params[2] * 1e4
    f["twin_beta"] = twin.params[5] * 100
    f["twin_Gb"] = twin.params[0]
    return f


def build_patient_frame(df: pd.DataFrame, static_row: pd.Series, twin, with_labels: bool = True,
                        onboarding_twin=None, switch_step: int = 0) -> pd.DataFrame:
    """Meta + all features (+ labels) for one patient. ``twin`` is a PatientTwin.

    If ``onboarding_twin`` is given (an EHR-prior twin), it drives rows before ``switch_step``
    and the calibrated ``twin`` takes over afterwards, continuing from the same state. This
    mirrors deployment: the calibrated twin only exists once the onboarding week is over.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    if onboarding_twin is None or switch_step <= 0:
        states = twin.track(df)
        f = dynamic_features(df, twin.forecast_columns(df, states))
        f = add_twin_static(f, twin)
    else:
        a, b = df.iloc[:switch_step], df.iloc[switch_step:]
        st_a = onboarding_twin.track(a)
        st_b = twin.track(b, s0=st_a[-1])
        tcols = pd.concat([onboarding_twin.forecast_columns(a, st_a), twin.forecast_columns(b, st_b)])
        f = dynamic_features(df, tcols)
        f_a = add_twin_static(f.iloc[:switch_step].copy(), onboarding_twin)
        f_b = add_twin_static(f.iloc[switch_step:].copy(), twin)
        f = pd.concat([f_a, f_b])
    for k, v in static_row.items():
        f[k] = v
    meta = pd.DataFrame({"patient_id": df["patient_id"], "timestamp": df["timestamp"], "cgm": df["cgm"],
                         "day": np.arange(len(df)) // DAY})
    parts = [meta, f[ALL_FEATURES].astype(np.float32)]
    if with_labels:
        parts.append(labels(df))
    return pd.concat(parts, axis=1)
