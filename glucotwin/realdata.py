"""Loader for the real ShanghaiT2DM dataset (Zhao et al., Scientific Data 10:35, 2023; CC BY 4.0).

Each recording has CGM every 15 min plus time-stamped meals, insulin doses and oral agents; a summary
sheet holds the clinical record. This module maps both onto GlucoTwin's schemas:

* EHR    -> same columns as data/raw/ehr.csv (units converted: HbA1c IFCC->NGSP, lipids mmol/L->mg/dL)
* stream -> 5-min grid (CGM interpolated between 15-min samples), carbs estimated from the meal text,
            plus two extra columns used only by the real-data experiment: insulin_units, oha_taken.

There are no step, heart-rate, HRV or sleep data in this cohort; those columns are left missing.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .interop import EHR_COLUMNS

SUMMARY = "Shanghai_T2DM_Summary.xlsx"
FOLDER = "Shanghai_T2DM"
NOMINAL_MEAL_CARBS = 50.0   # grams, when a meal is recorded but its content is "data not available"

# grams of carbohydrate per gram of food (cooked/as eaten), matched by keyword in the meal description
CARB_DENSITY = [
    ("porridge", 0.10), ("congee", 0.10), ("fried rice", 0.30), ("rice noodle", 0.25), ("rice", 0.28),
    ("noodle", 0.25), ("steamed bread", 0.47), ("steamed bun", 0.45), ("bun", 0.45), ("bread", 0.50),
    ("dumpling", 0.25), ("wonton", 0.22), ("pancake", 0.45), ("biscuit", 0.70), ("cracker", 0.70),
    ("cake", 0.55), ("oat", 0.60), ("cereal", 0.65), ("corn", 0.19), ("sweet potato", 0.20), ("potato", 0.17),
    ("taro", 0.18), ("yam", 0.20), ("pumpkin", 0.07), ("banana", 0.22), ("apple", 0.14), ("orange", 0.12),
    ("pear", 0.15), ("grape", 0.17), ("watermelon", 0.08), ("fruit", 0.13), ("soy milk", 0.03), ("milk", 0.05),
    ("yogurt", 0.12), ("sugar", 1.00), ("lotus root", 0.16), ("bean", 0.10),
]
DEFAULT_DENSITY = 0.03      # vegetables, meat, eggs, tofu ...

MED_PATTERNS = {
    "med_metformin": ["metformin"],
    "med_sulfonylurea": ["gliclazide", "glimepiride", "glipizide", "gliquidone", "glibenclamide", "glyburide"],
    "med_dpp4i": ["gliptin"],
    "med_sglt2i": ["gliflozin"],
    # any injected insulin -> exogenous insulin term in the twin (premixed regimens included)
    "med_basal_insulin": ["insulin", "novolin", "humulin", "gansulin", "lantus", "levemir", "tresiba"],
    "med_statin": ["statin"],
    "med_betablocker": ["olol"],
}


def estimate_carbs(text: str) -> float:
    """Rough carbohydrate estimate (g) from a free-text meal description like 'Rice 150 g\\nEgg 40 g'."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    low = text.lower()
    if "not available" in low:
        return NOMINAL_MEAL_CARBS
    total, found = 0.0, False
    for line in re.split(r"[\n;,]+", low):
        m = re.search(r"(\d+(?:\.\d+)?)\s*(g|ml)\b", line)
        if not m:
            continue
        found = True
        density = next((d for k, d in CARB_DENSITY if k in line), DEFAULT_DENSITY)
        total += float(m.group(1)) * density
    return round(total, 1) if found else NOMINAL_MEAL_CARBS


def insulin_units(*cells) -> float:
    """Sum the insulin units written in one row (e.g. 'insulin aspart 70/30, 18 IU' or a bare number)."""
    total = 0.0
    for c in cells:
        if c is None or (isinstance(c, float) and np.isnan(c)):
            continue
        if isinstance(c, (int, float, np.number)):
            total += float(c)
            continue
        total += sum(float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*iu", str(c).lower()))
    return total


def _num(v) -> float:
    return float(pd.to_numeric(v, errors="coerce"))


def ehr_from_summary(s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in s.iterrows():
        e = {c: np.nan for c in EHR_COLUMNS}
        meds = f"{r.get('Hypoglycemic Agents', '')} {r.get('Other Agents', '')}".lower()
        comorb = str(r.get("Comorbidities", "")).lower()
        micro = str(r.get("Diabetic Microvascular Complications", "")).lower()
        macro = str(r.get("Diabetic Macrovascular  Complications", "")).lower()
        a1c = _num(r["HbA1c (mmol/mol)"])
        egfr = _num(r["Estimated Glomerular Filtration Rate  (ml/min/1.73m2) "])
        e.update(
            patient_id=f"SH-{r['Patient Number']}", age=_num(r["Age (years)"]),
            sex="M" if _num(r["Gender (Female=1, Male=2)"]) == 2 else "F", city="Shanghai",
            height_cm=_num(r["Height (m)"]) * 100, weight_kg=_num(r["Weight (kg)"]), bmi=_num(r["BMI (kg/m2)"]),
            diabetes_duration_y=_num(r["Duration of diabetes (years)"]),
            hba1c_pct=round(0.0915 * a1c + 2.15, 1) if not np.isnan(a1c) else np.nan,
            fpg_mg_dl=_num(r["Fasting Plasma Glucose (mg/dl)"]),
            ldl_mg_dl=_num(r["Low-Density Lipoprotein Cholesterol (mmol/L)"]) * 38.67,
            hdl_mg_dl=_num(r["High-Density Lipoprotein Cholesterol (mmol/L)"]) * 38.67,
            tg_mg_dl=_num(r["Triglyceride (mmol/L)"]) * 88.57, egfr=egfr,
            hypertension=int("hypertension" in comorb),
            dyslipidemia=int("hyperlipidemia" in comorb or "dyslipidemia" in comorb or "statin" in meds),
            ckd=int("nephropathy" in micro or (not np.isnan(egfr) and egfr < 60)),
            neuropathy=int("neuropathy" in micro), coronary_artery_disease=int("coronary" in macro),
            diet_staple="rice",
        )
        for flag, pats in MED_PATTERNS.items():
            e[flag] = int(any(p in meds for p in pats))
        rows.append(e)
    return pd.DataFrame(rows, columns=EHR_COLUMNS)


def _col(d: pd.DataFrame, prefix: str) -> str | None:
    """Column names vary slightly between files ('CGM (mg / dl)' vs 'CGM '); match by prefix."""
    return next((c for c in d.columns if str(c).strip().lower().startswith(prefix.lower())), None)


def stream_from_file(path: Path) -> pd.DataFrame:
    d = pd.read_excel(path)
    rename = {_col(d, p): name for p, name in (
        ("CGM", "CGM (mg / dl)"), ("Dietary intake", "Dietary intake"), ("Insulin dose - s.c", "Insulin dose - s.c."),
        ("Insulin dose - i.v", "Insulin dose - i.v."), ("CSII - bolus", "CSII - bolus insulin (Novolin R, IU)"),
        ("Non-insulin", "Non-insulin hypoglycemic agents")) if _col(d, p)}
    d = d.rename(columns=rename)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    step = pd.Timedelta(minutes=C.STEP_MIN)
    t0, t1 = d["Date"].min().floor(step), d["Date"].max().floor(step) + step
    grid = pd.DataFrame({"timestamp": pd.date_range(t0, t1, freq=step, inclusive="left")})
    b = ((d["Date"] - t0) // step).astype(int).to_numpy()
    n = len(grid)

    cgm = np.full(n, np.nan)
    vals = pd.to_numeric(d["CGM (mg / dl)"], errors="coerce").to_numpy()
    ok = ~np.isnan(vals)
    cgm[b[ok]] = vals[ok]
    grid["cgm"] = pd.Series(cgm).interpolate(limit=2, limit_area="inside").round().to_numpy()

    carbs, ins, oha = np.zeros(n), np.zeros(n), np.zeros(n)
    cols = [c for c in ("Insulin dose - s.c.", "CSII - bolus insulin (Novolin R, IU)", "Insulin dose - i.v.") if c in d]
    for i, r in zip(b, d.to_dict("records")):
        if isinstance(r.get("Dietary intake"), str):
            carbs[i] += estimate_carbs(r["Dietary intake"])
        ins[i] += insulin_units(*[r.get(c) for c in cols])
        agent = r.get("Non-insulin hypoglycemic agents")
        if isinstance(agent, str) and agent.strip():
            oha[i] = 1
    grid["steps"] = np.nan
    grid["hr"] = np.nan
    grid["hrv_rmssd"] = np.nan
    grid["in_sleep"] = 0
    grid["sleep_stage"] = 0
    grid["carbs_logged"] = carbs
    grid["insulin_units"] = ins
    grid["oha_taken"] = oha
    return grid


def load_shanghai_t2dm(root: str | Path = C.DATA_DIR / "external" / "shanghai") -> tuple[pd.DataFrame, dict]:
    """Returns (ehr, {patient_id: 5-min stream}). One 'patient_id' per recording, e.g. SH-2001_1_20201117;
    the subject is the leading number (recordings of the same person share it)."""
    root = Path(root)
    ehr = ehr_from_summary(pd.read_excel(root / SUMMARY))
    known = set(ehr.patient_id)
    streams = {}
    for f in sorted((root / FOLDER).glob("*.xls*")):
        pid = f"SH-{f.stem}"
        if f.name.startswith("~$") or pid not in known:
            continue
        s = stream_from_file(f)
        s.insert(0, "patient_id", pid)
        streams[pid] = s
    ehr = ehr[ehr.patient_id.isin(streams)].reset_index(drop=True)
    return ehr, streams


def subject_of(pid: str) -> str:
    return pid.split("-", 1)[1].split("_")[0]


# --------------------------------------------------------------------------------------
# Feature frames for the real-data experiments
# --------------------------------------------------------------------------------------
CAL_DAYS = 3
MEDS_DYNAMIC = ["insulin_2h", "insulin_4h", "oha_6h"]


def build_record(pid: str, df: pd.DataFrame, erow: pd.Series, static_row: pd.Series, cal_days: int = CAL_DAYS):
    """Twin calibrated on the first ``cal_days`` (EHR priors before that) + all features + labels."""
    from . import features as F
    from . import twin as T

    day = C.STEPS_PER_DAY
    if len(df) < (cal_days + 1) * day:
        return None, None
    prior = T.PatientTwin(pid, T.prior_params(erow))
    fit = T.calibrate_patient(erow, df, days=cal_days)
    calib = T.PatientTwin.from_row(pd.Series(fit))
    fr = F.build_patient_frame(df, static_row, calib, onboarding_twin=prior, switch_step=cal_days * day)
    ins = df["insulin_units"].reset_index(drop=True)
    fr["insulin_2h"] = ins.rolling(24, min_periods=1).sum().to_numpy()
    fr["insulin_4h"] = ins.rolling(48, min_periods=1).sum().to_numpy()
    fr["oha_6h"] = df["oha_taken"].reset_index(drop=True).rolling(72, min_periods=1).max().to_numpy()
    fr["subject"] = subject_of(pid)
    fr = pd.concat([fr, hypo_labels(df["cgm"])], axis=1)
    return fr[fr.day >= 1].reset_index(drop=True), fit


def build_frames(ehr: pd.DataFrame, streams: dict, n_jobs: int = -1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All usable recordings -> (feature frame, twin fit table)."""
    from joblib import Parallel, delayed

    from . import features as F

    static = F.static_features(ehr)
    erows = ehr.set_index("patient_id", drop=False)
    out = Parallel(n_jobs=n_jobs)(delayed(build_record)(pid, df, erows.loc[pid], static.loc[pid])
                                  for pid, df in streams.items())
    frames = [f for f, _ in out if f is not None]
    fits = pd.DataFrame([fit for _, fit in out if fit is not None])
    return pd.concat(frames, ignore_index=True), fits


def hypo_labels(cgm: pd.Series, horizon: int = C.HORIZON_STEPS) -> pd.DataFrame:
    """y_hypo = CGM falls below 70 mg/dL within the next 2 h, for rows currently at or above 70."""
    s = cgm.astype(float).reset_index(drop=True)
    fmin = s[::-1].rolling(horizon, min_periods=max(1, horizon // 4)).min()[::-1].shift(-1)
    n_future = s.notna()[::-1].astype(float).rolling(horizon, min_periods=1).sum()[::-1].shift(-1)
    eligible = s.notna() & (s >= C.HYPO_THRESHOLD) & (n_future >= horizon // 2)
    return pd.DataFrame({"y_hypo": (fmin < C.HYPO_THRESHOLD).astype(float).where(eligible)})
