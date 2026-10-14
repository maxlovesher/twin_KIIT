"""Interoperability: import EHRs from Synthea, wearables from Apple Health, export FHIR R4.

* ``ehr_from_synthea``      Synthea CSV export  -> GlucoTwin EHR table (same columns as data/raw/ehr.csv)
* ``wearables_from_apple_health``  Apple Health export.xml -> 5-min wearable stream (engine input schema)
* ``fhir_bundle``           EHR row (+ latest assessment) -> FHIR R4 Bundle with a RiskAssessment, the format
                            used by India's Ayushman Bharat Digital Mission (ABDM) health-information exchange

Fields a source does not provide (e.g. genetics from Synthea) are left as NaN: the gradient-boosted
models handle missing values natively, and the twin falls back to population priors.
"""
from __future__ import annotations

import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

EHR_COLUMNS = [
    "patient_id", "age", "sex", "city", "height_cm", "weight_kg", "bmi", "waist_cm", "diabetes_duration_y",
    "family_history_t2d", "tcf7l2_risk_alleles", "kcnq1_risk_alleles", "polygenic_risk_z", "diet_staple",
    "vegetarian", "hypertension", "dyslipidemia", "coronary_artery_disease", "neuropathy", "med_metformin",
    "med_sulfonylurea", "med_dpp4i", "med_sglt2i", "med_basal_insulin", "med_statin", "med_betablocker",
    "hba1c_pct", "fpg_mg_dl", "ldl_mg_dl", "hdl_mg_dl", "tg_mg_dl", "egfr", "ckd", "sbp_mmhg", "dbp_mmhg",
    "lab_date", "diagnosis_year",
]

# --------------------------------------------------------------------------------------
# Synthea
# --------------------------------------------------------------------------------------
LOINC_TO_FIELD = {
    "8302-2": "height_cm", "29463-7": "weight_kg", "39156-5": "bmi", "8480-6": "sbp_mmhg",
    "8462-4": "dbp_mmhg", "4548-4": "hba1c_pct", "2339-0": "fpg_mg_dl", "1558-6": "fpg_mg_dl",
    "18262-6": "ldl_mg_dl", "2085-9": "hdl_mg_dl", "2571-8": "tg_mg_dl", "33914-3": "egfr", "62238-1": "egfr",
}
SNOMED_T2D = {"44054006"}
SNOMED_FLAGS = {
    "hypertension": {"59621000", "38341003"},
    "dyslipidemia": {"55822004", "267432004", "13644009"},
    "ckd": {"431855005", "431856006", "433144002", "431857002", "46177005", "709044004"},
    "neuropathy": {"368581000119106", "230572002"},
    "coronary_artery_disease": {"53741008", "22298006", "414545008", "399211009"},
}
MED_PATTERNS = {
    "med_metformin": ["metformin"],
    "med_sulfonylurea": ["glipizide", "glimepiride", "glyburide", "gliclazide", "glibenclamide"],
    "med_dpp4i": ["sitagliptin", "linagliptin", "saxagliptin", "alogliptin", "vildagliptin", "teneligliptin"],
    "med_sglt2i": ["canagliflozin", "dapagliflozin", "empagliflozin", "ertugliflozin"],
    "med_basal_insulin": ["insulin glargine", "insulin detemir", "insulin degludec", "isophane", "nph insulin"],
    "med_statin": ["statin"],
    "med_betablocker": ["metoprolol", "atenolol", "bisoprolol", "carvedilol", "nebivolol", "propranolol"],
}


def _read(csv_dir: Path, name: str) -> pd.DataFrame:
    p = Path(csv_dir) / name
    return pd.read_csv(p, dtype=str) if p.exists() else pd.DataFrame()


def _dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_localize(None)


def ehr_from_synthea(csv_dir: str | Path, as_of: str | None = None, diabetic_only: bool = True) -> pd.DataFrame:
    """Map a Synthea CSV export to the GlucoTwin EHR schema (one row per patient)."""
    pts = _read(csv_dir, "patients.csv")
    cond = _read(csv_dir, "conditions.csv")
    obs = _read(csv_dir, "observations.csv")
    meds = _read(csv_dir, "medications.csv")
    if pts.empty:
        raise FileNotFoundError(f"patients.csv not found in {csv_dir}")
    for df, cols in ((cond, ("START", "STOP")), (obs, ("DATE",)), (meds, ("START", "STOP"))):
        for c in cols:
            if c in df:
                df[c] = _dates(df[c])
    ref = pd.Timestamp(as_of) if as_of else (obs["DATE"].max() if not obs.empty else pd.Timestamp.now())

    def active(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        stop = df["STOP"] if "STOP" in df else pd.Series(pd.NaT, index=df.index)
        return df[(df["START"] <= ref) & (stop.isna() | (stop > ref))]

    cond_a, meds_a = active(cond), active(meds)
    obs_r = obs[obs["DATE"] <= ref] if not obs.empty else obs
    rows = []
    for _, p in pts.iterrows():
        pid = p["Id"]
        pc = cond[cond["PATIENT"] == pid] if not cond.empty else cond
        t2d = pc[pc["CODE"].isin(SNOMED_T2D)] if not pc.empty else pc
        if diabetic_only and t2d.empty:
            continue
        r = {c: np.nan for c in EHR_COLUMNS}
        birth = pd.to_datetime(p["BIRTHDATE"], errors="coerce")
        r.update(patient_id=f"SY-{pid[:8]}", age=int((ref - birth).days // 365.25),
                 sex="M" if str(p.get("GENDER", "")).upper().startswith("M") else "F",
                 city=p.get("CITY", np.nan), diet_staple="unknown")
        if not t2d.empty:
            onset = t2d["START"].min()
            r["diabetes_duration_y"] = round((ref - onset).days / 365.25, 1)
            r["diagnosis_year"] = int(onset.year)
        codes = set(cond_a[cond_a["PATIENT"] == pid]["CODE"]) if not cond_a.empty else set()
        for flag, snomed in SNOMED_FLAGS.items():
            r[flag] = int(bool(codes & snomed))
        po = obs_r[obs_r["PATIENT"] == pid] if not obs_r.empty else obs_r
        if not po.empty:
            po = po.assign(num=pd.to_numeric(po["VALUE"], errors="coerce")).dropna(subset=["num"])
            latest = po.sort_values("DATE").groupby("CODE").tail(1).set_index("CODE")
            for code, field in LOINC_TO_FIELD.items():
                if code in latest.index and pd.isna(r[field]):
                    r[field] = float(latest.loc[code, "num"])
            if "4548-4" in latest.index:
                r["lab_date"] = latest.loc["4548-4", "DATE"].strftime("%Y-%m-%d")
        desc = " | ".join(meds_a[meds_a["PATIENT"] == pid]["DESCRIPTION"].str.lower()) if not meds_a.empty else ""
        for flag, pats in MED_PATTERNS.items():
            r[flag] = int(any(s in desc for s in pats))
        if pd.isna(r["bmi"]) and not pd.isna(r["weight_kg"]) and not pd.isna(r["height_cm"]):
            r["bmi"] = round(r["weight_kg"] / (r["height_cm"] / 100) ** 2, 1)
        if not pd.isna(r["egfr"]):
            r["ckd"] = int(r["ckd"] or r["egfr"] < 60)
        rows.append(r)
    return pd.DataFrame(rows, columns=EHR_COLUMNS)


# --------------------------------------------------------------------------------------
# Apple Health
# --------------------------------------------------------------------------------------
HK = "HKQuantityTypeIdentifier"
SLEEP_STAGE = {"HKCategoryValueSleepAnalysisAsleepCore": 1, "HKCategoryValueSleepAnalysisAsleepUnspecified": 1,
               "HKCategoryValueSleepAnalysisAsleep": 1, "HKCategoryValueSleepAnalysisAsleepDeep": 2,
               "HKCategoryValueSleepAnalysisAsleepREM": 3, "HKCategoryValueSleepAnalysisAwake": 0,
               "HKCategoryValueSleepAnalysisInBed": 0}
MMOL_TO_MGDL = 18.016


def _wall(ts: str) -> pd.Timestamp:
    """Apple Health dates look like '2026-08-01 08:00:00 +0530'; keep the local wall-clock time."""
    return pd.Timestamp(ts[:19])


def wearables_from_apple_health(xml_path: str | Path, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Convert an Apple Health export.xml into the engine's 5-min stream.

    Uses blood glucose (CGM apps write it here), heart rate, HRV (Apple reports SDNN, used as a proxy for
    RMSSD), step count, dietary carbohydrates and sleep analysis. Returns columns
    timestamp, cgm, steps, hr, hrv_rmssd, in_sleep, sleep_stage, carbs_logged.
    """
    recs = {"glucose": [], "hr": [], "hrv": [], "steps": [], "carbs": [], "sleep": []}
    for _, el in ET.iterparse(str(xml_path), events=("end",)):
        if el.tag != "Record":
            continue
        t, a = el.get("type", ""), el.attrib
        s0, s1 = _wall(a["startDate"]), _wall(a.get("endDate", a["startDate"]))
        if t == HK + "BloodGlucose":
            v = float(a["value"]) * (MMOL_TO_MGDL if "mol" in a.get("unit", "") else 1.0)
            recs["glucose"].append((s0, v))
        elif t == HK + "HeartRate":
            recs["hr"].append((s0, float(a["value"])))
        elif t == HK + "HeartRateVariabilitySDNN":
            recs["hrv"].append((s0, float(a["value"])))
        elif t == HK + "StepCount":
            recs["steps"].append((s0, s1, float(a["value"])))
        elif t == HK + "DietaryCarbohydrates":
            recs["carbs"].append((s0, float(a["value"])))
        elif t == "HKCategoryTypeIdentifierSleepAnalysis":
            recs["sleep"].append((s0, s1, SLEEP_STAGE.get(a.get("value"), 1), a.get("value", "")))
        el.clear()

    times = [r[0] for v in recs.values() for r in v]
    if not times:
        raise ValueError("no supported Apple Health records found")
    step = pd.Timedelta(minutes=C.STEP_MIN)
    t0 = pd.Timestamp(start) if start else min(times).floor(step)
    t1 = pd.Timestamp(end) if end else max(times).floor(step) + step
    grid = pd.date_range(t0, t1, freq=step, inclusive="left")
    n = len(grid)

    def idx(ts: pd.Timestamp) -> int:
        return int((ts - t0) // step)

    def mean_series(pairs):
        acc, cnt = np.zeros(n), np.zeros(n)
        for ts, v in pairs:
            i = idx(ts)
            if 0 <= i < n:
                acc[i] += v
                cnt[i] += 1
        return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)

    out = pd.DataFrame({"timestamp": grid})
    out["cgm"] = np.round(mean_series(recs["glucose"]))
    out["hr"] = mean_series(recs["hr"])
    out["hrv_rmssd"] = mean_series(recs["hrv"])
    steps = np.zeros(n)
    for s0, s1, v in recs["steps"]:  # spread each step sample over the bins it covers
        bins = [i for i in range(idx(s0), max(idx(s1 - pd.Timedelta(seconds=1)), idx(s0)) + 1) if 0 <= i < n]
        for i in bins:
            steps[i] += v / len(bins)
    out["steps"] = np.round(steps)
    carbs = np.zeros(n)
    for ts, v in recs["carbs"]:
        if 0 <= idx(ts) < n:
            carbs[idx(ts)] += v
    out["carbs_logged"] = carbs
    in_sleep, stage = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    # "InBed" first so that staged records (core/deep/REM/awake) overwrite it
    for s0, s1, st, raw in sorted(recs["sleep"], key=lambda r: r[3] != "HKCategoryValueSleepAnalysisInBed"):
        for i in range(max(idx(s0), 0), min(idx(s1), n)):
            in_sleep[i] = 1
            if raw != "HKCategoryValueSleepAnalysisInBed":
                stage[i] = st
    out["in_sleep"], out["sleep_stage"] = in_sleep, stage
    return out[["timestamp", "cgm", "steps", "hr", "hrv_rmssd", "in_sleep", "sleep_stage", "carbs_logged"]]


# --------------------------------------------------------------------------------------
# FHIR R4
# --------------------------------------------------------------------------------------
LOINC = "http://loinc.org"
SNOMED = "http://snomed.info/sct"
ATC = "http://www.whocc.no/atc"
UCUM = "http://unitsofmeasure.org"
CLINICAL_STATUS = "http://terminology.hl7.org/CodeSystem/condition-clinical"
OBS_FIELDS = [  # field, LOINC, display, unit
    ("hba1c_pct", "4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", "%"),
    ("fpg_mg_dl", "1558-6", "Fasting glucose [Mass/volume] in Serum or Plasma", "mg/dL"),
    ("bmi", "39156-5", "Body mass index (BMI) [Ratio]", "kg/m2"),
    ("weight_kg", "29463-7", "Body weight", "kg"),
    ("height_cm", "8302-2", "Body height", "cm"),
    ("waist_cm", "8280-0", "Waist Circumference at umbilicus by Tape measure", "cm"),
    ("ldl_mg_dl", "18262-6", "Cholesterol in LDL [Mass/volume] in Serum or Plasma by Direct assay", "mg/dL"),
    ("hdl_mg_dl", "2085-9", "Cholesterol in HDL [Mass/volume] in Serum or Plasma", "mg/dL"),
    ("tg_mg_dl", "2571-8", "Triglyceride [Mass/volume] in Serum or Plasma", "mg/dL"),
    ("egfr", "33914-3", "Glomerular filtration rate/1.73 sq M.predicted", "mL/min/{1.73_m2}"),
]
CONDITIONS = [  # flag, SNOMED, display
    ("hypertension", "59621000", "Essential hypertension"),
    ("dyslipidemia", "55822004", "Hyperlipidemia"),
    ("ckd", "709044004", "Chronic kidney disease"),
    ("neuropathy", "230572002", "Diabetic neuropathy"),
    ("coronary_artery_disease", "53741008", "Coronary arteriosclerosis"),
]
MEDS = [  # flag, ATC, display
    ("med_metformin", "A10BA02", "Metformin"),
    ("med_sulfonylurea", "A10BB", "Sulfonylureas"),
    ("med_dpp4i", "A10BH", "Dipeptidyl peptidase 4 (DPP-4) inhibitors"),
    ("med_sglt2i", "A10BK", "Sodium-glucose co-transporter 2 (SGLT2) inhibitors"),
    ("med_basal_insulin", "A10AE", "Insulins and analogues for injection, long-acting"),
    ("med_statin", "C10AA", "HMG CoA reductase inhibitors"),
    ("med_betablocker", "C07A", "Beta blocking agents"),
]
NS = uuid.UUID("6f1c2a52-7d4e-4b8e-9a51-1d2b9c0e7f10")


def _uid(*parts) -> str:
    return str(uuid.uuid5(NS, "/".join(map(str, parts))))


def _cc(system: str, code: str, display: str) -> dict:
    return {"coding": [{"system": system, "code": code, "display": display}], "text": display}


def _qty(value: float, unit: str) -> dict:
    return {"value": float(value), "unit": unit, "system": UCUM, "code": unit}


def fhir_bundle(ehr: pd.Series | dict, assessment: dict | None = None) -> dict:
    """FHIR R4 'collection' Bundle: Patient, Conditions, Observations, MedicationStatements and, if an
    assessment is given, the latest CGM Observation plus a RiskAssessment for the 2-h spike prediction."""
    e = pd.Series(ehr)

    def val(key):
        v = e.get(key)
        return None if v is None or (not isinstance(v, str) and pd.isna(v)) else v

    pid = str(e["patient_id"])
    pat_id = _uid("Patient", pid)
    subject = {"reference": f"urn:uuid:{pat_id}"}
    active = _cc(CLINICAL_STATUS, "active", "Active")
    entries = []

    def add(resource: dict, *key) -> str:
        rid = _uid(resource["resourceType"], pid, *key)
        entries.append({"fullUrl": f"urn:uuid:{rid}", "resource": {"id": rid, **resource}})
        return rid

    patient = {"resourceType": "Patient", "identifier": [{"system": "urn:glucotwin:patient", "value": pid}],
               "gender": {"M": "male", "F": "female"}.get(val("sex"), "unknown")}
    if val("age") is not None:
        patient["birthDate"] = str(pd.Timestamp(C.SIM_START).year - int(val("age")))
    if isinstance(val("city"), str):
        patient["address"] = [{"city": val("city"), "country": "IN"}]
    entries.append({"fullUrl": f"urn:uuid:{pat_id}", "resource": {"id": pat_id, **patient}})

    t2d = {"resourceType": "Condition", "clinicalStatus": active, "subject": subject,
           "code": _cc(SNOMED, "44054006", "Diabetes mellitus type 2")}
    if val("diagnosis_year") is not None:
        t2d["onsetDateTime"] = str(int(val("diagnosis_year")))
    add(t2d, "t2d")
    for flag, code, disp in CONDITIONS:
        if val(flag) == 1:
            add({"resourceType": "Condition", "clinicalStatus": active, "subject": subject,
                 "code": _cc(SNOMED, code, disp)}, flag)

    lab_cat = [_cc("http://terminology.hl7.org/CodeSystem/observation-category", "laboratory", "Laboratory")]
    for field, code, disp, unit in OBS_FIELDS:
        if val(field) is None:
            continue
        ob = {"resourceType": "Observation", "status": "final", "category": lab_cat,
              "code": _cc(LOINC, code, disp), "subject": subject, "valueQuantity": _qty(val(field), unit)}
        if isinstance(val("lab_date"), str):
            ob["effectiveDateTime"] = val("lab_date")
        add(ob, field)
    if val("sbp_mmhg") is not None and val("dbp_mmhg") is not None:
        add({"resourceType": "Observation", "status": "final", "subject": subject,
             "code": _cc(LOINC, "85354-9", "Blood pressure panel with all children optional"),
             "component": [{"code": _cc(LOINC, "8480-6", "Systolic blood pressure"),
                            "valueQuantity": _qty(val("sbp_mmhg"), "mm[Hg]")},
                           {"code": _cc(LOINC, "8462-4", "Diastolic blood pressure"),
                            "valueQuantity": _qty(val("dbp_mmhg"), "mm[Hg]")}]}, "bp")
    for flag, code, disp in MEDS:
        if val(flag) == 1:
            add({"resourceType": "MedicationStatement", "status": "active", "subject": subject,
                 "medicationCodeableConcept": _cc(ATC, code, disp)}, flag)

    if assessment:
        ts = pd.Timestamp(assessment["timestamp"])
        basis = []
        if assessment.get("cgm") is not None:
            cgm_id = add({"resourceType": "Observation", "status": "final", "subject": subject,
                          "code": _cc(LOINC, "99504-3", "Glucose [Mass/volume] in Interstitial fluid"),
                          "effectiveDateTime": ts.isoformat(), "valueQuantity": _qty(assessment["cgm"], "mg/dL")},
                         "cgm", ts.isoformat())
            basis.append({"reference": f"urn:uuid:{cgm_id}"})
        notes = [{"text": f"Driver: {d['group']}: {d['text']}"} for d in assessment.get("drivers") or []]
        if assessment.get("suggestion"):
            sg = assessment["suggestion"]
            notes.append({"text": f"Twin-tested suggestion: {sg['action']} (predicted peak "
                                  f"{sg['peak_without']:.0f} -> {sg['peak_with']:.0f} mg/dL)"})
        risk = {"resourceType": "RiskAssessment", "status": "final", "subject": subject,
                "occurrenceDateTime": ts.isoformat(),
                "method": {"text": "GlucoTwin hybrid digital twin (mechanistic glucose-insulin model + gradient boosting)"},
                "prediction": [{
                    "outcome": {"text": "Hyperglycaemic excursion: CGM > 180 mg/dL and >= 40 mg/dL rise"},
                    "probabilityDecimal": round(float(assessment["spike_probability_2h"]), 4),
                    "whenPeriod": {"start": ts.isoformat(),
                                   "end": (ts + pd.Timedelta(minutes=C.HORIZON_MIN)).isoformat()}}],
                "basis": basis, "note": notes}
        if assessment.get("alert"):
            risk["mitigation"] = "Alert raised: review patient; consider post-meal activity or meal adjustment."
        add(risk, "risk", ts.isoformat())

    key = assessment["timestamp"] if assessment else "ehr"
    return {"resourceType": "Bundle", "id": _uid("Bundle", pid, key), "type": "collection",
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "entry": entries}
