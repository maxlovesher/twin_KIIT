"""Parsing tests for the ShanghaiT2DM loader (unit conversions, carbs, insulin) on made-up rows."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from glucotwin import config as C  # noqa: E402
from glucotwin import realdata as RD  # noqa: E402

DATA = C.DATA_DIR / "external" / "shanghai"


def test_estimate_carbs():
    assert RD.estimate_carbs("Rice 150 g\nVegetable 100 g") == pytest.approx(150 * 0.28 + 100 * 0.03)
    assert RD.estimate_carbs("data not available") == RD.NOMINAL_MEAL_CARBS
    assert RD.estimate_carbs("Coarse grain steamed bread 140 g") == pytest.approx(140 * 0.47)
    assert RD.estimate_carbs("Fried rice 100 g") == pytest.approx(30.0)      # 'fried rice' beats 'rice'
    assert RD.estimate_carbs("") == 0.0


def test_insulin_units():
    assert RD.insulin_units("insulin aspart 70/30, 18 IU") == 18
    assert RD.insulin_units("Novolin 30R, 6 IU", 4.0, np.nan) == 10
    assert RD.insulin_units(None, np.nan) == 0


def test_ehr_unit_conversions():
    row = {
        "Patient Number": "9999_0_20200101", "Gender (Female=1, Male=2)": 2, "Age (years)": 60,
        "Height (m)": 1.70, "Weight (kg)": 70, "BMI (kg/m2)": 24.2, "Duration of diabetes (years)": 10,
        "Diabetic Microvascular Complications": "neuropathy, retinopathy",
        "Diabetic Macrovascular  Complications": "coronary heart disease", "Comorbidities": "hypertension",
        "Hypoglycemic Agents": "metformin, insulin degludec, sitagliptin", "Other Agents": "atorvastatin, metoprolol",
        "Fasting Plasma Glucose (mg/dl)": 150, "HbA1c (mmol/mol)": 64, "Triglyceride (mmol/L)": 2.0,
        "High-Density Lipoprotein Cholesterol (mmol/L)": 1.0, "Low-Density Lipoprotein Cholesterol (mmol/L)": 3.0,
        "Estimated Glomerular Filtration Rate  (ml/min/1.73m2) ": "/",
    }
    e = RD.ehr_from_summary(pd.DataFrame([row])).iloc[0]
    assert e.patient_id == "SH-9999_0_20200101" and e.sex == "M" and e.height_cm == pytest.approx(170)
    assert e.hba1c_pct == pytest.approx(8.0, abs=0.05)                # 64 mmol/mol (IFCC) = 8.0% (NGSP)
    assert e.tg_mg_dl == pytest.approx(177.1, abs=0.1) and e.ldl_mg_dl == pytest.approx(116.0, abs=0.1)
    assert np.isnan(e.egfr) and e.ckd == 0                              # '/' = not measured
    assert e.neuropathy == 1 and e.coronary_artery_disease == 1 and e.hypertension == 1
    assert (e.med_metformin, e.med_dpp4i, e.med_basal_insulin, e.med_statin, e.med_betablocker) == (1, 1, 1, 1, 1)
    assert e.med_sulfonylurea == 0 and e.med_sglt2i == 0
    assert RD.subject_of(e.patient_id) == "9999"


@pytest.mark.skipif(not (DATA / RD.SUMMARY).exists(), reason="ShanghaiT2DM not downloaded")
def test_load_real_cohort():
    ehr, streams = RD.load_shanghai_t2dm()
    assert len(streams) > 100 and set(ehr.patient_id) == set(streams)
    s = next(iter(streams.values()))
    assert (s.timestamp.diff().dropna() == pd.Timedelta(minutes=5)).all()
    assert s.cgm.notna().mean() > 0.9 and s.cgm.dropna().between(20, 600).all()
