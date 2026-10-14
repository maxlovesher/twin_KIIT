"""Interoperability and patient-lifecycle tests: Synthea import, Apple Health import, FHIR export, API."""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from glucotwin import config as C  # noqa: E402
from glucotwin import interop  # noqa: E402

HAVE_MODELS = (C.MODELS_DIR / "glucotwin_bundle.joblib").exists()


# ------------------------------------------------------------------------------ fixtures
@pytest.fixture()
def synthea_dir(tmp_path: Path) -> Path:
    (tmp_path / "patients.csv").write_text(textwrap.dedent("""\
        Id,BIRTHDATE,DEATHDATE,GENDER,CITY
        aaaaaaaa-1111-2222-3333-444444444444,1970-03-15,,M,Pune
        bbbbbbbb-1111-2222-3333-444444444444,1985-07-01,,F,Mysuru
        """))
    (tmp_path / "conditions.csv").write_text(textwrap.dedent("""\
        START,STOP,PATIENT,ENCOUNTER,CODE,DESCRIPTION
        2018-01-10,,aaaaaaaa-1111-2222-3333-444444444444,e1,44054006,Diabetes mellitus type 2 (disorder)
        2015-06-01,,aaaaaaaa-1111-2222-3333-444444444444,e2,59621000,Essential hypertension (disorder)
        2016-01-01,,bbbbbbbb-1111-2222-3333-444444444444,e3,59621000,Essential hypertension (disorder)
        """))
    (tmp_path / "observations.csv").write_text(textwrap.dedent("""\
        DATE,PATIENT,ENCOUNTER,CATEGORY,CODE,DESCRIPTION,VALUE,UNITS,TYPE
        2025-11-01T09:00:00Z,aaaaaaaa-1111-2222-3333-444444444444,e4,laboratory,4548-4,Hemoglobin A1c,7.2,%,numeric
        2026-05-01T09:00:00Z,aaaaaaaa-1111-2222-3333-444444444444,e5,laboratory,4548-4,Hemoglobin A1c,8.1,%,numeric
        2026-05-01T09:00:00Z,aaaaaaaa-1111-2222-3333-444444444444,e5,laboratory,2339-0,Glucose,165,mg/dL,numeric
        2026-05-01T09:00:00Z,aaaaaaaa-1111-2222-3333-444444444444,e5,vital-signs,29463-7,Body Weight,78,kg,numeric
        2026-05-01T09:00:00Z,aaaaaaaa-1111-2222-3333-444444444444,e5,vital-signs,8302-2,Body Height,165,cm,numeric
        2026-05-01T09:00:00Z,aaaaaaaa-1111-2222-3333-444444444444,e5,laboratory,33914-3,eGFR,55,mL/min,numeric
        2026-05-01T09:00:00Z,bbbbbbbb-1111-2222-3333-444444444444,e6,laboratory,4548-4,Hemoglobin A1c,5.4,%,numeric
        """))
    (tmp_path / "medications.csv").write_text(textwrap.dedent("""\
        START,STOP,PATIENT,PAYER,ENCOUNTER,CODE,DESCRIPTION
        2018-01-10,,aaaaaaaa-1111-2222-3333-444444444444,p,e1,860975,24 HR Metformin hydrochloride 500 MG Extended Release Oral Tablet
        2019-01-01,,aaaaaaaa-1111-2222-3333-444444444444,p,e1,314231,Simvastatin 10 MG Oral Tablet
        2018-01-10,2020-01-01,aaaaaaaa-1111-2222-3333-444444444444,p,e1,310488,glipiZIDE 5 MG Oral Tablet
        """))
    return tmp_path


@pytest.fixture()
def apple_xml(tmp_path: Path) -> Path:
    rec = 'type="{t}" unit="{u}" startDate="{s} +0530" endDate="{e} +0530" value="{v}"'
    rows = [
        ("HKQuantityTypeIdentifierBloodGlucose", "mg/dL", "2026-08-01 08:01:00", "2026-08-01 08:01:00", "120"),
        ("HKQuantityTypeIdentifierBloodGlucose", "mmol&lt;180.1558800000541&gt;/L", "2026-08-01 08:06:00",
         "2026-08-01 08:06:00", "7.5"),
        ("HKQuantityTypeIdentifierHeartRate", "count/min", "2026-08-01 08:02:00", "2026-08-01 08:02:00", "80"),
        ("HKQuantityTypeIdentifierHeartRate", "count/min", "2026-08-01 08:03:00", "2026-08-01 08:03:00", "90"),
        ("HKQuantityTypeIdentifierStepCount", "count", "2026-08-01 08:05:00", "2026-08-01 08:15:00", "600"),
        ("HKQuantityTypeIdentifierDietaryCarbohydrates", "g", "2026-08-01 08:07:00", "2026-08-01 08:07:00", "60"),
        ("HKQuantityTypeIdentifierHeartRateVariabilitySDNN", "ms", "2026-08-01 08:20:00", "2026-08-01 08:20:00", "42"),
    ]
    sleep = [("HKCategoryValueSleepAnalysisInBed", "2026-08-01 07:00:00", "2026-08-01 08:00:00"),
             ("HKCategoryValueSleepAnalysisAsleepDeep", "2026-08-01 07:10:00", "2026-08-01 07:30:00")]
    body = "\n".join(f"<Record {rec.format(t=t, u=u, s=s, e=e, v=v)}/>" for t, u, s, e, v in rows)
    body += "\n" + "\n".join(f'<Record type="HKCategoryTypeIdentifierSleepAnalysis" value="{v}" '
                             f'startDate="{s} +0530" endDate="{e} +0530"/>' for v, s, e in sleep)
    p = tmp_path / "export.xml"
    p.write_text(f'<?xml version="1.0" encoding="UTF-8"?>\n<HealthData locale="en_IN">\n{body}\n</HealthData>\n')
    return p


# ------------------------------------------------------------------------------ Synthea
def test_synthea_import(synthea_dir):
    ehr = interop.ehr_from_synthea(synthea_dir, as_of="2026-07-01")
    assert len(ehr) == 1                                   # only the patient with T2D
    r = ehr.iloc[0]
    assert r.patient_id == "SY-aaaaaaaa" and r.sex == "M" and r.age == 56
    assert r.hba1c_pct == 8.1                              # latest value wins
    assert r.fpg_mg_dl == 165 and r.weight_kg == 78
    assert r.bmi == pytest.approx(78 / 1.65 ** 2, abs=0.1)
    assert r.hypertension == 1 and r.ckd == 1              # eGFR 55 -> CKD
    assert r.med_metformin == 1 and r.med_statin == 1
    assert r.med_sulfonylurea == 0                         # glipizide was stopped in 2020
    assert r.diabetes_duration_y == pytest.approx(8.5, abs=0.1)
    assert np.isnan(r.tcf7l2_risk_alleles)                 # genetics not in Synthea
    assert list(ehr.columns) == interop.EHR_COLUMNS


# ------------------------------------------------------------------------------ Apple Health
def test_apple_health_import(apple_xml):
    w = interop.wearables_from_apple_health(apple_xml).set_index("timestamp")
    assert (w.index.to_series().diff().dropna() == pd.Timedelta(minutes=5)).all()
    assert w.loc["2026-08-01 08:00", "cgm"] == 120
    assert w.loc["2026-08-01 08:05", "cgm"] == round(7.5 * interop.MMOL_TO_MGDL)
    assert w.loc["2026-08-01 08:00", "hr"] == 85
    assert w.loc["2026-08-01 08:05", "steps"] == 300 and w.loc["2026-08-01 08:10", "steps"] == 300
    assert w.loc["2026-08-01 08:05", "carbs_logged"] == 60
    assert w.loc["2026-08-01 07:15", "in_sleep"] == 1 and w.loc["2026-08-01 07:15", "sleep_stage"] == 2
    assert w.loc["2026-08-01 07:45", "in_sleep"] == 1 and w.loc["2026-08-01 07:45", "sleep_stage"] == 0
    assert w.loc["2026-08-01 08:20", "hrv_rmssd"] == 42


# ------------------------------------------------------------------------------ FHIR
def _check_bundle(b: dict) -> dict:
    assert b["resourceType"] == "Bundle" and b["type"] == "collection"
    urls = {e["fullUrl"] for e in b["entry"]}
    assert len(urls) == len(b["entry"])
    by_type: dict = {}
    for e in b["entry"]:
        r = e["resource"]
        assert e["fullUrl"] == f"urn:uuid:{r['id']}"
        if "subject" in r:
            assert r["subject"]["reference"] in urls          # references resolve inside the bundle
        by_type.setdefault(r["resourceType"], []).append(r)
    return by_type


def test_fhir_bundle_from_synthea(synthea_dir):
    ehr = interop.ehr_from_synthea(synthea_dir, as_of="2026-07-01").iloc[0]
    res = _check_bundle(interop.fhir_bundle(ehr))
    assert len(res["Patient"]) == 1 and res["Patient"][0]["gender"] == "male"
    codes = {o["code"]["coding"][0]["code"] for o in res["Observation"]}
    assert {"4548-4", "1558-6", "39156-5", "33914-3"} <= codes
    assert any(c["code"]["coding"][0]["code"] == "44054006" for c in res["Condition"])
    meds = {m["medicationCodeableConcept"]["coding"][0]["code"] for m in res["MedicationStatement"]}
    assert meds == {"A10BA02", "C10AA"}
    assert "RiskAssessment" not in res


# ------------------------------------------------------------------------------ lifecycle via the API
@pytest.mark.skipif(not HAVE_MODELS, reason="run scripts/build_all.py first")
def test_new_patient_lifecycle_api(synthea_dir):
    from fastapi.testclient import TestClient

    from glucotwin.api import app

    client = TestClient(app)
    ehr = interop.ehr_from_synthea(synthea_dir, as_of="2026-07-01").iloc[0]
    record = {k: v for k, v in ehr.items() if isinstance(v, str) or not pd.isna(v)}
    record = {k: (v.item() if hasattr(v, "item") else v) for k, v in record.items()}
    assert client.post("/patients", json=record).status_code == 201
    pid = record["patient_id"]

    # stream 3 days of a synthetic patient's wearables into the new patient
    wear = pd.read_parquet(C.RAW_DIR / "wearables.parquet", filters=[("patient_id", "==", "GT0001")])
    rows = wear.iloc[: 3 * C.STEPS_PER_DAY].drop(columns="patient_id")
    rows["timestamp"] = rows.timestamp.dt.strftime("%Y-%m-%dT%H:%M:%S")
    recs = rows.astype(object).where(rows.notna(), None).to_dict("records")
    assert client.post(f"/patients/{pid}/calibrate").status_code == 409     # no data yet
    for i in range(0, len(recs), 288):
        assert client.post(f"/patients/{pid}/observations", json=recs[i:i + 288]).status_code == 200

    a = client.get(f"/patients/{pid}/assessment").json()
    assert 0 <= a["spike_probability_2h"] <= 1
    cal = client.post(f"/patients/{pid}/calibrate").json()
    assert cal["days_used"] == 3 and cal["fit_rmse_30"] < 60
    assert next(p for p in client.get("/patients").json() if p["patient_id"] == pid)["twin_calibrated"]

    res = _check_bundle(client.get(f"/patients/{pid}/fhir").json())
    risk = res["RiskAssessment"][0]["prediction"][0]
    assert 0 <= risk["probabilityDecimal"] <= 1
    assert any(o["code"]["coding"][0]["code"] == "99504-3" for o in res["Observation"])
