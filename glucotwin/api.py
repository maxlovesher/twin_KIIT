"""REST API for the GlucoTwin engine.

    uvicorn glucotwin.api:app --port 8000
    # then stream data:  python scripts/replay_stream.py --patient GT0007
"""
from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .engine import TwinEngine

app = FastAPI(title="GlucoTwin API", version=__version__,
              description="Hybrid digital twin for predicting glucose spikes in Type 2 Diabetes. "
                          "Research prototype on synthetic data. Not a medical device.")
_engine: TwinEngine | None = None


def engine() -> TwinEngine:
    global _engine
    if _engine is None:
        _engine = TwinEngine()
    return _engine


class Observation(BaseModel):
    timestamp: datetime
    cgm: float | None = Field(None, ge=20, le=500, description="mg/dL")
    steps: float = Field(0, ge=0, description="steps in the 5-min interval")
    hr: float | None = Field(None, ge=25, le=230)
    hrv_rmssd: float | None = Field(None, ge=0, le=400, description="ms")
    in_sleep: int = Field(0, ge=0, le=1)
    sleep_stage: int = Field(0, ge=0, le=3, description="0 awake, 1 light, 2 deep, 3 REM")
    carbs_logged: float = Field(0, ge=0, le=400, description="grams logged in this interval")


class WhatIf(BaseModel):
    horizon_min: int = Field(360, ge=30, le=720)
    meal_carbs: float = Field(0, ge=0, le=300)
    meal_in_min: int = Field(0, ge=0, le=360)
    fast_frac: float = Field(0.6, ge=0.1, le=0.95, description="share of rapidly absorbed carbs")
    walk_min: int = Field(0, ge=0, le=120)
    walk_delay_min: int = Field(30, ge=0, le=240)
    walk_intensity: float = Field(0.7, ge=0.1, le=1.2)
    si_mult: float = Field(1.0, ge=0.5, le=1.5, description="insulin-sensitivity multiplier (e.g. 0.8 after poor sleep)")


def _check(pid: str) -> TwinEngine:
    eng = engine()
    if pid not in eng.twins:
        raise HTTPException(404, f"unknown patient {pid}")
    return eng


@app.get("/health")
def health():
    eng = engine()
    return {"status": "ok", "version": __version__, "patients": len(eng.twins),
            "streaming": len(eng.buffers), "alert_threshold": eng.bundle.risk.threshold}


class PatientRecord(BaseModel):
    """Health record used to onboard a new patient (any field may be omitted except patient_id)."""
    patient_id: str = Field(..., min_length=1, max_length=64)
    age: int | None = Field(None, ge=18, le=110)
    sex: str | None = Field(None, pattern="^(M|F)$")
    city: str | None = None
    height_cm: float | None = Field(None, ge=100, le=230)
    weight_kg: float | None = Field(None, ge=25, le=300)
    bmi: float | None = Field(None, ge=12, le=80)
    waist_cm: float | None = None
    diabetes_duration_y: float | None = Field(None, ge=0, le=80)
    hba1c_pct: float | None = Field(None, ge=3, le=20)
    fpg_mg_dl: float | None = Field(None, ge=40, le=600)
    ldl_mg_dl: float | None = None
    hdl_mg_dl: float | None = None
    tg_mg_dl: float | None = None
    egfr: float | None = None
    sbp_mmhg: float | None = None
    dbp_mmhg: float | None = None
    hypertension: int = 0
    dyslipidemia: int = 0
    ckd: int = 0
    neuropathy: int = 0
    coronary_artery_disease: int = 0
    med_metformin: int = 0
    med_sulfonylurea: int = 0
    med_dpp4i: int = 0
    med_sglt2i: int = 0
    med_basal_insulin: int = 0
    med_statin: int = 0
    med_betablocker: int = 0
    family_history_t2d: int | None = None
    tcf7l2_risk_alleles: int | None = Field(None, ge=0, le=2)
    kcnq1_risk_alleles: int | None = Field(None, ge=0, le=2)
    polygenic_risk_z: float | None = None
    diet_staple: str | None = None
    lab_date: str | None = None
    diagnosis_year: int | None = None


def _num(v):
    """JSON-friendly scalar: numpy -> Python, NaN -> None."""
    if hasattr(v, "item"):
        v = v.item()
    return None if v is None or (isinstance(v, float) and v != v) else v


@app.get("/patients")
def patients():
    eng = engine()
    return [{"patient_id": pid, "age": _num(eng.ehr.loc[pid, "age"]), "sex": _num(eng.ehr.loc[pid, "sex"]),
             "hba1c_pct": _num(eng.ehr.loc[pid, "hba1c_pct"]), "streaming": pid in eng.buffers,
             "twin_calibrated": eng.calibrated.get(pid, False)}
            for pid in eng.patients()]


@app.post("/patients", status_code=201)
def register(record: PatientRecord):
    """Onboard a new patient: their twin starts from EHR-derived priors (cold start)."""
    return engine().register(record.model_dump())


@app.post("/patients/{pid}/calibrate")
def calibrate(pid: str):
    """Personalise the patient's twin from the data streamed so far (needs >= 2 days)."""
    eng = _check(pid)
    try:
        return eng.calibrate(pid)
    except ValueError as e:
        raise HTTPException(409, str(e)) from e


@app.get("/patients/{pid}/fhir")
def fhir(pid: str):
    """FHIR R4 Bundle (Patient, Conditions, Observations, MedicationStatements, RiskAssessment)."""
    return _check(pid).fhir(pid)


@app.get("/patients/{pid}/twin")
def twin_profile(pid: str):
    eng = _check(pid)
    return {"patient_id": pid, "phenotype": eng.twins[pid].phenotype(eng.cohort_params)}


@app.post("/patients/{pid}/observations")
def ingest(pid: str, obs: list[Observation]):
    eng = _check(pid)
    n = eng.ingest(pid, [o.model_dump() for o in obs])
    return {"patient_id": pid, "buffer_rows": n}


@app.get("/patients/{pid}/assessment")
def assessment(pid: str, explain: bool = True):
    eng = _check(pid)
    try:
        return eng.assess(pid, explain=explain)
    except LookupError as e:
        raise HTTPException(409, str(e)) from e


@app.post("/patients/{pid}/what-if")
def what_if(pid: str, scenario: WhatIf):
    eng = _check(pid)
    if pid not in eng.buffers:
        raise HTTPException(409, f"no data ingested for {pid}")
    s = scenario.model_dump()
    return eng.what_if(pid, horizon_min=s.pop("horizon_min"), **s)
