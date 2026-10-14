"""Core tests: physiology sanity, simulator consistency, feature causality, metrics, engine parity."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from glucotwin import config as C  # noqa: E402
from glucotwin import features as F  # noqa: E402
from glucotwin import modeling as M  # noqa: E402
from glucotwin import physiology as ph  # noqa: E402
from glucotwin import simulate  # noqa: E402
from glucotwin import twin as T  # noqa: E402


@pytest.fixture(scope="module")
def cohort():
    return simulate.generate_cohort(n=4, days=9, seed=7, chunk=4)


def _rollout(p, carbs=0.0, walk=0.0, n=360):
    cf = np.zeros(n); cs = np.zeros(n); ex = np.zeros(n)
    cf[:15] = carbs * 0.6 / 15
    cs[:15] = carbs * 0.4 / 15
    ex[30:30 + int(walk)] = 0.75
    return ph.rollout(p, ph.initial_state(p), cf, cs, ex, np.ones(n), np.zeros(n), 12.0, 1.0)


# ------------------------------------------------------------------------------ physiology
def test_meal_raises_and_walk_lowers_glucose():
    p = ph.default_params(1)[0]
    fasting = _rollout(p)
    meal = _rollout(p, carbs=70)
    walk = _rollout(p, carbs=70, walk=20)
    assert meal.max() > fasting.max() + 40
    assert walk.max() < meal.max()
    assert 60 < meal.max() < 400


def test_sglt2_lowers_peak():
    p = ph.default_params(1)[0]
    q = p.copy()
    q[ph.P_KREN] = 0.005
    assert _rollout(q, carbs=90).max() < _rollout(p, carbs=90).max()


# ------------------------------------------------------------------------------ simulator
def test_cohort_shapes_and_consistency(cohort):
    ehr, wear, truth, tp = cohort
    assert len(ehr) == 4 and ehr.patient_id.is_unique
    assert len(wear) == 4 * 9 * C.STEPS_PER_DAY
    assert wear.cgm.dropna().between(40, 400).all()
    assert wear.cgm.isna().mean() < 0.1
    # EHR HbA1c is derived from the simulated glucose -> should track the CGM mean
    gmi = 3.31 + 0.02392 * wear.groupby("patient_id").cgm.mean()
    a1c = ehr.set_index("patient_id").hba1c_pct.loc[gmi.index]
    assert np.all(np.abs(gmi - a1c) < 1.2)
    assert set(F.STATIC_CLINICAL + F.STATIC_GENETIC) <= set(F.static_features(ehr).columns)


# ------------------------------------------------------------------------------ features
def test_features_are_causal(cohort):
    """Changing the future must not change any feature at time t (no look-ahead leakage)."""
    ehr, wear, *_ = cohort
    pid = ehr.patient_id[0]
    df = wear[wear.patient_id == pid].reset_index(drop=True)
    tw = T.PatientTwin(pid, T.prior_params(ehr.iloc[0]))
    st_ = F.static_features(ehr).loc[pid]
    t = 5 * C.STEPS_PER_DAY + 150
    a = F.build_patient_frame(df, st_, tw, with_labels=False)
    future = df.copy()
    rng = np.random.default_rng(0)
    n_future = len(future) - t - 1
    future.loc[t + 1:, "cgm"] = rng.uniform(60, 350, n_future).astype(np.float32)
    future.loc[t + 1:, "steps"] = rng.integers(0, 600, n_future).astype(np.int32)
    future.loc[t + 1:, "carbs_logged"] = 50.0
    future.loc[t + 1:, "in_sleep"] = 1
    b = F.build_patient_frame(future, st_, tw, with_labels=False)
    pd.testing.assert_series_equal(a.loc[t, F.ALL_FEATURES], b.loc[t, F.ALL_FEATURES], check_names=False)


def test_labels_definition():
    df = pd.DataFrame({"cgm": [150.0] * 30 + [190.0] * 30})
    lab = F.labels(df)
    assert lab.y.iloc[5] == 0.0            # next 120 min = rows 6..29, all 150
    assert lab.y.iloc[6] == 1.0            # next 120 min reaches row 30 (190)
    assert np.isnan(lab.y.iloc[40])        # already above 180 -> not eligible


# ------------------------------------------------------------------------------ metrics
def test_event_metrics_single_correct_alarm():
    cgm = np.r_[np.full(60, 140.0), np.full(20, 220.0), np.full(60, 140.0)]
    frame = pd.DataFrame({"patient_id": "P", "cgm": cgm, "y": F.labels(pd.DataFrame({"cgm": cgm})).y})
    score = np.zeros(len(cgm))
    score[40] = 1.0                         # one alarm 100 min before the onset at row 60
    m = M.event_metrics(frame, score, 0.5)
    assert m["n_onsets"] == 1 and m["event_sensitivity"] == 1.0
    assert m["median_lead_min"] == 100.0 and m["alert_precision"] == 1.0


# ------------------------------------------------------------------------------ engine
@pytest.mark.skipif(not (C.MODELS_DIR / "glucotwin_bundle.joblib").exists(), reason="run scripts/build_all.py first")
def test_engine_matches_batch_features():
    from glucotwin.engine import TwinEngine

    eng = TwinEngine()
    pid = eng.patients()[0]
    wear = pd.read_parquet(C.RAW_DIR / "wearables.parquet", filters=[("patient_id", "==", pid)])
    t = 10 * C.STEPS_PER_DAY + 100
    for s in range(0, t + 1, 37):  # stream in chunks, the way a device gateway would
        eng.ingest(pid, wear.iloc[s: min(s + 37, t + 1)].drop(columns="patient_id"))
    online = eng.feature_row(pid).iloc[0]
    batch = F.build_patient_frame(wear, eng.static.loc[pid], eng.twins[pid], with_labels=False).iloc[t]
    pd.testing.assert_series_equal(online[F.ALL_FEATURES].astype(float), batch[F.ALL_FEATURES].astype(float),
                                   check_names=False, rtol=1e-4)
    res = eng.assess(pid)
    assert 0 <= res["spike_probability_2h"] <= 1 and len(res["drivers"]) == 4
    wi = eng.what_if(pid, horizon_min=180, meal_carbs=80, walk_min=20)
    assert wi["scenario_peak"] > wi["baseline_peak"]


@pytest.mark.skipif(not (C.MODELS_DIR / "glucotwin_bundle.joblib").exists(), reason="run scripts/build_all.py first")
def test_assessment_is_valid_json_during_cgm_dropout():
    import json

    from glucotwin.engine import TwinEngine

    eng = TwinEngine()
    pid = eng.patients()[1]
    wear = pd.read_parquet(C.RAW_DIR / "wearables.parquet", filters=[("patient_id", "==", pid)])
    rows = wear.iloc[: 3 * C.STEPS_PER_DAY].drop(columns="patient_id").copy()
    rows.loc[rows.index[-6:], "cgm"] = np.nan          # last 30 min of CGM missing
    eng.ingest(pid, rows)
    json.dumps(eng.assess(pid), allow_nan=False)        # raises if any NaN leaks into the response


def test_injected_insulin_lowers_glucose():
    p = ph.default_params(1)[0]
    n = 300
    cf = np.zeros(n); cs = np.zeros(n); ex = np.zeros(n); ins = np.zeros(n)
    cf[:15] = 70 * 0.6 / 15
    cs[:15] = 70 * 0.4 / 15
    base = ph.rollout(p, ph.initial_state(p), cf, cs, ex, np.ones(n), ins, 12.0, 1.0)
    ins[0] = 8.0                                   # 8 IU s.c. with the meal
    dosed = ph.rollout(p, ph.initial_state(p), cf, cs, ex, np.ones(n), ins, 12.0, 1.0)
    assert dosed.max() < base.max() - 10 and dosed.min() > 40
