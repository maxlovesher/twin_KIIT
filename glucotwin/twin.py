"""Per-patient mechanistic digital twin: calibration, state tracking, forecasting, what-if.

The twin only uses information a real deployment would have:
  * EHR → physiological priors (weight → glucose volume, meds → secretion/excretion terms,
    FPG → basal glucose, BMI → insulin sensitivity, duration → beta-cell function)
  * logged meals, step counts and CGM → personalised parameters (onboarding week) and
    a continuously updated physiological state (CGM is assimilated every 5 min).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import config as C
from . import physiology as ph

DT = float(C.STEP_MIN)
DEFAULT_FAST_FRAC = 0.6          # assumed GI split for logged meals
STEPS_PER_MIN_AT_FULL = 110.0    # cadence that maps to exercise intensity 1.0
FIT_IDX = [ph.P_GB, ph.P_SI, ph.P_BETA, ph.P_DAWN, ph.P_SG]
FIT_NAMES = ["Gb", "SI", "beta", "dawn", "SG"]
PRIOR_SD = np.array([0.15, 0.6, 0.6, 0.8, 0.6])  # on the log scale of each fitted parameter


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------
@dataclass
class TwinInputs:
    cgm: np.ndarray
    carbs_fast: np.ndarray
    carbs_slow: np.ndarray
    ex: np.ndarray
    insulin: np.ndarray
    hour: np.ndarray


def make_inputs(df: pd.DataFrame) -> TwinInputs:
    """Twin inputs from one patient's 5-min wearable rows (sorted by time)."""
    carbs = np.nan_to_num(df["carbs_logged"].to_numpy(np.float64))  # missing log / no step sensor = 0
    steps = np.nan_to_num(df["steps"].to_numpy(np.float64))
    ts = pd.DatetimeIndex(df["timestamp"])
    return TwinInputs(
        cgm=df["cgm"].to_numpy(np.float64),
        carbs_fast=carbs * DEFAULT_FAST_FRAC,
        carbs_slow=carbs * (1 - DEFAULT_FAST_FRAC),
        ex=np.clip(steps / (C.STEP_MIN * STEPS_PER_MIN_AT_FULL), 0, 1.2),
        insulin=(np.nan_to_num(df["insulin_units"].to_numpy(np.float64)) if "insulin_units" in df
                 else np.zeros(len(df))),
        hour=(ts.hour + ts.minute / 60.0).to_numpy(np.float64),
    )


def prior_params(ehr: pd.Series) -> np.ndarray:
    """Population priors for one patient, personalised from their EHR."""
    def get(key: str, default: float) -> float:
        v = ehr.get(key, default)
        return default if v is None or pd.isna(v) else float(v)

    p = ph.default_params(1)[0]
    p[ph.P_VG] = 1.9 * get("weight_kg", 70.0)
    p[ph.P_GB] = 0.97 * get("fpg_mg_dl", 140.0)
    p[ph.P_SI] = 2.6e-4 * np.exp(-0.07 * (get("bmi", 25.0) - 25)) * (1.1 if get("med_metformin", 0) else 1.0)
    p[ph.P_BETA] = (0.075 * np.exp(-0.045 * get("diabetes_duration_y", 5.0))
                    * (1.25 if get("med_dpp4i", 0) else 1.0) * (1.2 if get("med_sulfonylurea", 0) else 1.0))
    p[ph.P_S0] = 0.5 if get("med_sulfonylurea", 0) else 0.0
    p[ph.P_IEXO] = 6.0 if get("med_basal_insulin", 0) else 0.0
    p[ph.P_KREN] = 0.005 if get("med_sglt2i", 0) else 0.0
    p[ph.P_DAWN] = 0.15
    return p


def _pack(p: np.ndarray) -> np.ndarray:
    return np.log(p[FIT_IDX])


def _unpack(theta: np.ndarray, base: np.ndarray) -> np.ndarray:
    p = base.copy()
    p[FIT_IDX] = np.exp(theta)
    return p


def _start_state(p: np.ndarray, first_cgm: float) -> np.ndarray:
    s = ph.initial_state(p)
    if not np.isnan(first_cgm):
        s[ph.S_G] = first_cgm
    return s


# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------
def _fit_errors(p: np.ndarray, inp: TwinInputs, horizons=(6, 12), stride: int = 2) -> list[np.ndarray]:
    first = inp.cgm[~np.isnan(inp.cgm)][0]
    states = ph.observe(p, _start_state(p, first), inp.cgm, inp.carbs_fast, inp.carbs_slow,
                        inp.ex, inp.insulin, inp.hour, DT)
    fc = ph.forecast_with_inputs(p, states, inp.carbs_fast, inp.carbs_slow, inp.ex, inp.insulin, inp.hour,
                                 DT, max(horizons), stride)
    starts = np.arange(0, len(inp.cgm), stride)
    errs = []
    for k in horizons:
        tgt = starts + k
        ok = tgt < len(inp.cgm)
        e = fc[ok, k - 1] - inp.cgm[tgt[ok]]
        errs.append(e[~np.isnan(e)])
    return errs


def _huber(e: np.ndarray, delta: float = 25.0) -> float:
    a = np.abs(e)
    return float(np.mean(np.where(a <= delta, 0.5 * a ** 2, delta * (a - 0.5 * delta))))


def calibrate_patient(ehr: pd.Series, df: pd.DataFrame, days: int = C.CALIBRATION_DAYS) -> dict:
    """MAP fit of (Gb, SI, beta, dawn) on the first ``days`` days of data."""
    sub = df.iloc[: days * C.STEPS_PER_DAY]
    inp = make_inputs(sub)
    base = prior_params(ehr)
    theta0 = _pack(base)

    def objective(theta):
        p = _unpack(theta, base)
        data = sum(_huber(e) for e in _fit_errors(p, inp))
        prior = float(np.sum(((theta - theta0) / PRIOR_SD) ** 2))
        return data / 100.0 + prior

    res = minimize(objective, theta0, method="Nelder-Mead",
                   options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-4})
    p = _unpack(res.x, base)
    e30, e60 = _fit_errors(p, inp)
    out = {"patient_id": ehr.patient_id}
    out.update({name: float(p[i]) for i, name in enumerate(ph.PARAM_NAMES)})
    out.update(fit_rmse_30=float(np.sqrt(np.mean(e30 ** 2))), fit_rmse_60=float(np.sqrt(np.mean(e60 ** 2))))
    return out


def calibrate_cohort(ehr: pd.DataFrame, wear: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    from joblib import Parallel, delayed

    groups = {pid: g.reset_index(drop=True) for pid, g in wear.groupby("patient_id", sort=False)}
    rows = Parallel(n_jobs=n_jobs)(
        delayed(calibrate_patient)(r, groups[r.patient_id]) for _, r in ehr.iterrows())
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# The twin
# --------------------------------------------------------------------------------------
TWIN_FORECAST_STEPS = C.HORIZON_STEPS


@dataclass
class PatientTwin:
    patient_id: str
    params: np.ndarray

    @classmethod
    def from_row(cls, row: pd.Series) -> "PatientTwin":
        return cls(row.patient_id, np.array([row[k] for k in ph.PARAM_NAMES], dtype=np.float64))

    # -- state tracking ----------------------------------------------------------------
    def initial_state(self, first_cgm: float = np.nan) -> np.ndarray:
        return _start_state(self.params, first_cgm)

    def track(self, df: pd.DataFrame, s0: np.ndarray | None = None) -> np.ndarray:
        """Assimilate observed rows; returns post-update states (T, N_STATES)."""
        inp = make_inputs(df)
        if s0 is None:
            ok = inp.cgm[~np.isnan(inp.cgm)]
            s0 = self.initial_state(ok[0] if len(ok) else np.nan)
        return ph.observe(self.params, s0, inp.cgm, inp.carbs_fast, inp.carbs_slow, inp.ex, inp.insulin,
                          inp.hour, DT)

    def forecast_columns(self, df: pd.DataFrame, states: np.ndarray) -> pd.DataFrame:
        """Physiology-only forecasts (no future meals/exercise assumed) for every row."""
        hour = make_inputs(df).hour
        fc = ph.forecast(self.params, states, hour, DT, TWIN_FORECAST_STEPS)
        return pd.DataFrame({
            "twin_g30": fc[:, 5], "twin_g60": fc[:, 11], "twin_g120": fc[:, 23],
            "twin_max120": fc.max(axis=1),
            "twin_insulin": states[:, ph.S_I],
            "twin_gut_carbs": states[:, [ph.S_Q1F, ph.S_Q2F, ph.S_Q1S, ph.S_Q2S]].sum(axis=1),
            "twin_ex_memory": states[:, ph.S_EXM],
        }, index=df.index)

    # -- what-if -------------------------------------------------------------------------
    def what_if(self, state: np.ndarray, hour0: float, horizon_min: int = 360,
                meal_carbs: float = 0.0, meal_in_min: int = 0, fast_frac: float = DEFAULT_FAST_FRAC,
                walk_min: int = 0, walk_delay_min: int = 30, walk_intensity: float = 0.7,
                si_mult: float = 1.0, insulin_units: float = 0.0, insulin_in_min: int = 0,
                dt: float = 1.0) -> np.ndarray:
        """Simulate a scenario from the current state. Returns glucose every ``dt`` minutes."""
        n = int(horizon_min / dt)
        cf = np.zeros(n); cs = np.zeros(n); ex = np.zeros(n)
        if meal_carbs > 0:
            k0, k1 = int(meal_in_min / dt), int((meal_in_min + 15) / dt)
            k1 = max(k1, k0 + 1)
            cf[k0:k1] = meal_carbs * fast_frac / (k1 - k0)
            cs[k0:k1] = meal_carbs * (1 - fast_frac) / (k1 - k0)
        if walk_min > 0:
            w0 = int((meal_in_min + walk_delay_min) / dt)
            ex[w0:w0 + int(walk_min / dt)] = walk_intensity
        ins = np.zeros(n)
        if insulin_units > 0:
            ins[min(int(insulin_in_min / dt), n - 1)] = insulin_units
        return ph.rollout(self.params, state.astype(np.float64), cf, cs, ex,
                          np.full(n, float(si_mult)), ins, float(hour0), float(dt))

    # -- interpretation ------------------------------------------------------------------
    def phenotype(self, cohort: pd.DataFrame | None = None) -> dict:
        """Plain-language phenotype; percentiles relative to the cohort if given."""
        p = dict(zip(ph.PARAM_NAMES, self.params))
        out = {"insulin_sensitivity": p["SI"] * 1e4, "beta_cell_response": p["beta"] * 100,
               "basal_glucose": p["Gb"], "dawn_effect": p["dawn"] * 60}
        if cohort is not None:
            for key, col in [("insulin_sensitivity", "SI"), ("beta_cell_response", "beta"),
                             ("basal_glucose", "Gb"), ("dawn_effect", "dawn")]:
                out[key + "_pct"] = float((cohort[col] < p[col]).mean() * 100)
        return out
