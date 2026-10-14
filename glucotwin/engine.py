"""Real-time digital-twin engine: ingest streaming wearable data, keep each twin in sync,
and return risk, forecasts, explanations and what-if simulations on demand."""
from __future__ import annotations

import threading
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from . import modeling as M
from .twin import PatientTwin, calibrate_patient, prior_params

STREAM_COLUMNS = ["timestamp", "cgm", "steps", "hr", "hrv_rmssd", "in_sleep", "sleep_stage", "carbs_logged"]
_DEFAULTS = {"cgm": np.nan, "steps": 0, "hr": np.nan, "hrv_rmssd": np.nan, "in_sleep": 0,
             "sleep_stage": 0, "carbs_logged": 0.0}


def _json_safe(obj):
    """Replace NaN/inf (e.g. forecasts during a CGM dropout) with None so the result is valid JSON."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


class TwinEngine:
    def __init__(self, models_dir: Path = C.MODELS_DIR, raw_dir: Path = C.RAW_DIR):
        self.bundle: M.Bundle = joblib.load(models_dir / "glucotwin_bundle.joblib")
        ehr = pd.read_csv(raw_dir / "ehr.csv")
        self.ehr = ehr.set_index("patient_id")
        self.static = F.static_features(ehr)
        params = pd.read_csv(models_dir / "twin_params.csv")
        self.cohort_params = params
        self.twins = {r.patient_id: PatientTwin.from_row(r) for _, r in params.iterrows()}
        self.buffers: dict[str, pd.DataFrame] = {}
        self.states: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

        self.calibrated = {pid: True for pid in self.twins}

    # ------------------------------------------------------------------------------
    def patients(self) -> list[str]:
        return list(self.twins)

    def register(self, ehr: dict) -> dict:
        """Onboard a new patient from their health record. The twin starts from EHR-derived priors
        (cold start) and can be personalised later with ``calibrate`` once data has streamed in."""
        from .interop import EHR_COLUMNS

        pid = str(ehr["patient_id"])
        row = pd.Series({c: ehr.get(c, np.nan) for c in EHR_COLUMNS})
        with self._lock, warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # enlarging with NaN fields is intended
            self.ehr.loc[pid] = row.drop("patient_id")
            self.static.loc[pid] = F.static_features(pd.DataFrame([row])).loc[pid]
            self.twins[pid] = PatientTwin(pid, prior_params(row))
            self.calibrated[pid] = False
            self.buffers.pop(pid, None)
            self.states.pop(pid, None)
        return {"patient_id": pid, "twin": "EHR priors (uncalibrated)"}

    def calibrate(self, pid: str, min_days: float = 2.0) -> dict:
        """Personalise the twin (insulin sensitivity, beta-cell response, basal glucose, dawn effect) from the
        buffered stream, then re-track the buffer with the new parameters."""
        buf = self.buffers.get(pid)
        if buf is None or len(buf) < min_days * C.STEPS_PER_DAY:
            have = 0 if buf is None else len(buf) / C.STEPS_PER_DAY
            raise ValueError(f"need at least {min_days:g} days of data to calibrate, have {have:.1f}")
        days = len(buf) / C.STEPS_PER_DAY
        ehr_row = pd.Series({"patient_id": pid, **self.ehr.loc[pid].to_dict()})
        fit = calibrate_patient(ehr_row, buf, days=int(days))
        twin = PatientTwin.from_row(pd.Series(fit))
        with self._lock:
            self.twins[pid] = twin
            self.states[pid] = twin.track(buf)
            self.calibrated[pid] = True
        return {"patient_id": pid, "days_used": int(days), "phenotype": twin.phenotype(self.cohort_params),
                "fit_rmse_30": fit["fit_rmse_30"], "fit_rmse_60": fit["fit_rmse_60"]}

    def fhir(self, pid: str) -> dict:
        """FHIR R4 Bundle of the patient's record plus the latest RiskAssessment (if streaming)."""
        from .interop import fhir_bundle

        assessment = self.assess(pid) if pid in self.buffers else None
        return fhir_bundle({"patient_id": pid, **self.ehr.loc[pid].to_dict()}, assessment)

    def reset(self, pid: str) -> None:
        with self._lock:
            self.buffers.pop(pid, None)
            self.states.pop(pid, None)

    def ingest(self, pid: str, records) -> int:
        """Append 5-min records (list of dicts or DataFrame) and advance the twin. Returns buffer size."""
        if pid not in self.twins:
            raise KeyError(pid)
        new = pd.DataFrame(records).copy()
        for col, default in _DEFAULTS.items():
            if col not in new:
                new[col] = default
        new["timestamp"] = pd.to_datetime(new["timestamp"])
        new["patient_id"] = pid
        new = new.sort_values("timestamp")[["patient_id"] + STREAM_COLUMNS].astype(
            {"cgm": float, "steps": float, "hr": float, "hrv_rmssd": float, "carbs_logged": float,
             "in_sleep": int, "sleep_stage": int})
        twin = self.twins[pid]
        with self._lock:
            buf = self.buffers.get(pid)
            prev = self.states.get(pid)
            if buf is not None and len(buf):
                new = new[new.timestamp > buf.timestamp.iloc[-1]]
                if new.empty:
                    return len(buf)
            s0 = prev[-1] if prev is not None else None
            st = twin.track(new, s0=s0)
            buf = new if buf is None else pd.concat([buf, new], ignore_index=True)
            states = st if prev is None else np.vstack([prev, st])
            keep = F.ENGINE_BUFFER_STEPS
            self.buffers[pid] = buf.iloc[-keep:].reset_index(drop=True)
            self.states[pid] = states[-keep:]
            return len(self.buffers[pid])

    # ------------------------------------------------------------------------------
    def feature_row(self, pid: str) -> pd.DataFrame:
        buf, states = self.buffers[pid], self.states[pid]
        twin = self.twins[pid]
        f = F.dynamic_features(buf, twin.forecast_columns(buf, states))
        f = F.add_twin_static(f, twin)
        for k, v in self.static.loc[pid].items():
            f[k] = v
        return f[F.ALL_FEATURES].iloc[[-1]].astype(np.float32)

    def assess(self, pid: str, explain: bool = True) -> dict:
        if pid not in self.buffers:
            raise LookupError(f"no data ingested for {pid}")
        X = self.feature_row(pid)
        row = X.iloc[0]
        p = float(self.bundle.risk.predict_proba(X)[0])
        thr = self.bundle.risk.threshold
        cgm = None if pd.isna(row.cgm_now) else float(row.cgm_now)
        fc = self.bundle.forecast(X).iloc[0]
        out = {
            "patient_id": pid,
            "timestamp": self.buffers[pid].timestamp.iloc[-1].isoformat(),
            "cgm": cgm,
            "trend_mg_dl_per_min": None if pd.isna(row.cgm_roc) else float(row.cgm_roc),
            "spike_probability_2h": p,
            "alert_threshold": thr,
            "alert": bool(p >= thr and cgm is not None and cgm <= C.HYPER_THRESHOLD),
            "forecast": {str(h): {"glucose": float(fc[f"fc{h}"]),
                                  "low": float(fc[f"fc{h}"] + self.bundle.resid_q[h][0]),
                                  "high": float(fc[f"fc{h}"] + self.bundle.resid_q[h][1])}
                         for h in M.FORECAST_HORIZONS},
            "twin": {"g30": float(row.twin_g30), "g60": float(row.twin_g60), "g120": float(row.twin_g120),
                     "peak_2h": float(row.twin_max120), "gut_carbs_g": float(row.twin_gut_carbs)},
        }
        if explain:
            contrib = M.explain_rows(self.bundle, X).iloc[0]
            out["drivers"] = [{"group": g, "log_odds": c, "text": t} for g, c, t in M.narrate(row, contrib)]
            out["suggestion"] = self.suggest(pid)
        return _json_safe(out)

    # ------------------------------------------------------------------------------
    def _now(self, pid: str) -> tuple[np.ndarray, float]:
        ts = self.buffers[pid].timestamp.iloc[-1]
        return self.states[pid][-1], ts.hour + ts.minute / 60.0

    def what_if(self, pid: str, horizon_min: int = 360, **scenario) -> dict:
        state, hour = self._now(pid)
        twin = self.twins[pid]
        base = twin.what_if(state, hour, horizon_min=horizon_min)
        alt = twin.what_if(state, hour, horizon_min=horizon_min, **scenario)
        return {"minutes": list(range(1, horizon_min + 1)), "baseline": base.round(1).tolist(),
                "scenario": alt.round(1).tolist(), "baseline_peak": float(base.max()),
                "scenario_peak": float(alt.max()),
                "scenario_minutes_above_180": int((alt > C.HYPER_THRESHOLD).sum())}

    def suggest(self, pid: str) -> dict | None:
        """Twin-tested micro-intervention: what would a 15-min brisk walk now do to the 2-h peak?"""
        state, hour = self._now(pid)
        twin = self.twins[pid]
        if self.buffers[pid].in_sleep.iloc[-1] == 1:
            return None
        base = twin.what_if(state, hour, horizon_min=120)
        walk = twin.what_if(state, hour, horizon_min=120, walk_min=15, walk_delay_min=0, walk_intensity=0.75)
        drop = float(base.max() - walk.max())
        if drop < 5:
            return None
        return {"action": "15-minute brisk walk now", "peak_without": float(base.max()),
                "peak_with": float(walk.max()), "peak_reduction": drop}
