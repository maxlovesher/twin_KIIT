"""Project-wide constants and paths."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
DOCS_DIR = ROOT / "docs"

# Cohort / simulation
N_PATIENTS = 120
N_DAYS = 28
SIM_START = "2026-08-03 00:00"  # a Monday
STEP_MIN = 5                    # resolution of the wearable stream
STEPS_PER_DAY = 24 * 60 // STEP_MIN
SEED = 2026

# Digital-twin calibration window (the patient's "onboarding week")
CALIBRATION_DAYS = 7
# Rows used for modelling start on day 1 (day 0 is warm-up). Before the calibration week
# ends, the twin runs on EHR-derived priors ("cold start"); afterwards on calibrated params.
FEATURE_START_DAY = 1
HISTORY_BUCKETS = {"Days 1-2 (cold start)": (1, 2), "Days 3-6 (onboarding)": (3, 6),
                   "Day 7+ (calibrated twin)": (7, 10_000)}

# Prediction task
HORIZON_MIN = 120
HORIZON_STEPS = HORIZON_MIN // STEP_MIN
HYPER_THRESHOLD = 180.0   # mg/dL, ADA "time above range" cut-off
SPIKE_RISE = 40.0         # mg/dL rise above the current level that makes a crossing a real excursion
HYPO_THRESHOLD = 70.0     # mg/dL, ADA level-1 hypoglycaemia
TIR_LOW, TIR_HIGH = 70.0, 180.0

# Patient-level split
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)  # train / val / test


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, DOCS_DIR):
        d.mkdir(parents=True, exist_ok=True)
