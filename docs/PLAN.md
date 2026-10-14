# GlucoTwin — Build Plan

**Challenge:** Happiest Health Digital Twin Challenge 2026 (Phase 1 deadline: 20 Oct 2026, 7 PM IST)

**Condition:** Type 2 Diabetes (T2D). India has about 100M+ adults with diabetes, and post-meal spikes driven by high-carb diets are a major driver of complications.

**Adverse event predicted:** a **hyperglycaemic excursion**. CGM glucose goes above 180 mg/dL **and** rises at least 40 mg/dL above its current level within the next **120 minutes**, while the patient is currently in range.

*Changed during the build:* hypoglycaemia was dropped as a secondary target because it is too rare in the cohort (time below range ≈0.02%). The plain "crosses 180" definition was also replaced. It counted sensor noise around 180 as spikes.

## Core idea: a *hybrid* digital twin

| Layer | What it is | Why |
|---|---|---|
| **Mechanistic twin** | A personalised glucose–insulin ODE (Bergman minimal model + gut absorption + exercise, dawn phenomenon, medication effects). Its parameters (insulin sensitivity, β-cell response, basal glucose, dawn amplitude) are calibrated per patient from their onboarding week. | Gives a physiologically meaningful, explainable "virtual patient" that supports **what-if simulation** (meal size, post-meal walk, sleep). |
| **ML layer** | Gradient-boosted model over fused features: static EHR + dynamic wearable streams (CGM, steps, HR, HRV, sleep) + the twin's own forecast. | Learns what the ODE cannot see: unlogged meals, stress, sleep debt, and personal routines. |
| **Clinician dashboard** | Streamlit ward view, virtual-patient view, explanation of drivers, what-if simulator, model performance. | Required conceptual UI showing how a doctor interacts with the twin. |

## Data (sandbox rules: synthetic only)

- **EHR (static):** synthetic Indian T2D cohort with demographics, BMI and waist, diabetes duration, HbA1c, FPG, lipids, eGFR, BP, comorbidities, medications, and genetic markers (TCF7L2 rs7903146 and KCNQ1 risk alleles, polygenic risk score). There is also a **Synthea CSV adapter**, so a real Synthea export can be dropped in.
- **Wearables (dynamic):** a 28-day, 5-minute resolution stream per patient. It includes CGM (sensor lag, noise, dropouts), steps, heart rate, HRV (RMSSD), sleep sessions and stages, and partial meal logs.
- **Causal generator:** demographics, genetics and meds → physiology → simulated glucose → lab values. As a result, HbA1c in the EHR is *consistent* with the simulated CGM, which is a validation check.
- Hidden drivers the model must infer from wearables: stress episodes (HR↑ HRV↓), poor sleep (next-day insulin resistance), unlogged meals.

## Pipeline

1. `simulate`: cohort of 120 patients × 28 days.
2. `calibrate`: fit each patient's twin on **days 1–7 only** (the onboarding week).
3. `features`: causal features only (no look-ahead), from day 8 onward.
4. `train`: patient-level split 60/20/20 (train/val/test). The model is HistGradientBoosting with isotonic calibration and a threshold picked on val.
5. `evaluate` on unseen test patients:
   - AUROC, AUPRC, Brier score
   - sensitivity and false alerts per day
   - **lead time**
   - **ablations**: static-only / dynamic-only / CGM-only / fusion / fusion + twin / rule baseline
   - glucose forecast RMSE at 30, 60 and 120 min
6. `explain`: feature-group occlusion attributions per prediction, plus global permutation importance.
7. `engine` + `api`: a real-time ingestion engine (FastAPI) and a replay streamer.
8. `dashboard`: Streamlit + Plotly.
9. `docs`: the README in the required format, architecture diagram (PDF), presentation (PPTX/PDF), 20-min video script, and MIT license.
