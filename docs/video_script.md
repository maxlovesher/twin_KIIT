# Demo video script (target 22–24 minutes)

**Requirement:** a video of at least 20 minutes, uploaded as an **unlisted YouTube video** (or recorded as a
PowerPoint presentation).

**Setup before recording:**
- Run `python scripts/build_all.py` and `python scripts/make_docs.py`.
- Open `docs/GlucoTwin_presentation.pptx`.
- Start `streamlit run dashboard/app.py`.
- Start `uvicorn glucotwin.api:app --port 8000`.
- Keep a terminal ready.

**Tips:**
- Record at 1080p with system zoom at 125%.
- Have each speaker introduce themselves once.
- Pause on every chart long enough for a judge to read it.

| # | Time | Section | Who | Show | Say (key points) |
|---|---|---|---|---|---|
| 1 | 0:00–1:30 | Intro | Lead | Title slide | Team, college, one-sentence pitch: "GlucoTwin warns a person with Type 2 Diabetes about a glucose spike up to 2 hours before it happens, explains why, and lets the doctor test what would prevent it." |
| 2 | 1:30–4:00 | Problem | Member 2 | Problem slide | 101M Indians with diabetes. HbA1c every 3 months misses daily spikes. Rice/roti-heavy meals cause sharp post-meal excursions. CGMs only look backward. Doctors can't test "what if". Define the adverse event (>180 mg/dL and ≥40 mg/dL rise within 2 h). |
| 3 | 4:00–6:30 | Solution & architecture | Lead | Architecture slide / `docs/architecture.pdf` | The three layers: mechanistic twin, causal fusion, ML + explanations. Static vs dynamic streams. Why hybrid: physiology gives what-if and interpretability; ML learns what physiology can't see (unlogged meals, routine, stress). |
| 4 | 6:30–9:00 | Data & sandbox rules | Member 3 | Data slide, then `data/raw/ehr.csv` and a few rows of `wearables.parquet` | DPDP Act/HIPAA → synthetic data only. Causal generator: genetics + meds → physiology → glucose → CGM + labs. Show HbA1c-vs-GMI agreement and TIR distribution. Hidden drivers: unlogged meals, stress, sleep debt. |
| 5 | 9:00–11:30 | The mechanistic twin | Member 2 | Twin slide; open `glucotwin/physiology.py` and `twin.py` briefly | Equations in plain words. EHR → priors. Onboarding-week calibration. CGM assimilation every 5 min. Parameter recovery vs EHR-only priors (read the numbers off the twin-recovery figure / README). Be honest: the dawn parameter is not identifiable in a week. |
| 6 | 11:30–13:00 | Features & model | Member 3 | `glucotwin/features.py` FEATURE_GROUPS | 10 feature groups. The twin-residual feature as an unlogged-meal detector. Leakage-proof (run `pytest -q` live). Gradient boosting + isotonic calibration; forecasters. |
| 7 | 13:00–16:30 | **Live dashboard demo** | Lead | Streamlit | **Ward overview:** risk-sorted list, active alerts, select a patient. **Virtual patient:** CGM + twin forecast (dotted) + hybrid forecast with band; risk panel; steps/HR/HRV/sleep; drivers; twin-tested walk suggestion; EHR card; AGP; phenotype percentiles. Move the clock with +5m/+1h to show an alert appear **before** the spike, then turn on **Live replay**. |
| 8 | 16:30–18:30 | What-if simulator | Member 2 | What-if tab | Same patient. "White rice + sambar" vs "Ragi mudde". Add a 15-min walk 20 min after eating. Show a short-sleep effect. Read out the peak and minutes-above-180 changes. Stress that this is counselling, not dosing. |
| 9 | 18:30–20:30 | Results | Member 3 | Results, ablation, cold-start and forecast slides | Test = 24 unseen patients. AUROC, % spikes warned, median lead time, false alerts/day. Ablations: CGM-only vs all wearables vs fusion vs GlucoTwin. EHR matters most at cold start; the twin cuts false alarms and halves forecast error. |
| 10 | 20:30–21:45 | Real-time engineering | Lead | Terminal: `python scripts/replay_stream.py --patient GT0007 --days 2`, then `/docs` (Swagger) | Device-gateway-style streaming, assessment JSON with drivers and suggestion. Online = offline features (parity test). |
| 11 | 21:45–23:00 | Limitations, impact, next steps | Member 2 | Limitations slide | Synthetic data. Hypo not yet modelled. Validation on real CGM cohorts. ABDM/FHIR integration. Clinician pilot. CDSCO SaMD pathway. |
| 12 | 23:00–23:30 | Close | Lead | Thank-you slide | Repo link, thanks to Happiest Health. |

### Moments to rehearse

1. **Alert before the spike.**
   - The dashboard opens on a held-out test patient a few minutes before a correctly predicted spike.
   - Click **+5m** a few times: the alert fires while glucose is still in range, then glucose crosses 180 later.
2. **Unlogged-meal detection.**
   - Find a moment where the driver reads "glucose X mg/dL above twin expectation: possible unlogged meal".
3. **What-if contrast.**
   - Show rice + sambar with no walk against ragi mudde with a 20-minute walk. Read out the peak difference.
