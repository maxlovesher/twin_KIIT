### How GlucoTwin works

GlucoTwin keeps a **virtual replica of each patient** in sync with their data and uses it to look two hours
ahead.

**1. Two data streams are fused**
- **Static:** the electronic health record. This covers demographics, diagnoses, labs (HbA1c, FPG, lipids,
  eGFR), medications and genetic risk markers.
- **Dynamic:** wearables every 5 minutes. This covers CGM glucose, steps, heart rate, HRV, sleep stages and
  meal logs.

**2. A personalised physiological twin**

A glucose–insulin model (an extended Bergman minimal model) describes how *this* patient absorbs carbohydrates,
secretes and responds to insulin, reacts to exercise and medication, and how their glucose drifts at dawn.

- The **health record sets the starting point.** Body weight, fasting glucose, BMI, diabetes duration and
  medications give the initial values.
- During the **onboarding week**, the twin learns the patient's own insulin sensitivity, β-cell response, basal
  glucose and dawn effect.
- After that, every new CGM reading keeps the twin's internal state (glucose, insulin action, carbohydrates
  still in the gut) up to date.

**3. Machine learning on top of the twin**

About 100 features feed a gradient-boosted model that outputs a calibrated **probability of a glucose spike in
the next 2 hours**. The features include:
- the glucose trend;
- the patient's own routine (what happened at this hour on previous days);
- meals, activity, sleep and stress signals from HR/HRV;
- the twin's forecast, and how far reality is drifting from it.

Separate models forecast glucose at +30, +60 and +120 minutes.

**What counts as a spike?** Glucose goes above **180 mg/dL** *and* rises at least **40 mg/dL** above its
current level within 2 hours.

**4. Explanations and what-if**
- **Explanations.** Each risk score is decomposed into data-stream contributions, shown as plain-language
  "drivers".
- **What-if.** The twin can be run forward from the current moment to test a meal, a carbohydrate swap, a
  post-meal walk or a short night's sleep.
- **Suggestions.** When a short walk would meaningfully lower the predicted peak, the dashboard says so.

**5. How it was validated**
- Data comes from a synthetic cohort of 120 patients over 28 days (sandbox rules: no real patient data).
- Models were trained on 72 patients, tuned on 24 and **tested on 24 patients they never saw**.
- Alerts are scored as real notifications: how many spikes were warned about in advance, how early, and how
  many alerts were false.

> Research prototype on synthetic data. Not a medical device; all outputs are decision support for clinicians.
