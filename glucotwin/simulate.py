"""Synthetic T2D cohort: EHR + lifestyle + ground-truth physiology + wearable sensors.

Generative (causal) order, so every data stream is mutually consistent:

    demographics, genetics, duration, meds ──► physiology parameters
    habits (meals, walks, sleep, stress) ──► minute-level inputs
    physiology × inputs ──► true plasma glucose (numba ODE, 1-min resolution)
    true glucose ──► CGM sensor (lag, noise, calibration bias, dropouts)
    true glucose ──► EHR lab values (HbA1c, FPG) recorded before monitoring
    activity/sleep/stress/glucose ──► steps, heart rate, HRV, sleep stages

Hidden drivers the downstream model must *infer*: unlogged meals, stress episodes
(visible only as HR↑/HRV↓) and sleep debt (next-day insulin resistance).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import physiology as ph

CITIES = ["Bengaluru", "Chennai", "Hyderabad", "Mumbai", "Delhi", "Kolkata", "Pune", "Kochi",
          "Ahmedabad", "Lucknow"]
MIN_PER_DAY = 1440


# --------------------------------------------------------------------------------------
# 1. Patients: demographics, genetics, physiology, medications, habits
# --------------------------------------------------------------------------------------
def make_patients(n: int, rng: np.random.Generator):
    male = rng.random(n) < 0.55
    age = np.clip(rng.normal(52, 10, n), 28, 78).round()
    height = np.where(male, rng.normal(168, 6, n), rng.normal(155, 6, n))
    bmi = np.clip(rng.normal(27.0, 4.2, n), 19, 42)
    weight = bmi * (height / 100) ** 2
    waist = 2.2 * bmi + 30 + 3 * male + rng.normal(0, 5, n)
    duration = np.clip(rng.gamma(2.0, 3.5, n), 0.3, age - 25)
    family_history = rng.random(n) < 0.6
    tcf7l2 = rng.binomial(2, 0.30, n)        # rs7903146 T allele
    kcnq1 = rng.binomial(2, 0.60, n)         # rs2237892 C allele (common in South Asians)
    prs_raw = 0.35 * tcf7l2 + 0.2 * kcnq1 + 0.3 * family_history + rng.normal(0, 0.8, n)
    prs = (prs_raw - prs_raw.mean()) / prs_raw.std()
    staple = rng.choice(["rice", "wheat", "mixed"], n, p=[0.45, 0.30, 0.25])
    vegetarian = rng.random(n) < 0.4
    city = rng.choice(CITIES, n)

    # Untreated physiology
    si = 2.4e-4 * np.exp(-0.07 * (bmi - 25)) * np.exp(-0.12 * prs) * rng.lognormal(0, 0.3, n)
    beta = (0.075 * np.exp(-0.045 * duration) * np.exp(-0.12 * tcf7l2) * np.exp(-0.08 * kcnq1)
            * rng.lognormal(0, 0.3, n))
    gb = np.clip(122 + 4.0 * duration + 7 * tcf7l2 + 1.0 * (bmi - 25) + rng.normal(0, 20, n), 100, 240)

    # Medications (more intensive therapy for more advanced disease)
    severity = (gb - 100) / 100 + duration / 20
    metformin = rng.random(n) < 0.9
    su = rng.random(n) < np.clip(0.15 + 0.5 * severity, 0, 0.7)
    dpp4 = rng.random(n) < 0.35
    sglt2 = rng.random(n) < 0.25
    insulin = (duration > 8) & (gb > 155) & (rng.random(n) < 0.6)
    htn = rng.random(n) < np.clip(0.25 + 0.01 * (age - 40) + 0.03 * (bmi - 25), 0.1, 0.85)
    dyslip = rng.random(n) < 0.55
    statin = dyslip & (rng.random(n) < 0.8)
    betablocker = htn & (rng.random(n) < 0.2)
    cad = rng.random(n) < np.clip(0.04 + 0.004 * (age - 40), 0.02, 0.25)
    neuropathy = rng.random(n) < np.clip(0.03 * duration, 0, 0.5)

    p = ph.default_params(n)
    gastric = rng.lognormal(0, 0.15, n) * np.where(duration > 12, 0.85, 1.0)
    p[:, ph.P_GB] = gb * np.where(metformin, 0.88, 1.0)
    p[:, ph.P_SI] = si * np.where(metformin, 1.1, 1.0)
    p[:, ph.P_SG] = 0.014 * rng.lognormal(0, 0.15, n)
    p[:, ph.P_N] = 0.14 * rng.lognormal(0, 0.1, n)
    p[:, ph.P_BETA] = beta * np.where(dpp4, 1.25, 1.0) * np.where(su, 1.2, 1.0)
    p[:, ph.P_S0] = np.where(su, rng.uniform(0.35, 0.7, n), 0.0)
    p[:, ph.P_IEXO] = np.where(insulin, rng.uniform(4, 9, n), 0.0)
    p[:, ph.P_KF] = 0.035 * gastric
    p[:, ph.P_KS] = 0.014 * gastric
    p[:, ph.P_VG] = 1.9 * weight
    p[:, ph.P_DAWN] = rng.uniform(0.05, 0.35, n)
    p[:, ph.P_KEX] = 0.007 * rng.lognormal(0, 0.2, n)
    p[:, ph.P_KREN] = np.where(sglt2, 0.005, 0.0)

    ids = [f"GT{i:04d}" for i in range(1, n + 1)]
    ehr = pd.DataFrame({
        "patient_id": ids, "age": age.astype(int), "sex": np.where(male, "M", "F"),
        "city": city, "height_cm": height.round(1), "weight_kg": weight.round(1),
        "bmi": bmi.round(1), "waist_cm": waist.round(0),
        "diabetes_duration_y": duration.round(1), "family_history_t2d": family_history.astype(int),
        "tcf7l2_risk_alleles": tcf7l2, "kcnq1_risk_alleles": kcnq1, "polygenic_risk_z": prs.round(2),
        "diet_staple": staple, "vegetarian": vegetarian.astype(int),
        "hypertension": htn.astype(int), "dyslipidemia": dyslip.astype(int),
        "coronary_artery_disease": cad.astype(int), "neuropathy": neuropathy.astype(int),
        "med_metformin": metformin.astype(int), "med_sulfonylurea": su.astype(int),
        "med_dpp4i": dpp4.astype(int), "med_sglt2i": sglt2.astype(int),
        "med_basal_insulin": insulin.astype(int), "med_statin": statin.astype(int),
        "med_betablocker": betablocker.astype(int),
    })

    fast_base = np.select([staple == "rice", staple == "wheat"], [0.66, 0.55], 0.60)
    traits = pd.DataFrame({
        "patient_id": ids,
        "breakfast_min": np.clip(rng.normal(8.0 * 60, 35, n), 6.5 * 60, 10 * 60),
        "lunch_min": np.clip(rng.normal(13.5 * 60, 35, n), 12 * 60, 15 * 60),
        "dinner_min": np.clip(rng.normal(21.0 * 60, 40, n), 19.5 * 60, 22.5 * 60),
        "carb_scale": rng.lognormal(0, 0.15, n) * np.where(staple == "rice", 1.1, 1.0),
        "fast_frac": fast_base,
        "snack_prob": rng.uniform(0.3, 0.9, n),
        "morning_walk_prob": rng.uniform(0, 0.7, n),
        "post_dinner_walk_prob": rng.uniform(0, 0.5, n),
        "activity_level": rng.lognormal(0, 0.35, n),
        "bedtime_min": np.clip(rng.normal(23.25 * 60, 35, n), 22 * 60, 24.75 * 60),
        "sleep_h": np.clip(rng.normal(6.8, 0.6, n), 5.3, 8.2),
        "poor_night_prob": rng.uniform(0.08, 0.35, n),
        "log_compliance": rng.uniform(0.3, 0.95, n),
        "stress_prob": rng.uniform(0.05, 0.35, n),
        "rhr": rng.normal(74, 7, n) - 6 * betablocker,
        "hrv_base": np.clip(55 - 0.5 * (age - 30) - 0.8 * duration + rng.normal(0, 6, n), 12, 70),
    })
    return ehr, traits, p


# --------------------------------------------------------------------------------------
# 2. Lifestyle: minute-level meals, activity, sleep, stress for one patient
# --------------------------------------------------------------------------------------
def _sleep_stages(n_min: int, rng: np.random.Generator, poor: bool) -> tuple[np.ndarray, int]:
    """Hypnogram for one night: 0 awake, 1 light, 2 deep, 3 REM. Returns (stages, n_awakenings)."""
    stages = np.ones(n_min, dtype=np.int8)
    pos, cycle = 0, 0
    while pos < n_min:
        clen = int(rng.normal(90, 10))
        deep = int(clen * max(0.05, 0.35 - 0.09 * cycle) * (0.7 if poor else 1.0))
        rem = int(clen * min(0.35, 0.10 + 0.06 * cycle))
        light1 = int(0.3 * (clen - deep - rem))
        seg = np.concatenate([np.full(light1, 1), np.full(deep, 2),
                              np.full(clen - light1 - deep - rem, 1), np.full(rem, 3)])
        end = min(n_min, pos + len(seg))
        stages[pos:end] = seg[: end - pos]
        pos, cycle = end, cycle + 1
    n_awak = rng.poisson(1.5 + (4.0 if poor else 0.0))
    for _ in range(n_awak):
        a = rng.integers(0, n_min)
        stages[a:a + rng.integers(3, 16)] = 0
    stages[:rng.integers(5, 25)] = 0  # sleep latency
    return stages, n_awak


def make_lifestyle(tr: pd.Series, n_days: int, rng: np.random.Generator) -> dict:
    T = n_days * MIN_PER_DAY
    cf = np.zeros(T); cs = np.zeros(T); ex = np.zeros(T)
    in_sleep = np.zeros(T, dtype=np.int8); stage = np.zeros(T, dtype=np.int8)
    stress = np.zeros(T, dtype=np.int8); si_mult = np.ones(T); egp = np.zeros(T)
    debt_arr = np.zeros(T)
    carbs_true_at = np.zeros(T); carbs_logged_at = np.zeros(T)

    # Sleep: night k starts on the evening of day k-1 (k=0 is the night before the study)
    nights = []  # (onset, wake, debt, frag)
    for k in range(n_days + 1):
        weekend_night = ((k - 1) % 7) in (4, 5)  # Fri & Sat nights
        onset = (k - 1) * MIN_PER_DAY + tr.bedtime_min + rng.normal(0, 25) + (35 if weekend_night else 0)
        poor = rng.random() < tr.poor_night_prob
        dur_h = tr.sleep_h + rng.normal(0, 0.4) - (rng.uniform(1.0, 2.2) if poor else 0.0)
        dur_h = float(np.clip(dur_h, 3.5, 9.5))
        o, w = int(onset), int(onset + dur_h * 60)
        st, n_awak = _sleep_stages(w - o, rng, poor)
        lo, hi = max(o, 0), min(w, T)
        if hi > lo:
            in_sleep[lo:hi] = 1
            stage[lo:hi] = st[lo - o: hi - o]
        asleep_h = (st > 0).sum() / 60.0
        nights.append((o, w, float(np.clip((7.0 - asleep_h) / 2.5, 0, 1)), min(n_awak / 8.0, 1.0)))

    # Next-day insulin resistance after poor sleep (from wake until the next sleep onset)
    for k, (_, w, debt, frag) in enumerate(nights):
        nxt = nights[k + 1][0] if k + 1 < len(nights) else T
        lo, hi = max(w, 0), min(max(nxt, w), T)
        if hi > lo:
            si_mult[lo:hi] *= 1.0 - 0.22 * debt - 0.06 * frag
            debt_arr[lo:hi] = debt

    def add_meal(t0: float, grams: float, fast: float):
        t0 = int(np.clip(t0, 0, T - 16))
        cf[t0:t0 + 15] += grams * fast / 15
        cs[t0:t0 + 15] += grams * (1 - fast) / 15
        carbs_true_at[t0] += grams
        if rng.random() < tr.log_compliance:
            carbs_logged_at[t0] += max(5.0, round(grams * rng.lognormal(0, 0.25) / 5) * 5)

    def add_bout(t0: float, dur: int, inten: float):
        t0 = int(np.clip(t0, 0, T - 1))
        ex[t0:t0 + dur] = np.maximum(ex[t0:t0 + dur], inten)

    def frac() -> float:
        return float(np.clip(tr.fast_frac + rng.normal(0, 0.08), 0.3, 0.9))

    for d in range(n_days):
        base = d * MIN_PER_DAY
        weekend = (d % 7) >= 5
        wake = nights[d][1]
        bed = nights[d + 1][0] - 60  # movement stops ~1 h before sleep onset
        wk = 1.15 if weekend else 1.0

        if rng.random() > 0.1:
            t = max(base + tr.breakfast_min + rng.normal(0, 25) + (45 if weekend else 0), wake + 15)
            add_meal(t, 45 * tr.carb_scale * rng.lognormal(0, 0.2) * wk, frac())
        add_meal(base + tr.lunch_min + rng.normal(0, 30), 70 * tr.carb_scale * rng.lognormal(0, 0.2) * wk, frac())
        if rng.random() < tr.snack_prob:
            add_meal(base + 17 * 60 + rng.normal(0, 40), 25 * rng.lognormal(0, 0.25), 0.8)
        dinner = base + tr.dinner_min + rng.normal(0, 30) + (30 if weekend else 0)
        festive = 1.5 if rng.random() < 0.05 else 1.0
        add_meal(dinner, 68 * tr.carb_scale * rng.lognormal(0, 0.2) * wk * festive, frac())
        if rng.random() < 0.1:
            add_meal(dinner + rng.uniform(60, 120), 30 * rng.lognormal(0, 0.2), 0.85)

        if rng.random() < tr.morning_walk_prob:
            add_bout(wake + rng.uniform(10, 60), int(rng.uniform(20, 45)), rng.uniform(0.65, 0.9))
        if rng.random() < tr.post_dinner_walk_prob:
            add_bout(dinner + rng.uniform(10, 40), int(rng.uniform(10, 25)), rng.uniform(0.5, 0.75))
        awake_h = max((bed - wake) / 60.0, 8.0)  # incidental movement while awake
        for _ in range(rng.poisson(2.0 * tr.activity_level * awake_h)):
            add_bout(wake + rng.uniform(0, awake_h * 60), int(rng.geometric(0.25)), rng.uniform(0.15, 0.5))

        if rng.random() < tr.stress_prob:
            s0 = int(base + rng.uniform(9, 18) * 60)
            s1 = min(T, s0 + int(rng.uniform(90, 300)))
            stress[s0:s1] = 1
            si_mult[s0:s1] *= 0.8
            egp[s0:s1] += 0.25

    ex[in_sleep == 1] = 0.0
    return dict(cf=cf, cs=cs, ex=ex, in_sleep=in_sleep, stage=stage, stress=stress,
                si_mult=si_mult, egp=egp, debt=debt_arr,
                carbs_true=carbs_true_at, carbs_logged=carbs_logged_at)


# --------------------------------------------------------------------------------------
# 3. Sensors
# --------------------------------------------------------------------------------------
def _bin(x: np.ndarray, how: str) -> np.ndarray:
    b = x.reshape(-1, C.STEP_MIN)
    return {"sum": b.sum(1), "mean": b.mean(1), "mid": b[:, C.STEP_MIN // 2]}[how]


def make_sensors(G: np.ndarray, life: dict, tr: pd.Series, rng: np.random.Generator) -> dict:
    T = G.shape[0]
    n5 = T // C.STEP_MIN
    ex, sl, stg, stress, debt = life["ex"], life["in_sleep"], life["stage"], life["stress"], life["debt"]

    steps = np.where(ex > 0, rng.poisson(np.maximum(ex, 0) * 110), 0)
    hr = (tr.rhr + 50 * ex - 7 * sl - 4 * (stg == 2) + 7 * stress + 5 * debt
          + 0.03 * np.maximum(G - 140, 0) + rng.normal(0, 2, T))

    ex5 = _bin(ex, "mean"); sl5 = _bin(sl.astype(float), "mean") >= 0.5
    stg5 = _bin(stg, "mid"); stress5 = _bin(stress.astype(float), "mean"); debt5 = _bin(debt, "mean")
    G5 = _bin(G, "mean")
    stage_mult = np.select([stg5 == 2, stg5 == 3, sl5], [1.5, 1.1, 1.3], 1.0)
    hrv = (tr.hrv_base * stage_mult * (1 - 0.3 * stress5) * (1 - 0.6 * np.minimum(ex5, 1))
           * (1 - 0.15 * debt5) * (1 - 0.0015 * np.maximum(G5 - 140, 0)) * rng.lognormal(0, 0.15, n5))
    day_missing = (~sl5) & ((rng.random(n5) < 0.6) | (ex5 > 0.25))
    hrv = np.where(day_missing, np.nan, hrv)

    # CGM: interstitial lag (tau 8 min), two 14-day sensors, AR(1) noise, dropouts
    isf = np.empty(T); acc = G[0]
    for t in range(T):
        acc += (G[t] - acc) / 8.0
        isf[t] = acc
    cgm = isf[C.STEP_MIN - 1::C.STEP_MIN].copy()
    sensor_len = 14 * C.STEPS_PER_DAY
    for s0 in range(0, n5, sensor_len):
        s1 = min(n5, s0 + sensor_len)
        cgm[s0:s1] = cgm[s0:s1] * rng.normal(1, 0.06) + rng.normal(0, 6)
        if s0 > 0:  # 1 h warm-up after the sensor change at 10:00
            cgm[s0 + 10 * 12: s0 + 11 * 12] = np.nan
    e = np.zeros(n5)
    for t in range(1, n5):
        e[t] = 0.7 * e[t - 1] + rng.normal(0, 5.0)
    cgm = cgm + e
    for d in range(n5 // C.STEPS_PER_DAY):
        if rng.random() < 0.35:
            g0 = d * C.STEPS_PER_DAY + rng.integers(0, C.STEPS_PER_DAY)
            cgm[g0:g0 + int(rng.uniform(4, 24))] = np.nan
    cgm = np.round(np.clip(cgm, 40, 400))

    return dict(
        cgm=cgm.astype(np.float32), steps=_bin(steps, "sum").astype(np.int32),
        hr=_bin(hr, "mean").round(1).astype(np.float32), hrv_rmssd=hrv.round(1).astype(np.float32),
        in_sleep=sl5.astype(np.int8), sleep_stage=np.where(sl5, stg5, 0).astype(np.int8),
        carbs_logged=_bin(life["carbs_logged"], "sum").astype(np.float32),
        glucose_true=G5.astype(np.float32), carbs_true=_bin(life["carbs_true"], "sum").astype(np.float32),
        stress=(stress5 >= 0.5).astype(np.int8), si_mult=_bin(life["si_mult"], "mean").astype(np.float32),
    )


# --------------------------------------------------------------------------------------
# 4. Labs recorded in the EHR (derived from the simulated physiology)
# --------------------------------------------------------------------------------------
def add_labs(ehr: pd.DataFrame, G_all: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    n = len(ehr)
    mean_g = G_all.mean(axis=1)
    days = G_all.shape[1] // MIN_PER_DAY
    fasting_idx = np.concatenate([np.arange(d * MIN_PER_DAY + 405, d * MIN_PER_DAY + 420) for d in range(days)])
    ehr = ehr.copy()
    ehr["hba1c_pct"] = ((mean_g + 46.7) / 28.7 + rng.normal(0, 0.2, n)).round(1)
    ehr["fpg_mg_dl"] = (G_all[:, fasting_idx].mean(axis=1) + rng.normal(0, 8, n)).round(0)
    ehr["ldl_mg_dl"] = (rng.normal(115, 30, n) - 35 * ehr.med_statin).clip(45, 220).round(0)
    ehr["hdl_mg_dl"] = (np.where(ehr.sex == "M", 40, 47) + rng.normal(0, 7, n)).clip(22, 80).round(0)
    ehr["tg_mg_dl"] = (110 + 6 * (ehr.bmi - 25) + 12 * (ehr.hba1c_pct - 7) + rng.normal(0, 35, n)).clip(60, 450).round(0)
    ehr["egfr"] = (105 - 0.7 * (ehr.age - 30) - 0.8 * ehr.diabetes_duration_y + rng.normal(0, 10, n)).clip(25, 125).round(0)
    ehr["ckd"] = (ehr.egfr < 60).astype(int)
    ehr["sbp_mmhg"] = (122 + 14 * ehr.hypertension + 0.3 * (ehr.age - 50) + rng.normal(0, 9, n)).round(0)
    ehr["dbp_mmhg"] = (78 + 7 * ehr.hypertension + rng.normal(0, 6, n)).round(0)
    start = pd.Timestamp(C.SIM_START)
    ehr["lab_date"] = [(start - pd.Timedelta(days=int(d))).strftime("%Y-%m-%d") for d in rng.integers(5, 60, n)]
    ehr["diagnosis_year"] = (start.year - ehr.diabetes_duration_y).astype(int)
    return ehr


# --------------------------------------------------------------------------------------
# 5. Orchestration
# --------------------------------------------------------------------------------------
def generate_cohort(n: int = C.N_PATIENTS, days: int = C.N_DAYS, seed: int = C.SEED, chunk: int = 40):
    """Returns (ehr, wearables, ground_truth, true_params) DataFrames."""
    rng = np.random.default_rng(seed)
    ehr, traits, params = make_patients(n, rng)
    T = days * MIN_PER_DAY
    hour = (np.arange(T) % MIN_PER_DAY) / 60.0
    ts = pd.date_range(C.SIM_START, periods=T // C.STEP_MIN, freq=f"{C.STEP_MIN}min")

    wear_frames, truth_frames, G_minute = [], [], np.empty((n, T), dtype=np.float32)
    for c0 in range(0, n, chunk):
        idx = list(range(c0, min(n, c0 + chunk)))
        lives = [make_lifestyle(traits.iloc[i], days, rng) for i in idx]

        def stack(k):
            return np.stack([lf[k] for lf in lives])

        G, _ = ph.simulate_truth(params[idx], stack("cf"), stack("cs"), stack("ex"),
                                 stack("si_mult"), stack("egp"), hour, 1.0, MIN_PER_DAY)
        for j, i in enumerate(idx):
            G_minute[i] = G[j]
            s = make_sensors(G[j].astype(np.float64), lives[j], traits.iloc[i], rng)
            pid = ehr.patient_id.iloc[i]
            wear_frames.append(pd.DataFrame({
                "patient_id": pid, "timestamp": ts, "cgm": s["cgm"], "steps": s["steps"],
                "hr": s["hr"], "hrv_rmssd": s["hrv_rmssd"], "in_sleep": s["in_sleep"],
                "sleep_stage": s["sleep_stage"], "carbs_logged": s["carbs_logged"]}))
            truth_frames.append(pd.DataFrame({
                "patient_id": pid, "timestamp": ts, "glucose_true": s["glucose_true"],
                "carbs_true": s["carbs_true"], "stress": s["stress"], "si_mult": s["si_mult"]}))

    ehr = add_labs(ehr, G_minute, rng)
    wear = pd.concat(wear_frames, ignore_index=True)
    truth = pd.concat(truth_frames, ignore_index=True)
    true_params = pd.concat([traits, pd.DataFrame(params, columns=[f"true_{k}" for k in ph.PARAM_NAMES])], axis=1)
    return ehr, wear, truth, true_params


def write_cohort(out_dir=C.RAW_DIR, **kwargs) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ehr, wear, truth, tp = generate_cohort(**kwargs)
    ehr.to_csv(out_dir / "ehr.csv", index=False)
    wear.to_parquet(out_dir / "wearables.parquet", index=False)
    truth.to_parquet(out_dir / "ground_truth.parquet", index=False)
    tp.to_csv(out_dir / "true_params.csv", index=False)
