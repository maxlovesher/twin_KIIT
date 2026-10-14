"""Training, evaluation, ablations, forecasting and explanations for the spike predictor."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve, precision_recall_curve

from . import config as C
from .features import FEATURE_GROUPS, H

DYNAMIC = ["Glucose trend (CGM)", "Personal routine", "Meals (logged)", "Activity",
           "Heart rate & HRV", "Sleep", "Time of day"]
ABLATIONS: dict[str, list[str]] = {
    "Static EHR + time of day": ["Clinical (EHR)", "Genetics", "Time of day"],
    "CGM only": ["Glucose trend (CGM)", "Time of day"],
    "Wearables only (dynamic)": DYNAMIC,
    "Fusion: EHR + wearables": DYNAMIC + ["Clinical (EHR)", "Genetics"],
    "GlucoTwin: fusion + digital twin": DYNAMIC + ["Clinical (EHR)", "Genetics", "Digital twin"],
}
FINAL = "GlucoTwin: fusion + digital twin"
FORECAST_HORIZONS = (30, 60, 120)


def cols_for(groups: list[str]) -> list[str]:
    return [c for g in groups for c in FEATURE_GROUPS[g]]


def split_patients(ids: list[str], seed: int = C.SEED) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    ids = list(ids)
    rng.shuffle(ids)
    n_tr = int(round(C.SPLIT_FRACTIONS[0] * len(ids)))
    n_va = int(round(C.SPLIT_FRACTIONS[1] * len(ids)))
    return {pid: ("train" if i < n_tr else "val" if i < n_tr + n_va else "test") for i, pid in enumerate(ids)}


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
def _clf() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=600, max_leaf_nodes=48, min_samples_leaf=100,
        l2_regularization=1.0, max_features=0.7, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=30, random_state=C.SEED)


def _reg() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=500, max_leaf_nodes=48, min_samples_leaf=100,
        l2_regularization=1.0, max_features=0.7, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=30, random_state=C.SEED)


@dataclass
class RiskModel:
    clf: HistGradientBoostingClassifier
    iso: IsotonicRegression
    cols: list[str]
    threshold: float = 0.5

    def raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(X[self.cols])[:, 1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.iso.predict(self.raw_proba(X))


@dataclass
class Bundle:
    risk: RiskModel
    regressors: dict = field(default_factory=dict)     # horizon -> regressor (predicts delta)
    resid_q: dict = field(default_factory=dict)        # horizon -> (q10, q90) of residuals
    reference: pd.Series | None = None                 # per-feature medians for occlusion

    def forecast(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for h, reg in self.regressors.items():
            out[f"fc{h}"] = X["cgm_now"].to_numpy() + reg.predict(X[self.risk.cols])
        return pd.DataFrame(out, index=X.index)


def fit_risk(train: pd.DataFrame, val: pd.DataFrame, cols: list[str]) -> RiskModel:
    tr = train[train.y.notna()]
    va = val[val.y.notna()]
    clf = _clf().fit(tr[cols], tr.y.astype(int))
    p_val = clf.predict_proba(va[cols])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_val, va.y)
    model = RiskModel(clf, iso, cols)
    model.threshold = choose_event_threshold(val, model.predict_proba(val))
    return model


def choose_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising F1 on the validation patients."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-9)
    return float(thr[int(np.argmax(f1))])


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
def row_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "prevalence": float(np.mean(y))}


MIN_GAP_STEPS = 6     # never notify twice within 30 min
REARM_STEPS = 12      # while risk stays high, repeat the notification hourly (CGM-alarm style)


def onsets(cgm: np.ndarray) -> np.ndarray:
    """Excursion onsets: first reading > 180 mg/dL that is also >= 40 mg/dL above the 2-h nadir,
    with no such reading in the previous 30 min."""
    s = pd.Series(cgm)
    nadir = s.shift(1).rolling(H, min_periods=6).min()
    cond = (s > C.HYPER_THRESHOLD) & (s - nadir >= C.SPIKE_RISE)
    prev = cond.astype(float).shift(1).rolling(6, min_periods=1).max().fillna(0)
    return np.flatnonzero(cond.to_numpy() & (prev == 0).to_numpy())


def notifications(alert: np.ndarray) -> np.ndarray:
    out, last = [], -10 ** 9
    for i in np.flatnonzero(alert):
        rising = i == 0 or not alert[i - 1]
        if (rising and i - last >= MIN_GAP_STEPS) or i - last >= REARM_STEPS:
            out.append(i)
            last = i
    return np.array(out, dtype=int)


def event_metrics(frame: pd.DataFrame, score: np.ndarray, thr: float) -> dict:
    """Clinically framed metrics on full-resolution streams, treating alerts as notifications.

    * notification = alert rising edge (re-sent hourly while the risk stays above threshold)
    * a notification is *true* if glucose exceeds 180 within the next 120 min
    * an onset is *caught* if a notification was sent in the 120 min before it
    * lead time = onset time minus the earliest notification in that window
    """
    n_onsets = caught = n_notif = n_true = 0
    leads, days = [], 0.0
    frame = frame.assign(_score=score)
    for _, g in frame.groupby("patient_id", sort=False):
        y = g["y"].to_numpy()
        alert = (g["_score"].to_numpy() >= thr) & ~np.isnan(y)
        days += len(g) / C.STEPS_PER_DAY
        notif = notifications(alert)
        n_notif += len(notif)
        n_true += int(np.sum(y[notif] == 1)) if len(notif) else 0
        for o in onsets(g["cgm"].to_numpy(np.float64)):
            if o < H:
                continue
            n_onsets += 1
            win = notif[(notif >= o - H) & (notif < o)]
            if len(win):
                caught += 1
                leads.append((o - win[0]) * C.STEP_MIN)
    leads = np.array(leads)
    return {
        "event_sensitivity": caught / max(n_onsets, 1),
        "alert_precision": n_true / max(n_notif, 1),
        "median_lead_min": float(np.median(leads)) if len(leads) else 0.0,
        "pct_lead_ge_30min": float(np.mean(leads >= 30)) if len(leads) else 0.0,
        "alerts_per_day": n_notif / max(days, 1e-9),
        "false_alerts_per_day": (n_notif - n_true) / max(days, 1e-9),
        "n_onsets": n_onsets,
        "_leads": leads.tolist(),
    }


def choose_event_threshold(frame: pd.DataFrame, p: np.ndarray, grid=None) -> float:
    """Threshold maximising the event-level F1 (spikes caught vs. notification precision)."""
    best, best_thr = -1.0, 0.5
    for thr in (np.arange(0.20, 0.91, 0.025) if grid is None else grid):
        m = event_metrics(frame, p, thr)
        se, pr = m["event_sensitivity"], m["alert_precision"]
        f1 = 2 * se * pr / max(se + pr, 1e-9)
        if f1 > best:
            best, best_thr = f1, float(thr)
    return best_thr


def group_permutation_importance(model: RiskModel, X: pd.DataFrame, y: np.ndarray, groups: list[str],
                                 n_repeats: int = 3, seed: int = C.SEED) -> pd.Series:
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.raw_proba(X))
    drops = {}
    for g in groups:
        cols = [c for c in FEATURE_GROUPS[g] if c in model.cols]
        vals = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[perm]
            vals.append(base - roc_auc_score(y, model.raw_proba(Xp)))
        drops[g] = float(np.mean(vals))
    return pd.Series(drops).sort_values(ascending=False)


def _curve(y, p, n=60) -> dict:
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    i = np.linspace(0, len(fpr) - 1, n).astype(int)
    j = np.linspace(0, len(prec) - 1, n).astype(int)
    return {"fpr": fpr[i].round(4).tolist(), "tpr": tpr[i].round(4).tolist(),
            "precision": prec[j].round(4).tolist(), "recall": rec[j].round(4).tolist()}


def _calibration(y, p, bins: int = 10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    frac = [float(y[idx == b].mean()) for b in range(bins) if np.any(idx == b)]
    mean = [float(p[idx == b].mean()) for b in range(bins) if np.any(idx == b)]
    return frac, mean


def train_all(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, log=print) -> tuple[Bundle, dict]:
    """Train ablations + final model + forecasters; return the bundle and a metrics dict."""
    tr_fit = train.iloc[::2]  # every 10 min is plenty and halves training time
    te = test[test.y.notna()]
    y_te = te.y.to_numpy().astype(int)
    metrics: dict = {"ablations": {}, "curves": {}, "split_sizes": {
        "train_patients": int(train.patient_id.nunique()), "val_patients": int(val.patient_id.nunique()),
        "test_patients": int(test.patient_id.nunique()), "test_rows": int(len(te))}}

    metrics["by_history"] = {b: {} for b in C.HISTORY_BUCKETS}

    def bucket_eval(name: str, score_all: np.ndarray, thr: float) -> None:
        for b, (lo, hi) in C.HISTORY_BUCKETS.items():
            mask = test["day"].between(lo, hi).to_numpy()
            sub, sc = test[mask], score_all[mask]
            ok = sub.y.notna().to_numpy()
            r = row_metrics(sub.y.to_numpy()[ok].astype(int), sc[ok])
            r.update({k: v for k, v in event_metrics(sub, sc, thr).items() if k != "_leads"})
            metrics["by_history"][b][name] = r

    models, p_te = {}, {}
    for name, groups in ABLATIONS.items():
        cols = cols_for(groups)
        m = fit_risk(tr_fit, val, cols)
        p = m.predict_proba(te)
        res = row_metrics(y_te, p)
        res["brier"] = float(brier_score_loss(y_te, p))
        res["threshold"] = m.threshold
        res.update({k: v for k, v in event_metrics(test, m.predict_proba(test), m.threshold).items() if k != "_leads"})
        res["n_features"] = len(cols)
        bucket_eval(name, m.predict_proba(test), m.threshold)
        metrics["ablations"][name] = res
        metrics["curves"][name] = _curve(y_te, p)
        models[name] = m
        p_te[name] = p
        log(f"  {name:38s} AUROC {res['auroc']:.3f}  AUPRC {res['auprc']:.3f}  "
            f"caught {res['event_sensitivity']:.0%}  lead {res['median_lead_min']:.0f} min  "
            f"FA/day {res['false_alerts_per_day']:.2f}")

    # Non-learned baselines
    baselines = {
        "Rule: 30-min linear extrapolation": lambda d: (d["cgm_now"] + 2 * d["cgm_d15"]).fillna(0).to_numpy(),
        "Mechanistic twin only (no ML)": lambda d: d["twin_max120"].fillna(0).to_numpy(),
    }
    for name, score_fn in baselines.items():
        thr = choose_event_threshold(val, score_fn(val), grid=np.arange(140, 261, 5))
        res = row_metrics(y_te, score_fn(te))
        res["threshold"] = float(thr)
        res.update({k: v for k, v in event_metrics(test, score_fn(test), thr).items() if k != "_leads"})
        bucket_eval(name, score_fn(test), thr)
        metrics["ablations"][name] = res
        metrics["curves"][name] = _curve(y_te, score_fn(te))
        log(f"  {name:38s} AUROC {res['auroc']:.3f}  caught {res['event_sensitivity']:.0%}  "
            f"lead {res['median_lead_min']:.0f} min  FA/day {res['false_alerts_per_day']:.2f}")

    for b in C.HISTORY_BUCKETS:
        parts = [f"{n.split(':')[0]} {v['auroc']:.3f}/{v['false_alerts_per_day']:.2f}FA"
                 for n, v in metrics["by_history"][b].items()]
        log(f"  [{b}] AUROC/FA-per-day: " + ", ".join(parts))

    # Patient-level bootstrap: is GlucoTwin's AUROC really different from each ablation?
    pids = te.patient_id.to_numpy()
    rows_of = [np.flatnonzero(pids == u) for u in np.unique(pids)]
    rng = np.random.default_rng(C.SEED)
    diffs = {n: [] for n in ABLATIONS if n != FINAL}
    for _ in range(200):
        idx = np.concatenate([rows_of[i] for i in rng.integers(0, len(rows_of), len(rows_of))])
        a_final = roc_auc_score(y_te[idx], p_te[FINAL][idx])
        for n in diffs:
            diffs[n].append(a_final - roc_auc_score(y_te[idx], p_te[n][idx]))
    metrics["auroc_ci"] = {n: {"diff": float(np.mean(d)), "lo": float(np.quantile(d, 0.025)),
                               "hi": float(np.quantile(d, 0.975))} for n, d in diffs.items()}
    log("  AUROC difference GlucoTwin minus X (95% patient-bootstrap CI): " + ", ".join(
        f"{n.split(':')[0]} {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]" for n, v in metrics["auroc_ci"].items()))

    final = models[FINAL]
    p_final = final.predict_proba(te)
    ev = event_metrics(test, final.predict_proba(test), final.threshold)
    metrics["final"] = {"name": FINAL, "lead_times": ev["_leads"], **metrics["ablations"][FINAL]}
    frac_pos, mean_pred = _calibration(y_te, p_final)
    metrics["calibration"] = {"mean_pred": mean_pred, "frac_pos": frac_pos}

    # Group importance on a test subsample
    sub = te.sample(min(40000, len(te)), random_state=C.SEED)
    imp = group_permutation_importance(final, sub, sub.y.to_numpy().astype(int), ABLATIONS[FINAL])
    metrics["group_importance"] = imp.round(4).to_dict()
    log("  group importance (AUROC drop): " + ", ".join(f"{k} {v:.3f}" for k, v in imp.items()))

    # Glucose forecasters (delta from current CGM)
    bundle = Bundle(risk=final, reference=train[final.cols].median())
    no_twin_cols = cols_for(ABLATIONS["Fusion: EHR + wearables"])
    metrics["forecast"] = {}
    for h in FORECAST_HORIZONS:
        tgt = f"g_t{h}"
        ok_tr = tr_fit[tgt].notna() & tr_fit.cgm_now.notna()
        reg = _reg().fit(tr_fit.loc[ok_tr, final.cols], (tr_fit.loc[ok_tr, tgt] - tr_fit.loc[ok_tr, "cgm_now"]))
        bundle.regressors[h] = reg
        ok_va = val[tgt].notna() & val.cgm_now.notna()
        r_va = val.loc[ok_va, tgt] - (val.loc[ok_va, "cgm_now"] + reg.predict(val.loc[ok_va, final.cols]))
        bundle.resid_q[h] = (float(np.quantile(r_va, 0.1)), float(np.quantile(r_va, 0.9)))
        ok = test[tgt].notna() & test.cgm_now.notna()
        truth = test.loc[ok, tgt].to_numpy()
        reg_nt = _reg().fit(tr_fit.loc[ok_tr, no_twin_cols], (tr_fit.loc[ok_tr, tgt] - tr_fit.loc[ok_tr, "cgm_now"]))
        preds = {"GlucoTwin hybrid": test.loc[ok, "cgm_now"].to_numpy() + reg.predict(test.loc[ok, final.cols]),
                 "Hybrid without twin features": (test.loc[ok, "cgm_now"].to_numpy()
                                                  + reg_nt.predict(test.loc[ok, no_twin_cols])),
                 "Mechanistic twin": test.loc[ok, f"twin_g{h}"].to_numpy(),
                 "Persistence (no change)": test.loc[ok, "cgm_now"].to_numpy()}
        metrics["forecast"][str(h)] = {k: {"rmse": float(np.sqrt(np.nanmean((v - truth) ** 2))),
                                           "mae": float(np.nanmean(np.abs(v - truth)))} for k, v in preds.items()}
        log(f"  forecast +{h:3d} min RMSE: " + ", ".join(f"{k} {v['rmse']:.1f}" for k, v in metrics["forecast"][str(h)].items()))
    return bundle, metrics


# --------------------------------------------------------------------------------------
# Explanations
# --------------------------------------------------------------------------------------
def _logit(q: np.ndarray) -> np.ndarray:
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def explain_rows(bundle: Bundle, X: pd.DataFrame) -> pd.DataFrame:
    """Group-occlusion attributions on the log-odds scale (rows × groups).

    For each feature group, replace its features with cohort medians and measure how much
    the predicted log-odds drop. Positive = the group pushed risk *up* for this patient.
    """
    m = bundle.risk
    base = _logit(m.raw_proba(X))
    out = {}
    for g in ABLATIONS[FINAL]:
        Xr = X.copy()
        for c in FEATURE_GROUPS[g]:
            Xr[c] = bundle.reference[c]
        out[g] = base - _logit(m.raw_proba(Xr))
    return pd.DataFrame(out, index=X.index)


def narrate(row: pd.Series, contrib: pd.Series, top: int = 4) -> list[tuple[str, float, str]]:
    """Human-readable drivers: [(group, contribution, sentence)] sorted by |contribution|."""
    def f(x, fmt="{:.0f}"):
        return "n/a" if pd.isna(x) else fmt.format(x)

    texts = {
        "Glucose trend (CGM)": f"Glucose {f(row.cgm_now)} mg/dL, changing {f(row.cgm_roc, '{:+.1f}')} mg/dL/min",
        "Personal routine": (f"Spiked above 180 around this time on {f(row.routine_spike_3d * 3)} of the last 3 days"
                             if not pd.isna(row.routine_spike_3d) else "Not enough history at this time of day"),
        "Meals (logged)": (f"{f(row.carbs_120)} g carbs logged in the last 2 h ({f(row.min_since_meal)} min ago)"
                           if row.carbs_120 > 0 else "No meal logged in the last 2 h"),
        "Digital twin": (f"Glucose {f(row.twin_resid_30)} mg/dL above twin expectation: possible unlogged meal"
                         if row.twin_resid_30 > 15 else
                         f"Twin projects a peak of {f(row.twin_max120)} mg/dL from current gut carbs / insulin state"),
        "Sleep": f"Slept {f(row.last_sleep_h, '{:.1f}')} h last night ({f(row.sleep_vs_usual_h, '{:+.1f}')} h vs usual)",
        "Heart rate & HRV": (f"Daytime HRV {f((row.hrv_rel_day - 1) * 100, '{:+.0f}')}% vs baseline, "
                             f"resting HR {f(row.hr_still_elev, '{:+.0f}')} bpm vs usual"),
        "Activity": f"{f(row.steps_60)} steps in the last hour, {f(row.sedentary_min_2h)} sedentary min in 2 h",
        "Clinical (EHR)": f"HbA1c {f(row.hba1c_pct, '{:.1f}')}%, {f(row.diabetes_duration_y, '{:.0f}')} y since diagnosis",
        "Genetics": f"TCF7L2 risk alleles {f(row.tcf7l2_risk_alleles)}, polygenic risk z={f(row.polygenic_risk_z, '{:+.1f}')}",
        "Time of day": "Time-of-day pattern",
    }
    order = contrib.abs().sort_values(ascending=False).index[:top]
    return [(g, float(contrib[g]), texts.get(g, g)) for g in order]


def save_metrics(metrics: dict, path=C.REPORTS_DIR / "metrics.json") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
