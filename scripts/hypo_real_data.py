"""Second adverse event: HYPOGLYCAEMIA (CGM < 70 mg/dL within 2 h), trained and validated on REAL patients.

    python scripts/hypo_real_data.py     (needs data/external/shanghai, see scripts/download_real_data.py)

Lows are the dangerous side of glucose control, mostly in people on insulin or sulfonylureas. The synthetic
cohort has almost none, so this model is built only from the real ShanghaiT2DM patients:
  * 5-fold cross-validation grouped by person; alert thresholds chosen by inner cross-validation
  * ablation: CGM -> + routine & meals -> + insulin/oral-agent doses -> + EHR -> + digital twin
  * baseline: the classic "urgent low soon" rule (30-min linear extrapolation below 70 mg/dL)
  * a "deployable" variant without the dose stream (doses are not always recorded) is saved for the
    dashboard/engine as models/hypo_model.joblib
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from glucotwin import config as C  # noqa: E402
from glucotwin import features as F  # noqa: E402
from glucotwin import modeling as M  # noqa: E402
from glucotwin import realdata as RD  # noqa: E402

H = C.HORIZON_STEPS
G = F.FEATURE_GROUPS
BASE = G["Glucose trend (CGM)"] + G["Time of day"]
RM = BASE + G["Personal routine"] + G["Meals (logged)"]
SETS = {
    "CGM only": BASE,
    "+ routine & meals": RM,
    "+ insulin & oral-agent doses": RM + RD.MEDS_DYNAMIC,
    "+ EHR (incl. medication classes)": RM + RD.MEDS_DYNAMIC + G["Clinical (EHR)"],
    "GlucoTwin (+ digital twin)": RM + RD.MEDS_DYNAMIC + G["Clinical (EHR)"] + G["Digital twin"],
    "Deployable (no dose stream)": RM + G["Clinical (EHR)"] + G["Digital twin"],
}
TUNED = ["CGM only", "GlucoTwin (+ digital twin)", "Deployable (no dose stream)"]
GRID = np.arange(0.05, 0.91, 0.025)


def hypo_onsets(cgm: np.ndarray) -> np.ndarray:
    s = pd.Series(cgm)
    prev_min = s.shift(1).rolling(6, min_periods=3).min()
    return np.flatnonzero((s < C.HYPO_THRESHOLD).to_numpy() & (prev_min >= C.HYPO_THRESHOLD).to_numpy())


def hypo_events(frame: pd.DataFrame, score: np.ndarray, thr: float) -> dict:
    """Same notification logic as the spike evaluation (see modeling.event_metrics), for lows."""
    n_on = caught = n_notif = n_true = 0
    leads, days = [], 0.0
    frame = frame.assign(_s=score)
    for _, g in frame.groupby("patient_id", sort=False):
        y = g["y_hypo"].to_numpy()
        alert = (g["_s"].to_numpy() >= thr) & ~np.isnan(y)
        days += len(g) / C.STEPS_PER_DAY
        notif = M.notifications(alert)
        n_notif += len(notif)
        n_true += int(np.sum(y[notif] == 1)) if len(notif) else 0
        for o in hypo_onsets(g["cgm"].to_numpy(np.float64)):
            if o < H:
                continue
            n_on += 1
            win = notif[(notif >= o - H) & (notif < o)]
            if len(win):
                caught += 1
                leads.append((o - win[0]) * C.STEP_MIN)
    leads = np.array(leads)
    return {"event_sensitivity": caught / max(n_on, 1), "alert_precision": n_true / max(n_notif, 1),
            "median_lead_min": float(np.median(leads)) if len(leads) else 0.0,
            "false_alerts_per_day": (n_notif - n_true) / max(days, 1e-9), "n_onsets": n_on}


def best_threshold(frame: pd.DataFrame, score: np.ndarray) -> float:
    best, thr = -1.0, 0.5
    for t in GRID:
        m = hypo_events(frame, score, t)
        se, pr = m["event_sensitivity"], m["alert_precision"]
        f1 = 2 * se * pr / max(se + pr, 1e-9)
        if f1 > best:
            best, thr = f1, float(t)
    return thr


def scores_for(frame: pd.DataFrame, lab: pd.DataFrame, p: np.ndarray) -> np.ndarray:
    key = ["patient_id", "timestamp"]
    return frame[key].merge(lab[key].assign(_p=p), on=key, how="left")["_p"].fillna(0).to_numpy()


def bootstrap_diff(lab: pd.DataFrame, y: np.ndarray, a: np.ndarray, b: np.ndarray, n: int = 200) -> dict:
    subj = lab.subject.to_numpy()
    rows_of = [np.flatnonzero(subj == s) for s in np.unique(subj)]
    rng = np.random.default_rng(C.SEED)
    d = []
    for _ in range(n):
        idx = np.concatenate([rows_of[i] for i in rng.integers(0, len(rows_of), len(rows_of))])
        if y[idx].min() != y[idx].max():
            d.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    d = np.array(d)
    return {"diff": float(d.mean()), "lo": float(np.quantile(d, 0.025)), "hi": float(np.quantile(d, 0.975))}


def rule_curve(allf: pd.DataFrame) -> list[dict]:
    """The 'urgent low soon' rule at different cut-offs: alert when 30-min projected glucose < cutoff."""
    proj = (allf.cgm_now + 2 * allf.cgm_d15).fillna(999).to_numpy()
    out = []
    for cut in range(50, 111, 5):
        m = hypo_events(allf, cut - proj, 0.0)
        out.append({"cutoff": cut, **{k: v for k, v in m.items() if k != "n_onsets"}})
    return out


def add_rule_curve(res: dict, allf: pd.DataFrame) -> None:
    """Compare the model and the rule at the model's own quiet operating points (same false-alarm rate)."""
    rc = rule_curve(allf)
    res["tradeoff"]["rule_curve"] = rc
    model = res["tradeoff"]["GlucoTwin (+ digital twin)"]
    rows = []
    for fa in (0.25, 0.5, 0.75, 1.0):
        m_best = max((c for c in model if c["false_alerts_per_day"] <= fa), key=lambda c: c["event_sensitivity"],
                     default=None)
        r_best = max((c for c in rc if c["false_alerts_per_day"] <= fa), key=lambda c: c["event_sensitivity"],
                     default=None)
        rows.append({"false_alerts_per_day_budget": fa,
                     "glucotwin_sensitivity": m_best["event_sensitivity"] if m_best else 0.0,
                     "rule_sensitivity": r_best["event_sensitivity"] if r_best else 0.0})
        print(f"  with <= {fa:.2f} false alarms/day: GlucoTwin catches "
              f"{rows[-1]['glucotwin_sensitivity']:.0%} of lows, best rule cut-off {rows[-1]['rule_sensitivity']:.0%}",
              flush=True)
    res["tradeoff"]["matched"]["by_alarm_budget"] = rows


def fit(df: pd.DataFrame, cols: list[str]):
    return M._clf().fit(df[cols], df.y_hypo.astype(int))


def main() -> None:
    ehr, streams = RD.load_shanghai_t2dm()
    allf, _ = RD.build_frames(ehr, streams)
    lab = allf[allf.y_hypo.notna()].reset_index(drop=True)
    y = lab.y_hypo.to_numpy().astype(int)
    n_lows = sum(len(hypo_onsets(g.cgm.to_numpy(np.float64))) for _, g in allf.groupby("patient_id"))
    print(f"real cohort: {allf.subject.nunique()} people, {len(lab)} labelled rows, "
          f"low-within-2h rate {y.mean():.2%}, {n_lows} low episodes", flush=True)

    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(lab, y, groups=lab.subject))
    oof = {n: np.zeros(len(lab)) for n in SETS}
    thresholds = {n: [] for n in TUNED}
    for k, (tr, te) in enumerate(folds):
        trl = lab.iloc[tr]
        for name, cols in SETS.items():
            oof[name][te] = fit(trl.iloc[::2], cols).predict_proba(lab.iloc[te][cols])[:, 1]
        # inner CV on the training people picks each tuned model's alert threshold
        for name in TUNED:
            cols = SETS[name]
            inner = np.zeros(len(trl))
            for itr, ite in GroupKFold(n_splits=3).split(trl, groups=trl.subject):
                inner[ite] = fit(trl.iloc[itr].iloc[::2], cols).predict_proba(trl.iloc[ite][cols])[:, 1]
            tr_frame = allf[allf.subject.isin(set(trl.subject))]
            thresholds[name].append(best_threshold(tr_frame, scores_for(tr_frame, trl, inner)))
        print(f"  fold {k + 1}/5 done", flush=True)

    res: dict = {"cohort": {"people": int(allf.subject.nunique()), "labelled_rows": int(len(lab)),
                            "hypo_rate": float(y.mean()), "low_episodes": int(n_lows)}, "auroc": {}}
    for name in SETS:
        r = {"auroc": float(roc_auc_score(y, oof[name])), "auprc": float(average_precision_score(y, oof[name]))}
        if name in TUNED:
            evs = []
            for k, (_, te) in enumerate(folds):     # each fold's own threshold on that fold's people
                fr = allf[allf.subject.isin(set(lab.iloc[te].subject))]
                evs.append(hypo_events(fr, scores_for(fr, lab.iloc[te], oof[name][te]), thresholds[name][k]))
            n_on = sum(e["n_onsets"] for e in evs)
            r.update(threshold=float(np.median(thresholds[name])),
                     event_sensitivity=sum(e["event_sensitivity"] * e["n_onsets"] for e in evs) / max(n_on, 1),
                     median_lead_min=float(np.median([e["median_lead_min"] for e in evs])),
                     alert_precision=float(np.mean([e["alert_precision"] for e in evs])),
                     false_alerts_per_day=float(np.mean([e["false_alerts_per_day"] for e in evs])))
        res["auroc"][name] = r
        extra = (f"  caught {r['event_sensitivity']:.0%} lead {r['median_lead_min']:.0f} min "
                 f"FA/day {r['false_alerts_per_day']:.2f}") if name in TUNED else ""
        print(f"  {name:34s} AUROC {r['auroc']:.3f} AUPRC {r['auprc']:.3f}{extra}", flush=True)

    rule_all = 70.0 - (allf.cgm_now + 2 * allf.cgm_d15).fillna(999).to_numpy()   # >= 0: projected below 70
    rule_lab = 70.0 - (lab.cgm_now + 2 * lab.cgm_d15).fillna(999).to_numpy()
    ev = hypo_events(allf, rule_all, 0.0)
    res["auroc"]["Rule: 'urgent low soon' (30-min extrapolation)"] = {
        "auroc": float(roc_auc_score(y, rule_lab)), "auprc": float(average_precision_score(y, rule_lab)), **ev}
    print(f"  rule baseline: AUROC {roc_auc_score(y, rule_lab):.3f} caught {ev['event_sensitivity']:.0%} "
          f"lead {ev['median_lead_min']:.0f} min FA/day {ev['false_alerts_per_day']:.2f}", flush=True)

    # Alarm trade-off: lows caught vs false alarms/day across thresholds (pooled out-of-fold scores).
    # This is the fair comparison with the fixed rule, which sits at a single, alarm-heavy operating point.
    res["tradeoff"] = {"rule": {k: ev[k] for k in ("event_sensitivity", "false_alerts_per_day", "median_lead_min")}}
    for name in ("CGM only", "GlucoTwin (+ digital twin)"):
        sc = scores_for(allf, lab, oof[name])
        res["tradeoff"][name] = [{"thr": float(t), **{k: v for k, v in hypo_events(allf, sc, t).items() if k != "n_onsets"}}
                                 for t in np.arange(0.02, 0.81, 0.02)]
    curve = res["tradeoff"]["GlucoTwin (+ digital twin)"]
    rule_fa, rule_se = ev["false_alerts_per_day"], ev["event_sensitivity"]
    at_rule_fa = max((c for c in curve if c["false_alerts_per_day"] <= rule_fa), key=lambda c: c["event_sensitivity"])
    at_95 = min((c for c in curve if c["event_sensitivity"] >= 0.95), key=lambda c: c["false_alerts_per_day"],
                default=None)
    res["tradeoff"]["matched"] = {
        "at_rule_false_alarm_rate": at_rule_fa,
        "at_95pct_sensitivity": at_95,
        "rule": {"event_sensitivity": rule_se, "false_alerts_per_day": rule_fa},
    }
    print(f"  at the rule's false-alarm rate ({rule_fa:.2f}/day) GlucoTwin catches "
          f"{at_rule_fa['event_sensitivity']:.0%} of lows (rule {rule_se:.0%})", flush=True)
    if at_95:
        print(f"  catching >=95% of lows, GlucoTwin needs {at_95['false_alerts_per_day']:.2f} false alarms/day "
              f"(rule {rule_fa:.2f})", flush=True)

    add_rule_curve(res, allf)

    res["diff_ci"] = {
        "GlucoTwin vs CGM only": bootstrap_diff(lab, y, oof["GlucoTwin (+ digital twin)"], oof["CGM only"]),
        "+ doses vs routine & meals": bootstrap_diff(lab, y, oof["+ insulin & oral-agent doses"],
                                                     oof["+ routine & meals"]),
        "+ twin vs + EHR": bootstrap_diff(lab, y, oof["GlucoTwin (+ digital twin)"],
                                          oof["+ EHR (incl. medication classes)"]),
    }
    for k, v in res["diff_ci"].items():
        print(f"  {k}: {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]", flush=True)

    dep = "Deployable (no dose stream)"
    model = {"clf": fit(lab.iloc[::2], SETS[dep]), "cols": SETS[dep], "threshold": res["auroc"][dep]["threshold"],
             "trained_on": "ShanghaiT2DM (real patients)", "auroc_cv": res["auroc"][dep]["auroc"]}
    joblib.dump(model, C.MODELS_DIR / "hypo_model.joblib", compress=3)
    (C.REPORTS_DIR / "hypo_metrics.json").write_text(json.dumps(res, indent=2))
    figure(res)


def figure(res: dict) -> None:
    import matplotlib.pyplot as plt

    from glucotwin import reporting as R

    R._style()
    a = res["auroc"]
    names = list(a)[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 3.9), gridspec_kw={"width_ratios": [1.4, 1]})
    vals = [a[n]["auroc"] for n in names]
    colors = [R.SERIES[0] if n.startswith("GlucoTwin") else R.NEUTRAL_BAR for n in names]
    bars = ax[0].barh(names, vals, color=colors, height=0.6)
    for b_, v in zip(bars, vals):
        ax[0].text(v, b_.get_y() + b_.get_height() / 2, f" {v:.3f}", va="center", color=R.INK2, fontsize=9)
    ax[0].set(xlim=(0.5, 1.0), title=f"Low glucose (<70) within 2 h, real patients (n={res['cohort']['people']})")
    ax[0].grid(axis="y", visible=False)
    ax[0].tick_params(axis="y", colors=R.INK2)
    tr = res["tradeoff"]
    for i, name in enumerate(("CGM only", "GlucoTwin (+ digital twin)")):
        pts = sorted(tr[name], key=lambda c: c["false_alerts_per_day"])
        ax[1].plot([c["false_alerts_per_day"] for c in pts], [c["event_sensitivity"] * 100 for c in pts],
                   color=(R.MUTED, R.SERIES[0])[i], lw=2, label=name)
    if "rule_curve" in tr:
        rc = sorted(tr["rule_curve"], key=lambda c: c["false_alerts_per_day"])
        ax[1].plot([c["false_alerts_per_day"] for c in rc], [c["event_sensitivity"] * 100 for c in rc],
                   color=R.SERIES[1], lw=2, ls="--", label="Rule, varying cut-off")
    rule = tr["rule"]
    ax[1].scatter([rule["false_alerts_per_day"]], [rule["event_sensitivity"] * 100], s=70, color=R.SERIES[1],
                  edgecolor=R.SURFACE, linewidth=1.5, zorder=3, label="'Urgent low soon' rule (<70)")
    ax[1].set(xlabel="False alarms per person-day", ylabel="Lows warned in advance (%)",
              title="Alarm trade-off on unseen people", ylim=(0, 102), xlim=(0, 2))
    ax[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    R._save(fig, "hypo_real_data.png")


def rule_curve_only() -> None:
    """Add the rule's curve to an existing hypo_metrics.json without retraining (frames are rebuilt)."""
    path = C.REPORTS_DIR / "hypo_metrics.json"
    res = json.loads(path.read_text())
    ehr, streams = RD.load_shanghai_t2dm()
    allf, _ = RD.build_frames(ehr, streams)
    add_rule_curve(res, allf)
    path.write_text(json.dumps(res, indent=2))
    figure(res)


if __name__ == "__main__":
    if "--rule-curve-only" in sys.argv:
        rule_curve_only()
    else:
        main()
