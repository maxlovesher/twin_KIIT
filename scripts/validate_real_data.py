"""Validate GlucoTwin on REAL patients: the ShanghaiT2DM dataset (100 people with T2D, CGM + clinical
record + meals + insulin/oral-agent doses; Zhao et al., Sci Data 2023, CC BY 4.0).

    python scripts/validate_real_data.py      (expects data/external/shanghai/, see README)

Design (mirrors the synthetic evaluation as far as the data allow):
  * each recording's twin is calibrated on its first 3 days; before that it runs on EHR priors
  * 5-fold cross-validation grouped by SUBJECT (no person in both train and test)
  * ablation: CGM -> + routine -> + meals & insulin -> + EHR -> + digital twin
  * subject-level bootstrap CIs for the AUROC differences
  * zero-shot transfer: the model trained only on synthetic patients, applied unchanged
  * forecasts: persistence vs calibrated twin vs ML (out-of-fold)
There are no steps / heart-rate / HRV / sleep data in this cohort, so those feature groups are absent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from glucotwin import config as C  # noqa: E402
from glucotwin import features as F  # noqa: E402
from glucotwin import modeling as M  # noqa: E402
from glucotwin import realdata as RD  # noqa: E402
from glucotwin import twin as T  # noqa: E402

CAL_DAYS = 3
DAY = C.STEPS_PER_DAY
MEDS_DYNAMIC = ["insulin_2h", "insulin_4h", "oha_6h"]
G = F.FEATURE_GROUPS
BASE = G["Glucose trend (CGM)"] + G["Time of day"]
SETS = {
    "CGM only": BASE,
    "+ personal routine": BASE + G["Personal routine"],
    "+ meals & insulin doses": BASE + G["Personal routine"] + G["Meals (logged)"] + MEDS_DYNAMIC,
    "+ EHR (clinical record)": BASE + G["Personal routine"] + G["Meals (logged)"] + MEDS_DYNAMIC + G["Clinical (EHR)"],
    "GlucoTwin (+ digital twin)": (BASE + G["Personal routine"] + G["Meals (logged)"] + MEDS_DYNAMIC
                                   + G["Clinical (EHR)"] + G["Digital twin"]),
}
COMPARE = [("GlucoTwin (+ digital twin)", "CGM only"), ("+ meals & insulin doses", "+ personal routine"),
           ("+ EHR (clinical record)", "+ meals & insulin doses"),
           ("GlucoTwin (+ digital twin)", "+ EHR (clinical record)")]


def build_record(pid: str, df: pd.DataFrame, erow: pd.Series, static_row: pd.Series):
    if len(df) < (CAL_DAYS + 1) * DAY:
        return None, None
    prior = T.PatientTwin(pid, T.prior_params(erow))
    fit = T.calibrate_patient(erow, df, days=CAL_DAYS)
    calib = T.PatientTwin.from_row(pd.Series(fit))
    fr = F.build_patient_frame(df, static_row, calib, onboarding_twin=prior, switch_step=CAL_DAYS * DAY)
    ins = df["insulin_units"].reset_index(drop=True)
    fr["insulin_2h"] = ins.rolling(24, min_periods=1).sum().to_numpy()
    fr["insulin_4h"] = ins.rolling(48, min_periods=1).sum().to_numpy()
    fr["oha_6h"] = df["oha_taken"].reset_index(drop=True).rolling(72, min_periods=1).max().to_numpy()
    fr["subject"] = RD.subject_of(pid)
    return fr[fr.day >= 1].reset_index(drop=True), fit


def bootstrap_diff(frame: pd.DataFrame, a: np.ndarray, b: np.ndarray, n: int = 200) -> dict:
    y = frame.y.to_numpy().astype(int)
    subj = frame.subject.to_numpy()
    rows_of = [np.flatnonzero(subj == s) for s in np.unique(subj)]
    rng = np.random.default_rng(C.SEED)
    d = []
    for _ in range(n):
        idx = np.concatenate([rows_of[i] for i in rng.integers(0, len(rows_of), len(rows_of))])
        if y[idx].min() == y[idx].max():
            continue
        d.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    d = np.array(d)
    return {"diff": float(d.mean()), "lo": float(np.quantile(d, 0.025)), "hi": float(np.quantile(d, 0.975))}


def main() -> None:
    ehr, streams = RD.load_shanghai_t2dm()
    static = F.static_features(ehr)
    erows = ehr.set_index("patient_id", drop=False)
    print(f"real cohort: {len(streams)} recordings, {len({RD.subject_of(p) for p in streams})} subjects", flush=True)
    out = Parallel(n_jobs=-1)(delayed(build_record)(pid, df, erows.loc[pid], static.loc[pid])
                              for pid, df in streams.items())
    frames = [f for f, _ in out if f is not None]
    fits = pd.DataFrame([fit for _, fit in out if fit is not None])
    allf = pd.concat(frames, ignore_index=True)
    lab = allf[allf.y.notna()].reset_index(drop=True)
    print(f"usable recordings {len(frames)}, labelled rows {len(lab)}, spike rate {lab.y.mean():.3f}", flush=True)

    res: dict = {"cohort": {
        "recordings": len(frames), "subjects": int(allf.subject.nunique()), "labelled_rows": int(len(lab)),
        "spike_rate": float(lab.y.mean()), "hba1c_mean": float(ehr.hba1c_pct.mean()),
        "tir": float(allf.cgm.between(70, 180).mean()),
        "twin_fit_rmse_30_median": float(fits.fit_rmse_30.median()),
        "twin_fit_rmse_60_median": float(fits.fit_rmse_60.median())}}

    # ---- grouped cross-validation over the ablation sets
    gkf = GroupKFold(n_splits=5)
    oof = {name: np.zeros(len(lab)) for name in SETS}
    for k, (tr, te) in enumerate(gkf.split(lab, lab.y, groups=lab.subject)):
        trf = lab.iloc[tr].iloc[::2]
        for name, cols in SETS.items():
            clf = M._clf().fit(trf[cols], trf.y.astype(int))
            oof[name][te] = clf.predict_proba(lab.iloc[te][cols])[:, 1]
        print(f"  fold {k + 1}/5 done", flush=True)
    y = lab.y.to_numpy().astype(int)
    res["auroc"] = {}
    key = ["patient_id", "timestamp"]
    for name in SETS:
        scores = allf[key].merge(lab[key].assign(_p=oof[name]), on=key, how="left")["_p"].fillna(0).to_numpy()
        ev = M.event_metrics(allf, scores, 0.5)
        res["auroc"][name] = {"auroc": float(roc_auc_score(y, oof[name])),
                              "auprc": float(average_precision_score(y, oof[name])),
                              **{k: v for k, v in ev.items() if k != "_leads"}}
        print(f"  {name:28s} AUROC {res['auroc'][name]['auroc']:.3f}  caught {ev['event_sensitivity']:.0%}  "
              f"lead {ev['median_lead_min']:.0f} min  FA/day {ev['false_alerts_per_day']:.2f}", flush=True)
    res["diff_ci"] = {f"{a} vs {b}": bootstrap_diff(lab, oof[a], oof[b]) for a, b in COMPARE}
    for k, v in res["diff_ci"].items():
        print(f"  {k}: {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]", flush=True)

    # ---- zero-shot transfer of the synthetic-trained model
    bundle: M.Bundle = joblib.load(C.MODELS_DIR / "glucotwin_bundle.joblib")
    zs = bundle.risk.predict_proba(lab)
    res["zero_shot"] = {"auroc": float(roc_auc_score(y, zs)), "auprc": float(average_precision_score(y, zs))}
    print(f"  zero-shot (trained on synthetic only): AUROC {res['zero_shot']['auroc']:.3f}", flush=True)

    # ---- forecasts after the calibration window
    post = allf[allf.day >= CAL_DAYS].reset_index(drop=True)
    res["forecast"] = {}
    cols = SETS["GlucoTwin (+ digital twin)"]
    for h in (30, 60):
        ok = post[f"g_t{h}"].notna() & post.cgm_now.notna() & post[f"twin_g{h}"].notna()
        P = post[ok].reset_index(drop=True)
        truth = P[f"g_t{h}"].to_numpy()
        ml = np.zeros(len(P))
        for tr, te in gkf.split(P, groups=P.subject):
            trf = P.iloc[tr].iloc[::2]
            reg = M._reg().fit(trf[cols], trf[f"g_t{h}"] - trf.cgm_now)
            ml[te] = P.iloc[te].cgm_now + reg.predict(P.iloc[te][cols])
        preds = {"Persistence (no change)": P.cgm_now.to_numpy(),
                 "Calibrated twin (physiology only)": P[f"twin_g{h}"].to_numpy(),
                 "GlucoTwin hybrid (ML, out-of-fold)": ml}
        res["forecast"][str(h)] = {k: float(np.sqrt(np.mean((v - truth) ** 2))) for k, v in preds.items()}
        print(f"  forecast +{h} RMSE: " + ", ".join(f"{k} {v:.1f}" for k, v in res["forecast"][str(h)].items()),
              flush=True)

    (C.REPORTS_DIR / "real_data_metrics.json").write_text(json.dumps(res, indent=2))
    figure(res)


def figure(res: dict) -> None:
    import matplotlib.pyplot as plt

    from glucotwin import reporting as R

    R._style()
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
    names = list(res["auroc"])[::-1]
    vals = [res["auroc"][n]["auroc"] for n in names]
    colors = [R.SERIES[0] if n.startswith("GlucoTwin") else R.NEUTRAL_BAR for n in names]
    bars = ax[0].barh(names, vals, color=colors, height=0.6)
    for b_, v in zip(bars, vals):
        ax[0].text(v, b_.get_y() + b_.get_height() / 2, f" {v:.3f}", va="center", color=R.INK2, fontsize=9)
    zs = res["zero_shot"]["auroc"]
    ax[0].axvline(zs, color=R.SERIES[1], lw=1.2, ls="--")
    ax[0].text(zs + 0.005, len(names) - 0.4, f"zero-shot from synthetic: {zs:.3f}", color=R.INK2, fontsize=8.5)
    ax[0].set(xlim=(0.5, 1.0), title=f"Real T2D patients (n={res['cohort']['subjects']}): spike-prediction AUROC")
    ax[0].grid(axis="y", visible=False)
    ax[0].tick_params(axis="y", colors=R.INK2)
    fc = res["forecast"]
    models = list(fc["30"])
    x = np.arange(2)
    for i, m in enumerate(models):
        ax[1].bar(x + (i - 1) * 0.26, [fc["30"][m], fc["60"][m]], width=0.23, color=R.SERIES[i], label=m)
    ax[1].set_xticks(x, ["+30 min", "+60 min"])
    ax[1].grid(axis="x", visible=False)
    ax[1].set(ylabel="RMSE (mg/dL)", title="Real-data glucose forecast error")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    R._save(fig, "real_data.png")


if __name__ == "__main__":
    main()
