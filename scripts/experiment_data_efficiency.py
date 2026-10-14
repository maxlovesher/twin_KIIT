"""Does the digital twin help most when training data is scarce (e.g. a new clinic)?

Trains the spike classifier and the +60-min forecaster on 6 / 12 / 24 / 72 training patients, with and
without the twin feature group, and evaluates on the 24 held-out test patients (3 random draws per size).

    python scripts/experiment_data_efficiency.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from glucotwin import config as C  # noqa: E402
from glucotwin import modeling as M  # noqa: E402
from glucotwin import reporting as R  # noqa: E402

SIZES = (6, 12, 24, 72)
DRAWS = 3
SETS = {"with twin": M.cols_for(M.ABLATIONS[M.FINAL]),
        "without twin": M.cols_for(M.ABLATIONS["Fusion: EHR + wearables"])}


def main() -> None:
    train = pd.read_parquet(C.PROCESSED_DIR / "features_train.parquet")
    test = pd.read_parquet(C.PROCESSED_DIR / "features_test.parquet")
    te = test[test.y.notna()]
    fc = test[test.g_t60.notna() & test.cgm_now.notna()]
    pids = train.patient_id.unique()
    rng = np.random.default_rng(C.SEED)
    rows = []
    for n in SIZES:
        for d in range(DRAWS if n < len(pids) else 1):
            chosen = rng.choice(pids, n, replace=False)
            tr = train[train.patient_id.isin(chosen)].iloc[::2]
            for name, cols in SETS.items():
                lab = tr[tr.y.notna()]
                clf = M._clf().fit(lab[cols], lab.y.astype(int))
                auc = roc_auc_score(te.y.astype(int), clf.predict_proba(te[cols])[:, 1])
                ok = tr.g_t60.notna() & tr.cgm_now.notna()
                reg = M._reg().fit(tr.loc[ok, cols], tr.loc[ok, "g_t60"] - tr.loc[ok, "cgm_now"])
                rmse = float(np.sqrt(np.mean((fc.cgm_now + reg.predict(fc[cols]) - fc.g_t60) ** 2)))
                rows.append({"n_patients": n, "draw": d, "features": name, "auroc": auc, "rmse60": rmse})
                print(f"n={n:2d} draw={d} {name:12s} AUROC {auc:.3f}  RMSE+60 {rmse:.1f}", flush=True)
    res = pd.DataFrame(rows)
    out = {str(n): {f: {"auroc": float(g.auroc.mean()), "auroc_sd": float(g.auroc.std(ddof=0)),
                        "rmse60": float(g.rmse60.mean()), "rmse60_sd": float(g.rmse60.std(ddof=0))}
                    for f, g in res[res.n_patients == n].groupby("features")} for n in SIZES}
    (C.REPORTS_DIR / "data_efficiency.json").write_text(json.dumps(out, indent=2))

    R._style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for i, f in enumerate(SETS):
        for a, key, label in ((ax[0], "auroc", "AUROC"), (ax[1], "rmse60", "+60 min forecast RMSE (mg/dL)")):
            m = [out[str(n)][f][key] for n in SIZES]
            sd = [out[str(n)][f][key + "_sd"] for n in SIZES]
            a.errorbar(SIZES, m, yerr=sd, color=R.SERIES[i], marker="o", ms=7, capsize=3, label=f,
                       markeredgecolor=R.SURFACE, markeredgewidth=1.5)
            a.set(xscale="log", xlabel="Training patients", ylabel=label)
            a.set_xticks(SIZES, [str(n) for n in SIZES])
    ax[0].set_title("Spike prediction vs. training-set size")
    ax[1].set_title("Forecast error vs. training-set size")
    ax[0].legend()
    fig.tight_layout()
    R._save(fig, "data_efficiency.png")


if __name__ == "__main__":
    main()
