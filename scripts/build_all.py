"""End-to-end GlucoTwin pipeline.

    python scripts/build_all.py            # run everything (reuses existing synthetic data)
    python scripts/build_all.py --fresh    # regenerate the synthetic cohort too
    python scripts/build_all.py --from train   # resume from a stage (simulate|calibrate|features|train|report)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from glucotwin import config as C  # noqa: E402
from glucotwin import features as F  # noqa: E402
from glucotwin import modeling as M  # noqa: E402
from glucotwin import simulate, twin  # noqa: E402

STAGES = ["simulate", "calibrate", "features", "train", "report"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stage_simulate(fresh: bool) -> None:
    if (C.RAW_DIR / "wearables.parquet").exists() and not fresh:
        log("simulate: using existing data/raw (pass --fresh to regenerate)")
        return
    log(f"simulate: {C.N_PATIENTS} patients x {C.N_DAYS} days")
    simulate.write_cohort()


def stage_calibrate() -> None:
    ehr = pd.read_csv(C.RAW_DIR / "ehr.csv")
    wear = pd.read_parquet(C.RAW_DIR / "wearables.parquet")
    log(f"calibrate: fitting {len(ehr)} patient twins on their first {C.CALIBRATION_DAYS} days")
    params = twin.calibrate_cohort(ehr, wear)
    params.to_csv(C.MODELS_DIR / "twin_params.csv", index=False)
    log(f"calibrate: median fit RMSE 30 min {params.fit_rmse_30.median():.1f} mg/dL, "
        f"60 min {params.fit_rmse_60.median():.1f} mg/dL")


def _patient_frame(df: pd.DataFrame, static_row: pd.Series, prow: pd.Series, ehr_row: pd.Series) -> pd.DataFrame:
    prior = twin.PatientTwin(ehr_row.patient_id, twin.prior_params(ehr_row))
    fr = F.build_patient_frame(df, static_row, twin.PatientTwin.from_row(prow), onboarding_twin=prior,
                               switch_step=C.CALIBRATION_DAYS * C.STEPS_PER_DAY)
    return fr[fr.day >= C.FEATURE_START_DAY]


def stage_features() -> None:
    ehr = pd.read_csv(C.RAW_DIR / "ehr.csv")
    wear = pd.read_parquet(C.RAW_DIR / "wearables.parquet")
    params = pd.read_csv(C.MODELS_DIR / "twin_params.csv").set_index("patient_id", drop=False)
    static = F.static_features(ehr)
    ehr_idx = ehr.set_index("patient_id", drop=False)
    split = M.split_patients(ehr.patient_id.tolist())
    pd.Series(split, name="split").rename_axis("patient_id").to_csv(C.PROCESSED_DIR / "splits.csv")
    log("features: building causal feature frames")
    frames = Parallel(n_jobs=-1)(
        delayed(_patient_frame)(g, static.loc[pid], params.loc[pid], ehr_idx.loc[pid])
        for pid, g in wear.groupby("patient_id", sort=False))
    allf = pd.concat(frames, ignore_index=True)
    allf["split"] = allf.patient_id.map(split)
    for s in ("train", "val", "test"):
        part = allf[allf.split == s].drop(columns="split")
        part.to_parquet(C.PROCESSED_DIR / f"features_{s}.parquet", index=False)
        log(f"features: {s:5s} {part.patient_id.nunique():3d} patients, {len(part):7d} rows, "
            f"spike rate {part.y.mean():.3f}")


def stage_train() -> None:
    tr, va, te = (pd.read_parquet(C.PROCESSED_DIR / f"features_{s}.parquet") for s in ("train", "val", "test"))
    log("train: ablations, final model, forecasters (test = unseen patients)")
    bundle, metrics = M.train_all(tr, va, te, log=lambda m: log(m))
    joblib.dump(bundle, C.MODELS_DIR / "glucotwin_bundle.joblib", compress=3)
    M.save_metrics(metrics)

    log("train: scoring every patient for the dashboard")
    split = pd.read_csv(C.PROCESSED_DIR / "splits.csv").set_index("patient_id")["split"]
    preds = []
    for df in (tr, va, te):
        out = df[["patient_id", "timestamp", "day", "cgm", "y", "twin_g30", "twin_g60", "twin_g120"]].copy()
        out["p_spike"] = bundle.risk.predict_proba(df).astype(np.float32)
        fc = bundle.forecast(df)
        for c in fc:
            out[c] = fc[c].astype(np.float32)
        out["alert"] = (out.p_spike >= bundle.risk.threshold) & df.y.notna().to_numpy()
        preds.append(out)
    pred = pd.concat(preds, ignore_index=True)
    pred["split"] = pred.patient_id.map(split)
    pred.to_parquet(C.PROCESSED_DIR / "predictions.parquet", index=False)


def stage_report() -> None:
    from glucotwin import reporting

    log("report: figures")
    reporting.make_all_figures()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="regenerate synthetic data")
    ap.add_argument("--from", dest="start", choices=STAGES, default="simulate")
    args = ap.parse_args()
    C.ensure_dirs()
    t0 = time.time()
    for st in STAGES[STAGES.index(args.start):]:
        if st == "simulate":
            stage_simulate(args.fresh)
        else:
            globals()[f"stage_{st}"]()
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
