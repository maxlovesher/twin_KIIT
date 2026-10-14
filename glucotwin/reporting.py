"""Static figures for the README, presentation and architecture docs (matplotlib → PNG)."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402
from .features import H  # noqa: E402
from .modeling import FINAL, notifications, onsets  # noqa: E402

FIG_DIR = C.REPORTS_DIR / "figures"

# Reference palette (validated categorical order) + chrome
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"
CRITICAL = "#d03b3b"
NEUTRAL_BAR = "#c3c2b7"
RANGE_FILL = "#cde2fb"


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
        "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
        "legend.labelcolor": INK2, "lines.linewidth": 2,
    })


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _metrics() -> dict:
    return json.loads((C.REPORTS_DIR / "metrics.json").read_text())


# --------------------------------------------------------------------------------------
def fig_cohort_validation() -> None:
    ehr = pd.read_csv(C.RAW_DIR / "ehr.csv")
    wear = pd.read_parquet(C.RAW_DIR / "wearables.parquet", columns=["patient_id", "cgm"])
    g = wear.groupby("patient_id").cgm
    gmi = 3.31 + 0.02392 * g.mean()
    tir = g.apply(lambda x: ((x >= 70) & (x <= 180)).mean() * 100)
    a1c = ehr.set_index("patient_id").hba1c_pct.loc[gmi.index]

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    r = np.corrcoef(a1c, gmi)[0, 1]
    ax[0].scatter(a1c, gmi, s=28, color=SERIES[0], edgecolor=SURFACE, linewidth=1.2)
    lo, hi = min(a1c.min(), gmi.min()) - 0.2, max(a1c.max(), gmi.max()) + 0.2
    ax[0].plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
    ax[0].set(xlabel="HbA1c in EHR (%)", ylabel="GMI from simulated CGM (%)",
              title=f"EHR labs agree with wearable stream (r = {r:.2f})")
    ax[1].hist(tir, bins=np.arange(0, 105, 5), color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    ax[1].axvline(70, color=MUTED, lw=1, ls="--")
    ax[1].text(71, ax[1].get_ylim()[1] * 0.92, "ADA target ≥ 70%", color=INK2, fontsize=9)
    ax[1].set(xlabel="Time in range 70–180 mg/dL (%)", ylabel="Patients",
              title=f"Cohort glycaemic control (median TIR {tir.median():.0f}%)")
    fig.tight_layout()
    _save(fig, "cohort_validation.png")


def fig_twin_recovery() -> None:
    fit = pd.read_csv(C.MODELS_DIR / "twin_params.csv").set_index("patient_id")
    tru = pd.read_csv(C.RAW_DIR / "true_params.csv").set_index("patient_id")
    items = [("SI", "Insulin sensitivity"), ("beta", "β-cell response"), ("Gb", "Basal glucose")]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
    for a, (k, label) in zip(ax, items):
        x, y = np.log(tru.loc[fit.index, f"true_{k}"]), np.log(fit[k])
        r = np.corrcoef(x, y)[0, 1]
        a.scatter(np.exp(x), np.exp(y), s=22, color=SERIES[0], edgecolor=SURFACE, linewidth=1.0)
        lo, hi = np.exp(min(x.min(), y.min())), np.exp(max(x.max(), y.max()))
        a.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
        a.set_xscale("log")
        a.set_yscale("log")
        a.set(title=f"{label}  (r = {r:.2f})", xlabel="Hidden true value", ylabel="Twin estimate")
    fig.suptitle("Digital-twin calibration recovers each patient's physiology from 7 days of data",
                 x=0.01, ha="left", fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout()
    _save(fig, "twin_recovery.png")


def fig_ablation() -> None:
    m = _metrics()["ablations"]
    order = sorted(m.keys(), key=lambda n: m[n]["auroc"])
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    cols = [("auroc", "AUROC (unseen patients)", "{:.3f}"),
            ("event_sensitivity", "Spikes caught ≤120 min ahead", "{:.0%}"),
            ("false_alerts_per_day", "False alerts per patient-day", "{:.2f}")]
    for a, (k, title, fmt) in zip(ax, cols):
        vals = [m[n][k] for n in order]
        colors = [SERIES[0] if n == FINAL else NEUTRAL_BAR for n in order]
        bars = a.barh(order, vals, color=colors, height=0.62)
        for b, v in zip(bars, vals):
            a.text(b.get_width(), b.get_y() + b.get_height() / 2, " " + fmt.format(v),
                   va="center", color=INK2, fontsize=9)
        a.set_title(title)
        a.grid(axis="y", visible=False)
        a.set_xlim(0.5 if k == "auroc" else 0, 1.0 if k == "auroc" else max(vals) * 1.25)
    ax[0].tick_params(axis="y", colors=INK2)
    fig.tight_layout()
    _save(fig, "ablation.png")


def fig_roc_pr() -> None:
    curves = _metrics()["curves"]
    show = ["Static EHR + time of day", "CGM only", "Fusion: EHR + wearables", FINAL]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i, name in enumerate(show):
        c = curves[name]
        ax[0].plot(c["fpr"], c["tpr"], color=SERIES[i], label=name)
        ax[1].plot(c["recall"], c["precision"], color=SERIES[i], label=name)
    ax[0].plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall")
    ax[1].legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    _save(fig, "roc_pr.png")


def fig_lead_time() -> None:
    leads = np.array(_metrics()["final"]["lead_times"])
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(leads, bins=np.arange(0, 125, 10), color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    med = float(np.median(leads))
    ax.axvline(med, color=INK2, lw=1.2, ls="--")
    ax.text(med + 2, ax.get_ylim()[1] * 0.9, f"median {med:.0f} min", color=INK2)
    ax.set(xlabel="Warning time before glucose crossed 180 mg/dL (min)", ylabel="Caught spikes",
           title="How early does GlucoTwin warn?")
    fig.tight_layout()
    _save(fig, "lead_time.png")


def fig_importance() -> None:
    imp = pd.Series(_metrics()["group_importance"]).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.barh(imp.index, imp.values, color=SERIES[0], height=0.6)
    for b, v in zip(bars, imp.values):
        ax.text(max(v, 0), b.get_y() + b.get_height() / 2, f" {v:.3f}", va="center", color=INK2, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", colors=INK2)
    ax.set(xlabel="Drop in AUROC when the group is shuffled", title="Which data streams matter?")
    ax.set_xlim(0, imp.max() * 1.25)
    fig.tight_layout()
    _save(fig, "group_importance.png")


def fig_calibration_forecast() -> None:
    m = _metrics()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    cal = m["calibration"]
    ax[0].plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax[0].plot(cal["mean_pred"], cal["frac_pos"], color=SERIES[0], marker="o", ms=7,
               markeredgecolor=SURFACE, markeredgewidth=2)
    ax[0].set(xlabel="Predicted spike probability", ylabel="Observed spike rate",
              title=f"Calibration (Brier {m['final']['brier']:.3f})")
    fc = m["forecast"]
    hs = list(fc.keys())
    models = ["Persistence (no change)", "Mechanistic twin", "GlucoTwin hybrid"]
    w = 0.26
    x = np.arange(len(hs))
    for i, name in enumerate(models):
        vals = [fc[h][name]["rmse"] for h in hs]
        ax[1].bar(x + (i - 1) * w, vals, width=w - 0.03, color=SERIES[i], label=name)
    ax[1].set_xticks(x, [f"+{h} min" for h in hs])
    ax[1].grid(axis="x", visible=False)
    ax[1].set(ylabel="RMSE (mg/dL)", title="Glucose forecast error (lower is better)")
    ax[1].legend(fontsize=8.5)
    fig.tight_layout()
    _save(fig, "calibration_forecast.png")


def fig_example_patient() -> None:
    pred = pd.read_parquet(C.PROCESSED_DIR / "predictions.parquet")
    test = pred[pred.split == "test"].copy()
    test["date"] = test.timestamp.dt.date
    # an illustrative day: mostly in range, several spikes, every spike warned about, few alarms
    rows = []
    for (pid_, date_), g_ in test.groupby(["patient_id", "date"]):
        cg = g_.cgm.to_numpy(np.float64)
        on = onsets(cg)
        nt = notifications(g_.alert.to_numpy())
        caught = sum(any((o - H <= n < o) for n in nt) for o in on if o >= H)
        false = int(np.sum(g_.y.to_numpy()[nt] == 0)) if len(nt) else 0
        tir = np.nanmean((cg >= 70) & (cg <= 180))
        rows.append((pid_, date_, caught - 2 * false + 2 * (tir > 0.6), len(on)))
    pid, date, *_ = max(rows, key=lambda r: (r[2], r[3]))
    g = test[(test.patient_id == pid) & (test.date == date)]
    fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    ax[0].axhspan(70, 180, color=RANGE_FILL, alpha=0.5, lw=0)
    ax[0].plot(g.timestamp, g.cgm, color=SERIES[0], label="CGM glucose")
    ax[0].axhline(180, color=MUTED, lw=1, ls="--")
    al = g.iloc[notifications(g.alert.to_numpy())]
    ax[0].scatter(al.timestamp, al.cgm, s=60, color=CRITICAL, zorder=3, label="Spike alert sent",
                  edgecolor=SURFACE, linewidth=1.5)
    ax[0].set(ylabel="mg/dL", title=f"Unseen test patient {pid}, {date}")
    ax[0].legend(loc="upper left", fontsize=9)
    ax[1].fill_between(g.timestamp, g.p_spike, color=SERIES[1], alpha=0.25, lw=0)
    ax[1].plot(g.timestamp, g.p_spike, color=SERIES[1])
    thr = _metrics()["final"]["threshold"]
    ax[1].axhline(thr, color=CRITICAL, lw=1, ls="--")
    ax[1].text(g.timestamp.iloc[0], thr + 0.03, f"alert threshold {thr:.0%}", color=INK2, fontsize=8.5)
    ax[1].set(ylabel="P(spike in 2 h)", ylim=(0, 1))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.tight_layout()
    _save(fig, "example_patient.png")


def fig_by_history() -> None:
    bh = _metrics()["by_history"]
    buckets = list(bh.keys())
    show = ["CGM only", "Wearables only (dynamic)", "Fusion: EHR + wearables", FINAL]
    labels = ["CGM only", "Wearables only", "EHR + wearables", "GlucoTwin (+ twin)"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.9))
    w = 0.2
    x = np.arange(len(buckets))
    for i, (name, lab) in enumerate(zip(show, labels)):
        for a, key in zip(ax, ("auroc", "false_alerts_per_day")):
            vals = [bh[b][name][key] for b in buckets]
            a.bar(x + (i - 1.5) * w, vals, width=w - 0.025, color=SERIES[i], label=lab)
    ax[0].set(ylim=(0.8, 0.97), ylabel="AUROC", title="Discrimination by monitoring history")
    ax[1].set(ylabel="False alerts / patient-day", title="False alarms by monitoring history")
    for a in ax:
        a.set_xticks(x, buckets, fontsize=9)
        a.grid(axis="x", visible=False)
    ax[1].legend(fontsize=8.5)
    fig.tight_layout()
    _save(fig, "by_history.png")


def make_all_figures() -> None:
    _style()
    for fn in (fig_by_history, fig_cohort_validation, fig_twin_recovery, fig_ablation, fig_roc_pr, fig_lead_time,
               fig_importance, fig_calibration_forecast, fig_example_patient):
        fn()
