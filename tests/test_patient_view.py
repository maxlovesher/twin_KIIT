"""Patient phone screen renders the right message in each state and language; low-glucose labels are correct."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dashboard"))

from patient_view import T, render_phone  # noqa: E402

from glucotwin.realdata import hypo_labels  # noqa: E402

BASE = {"time": "Wed 06:45", "cgm": 120.0, "roc": 0.8, "p_spike": 0.9, "alert": False, "p_low": 0.01,
        "low_alert": False, "drivers": [("Meals (logged)", 1.2), ("Sleep", -0.3)], "suggestion": None,
        "history": [110.0 + i for i in range(36)], "forecast": [(30, 150.0), (60, 190.0), (120, 170.0)],
        "tir_today": 0.75, "steps_today": 3200, "sleep_h": 6.4}


def test_spike_alert_message_reason_and_action():
    html = render_phone({**BASE, "alert": True, "suggestion": {"peak_without": 230.0, "peak_with": 212.0}}, "en")
    assert T["en"]["spike"] in html and "90% chance" in html
    assert T["en"]["reasons"]["Meals (logged)"] in html          # positive driver shown
    assert T["en"]["reasons"]["Sleep"] not in html                # driver that lowers risk is not a "why"
    assert "230" in html and "212" in html                        # twin-tested walk effect
    assert "<svg" in html


def test_low_alert_takes_priority_and_hindi():
    html = render_phone({**BASE, "alert": True, "low_alert": True, "p_low": 0.6}, "hi")
    assert T["hi"]["low"] in html and T["hi"]["spike"] not in html
    assert T["hi"]["eat_low"] in html


def test_all_clear_and_missing_values():
    html = render_phone({**BASE, "cgm": None, "sleep_h": float("nan"),
                         "history": [float("nan")] * 36}, "en")
    assert T["en"]["ok"] in html and "—" in html


def test_hypo_labels():
    s = pd.Series([100.0] * 30 + [65.0] * 10 + [100.0] * 30)
    y = hypo_labels(s).y_hypo
    assert y.iloc[5] == 0.0        # next 2 h (rows 6-29) stay at 100
    assert y.iloc[6] == 1.0        # row 30 drops to 65 within the window
    assert np.isnan(y.iloc[32])    # currently below 70 -> not eligible
