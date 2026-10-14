"""Replay a synthetic patient's wearable stream into the GlucoTwin API, as a device gateway would.

    uvicorn glucotwin.api:app --port 8000          # terminal 1
    python scripts/replay_stream.py --patient GT0007 --speed 0.2   # terminal 2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
import pandas as pd  # noqa: E402

from glucotwin import config as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="GT0007")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--start-day", type=int, default=0)
    ap.add_argument("--days", type=float, default=10)
    ap.add_argument("--chunk", type=int, default=12, help="5-min records per POST (12 = 1 hour)")
    ap.add_argument("--speed", type=float, default=0.0, help="seconds to sleep between POSTs")
    args = ap.parse_args()

    wear = pd.read_parquet(C.RAW_DIR / "wearables.parquet", filters=[("patient_id", "==", args.patient)])
    s0 = args.start_day * C.STEPS_PER_DAY
    s1 = s0 + int(args.days * C.STEPS_PER_DAY)
    rows = wear.iloc[s0:s1].drop(columns="patient_id")
    rows["timestamp"] = rows.timestamp.dt.strftime("%Y-%m-%dT%H:%M:%S")
    records = rows.astype(object).where(rows.notna(), None).to_dict("records")

    with httpx.Client(base_url=args.api, timeout=30) as client:
        print(client.get("/health").json())
        for i in range(0, len(records), args.chunk):
            client.post(f"/patients/{args.patient}/observations", json=records[i:i + args.chunk]).raise_for_status()
            if (i // args.chunk) % 3 == 0 or i + args.chunk >= len(records):
                a = client.get(f"/patients/{args.patient}/assessment").json()
                flag = "ALERT" if a["alert"] else "     "
                cgm = "  - " if a["cgm"] is None else f"{a['cgm']:4.0f}"
                top = a["drivers"][0]["text"] if a.get("drivers") else ""
                f60 = a["forecast"]["60"]["glucose"]
                f60 = "  - " if f60 is None else f"{f60:4.0f}"
                print(f"{a['timestamp'][5:16]}  CGM {cgm}  risk {a['spike_probability_2h']:5.1%}  {flag}  "
                      f"+60m {f60}  | {top}")
            if args.speed:
                time.sleep(args.speed)


if __name__ == "__main__":
    main()
