"""Bring real-world data in and send predictions out.

    # Synthea EHR export -> GlucoTwin EHR table (and optionally onboard the patients via the API)
    python scripts/interop_cli.py synthea path/to/synthea/csv --out ehr_synthea.csv [--register]

    # Apple Health export.xml -> 5-min stream (and optionally stream it to a patient via the API)
    python scripts/interop_cli.py apple-health export.xml --patient SY-1a2b3c4d --out stream.csv [--post]

    # FHIR R4 Bundle (record + latest RiskAssessment) for a patient
    python scripts/interop_cli.py fhir GT0007 --out GT0007_bundle.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
import pandas as pd  # noqa: E402

from glucotwin import interop  # noqa: E402


def _clean(rec: dict) -> dict:
    out = {}
    for k, v in rec.items():
        if hasattr(v, "item"):
            v = v.item()
        if v is None or (isinstance(v, float) and v != v):
            continue
        out[k] = v
    return out


def cmd_synthea(a) -> None:
    ehr = interop.ehr_from_synthea(a.csv_dir, as_of=a.as_of)
    ehr.to_csv(a.out, index=False)
    print(f"{len(ehr)} patients with type 2 diabetes -> {a.out}")
    if a.register:
        with httpx.Client(base_url=a.api, timeout=60) as c:
            for rec in ehr.to_dict("records"):
                c.post("/patients", json=_clean(rec)).raise_for_status()
                print("registered", rec["patient_id"])


def cmd_apple(a) -> None:
    stream = interop.wearables_from_apple_health(a.xml, start=a.start, end=a.end)
    if a.out:
        stream.to_csv(a.out, index=False)
    print(f"{len(stream)} five-minute rows, {stream.cgm.notna().sum()} with glucose, "
          f"{stream.timestamp.min()} -> {stream.timestamp.max()}")
    if a.post:
        rows = stream.assign(timestamp=stream.timestamp.dt.strftime("%Y-%m-%dT%H:%M:%S"))
        recs = rows.astype(object).where(rows.notna(), None).to_dict("records")
        with httpx.Client(base_url=a.api, timeout=60) as c:
            for i in range(0, len(recs), 288):
                c.post(f"/patients/{a.patient}/observations", json=recs[i:i + 288]).raise_for_status()
            print(json.dumps(c.get(f"/patients/{a.patient}/assessment").json(), indent=2)[:1500])


def cmd_fhir(a) -> None:
    with httpx.Client(base_url=a.api, timeout=60) as c:
        r = c.get(f"/patients/{a.patient}/fhir")
        r.raise_for_status()
        bundle = r.json()
    Path(a.out).write_text(json.dumps(bundle, indent=2))
    kinds = pd.Series([e["resource"]["resourceType"] for e in bundle["entry"]]).value_counts().to_dict()
    print(f"FHIR Bundle with {len(bundle['entry'])} resources {kinds} -> {a.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synthea")
    s.add_argument("csv_dir")
    s.add_argument("--out", default="ehr_synthea.csv")
    s.add_argument("--as-of", default=None)
    s.add_argument("--register", action="store_true")
    s.set_defaults(fn=cmd_synthea)
    h = sub.add_parser("apple-health")
    h.add_argument("xml")
    h.add_argument("--patient", required=False)
    h.add_argument("--out", default=None)
    h.add_argument("--start", default=None)
    h.add_argument("--end", default=None)
    h.add_argument("--post", action="store_true")
    h.set_defaults(fn=cmd_apple)
    f = sub.add_parser("fhir")
    f.add_argument("patient")
    f.add_argument("--out", default=None)
    f.set_defaults(fn=cmd_fhir)
    a = ap.parse_args()
    if a.cmd == "apple-health" and a.post and not a.patient:
        ap.error("--post needs --patient")
    if a.cmd == "fhir" and not a.out:
        a.out = f"{a.patient}_fhir_bundle.json"
    a.fn(a)


if __name__ == "__main__":
    main()
