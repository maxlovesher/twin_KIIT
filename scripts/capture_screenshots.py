"""Capture dashboard screenshots for the docs with headless Edge/Chrome over the DevTools protocol.

    streamlit run dashboard/app.py          # terminal 1
    python scripts/capture_screenshots.py   # terminal 2  -> docs/dashboard_*.png
"""
from __future__ import annotations

import asyncio
import base64
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from tornado.websocket import websocket_connect

ROOT = Path(__file__).resolve().parent.parent
URL = "http://localhost:8501"
PORT = 9333
# (tab, file, viewport height, optional deep-link query)
SHOTS = [("Ward overview", "dashboard_ward.png", 1000, ""), ("Virtual patient", "dashboard_patient.png", 1350, ""),
         ("What-if simulator", "dashboard_whatif.png", 1150, "?pid=GT0014&t=2026-08-12T11:30"),
         ("Model performance", "dashboard_performance.png", 1000, ""),
         ("Patient app", "dashboard_phone.png", 1000, "?pid=GT0014&t=2026-08-12T06:45"),
         ("Twin overview", "dashboard_body.png", 1080, "?pid=GT0014&t=2026-08-12T06:45")]
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe", "msedge", "google-chrome", "chromium"]


class CDP:
    def __init__(self, ws):
        self.ws, self.i = ws, 0

    async def call(self, method, **params):
        self.i += 1
        await self.ws.write_message(json.dumps({"id": self.i, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.read_message())
            if msg.get("id") == self.i:
                return msg.get("result", {})


async def main() -> None:
    exe = next((b for b in BROWSERS if Path(b).exists() or shutil.which(b)), None)
    if exe is None:
        raise SystemExit("Edge/Chrome not found")
    profile = tempfile.mkdtemp()
    proc = subprocess.Popen([exe, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
                             "--hide-scrollbars", "--window-size=1600,1000", "about:blank"])
    try:
        pages = []
        for _ in range(50):
            try:
                pages = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
                break
            except OSError:
                time.sleep(0.2)
        ws_url = next(p["webSocketDebuggerUrl"] for p in pages if p["type"] == "page")
        cdp = CDP(await websocket_connect(ws_url, max_message_size=200 * 1024 * 1024))
        await cdp.call("Page.enable")
        await cdp.call("Emulation.setDeviceMetricsOverride", width=1600, height=1000, deviceScaleFactor=1.5, mobile=False)
        current = None
        for tab, name, height, query in SHOTS:
            if query != current:
                await cdp.call("Page.navigate", url=URL + query)
                await asyncio.sleep(20)
                current = query
            await cdp.call("Emulation.setDeviceMetricsOverride", width=1600, height=height, deviceScaleFactor=1.5,
                           mobile=False)
            await cdp.call("Runtime.evaluate", expression=(
                "[...document.querySelectorAll('button[role=tab]')]"
                f".find(b => b.innerText.includes('{tab}')).click()"))
            await asyncio.sleep(6)
            shot = await cdp.call("Page.captureScreenshot", format="png")
            (ROOT / "docs" / name).write_bytes(base64.b64decode(shot["data"]))
            print("saved docs/" + name)
    finally:
        proc.terminate()


def crop_phone(path: Path, margin: int = 24) -> None:
    """Crop the patient-app screenshot to the phone frame (its near-black border)."""
    import numpy as np
    from PIL import Image

    img = Image.open(path).convert("RGB")
    a = np.asarray(img)
    dark = (a.max(axis=2) < 40)
    # the phone's side borders are long vertical dark lines; text never is
    cols = np.flatnonzero(dark.sum(axis=0) > 0.3 * img.height)
    if len(cols):
        col = dark[:, cols[0] + 2].astype(int)              # the left border: the longest unbroken dark run
        edges = np.flatnonzero(np.diff(np.r_[0, col, 0]))
        starts, ends = edges[0::2], edges[1::2]
        k = int(np.argmax(ends - starts))
        corner = int(0.12 * (cols[-1] - cols[0]))            # rounded corners extend beyond the straight edge
        rows = np.array([max(starts[k] - corner, 0), min(ends[k] + corner, img.height - 1)])
    else:
        rows = np.array([], dtype=int)
    if len(cols) and len(rows):
        box = (max(cols[0] - margin, 0), max(rows[0] - margin, 0),
               min(cols[-1] + margin, img.width), min(rows[-1] + margin, img.height))
        img.crop(box).save(path)


if __name__ == "__main__":
    asyncio.run(main())
    if (ROOT / "docs" / "dashboard_phone.png").exists():
        crop_phone(ROOT / "docs" / "dashboard_phone.png")
