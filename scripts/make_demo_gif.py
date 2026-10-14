"""Record an animated demo GIF of the dashboard for the README (headless Edge/Chrome + DevTools protocol).

    streamlit run dashboard/app.py        # terminal 1
    python scripts/make_demo_gif.py       # terminal 2  -> docs/demo.gif

Story: ward overview -> one held-out patient before, during and after a predicted spike -> the patient's
phone -> the what-if simulator. Captions match the data for patient GT0014 on 12 Aug (alert 06:45,
glucose crosses 180 at 07:50).
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from tornado.websocket import websocket_connect

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_screenshots import BROWSERS, CDP  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
URL = "http://localhost:8501"
PORT = 9334
W, H = 1440, 900
PID = "GT0014"
# (tab, deep-link query, caption, seconds on screen)
FRAMES = [
    ("Ward overview", f"?pid={PID}&t=2026-08-12T06:45", "Ward overview: every patient ranked by 2-hour spike risk", 2.6),
    ("Twin overview", f"?pid={PID}&t=2026-08-12T06:45", "Twin overview: every organ the twin models, live (sleep, pancreas, gut, ...)", 3.2),
    ("Virtual patient", f"?pid={PID}&t=2026-08-12T06:35", "06:35  Glucose in range, low risk", 2.0),
    ("Virtual patient", f"?pid={PID}&t=2026-08-12T06:45", "06:45  Meal logged: GlucoTwin alerts 'spike likely within 2 h' and says why", 3.0),
    ("Virtual patient", f"?pid={PID}&t=2026-08-12T07:30", "07:30  Glucose rising (159), still below 180", 2.0),
    ("Virtual patient", f"?pid={PID}&t=2026-08-12T07:55", "07:55  Glucose has crossed 180, about 65 minutes after the alert", 2.4),
    ("Patient app", f"?pid={PID}&t=2026-08-12T06:45", "The patient's phone: one number, one reason, one action (English / Hindi)", 3.0),
    ("What-if simulator", f"?pid={PID}&t=2026-08-12T11:30", "What-if: rice vs low-GI meal, with and without a 15-minute walk", 3.0),
]


async def capture() -> list[tuple[Image.Image, str, float]]:
    exe = next((b for b in BROWSERS if Path(b).exists() or shutil.which(b)), None)
    if exe is None:
        raise SystemExit("Edge/Chrome not found")
    proc = subprocess.Popen([exe, "--headless=new", f"--remote-debugging-port={PORT}",
                             f"--user-data-dir={tempfile.mkdtemp()}", "--hide-scrollbars", f"--window-size={W},{H}",
                             "about:blank"])
    shots = []
    try:
        pages = []
        for _ in range(50):
            try:
                pages = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
                break
            except OSError:
                time.sleep(0.2)
        ws = next(p["webSocketDebuggerUrl"] for p in pages if p["type"] == "page")
        cdp = CDP(await websocket_connect(ws, max_message_size=200 * 1024 * 1024))
        await cdp.call("Page.enable")
        await cdp.call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1, mobile=False)
        for tab, query, caption, secs in FRAMES:
            await cdp.call("Page.navigate", url=URL + query)
            await asyncio.sleep(14)
            await cdp.call("Runtime.evaluate", expression=(
                "[...document.querySelectorAll('button[role=tab]')]"
                f".find(b => b.innerText.includes('{tab}')).click()"))
            await asyncio.sleep(5)
            png = base64.b64decode((await cdp.call("Page.captureScreenshot", format="png"))["data"])
            shots.append((Image.open(io.BytesIO(png)).convert("RGB"), caption, secs))
            print("captured:", caption, flush=True)
    finally:
        proc.terminate()
    return shots


def captioned(img: Image.Image, text: str, width: int = 1100) -> Image.Image:
    img = img.resize((width, int(img.height * width / img.width)), Image.LANCZOS)
    bar = 46
    out = Image.new("RGB", (img.width, img.height + bar), "#0d366b")
    out.paste(img, (0, bar))
    try:
        font = ImageFont.truetype("segoeuib.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    ImageDraw.Draw(out).text((16, 12), text, fill="white", font=font)
    return out


def main() -> None:
    shots = asyncio.run(capture())
    frames = [captioned(img, cap) for img, cap, _ in shots]
    durations = [int(secs * 1000) for _, _, secs in shots]
    pal = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
    out = ROOT / "docs" / "demo.gif"
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
