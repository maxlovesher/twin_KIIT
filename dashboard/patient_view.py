"""Patient-facing phone screen ("your next 2 hours"), rendered as self-contained HTML/SVG.

Clinicians see the full console; the patient sees one number, one prediction, one reason and one action.
English and Hindi strings; dynamic values are filled from the same engine outputs the clinician sees.
"""
from __future__ import annotations

import html

T = {
    "en": {
        "hello": "Good to see you", "glucose": "Your glucose", "in": "In range", "above": "Above range",
        "below": "Below range", "spike": "Glucose likely to go HIGH in the next 2 hours",
        "low": "Glucose may go LOW in the next 2 hours", "ok": "All good for the next 2 hours",
        "why": "Why", "do": "What you can do", "walk": "Take a {m}-minute brisk walk now",
        "walk_effect": "Your twin predicts the peak drops from {a} to {b} mg/dL",
        "eat_low": "Keep 15 g of fast sugar handy (glucose tablets, juice) and re-check in 15 min",
        "stay": "Keep to your usual plan", "next2h": "Last 3 h and next 2 h", "today": "Today",
        "tir": "In range", "steps": "Steps", "sleep": "Sleep last night", "btn_ok": "I'll do it",
        "btn_later": "Remind me in 15 min", "disc": "Decision support only. Follow your doctor's advice.",
        "prob": "chance",
        "reasons": {
            "Meals (logged)": "You had a meal recently", "Personal routine": "You often spike around this time",
            "Glucose trend (CGM)": "Your glucose is rising", "Sleep": "You slept less than usual",
            "Heart rate & HRV": "Signs of stress from your watch", "Activity": "You've been sitting a while",
            "Digital twin": "Food is still being absorbed", "Clinical (EHR)": "Your health profile",
            "Genetics": "Your family/genetic risk", "Time of day": "Time of day",
        },
    },
    "hi": {
        "hello": "नमस्ते", "glucose": "आपका ग्लूकोज़", "in": "सामान्य सीमा में", "above": "सीमा से ऊपर",
        "below": "सीमा से नीचे", "spike": "अगले 2 घंटों में ग्लूकोज़ बहुत बढ़ सकता है",
        "low": "अगले 2 घंटों में ग्लूकोज़ कम हो सकता है", "ok": "अगले 2 घंटे सब ठीक दिख रहा है",
        "why": "क्यों", "do": "आप क्या कर सकते हैं", "walk": "अभी {m} मिनट तेज़ चलें",
        "walk_effect": "आपके ट्विन के अनुसार अधिकतम स्तर {a} से घटकर {b} mg/dL होगा",
        "eat_low": "15 ग्राम जल्दी असर करने वाली शक्कर पास रखें (ग्लूकोज़ गोली, जूस) और 15 मिनट बाद जाँचें",
        "stay": "अपनी सामान्य दिनचर्या जारी रखें", "next2h": "पिछले 3 घंटे और अगले 2 घंटे", "today": "आज",
        "tir": "सामान्य सीमा में", "steps": "क़दम", "sleep": "पिछली रात की नींद", "btn_ok": "ठीक है",
        "btn_later": "15 मिनट बाद याद दिलाएँ", "disc": "यह केवल सहायता के लिए है। अपने डॉक्टर की सलाह मानें।",
        "prob": "संभावना",
        "reasons": {
            "Meals (logged)": "आपने हाल ही में भोजन किया है", "Personal routine": "इस समय अक्सर आपका ग्लूकोज़ बढ़ता है",
            "Glucose trend (CGM)": "आपका ग्लूकोज़ बढ़ रहा है", "Sleep": "आप सामान्य से कम सोए",
            "Heart rate & HRV": "घड़ी में तनाव के संकेत", "Activity": "आप काफ़ी देर से बैठे हैं",
            "Digital twin": "भोजन अभी पच रहा है", "Clinical (EHR)": "आपकी स्वास्थ्य प्रोफ़ाइल",
            "Genetics": "पारिवारिक/आनुवंशिक जोखिम", "Time of day": "दिन का समय",
        },
    },
}


def _spark(history: list[float], forecast: list[tuple[int, float]], w: int = 300, h: int = 110) -> str:
    """SVG: last 3 h of CGM (36 points) + forecast markers at +30/+60/+120 min, with the 70-180 band."""
    total = 36 + 24                                      # 3 h history + 2 h ahead, 5-min steps
    vals = [v for v in history if v == v] + [v for _, v in forecast if v == v]
    lo, hi = 50.0, max([260.0] + [v + 15 for v in vals])

    def xy(i: float, v: float) -> tuple[float, float]:
        return 8 + i / total * (w - 16), h - 8 - (v - lo) / (hi - lo) * (h - 16)

    band_top, band_bot = xy(0, 180)[1], xy(0, 70)[1]
    pts = " ".join(f"{xy(i, v)[0]:.1f},{xy(i, v)[1]:.1f}" for i, v in enumerate(history) if v == v)
    n0 = len(history) - 1
    now_x = xy(n0, lo)[0]
    last = next((v for v in reversed(history) if v == v), None)
    fc = [(m, v) for m, v in forecast if v == v]
    fc_pts = ([f"{xy(n0, last)[0]:.1f},{xy(n0, last)[1]:.1f}"] if last is not None else []) + \
             [f"{xy(n0 + m / 5, v)[0]:.1f},{xy(n0 + m / 5, v)[1]:.1f}" for m, v in fc]
    dots = "".join(f'<circle cx="{xy(n0 + m / 5, v)[0]:.1f}" cy="{xy(n0 + m / 5, v)[1]:.1f}" r="4" '
                   f'fill="#eb6834" stroke="#fff" stroke-width="2"/>' for m, v in fc)
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" role="img" aria-label="glucose chart">'
            f'<rect x="8" y="{band_top:.1f}" width="{w - 16}" height="{band_bot - band_top:.1f}" fill="#2a78d6" opacity="0.08"/>'
            f'<line x1="8" x2="{w - 8}" y1="{band_top:.1f}" y2="{band_top:.1f}" stroke="#898781" stroke-dasharray="3 3"/>'
            f'<line x1="{now_x:.1f}" x2="{now_x:.1f}" y1="6" y2="{h - 6}" stroke="#c3c2b7"/>'
            f'<polyline points="{pts}" fill="none" stroke="#2a78d6" stroke-width="2.5" stroke-linejoin="round"/>'
            f'<polyline points="{" ".join(fc_pts)}" fill="none" stroke="#eb6834" stroke-width="2" stroke-dasharray="4 3"/>'
            f'{dots}</svg>')


def render_phone(s: dict, lang: str = "en") -> str:
    t = T[lang]
    cgm = s.get("cgm")
    roc = s.get("roc") or 0.0
    arrow = ("⇈" if roc > 2 else "↑" if roc > 1 else "↗" if roc > 0.5 else "⇊" if roc < -2 else "↓" if roc < -1
             else "↘" if roc < -0.5 else "→")
    if cgm is None:
        status, scol = "—", "#898781"
    elif cgm > 180:
        status, scol = t["above"], "#c2410c"
    elif cgm < 70:
        status, scol = t["below"], "#d03b3b"
    else:
        status, scol = t["in"], "#0f7a3d"

    if s.get("low_alert"):
        head, color, bg, prob = t["low"], "#9a3412", "#fff7ed", s.get("p_low", 0.0)
        action = f'<div class="act">🍬 {html.escape(t["eat_low"])}</div>'
        reasons = []
    elif s.get("alert"):
        head, color, bg, prob = t["spike"], "#b91c1c", "#fef2f2", s.get("p_spike", 0.0)
        sg = s.get("suggestion")
        effect = (f'<div class="sub">{html.escape(t["walk_effect"].format(a=round(sg["peak_without"]), b=round(sg["peak_with"])))}</div>'
                  if sg else "")
        action = f'<div class="act">🚶 {html.escape(t["walk"].format(m=15))}{effect}</div>'
        reasons = [g for g, c in s.get("drivers", []) if c > 0][:2]
    else:
        head, color, bg, prob = t["ok"], "#166534", "#f0fdf4", 0.0
        action = f'<div class="act">✅ {html.escape(t["stay"])}</div>'
        reasons = []
    why = "".join(f"<li>{html.escape(t['reasons'].get(g, g))}</li>" for g in reasons)
    why_block = f'<div class="why"><b>{t["why"]}</b><ul>{why}</ul></div>' if why else ""
    alerting = bool(s.get("alert") or s.get("low_alert"))
    buttons = (f'<div class="btns"><span class="b1">{t["btn_ok"]}</span><span class="b2">{t["btn_later"]}</span></div>'
               if alerting else "")
    pct = f'{round(prob * 100)}% {t["prob"]}' if alerting else ""
    sleep = s.get("sleep_h")
    sleep_txt = "—" if sleep is None or sleep != sleep else f"{sleep:.1f} h"
    cgm_txt = "—" if cgm is None else f"{cgm:.0f}"

    return f"""
<style>
  body {{ margin: 0; background: transparent; }}
  .phone {{ width: 340px; margin: 4px auto; border: 10px solid #111; border-radius: 38px; background: #fcfcfb;
           font-family: system-ui, -apple-system, 'Segoe UI', 'Nirmala UI', 'Noto Sans Devanagari', sans-serif;
           color: #0b0b0b; box-shadow: 0 12px 30px rgba(0,0,0,.18); overflow: hidden; }}
  .notch {{ width: 120px; height: 22px; background: #111; border-radius: 0 0 14px 14px; margin: 0 auto 6px; }}
  .wrap {{ padding: 6px 18px 18px; }}
  .hello {{ color: #52514e; font-size: 14px; }}
  .big {{ display: flex; align-items: baseline; gap: 8px; margin: 2px 0; }}
  .num {{ font-size: 58px; font-weight: 700; letter-spacing: -1px; }}
  .unit {{ color: #52514e; font-size: 14px; }} .arr {{ font-size: 30px; color: #2a78d6; }}
  .pill {{ display: inline-block; font-size: 12px; font-weight: 600; padding: 3px 10px; border-radius: 99px;
          color: {scol}; background: {scol}1a; }}
  .card {{ margin: 14px 0 10px; padding: 12px 14px; border-radius: 16px; background: {bg}; border: 1px solid {color}33; }}
  .card h3 {{ margin: 0 0 4px; font-size: 16px; color: {color}; line-height: 1.3; }}
  .pct {{ font-size: 12px; color: #52514e; }}
  .why ul {{ margin: 4px 0 0 18px; padding: 0; font-size: 13px; color: #333; }} .why b {{ font-size: 12px; color: #52514e; }}
  .act {{ margin-top: 4px; font-size: 14px; font-weight: 600; }}
  .sub {{ font-weight: 400; font-size: 12px; color: #52514e; margin-top: 2px; }}
  .btns {{ display: flex; gap: 8px; margin-top: 10px; }}
  .btns span {{ flex: 1; text-align: center; font-size: 12px; padding: 8px 4px; border-radius: 10px; }}
  .b1 {{ background: #2a78d6; color: #fff; font-weight: 600; }} .b2 {{ background: #eceae4; color: #333; }}
  .lbl {{ font-size: 12px; color: #52514e; margin-top: 8px; }}
  .stats {{ display: flex; gap: 8px; margin-top: 6px; }}
  .stat {{ flex: 1; background: #f3f2ee; border-radius: 12px; padding: 8px; text-align: center; }}
  .stat b {{ display: block; font-size: 17px; }} .stat span {{ font-size: 11px; color: #52514e; }}
  .disc {{ font-size: 10.5px; color: #898781; margin-top: 12px; text-align: center; }}
</style>
<div class="phone"><div class="notch"></div><div class="wrap">
  <div class="hello">{t["hello"]} 👋 · {html.escape(s.get("time", ""))}</div>
  <div class="lbl">{t["glucose"]}</div>
  <div class="big"><span class="num">{cgm_txt}</span><span class="unit">mg/dL</span><span class="arr">{arrow}</span></div>
  <span class="pill">{status}</span>
  <div class="card"><h3>{head}</h3><div class="pct">{pct}</div>{why_block}
    <div class="lbl" style="margin-top:8px">{t["do"]}</div>{action}{buttons}</div>
  <div class="lbl">{t["next2h"]}</div>
  {_spark(s.get("history", []), s.get("forecast", []))}
  <div class="lbl">{t["today"]}</div>
  <div class="stats">
    <div class="stat"><b>{round(s.get("tir_today", 0) * 100)}%</b><span>{t["tir"]}</span></div>
    <div class="stat"><b>{int(s.get("steps_today", 0)):,}</b><span>{t["steps"]}</span></div>
    <div class="stat"><b>{sleep_txt}</b><span>{t["sleep"]}</span></div>
  </div>
  <div class="disc">{t["disc"]}</div>
</div></div>
"""
