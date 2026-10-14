"""'Twin overview': an interactive human figure whose organs show the live state of the patient's twin.

Each organ maps to one part of the physiology the twin models (pancreas = beta-cell insulin output,
muscle = insulin sensitivity, liver = basal glucose output, gut = carbohydrate still being absorbed, ...).
Status colours follow the reserved status palette and always come with an icon + label.
"""
from __future__ import annotations

import html
import math

GOOD, WARN, SERIOUS, CRIT, INFO, NEUTRAL = "good", "warning", "serious", "critical", "info", "neutral"
COLOR = {GOOD: "#0ca30c", WARN: "#e3a008", SERIOUS: "#ec835a", CRIT: "#d03b3b", INFO: "#2a78d6", NEUTRAL: "#a8a69e"}
ICON = {GOOD: "✓", WARN: "!", SERIOUS: "▲", CRIT: "✕", INFO: "●", NEUTRAL: "–"}
LABEL = {GOOD: "Good", WARN: "Watch", SERIOUS: "Concern", CRIT: "Urgent", INFO: "Active", NEUTRAL: "No data"}


def _ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def _pct_status(p) -> str:
    if not _ok(p):
        return NEUTRAL
    return SERIOUS if p < 20 else WARN if p < 40 else GOOD


def _ordinal(p: float) -> str:
    n = int(round(p))
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def build_organs(v: dict) -> list[dict]:
    """Turn twin/feature values into organ cards: id, number, title, status, value, detail, meaning."""
    o = []
    cgm, roc = v.get("cgm"), v.get("roc") or 0.0
    arrow = "↑" if roc > 1 else "↗" if roc > 0.5 else "↓" if roc < -1 else "↘" if roc < -0.5 else "→"
    if not _ok(cgm):
        st, mean = NEUTRAL, "Sensor reading missing"
    elif cgm < 54:
        st, mean = CRIT, "Dangerously low"
    elif cgm < 70:
        st, mean = SERIOUS, "Below range"
    elif cgm <= 180:
        st, mean = GOOD, "In target range"
    elif cgm <= 250:
        st, mean = WARN, "Above range"
    else:
        st, mean = SERIOUS, "Well above range"
    o.append(dict(id="blood", title="Blood glucose · CGM sensor", status=st,
                  value=f"{cgm:.0f} mg/dL {arrow}" if _ok(cgm) else "—",
                  detail=f"Time in range (7 d): {v['tir7']:.0%}" if _ok(v.get("tir7")) else "", meaning=mean))

    bp = v.get("beta_pct")
    o.append(dict(id="pancreas", title="Pancreas · β-cells", status=_pct_status(bp),
                  value=f"{_ordinal(bp)} percentile insulin output" if _ok(bp) else "—",
                  detail=f"Insulin now: +{v.get('insulin_now', 0):.0f} µU/mL above basal"
                         + (" · sulfonylurea/DPP-4i boost" if v.get("secretagogue") else ""),
                  meaning="Makes less insulin than most patients" if _ok(bp) and bp < 40 else "Insulin output holding up"))

    sp = v.get("si_pct")
    steps = v.get("steps_1h")
    o.append(dict(id="muscle", title="Muscles · insulin sensitivity", status=_pct_status(sp),
                  value=f"{_ordinal(sp)} percentile sensitivity" if _ok(sp) else "—",
                  detail=(f"{steps:,.0f} steps in the last hour" if _ok(steps) else "No step data")
                         + (f" · exercise boost +{v['ex_boost']:.0f}%" if v.get("ex_boost", 0) >= 1 else ""),
                  meaning="Insulin-resistant: muscles take up glucose slowly" if _ok(sp) and sp < 40
                  else "Muscles respond reasonably to insulin"))

    gb = v.get("gb")
    st = NEUTRAL if not _ok(gb) else GOOD if gb < 130 else WARN if gb < 160 else SERIOUS
    o.append(dict(id="liver", title="Liver · glucose output", status=st,
                  value=f"Background level {gb:.0f} mg/dL" if _ok(gb) else "—",
                  detail=f"Dawn rise ≈ {v.get('dawn', 0):.0f} mg/dL/h" + (" · on metformin" if v.get("metformin") else ""),
                  meaning="Liver releases too much glucose between meals" if _ok(gb) and gb >= 130
                  else "Background glucose output near normal"))

    gut, since = v.get("gut_carbs") or 0.0, v.get("min_since_meal")
    st = INFO if gut >= 10 else NEUTRAL
    o.append(dict(id="gut", title="Gut · digestion", status=st,
                  value=f"≈ {gut:.0f} g carbs still absorbing" if gut >= 1 else "Nothing absorbing",
                  detail=f"Last logged meal {since:.0f} min ago" if _ok(since) and since < 720 else "No meal logged recently",
                  meaning="Glucose will keep entering the blood" if gut >= 10 else "Fasting state"))

    egfr = v.get("egfr")
    st = NEUTRAL if not _ok(egfr) else GOOD if egfr >= 90 else WARN if egfr >= 60 else SERIOUS
    o.append(dict(id="kidneys", title="Kidneys", status=st,
                  value=f"eGFR {egfr:.0f} mL/min" if _ok(egfr) else "eGFR not recorded",
                  detail="SGLT2 inhibitor: flushing excess glucose into urine" if v.get("sglt2") else "",
                  meaning="Reduced kidney function: check yearly" if _ok(egfr) and egfr < 60
                  else "Mildly reduced" if _ok(egfr) and egfr < 90 else "Kidney function fine"))

    hr, hrv, elev = v.get("hr"), v.get("hrv_rel"), v.get("hr_still_elev")
    stressed = (_ok(hrv) and hrv < 0.8) or (_ok(elev) and elev > 8)
    o.append(dict(id="heart", title="Heart & stress · smartwatch", status=NEUTRAL if not _ok(hr) else WARN if stressed else GOOD,
                  value=f"{hr:.0f} bpm" if _ok(hr) else "—",
                  detail=f"HRV {(hrv - 1) * 100:+.0f}% vs usual" if _ok(hrv) else "HRV not measured right now",
                  meaning="Signs of stress: stress hormones raise glucose" if stressed else "No stress signal"))

    sl, dv = v.get("sleep_h"), v.get("sleep_vs_usual")
    st = NEUTRAL if not _ok(sl) else SERIOUS if sl < 5.5 else WARN if sl < 6.5 else GOOD
    o.append(dict(id="brain", title="Brain · sleep", status=st, value=f"Slept {sl:.1f} h" if _ok(sl) else "—",
                  detail=f"{dv:+.1f} h vs usual" if _ok(dv) else "",
                  meaning="Short sleep lowers next-day insulin sensitivity" if _ok(sl) and sl < 6.5
                  else "Enough sleep"))

    neuro = v.get("neuropathy")
    o.append(dict(id="nerves", title="Nerves & feet", status=WARN if neuro else GOOD,
                  value="Neuropathy on record" if neuro else "No neuropathy recorded",
                  detail="Daily foot checks, yearly exam" if neuro else "Yearly foot exam",
                  meaning="Long-term high glucose damages nerves" if neuro else "Protect with time in range"))
    for i, d in enumerate(o, 1):
        d["n"] = i
    return o


# --------------------------------------------------------------------------------------
# SVG figure (front view, viewBox 0 0 300 640). Organ shapes are tagged with data-organ.
# --------------------------------------------------------------------------------------
SILHOUETTE = """
<ellipse cx="150" cy="62" rx="34" ry="40"/>
<rect x="137" y="96" width="26" height="26" rx="8"/>
<path d="M98 122 Q150 110 202 122 Q214 150 210 200 Q204 270 198 336 L102 336 Q96 270 90 200 Q86 150 98 122 Z"/>
<path d="M98 124 Q78 128 72 160 L60 250 Q54 300 56 322 Q60 336 72 332 Q76 300 80 262 L92 190 Z"/>
<path d="M202 124 Q222 128 228 160 L240 250 Q246 300 244 322 Q240 336 228 332 Q224 300 220 262 L208 190 Z"/>
<path d="M102 334 L198 334 Q206 360 200 388 L100 388 Q94 360 102 334 Z"/>
<path d="M102 384 L148 384 L142 470 L138 600 Q128 616 112 608 L110 470 Q100 420 102 384 Z"/>
<path d="M152 384 L198 384 Q200 420 190 470 L188 608 Q172 616 162 600 L158 470 Z"/>
"""

ORGANS = {
    "brain": ('<ellipse cx="150" cy="50" rx="24" ry="17"/>', (150, 50)),
    "heart": ('<path class="beat" d="M166 162 C162 152 148 152 148 164 C148 174 160 182 166 188 C172 182 184 174 '
              '184 164 C184 152 170 152 166 162 Z"/><rect x="50" y="292" width="20" height="16" rx="4"/>', (166, 168)),
    "liver": ('<path d="M100 200 Q120 186 156 194 Q160 206 150 216 Q126 228 104 220 Q96 212 100 200 Z"/>', (126, 206)),
    "gut": ('<path d="M162 206 Q186 196 194 216 Q198 240 178 246 Q164 244 166 232 Q170 222 162 214 Z"/>'
            '<rect x="124" y="272" width="52" height="50" rx="20"/>', (150, 297)),
    "pancreas": ('<path d="M118 252 Q150 242 188 249 Q186 262 150 264 Q124 266 118 252 Z"/>', (152, 255)),
    "kidneys": ('<ellipse cx="112" cy="282" rx="9" ry="15"/><ellipse cx="188" cy="282" rx="9" ry="15"/>', (112, 282)),
    "muscle": ('<ellipse cx="126" cy="440" rx="15" ry="44"/><ellipse cx="174" cy="440" rx="15" ry="44"/>'
               '<ellipse cx="80" cy="200" rx="8" ry="26"/>', (126, 440)),
    "blood": ('<circle cx="222" cy="176" r="9"/>', (222, 176)),
    "nerves": ('<ellipse cx="124" cy="613" rx="18" ry="8"/><ellipse cx="176" cy="613" rx="18" ry="8"/>', (176, 613)),
}


def render_body(organs: list[dict], header: dict) -> str:
    by_id = {o["id"]: o for o in organs}
    shapes, badges = [], []
    for oid, (svg, (bx, by)) in ORGANS.items():
        o = by_id.get(oid)
        if not o:
            continue
        c = COLOR[o["status"]]
        shapes.append(f'<g class="organ" data-organ="{oid}" fill="{c}" fill-opacity="0.55" stroke="{c}" '
                      f'stroke-width="2"><title>{html.escape(o["title"])}: {html.escape(o["value"])}</title>{svg}</g>')
        badges.append(f'<g class="badge" data-organ="{oid}"><circle cx="{bx}" cy="{by}" r="9" fill="#fff" '
                      f'stroke="{c}" stroke-width="2"/><text x="{bx}" y="{by + 4}" text-anchor="middle">{o["n"]}</text></g>')
    gut_flow = ('<path class="flow" d="M178 246 Q170 262 162 272" stroke="#2a78d6" stroke-width="3" fill="none" '
                'stroke-dasharray="4 5"/>' if by_id.get("gut", {}).get("status") == INFO else "")
    cards = "".join(
        f'<div class="card" data-organ="{o["id"]}"><div class="top"><span class="num">{o["n"]}</span>'
        f'<span class="t">{html.escape(o["title"])}</span>'
        f'<span class="chip" style="color:{COLOR[o["status"]]};background:{COLOR[o["status"]]}1f">'
        f'{ICON[o["status"]]} {LABEL[o["status"]]}</span></div>'
        f'<div class="v">{html.escape(o["value"])}</div>'
        + (f'<div class="d">{html.escape(o["detail"])}</div>' if o["detail"] else "")
        + f'<div class="m">{html.escape(o["meaning"])}</div></div>' for o in organs)
    hr = header.get("hr")
    beat = 60.0 / max(hr if _ok(hr) else 72.0, 30.0)
    risk = header.get("risk_text", "")
    rcol = COLOR[header.get("risk_status", NEUTRAL)]
    return f"""
<style>
  body {{ margin: 0; font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; color: #0b0b0b; }}
  .wrap {{ display: grid; grid-template-columns: 320px 1fr; gap: 18px; align-items: start; }}
  .fig {{ background: linear-gradient(180deg, #f7f9fc, #eef2f8); border: 1px solid #e1e0d9; border-radius: 18px;
          padding: 10px 6px 4px; }}
  .fig svg {{ width: 100%; height: auto; display: block; }}
  .sil {{ fill: #e4e9f1; stroke: #c8cfdb; stroke-width: 1.5; }}
  .organ {{ cursor: pointer; transition: fill-opacity .15s, stroke-width .15s; }}
  .organ.hl, .organ:hover {{ fill-opacity: .9; stroke-width: 3.5; }}
  .badge text {{ font-size: 10px; font-weight: 700; fill: #0b0b0b; pointer-events: none; }}
  .badge {{ cursor: pointer; }}
  .beat {{ transform-origin: 166px 170px; animation: beat {beat:.2f}s ease-in-out infinite; }}
  @keyframes beat {{ 0%,100% {{ transform: scale(1) }} 15% {{ transform: scale(1.12) }} 30% {{ transform: scale(1) }} }}
  .flow {{ animation: flow 1.2s linear infinite; }}
  @keyframes flow {{ to {{ stroke-dashoffset: -18 }} }}
  @media (prefers-reduced-motion: reduce) {{ .beat, .flow {{ animation: none }} }}
  .legend {{ font-size: 11px; color: #52514e; text-align: center; padding: 4px 0 6px; }}
  .lab {{ font-size: 9px; fill: #52514e; }}
  .head {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; gap: 10px; }}
  .who {{ font-size: 15px; font-weight: 700; }} .who span {{ font-weight: 400; color: #52514e; font-size: 13px; }}
  .risk {{ font-weight: 700; font-size: 13px; padding: 6px 12px; border-radius: 99px; color: {rcol};
           background: {rcol}1f; white-space: nowrap; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 10px; }}
  .card {{ background: #fff; border: 1px solid #e6e4dd; border-radius: 14px; padding: 10px 12px;
           transition: box-shadow .15s, border-color .15s; }}
  .card.hl {{ border-color: #2a78d6; box-shadow: 0 0 0 3px #2a78d633; }}
  .top {{ display: flex; align-items: center; gap: 6px; }}
  .num {{ width: 18px; height: 18px; border-radius: 50%; background: #0d366b; color: #fff; font-size: 10px;
          font-weight: 700; display: inline-flex; align-items: center; justify-content: center; flex: none; }}
  .t {{ font-size: 12px; font-weight: 600; color: #52514e; flex: 1; }}
  .chip {{ font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 99px; white-space: nowrap; }}
  .v {{ font-size: 16px; font-weight: 700; margin-top: 6px; }}
  .d {{ font-size: 12px; color: #52514e; margin-top: 2px; }}
  .m {{ font-size: 12px; color: #0b0b0b; margin-top: 6px; padding-top: 6px; border-top: 1px dashed #e1e0d9; }}
</style>
<div class="wrap">
  <div class="fig">
    <svg viewBox="0 0 300 640" role="img" aria-label="Digital twin body map">
      <g class="sil">{SILHOUETTE}</g>
      {''.join(shapes)}
      {gut_flow}
      {''.join(badges)}
      <text class="lab" x="236" y="168">CGM</text>
      <text class="lab" x="26" y="303">watch</text>
    </svg>
    <div class="legend">Hover an organ or a card · colours show the twin's live status</div>
  </div>
  <div>
    <div class="head"><div class="who">{html.escape(header.get("who", ""))}<br><span>{html.escape(header.get("sub", ""))}</span></div>
      <div class="risk">{html.escape(risk)}</div></div>
    <div class="cards">{cards}</div>
  </div>
</div>
<script>
  const all = (id) => document.querySelectorAll('[data-organ="' + id + '"]');
  document.querySelectorAll('[data-organ]').forEach(el => {{
    el.addEventListener('mouseenter', () => all(el.dataset.organ).forEach(x => x.classList.add('hl')));
    el.addEventListener('mouseleave', () => all(el.dataset.organ).forEach(x => x.classList.remove('hl')));
  }});
</script>
"""
