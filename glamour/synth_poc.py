"""Glamour — proof-of-concept synthesis pipeline.

Demonstrates the core trick end-to-end on one clean component:
  clean UI  --inject defect-->  broken UI  --Playwright render-->
  screenshot + DOM bboxes + computed styles  -->  MEASURED severity
  (WCAG contrast ratio / pixel offset)  -->  grounded critique + fix.

Because we INJECT the defect, the label is exact: we know what's wrong, where,
the measured severity, and the fix (revert the mutation). Run:

    python -m glamour.synth_poc

Writes output/renders/*.png and output/examples.jsonl.
"""
from __future__ import annotations
import json
import os
import re
from playwright.sync_api import sync_playwright

OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
RENDERS = os.path.join(OUT, "renders")

# --------------------------------------------------------------------------- #
# WCAG contrast helpers
# --------------------------------------------------------------------------- #
def _parse_rgb(s: str):
    nums = [int(x) for x in re.findall(r"\d+", s)[:3]]
    return tuple(nums) if len(nums) == 3 else (0, 0, 0)


def _lin(c: float) -> float:
    c /= 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def _lum(rgb) -> float:
    r, g, b = (_lin(x) for x in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(fg, bg) -> float:
    L1, L2 = _lum(fg), _lum(bg)
    hi, lo = max(L1, L2), min(L1, L2)
    return (hi + 0.05) / (lo + 0.05)


# --------------------------------------------------------------------------- #
# the clean component + the defect mutators
# --------------------------------------------------------------------------- #
BASE_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
  body { margin:0; background:#f3f4f6; font-family:-apple-system,Segoe UI,Roboto,sans-serif; }
  .wrap { padding:40px; display:flex; justify-content:center; }
  .card { width:360px; background:#ffffff; border-radius:12px; padding:24px;
          box-shadow:0 1px 3px rgba(0,0,0,.12); }
  .title { margin:0 0 8px; font-size:20px; font-weight:600; color:#111827; }
  .subtext { margin:0 0 20px; font-size:14px; color:#4b5563; }
  .actions { display:flex; justify-content:flex-end; }
  .btn { background:#4f46e5; color:#fff; border:none; border-radius:8px;
         padding:10px 18px; font-size:14px; font-weight:500; cursor:pointer; }
  __DEFECT_CSS__
</style></head><body><div class="wrap">
  <div class="card" data-glam="card">
    <h2 class="title" data-glam="title">Stay in the loop</h2>
    <p class="subtext" data-glam="subtext">Get product updates, no spam.</p>
    <div class="actions" data-glam="actions">
      <button class="btn" data-glam="cta">Subscribe</button>
    </div>
  </div></div></body></html>"""

# Each defect = (css mutation injected, the human fix instruction, the css diff).
DEFECTS = {
    "clean": dict(css="", kind=None),
    "contrast": dict(
        css=".subtext { color:#cfcfcf; }",
        kind="contrast",
        fix="Darken the subtext color so it meets WCAG AA (≥ 4.5:1). "
            "Restore it to a mid-gray like #4b5563.",
        diff="- .subtext { color:#cfcfcf; }\n+ .subtext { color:#4b5563; }",
    ),
    "alignment": dict(
        css=".btn { transform:translateX(-46px); }",
        kind="alignment",
        fix="Remove the horizontal offset on the button so it aligns flush "
            "with the right edge of its actions container.",
        diff="- .btn { transform:translateX(-46px); }\n+ .btn { /* no transform */ }",
    ),
}


# --------------------------------------------------------------------------- #
# render + extract DOM ground-truth
# --------------------------------------------------------------------------- #
EXTRACT_JS = """() => {
  const out = {};
  document.querySelectorAll('[data-glam]').forEach(el => {
    const r = el.getBoundingClientRect();
    const cs = getComputedStyle(el);
    out[el.dataset.glam] = {
      bbox: [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)],
      right: Math.round(r.right), bottom: Math.round(r.bottom),
      color: cs.color, background: cs.backgroundColor, fontSize: cs.fontSize
    };
  });
  return out;
}"""


def render(page, defect_css, png_path):
    page.set_content(BASE_HTML.replace("__DEFECT_CSS__", defect_css))
    page.set_viewport_size({"width": 440, "height": 240})
    dom = page.evaluate(EXTRACT_JS)
    page.locator('[data-glam="card"]').screenshot(path=png_path)
    return dom


# --------------------------------------------------------------------------- #
# measure severity + write the grounded critique (templated from ground-truth)
# --------------------------------------------------------------------------- #
def measure_and_critique(kind, dom):
    if kind == "contrast":
        fg = _parse_rgb(dom["subtext"]["color"])
        bg = _parse_rgb(dom["card"]["background"])  # subtext bg is transparent -> card
        ratio = contrast_ratio(fg, bg)
        sev = {"contrast_ratio": round(ratio, 2), "wcag_aa_pass": ratio >= 4.5}
        crit = (f"The subtext “Get product updates, no spam.” is "
                f"rgb{fg} on rgb{bg} — only {ratio:.1f}:1 contrast, below the "
                f"WCAG AA minimum of 4.5:1 for body text. It's hard to read.")
        return sev, crit, "subtext"
    if kind == "alignment":
        offset = dom["actions"]["right"] - dom["cta"]["right"]
        sev = {"right_edge_offset_px": offset}
        crit = (f"The “Subscribe” button is {offset}px inside the right edge "
                f"of its actions row instead of sitting flush against it — it "
                f"reads as misaligned with the card's content.")
        return sev, crit, "cta"
    return {}, "", None


def main():
    os.makedirs(RENDERS, exist_ok=True)
    examples = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(device_scale_factor=2)
        for name, d in DEFECTS.items():
            png = os.path.join(RENDERS, f"{name}.png")
            dom = render(page, d["css"], png)
            rec = {
                "id": f"poc-{name}",
                "component": "subscribe-card",
                "viewport": {"width": 440, "height": 240},
                "screenshot": os.path.relpath(png, OUT),
                "dom_bboxes": {k: v["bbox"] for k, v in dom.items()},
                "defect": d["kind"],
            }
            if d["kind"]:
                sev, crit, target = measure_and_critique(d["kind"], dom)
                rec.update(
                    injected_defect=d["kind"],
                    target_element=target,
                    target_bbox=dom[target]["bbox"],
                    measured_severity=sev,
                    critique_text=crit,
                    fix_instruction=d["fix"],
                    fix_diff=d["diff"],
                )
            examples.append(rec)
        browser.close()

    with open(os.path.join(OUT, "examples.jsonl"), "w") as f:
        for r in examples:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(examples)} examples -> {OUT}/examples.jsonl")
    for r in examples:
        if r.get("defect"):
            print(f"\n[{r['id']}]  defect={r['injected_defect']}  "
                  f"severity={r['measured_severity']}")
            print(f"  target {r['target_element']} @ {r['target_bbox']}")
            print(f"  critique: {r['critique_text']}")
            print(f"  fix: {r['fix_instruction']}")
        else:
            print(f"\n[{r['id']}]  (clean baseline, for re-render diffing)")


if __name__ == "__main__":
    main()
