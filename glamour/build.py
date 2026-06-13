"""Glamour corpus builder.

For each (component x defect x severity-variant): render the clean baseline and
the broken version, MEASURE the defect from DOM ground-truth, write a grounded
critique + fix, and run closed-loop verification (pixel-diff broken-vs-clean +
re-measure) to confirm the reward signal separates broken from clean.

    python -m glamour.build           # full batch
    python -m glamour.build --n 3     # quick smoke

Writes output/renders/*.png and output/corpus.jsonl.
"""
from __future__ import annotations
import argparse
import json
import os
import re

from .render import Renderer, pixel_diff

OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
RENDERS = os.path.join(OUT, "renders")

# --------------------------------------------------------------------------- #
# clean components — all expose the same data-glam anchors so any defect that
# targets {subtext, cta, actions, card} applies to every component.
# --------------------------------------------------------------------------- #
_BASE_CSS = """
  body { margin:0; background:#f3f4f6; font-family:-apple-system,Segoe UI,Roboto,sans-serif; }
  .wrap { padding:36px; display:flex; justify-content:center; }
  .card { width:360px; background:#fff; border-radius:12px; padding:24px;
          box-shadow:0 1px 3px rgba(0,0,0,.12); }
  .title { margin:0 0 8px; font-size:20px; font-weight:600; color:#111827; }
  .subtext { margin:0 0 20px; font-size:14px; color:#4b5563; }
  .actions { display:flex; justify-content:flex-end; gap:8px; }
  .btn { background:#4f46e5; color:#fff; border:none; border-radius:8px;
         padding:10px 18px; font-size:14px; font-weight:500; cursor:pointer; }
  .btn.ghost { background:#fff; color:#4f46e5; border:1px solid #c7d2fe; }
  __DEFECT_CSS__
"""

def _page(body: str) -> str:
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>{_BASE_CSS}'
            f'</style></head><body><div class="wrap">{body}</div></body></html>')

COMPONENTS = {
    "subscribe-card": _page("""
      <div class="card" data-glam="card">
        <h2 class="title" data-glam="title">Stay in the loop</h2>
        <p class="subtext" data-glam="subtext">Get product updates, no spam.</p>
        <div class="actions" data-glam="actions">
          <button class="btn" data-glam="cta">Subscribe</button>
        </div></div>"""),
    "pricing-tile": _page("""
      <div class="card" data-glam="card">
        <h2 class="title" data-glam="title">Pro — $29/mo</h2>
        <p class="subtext" data-glam="subtext">Billed annually. Cancel anytime.</p>
        <div class="actions" data-glam="actions">
          <button class="btn ghost" data-glam="alt">Compare</button>
          <button class="btn" data-glam="cta">Choose Pro</button>
        </div></div>"""),
    "confirm-dialog": _page("""
      <div class="card" data-glam="card">
        <h2 class="title" data-glam="title">Delete project?</h2>
        <p class="subtext" data-glam="subtext">This permanently removes all data.</p>
        <div class="actions" data-glam="actions">
          <button class="btn ghost" data-glam="alt">Cancel</button>
          <button class="btn" data-glam="cta">Delete</button>
        </div></div>"""),
}

# --------------------------------------------------------------------------- #
# measurement helpers
# --------------------------------------------------------------------------- #
def _rgb(s):
    n = [int(x) for x in re.findall(r"\d+", s)[:3]]
    return tuple(n) if len(n) == 3 else (0, 0, 0)

def _px(s):
    m = re.search(r"[\d.]+", s or "")
    return float(m.group()) if m else 0.0

def _lin(c):
    c /= 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

def _lum(rgb):
    r, g, b = (_lin(x) for x in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def contrast_ratio(fg, bg):
    L1, L2 = _lum(fg), _lum(bg)
    return (max(L1, L2) + 0.05) / (min(L1, L2) + 0.05)

# --------------------------------------------------------------------------- #
# defect taxonomy — each entry: css mutation + which element it measures +
# how to measure / critique / fix. `clean` is the clean-render DOM (for the
# fix target value and the contrast background).
# --------------------------------------------------------------------------- #
def _contrast(dom, clean, target="subtext"):
    fg, bg = _rgb(dom[target]["color"]), _rgb(clean["card"]["background"])
    ratio = contrast_ratio(fg, bg)
    sev = {"contrast_ratio": round(ratio, 2), "wcag_aa_pass": ratio >= 4.5}
    crit = (f"The subtext is rgb{fg} on rgb{bg} — only {ratio:.1f}:1 contrast, "
            f"below the WCAG AA minimum of 4.5:1 for body text; it's hard to read.")
    return sev, crit

def _alignment(dom, clean, target="cta"):
    offset = dom["actions"]["right"] - dom[target]["right"]
    sev = {"right_edge_offset_px": offset, "aligned": offset == 0}
    crit = (f"The primary button sits {offset}px inside the right edge of its "
            f"actions row instead of flush against it — it reads as misaligned.")
    return sev, crit

def _type_small(dom, clean, target="subtext"):
    fs = _px(dom[target]["fontSize"])
    sev = {"font_size_px": fs, "readable": fs >= 12}
    crit = (f"The subtext is {fs:.0f}px — under the ~12px floor for comfortable "
            f"body text; it strains readability, especially for small screens.")
    return sev, crit

def _cramped(dom, clean, target="subtext"):
    mb = _px(dom[target]["marginBottom"])
    # pass-style bool (True = good), consistent with the other defects so the
    # verifier's broken-fails / clean-passes check has one polarity convention.
    sev = {"gap_px": mb, "ok_spacing": mb >= 8}
    crit = (f"Only {mb:.0f}px separates the subtext from the action buttons — "
            f"the group feels cramped and the CTA crowds the copy above it.")
    return sev, crit

DEFECTS = {
    "contrast":   dict(measure=_contrast, target="subtext",
                       css=lambda v: f'[data-glam="subtext"]{{color:{v};}}',
                       variants=["#cfcfcf", "#bdbdbd"],
                       fix="Darken the subtext to meet WCAG AA (≥ 4.5:1), e.g. {clean}."),
    "alignment":  dict(measure=_alignment, target="cta",
                       css=lambda v: f'[data-glam="cta"]{{transform:translateX(-{v}px);}}',
                       variants=[46, 24],
                       fix="Remove the button's horizontal offset so it aligns flush right."),
    "type_small": dict(measure=_type_small, target="subtext",
                       css=lambda v: f'[data-glam="subtext"]{{font-size:{v}px;}}',
                       variants=[9, 10],
                       fix="Raise the subtext to ≥ 12px (clean was {clean})."),
    "cramped":    dict(measure=_cramped, target="subtext",
                       css=lambda v: f'[data-glam="subtext"]{{margin-bottom:{v}px;}}',
                       variants=[2],
                       fix="Restore breathing room below the subtext (clean was {clean})."),
}

# clean values used to phrase the fix, read off the clean DOM
def _clean_value(kind, clean, target):
    if kind == "contrast":   return clean[target]["color"]
    if kind == "type_small": return clean[target]["fontSize"]
    if kind == "cramped":    return clean[target]["marginBottom"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="cap examples (0 = all)")
    args = ap.parse_args()
    os.makedirs(RENDERS, exist_ok=True)

    corpus, n_emitted, n_discriminating = [], 0, 0
    with Renderer() as r:
        for comp_id, html in COMPONENTS.items():
            clean_png = os.path.join(RENDERS, f"{comp_id}__clean.png")
            clean = r.render(html.replace("__DEFECT_CSS__", ""), clean_png)
            for kind, spec in DEFECTS.items():
                tgt = spec["target"]
                for vi, val in enumerate(spec["variants"]):
                    if args.n and n_emitted >= args.n:
                        break
                    css = spec["css"](val)
                    png = os.path.join(RENDERS, f"{comp_id}__{kind}{vi}.png")
                    dom = r.render(html.replace("__DEFECT_CSS__", css), png)
                    sev, crit = spec["measure"](dom, clean, tgt)
                    cleanval = _clean_value(kind, clean, tgt)
                    fix = spec["fix"].format(clean=cleanval) if cleanval else spec["fix"]

                    # closed-loop verification: the clean render IS the canonical
                    # fix, so re-measure clean (must pass) + diff broken-vs-clean
                    # (must be visible). This is exactly the reward an agent's
                    # edit would be scored against.
                    sev_clean, _ = spec["measure"](clean, clean, tgt)
                    pass_keys = [k for k in sev if isinstance(sev[k], bool)]
                    broken_fails = any(sev[k] is False for k in pass_keys)
                    clean_passes = all(sev_clean[k] is True for k in pass_keys)
                    diff_vs_clean = pixel_diff(png, clean_png)
                    discriminates = broken_fails and clean_passes and diff_vs_clean > 0.001
                    n_discriminating += discriminates

                    corpus.append({
                        "id": f"{comp_id}-{kind}-{vi}",
                        "component": comp_id,
                        "injected_defect": kind,
                        "screenshot": os.path.relpath(png, OUT),
                        "clean_screenshot": os.path.relpath(clean_png, OUT),
                        "target_element": tgt,
                        "target_bbox": dom[tgt]["bbox"],
                        "dom_bboxes": {k: v["bbox"] for k, v in dom.items()},
                        "measured_severity": sev,
                        "critique_text": crit,
                        "fix_instruction": fix,
                        "verification": {
                            "pixel_diff_vs_clean": diff_vs_clean,
                            "broken_fails_metric": broken_fails,
                            "clean_passes_metric": clean_passes,
                            "reward_discriminates": discriminates,
                        },
                    })
                    n_emitted += 1

    with open(os.path.join(OUT, "corpus.jsonl"), "w") as f:
        for r_ in corpus:
            f.write(json.dumps(r_, ensure_ascii=False) + "\n")

    print(f"emitted {n_emitted} examples across {len(COMPONENTS)} components "
          f"x {len(DEFECTS)} defect kinds -> {OUT}/corpus.jsonl")
    print(f"reward signal discriminates broken-vs-clean on "
          f"{n_discriminating}/{n_emitted} ({100*n_discriminating/max(n_emitted,1):.0f}%)")
    from collections import Counter
    print("by defect:", dict(Counter(r_["injected_defect"] for r_ in corpus)))
    print("\nsample critiques:")
    seen = set()
    for r_ in corpus:
        if r_["injected_defect"] not in seen:
            seen.add(r_["injected_defect"])
            print(f"  [{r_['id']}] sev={r_['measured_severity']} "
                  f"diff={r_['verification']['pixel_diff_vs_clean']}")
            print(f"      {r_['critique_text']}")


if __name__ == "__main__":
    main()
