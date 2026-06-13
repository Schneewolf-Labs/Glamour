"""Glamour render engine — Playwright headless Chromium.

Renders an HTML string, screenshots a target element, and extracts DOM
ground-truth (bounding boxes + the computed styles the defect measures need).
Also a pixel-diff used by the closed-loop verifier to score how much a render
differs from the clean baseline.
"""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageChops
from playwright.sync_api import sync_playwright

# Pull bbox + the computed styles our measures rely on, for every tagged element.
EXTRACT_JS = """() => {
  const out = {};
  document.querySelectorAll('[data-glam]').forEach(el => {
    const r = el.getBoundingClientRect();
    const cs = getComputedStyle(el);
    out[el.dataset.glam] = {
      bbox: [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)],
      right: Math.round(r.right), bottom: Math.round(r.bottom),
      color: cs.color, background: cs.backgroundColor,
      fontSize: cs.fontSize, fontWeight: cs.fontWeight,
      marginBottom: cs.marginBottom, transform: cs.transform
    };
  });
  return out;
}"""


class Renderer:
    """A reusable headless-Chromium session."""

    def __init__(self, width=440, height=300, scale=2, clip='[data-glam="card"]'):
        self._p = sync_playwright().start()
        self.browser = self._p.chromium.launch()
        self.page = self.browser.new_page(device_scale_factor=scale)
        self.page.set_viewport_size({"width": width, "height": height})
        self.clip = clip

    def render(self, html: str, png_path: str) -> dict:
        """Set content, extract DOM ground-truth, screenshot the clip element."""
        self.page.set_content(html)
        dom = self.page.evaluate(EXTRACT_JS)
        self.page.locator(self.clip).screenshot(path=png_path)
        return dom

    def close(self):
        self.browser.close()
        self._p.stop()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def pixel_diff(a_path: str, b_path: str, thresh: int = 12) -> float:
    """Fraction of pixels differing beyond `thresh` (summed RGB delta).
    ~0 means visually identical to the baseline; larger means a visible change.
    This is the closed-loop reward signal for scoring an agent's edit."""
    a = Image.open(a_path).convert("RGB")
    b = Image.open(b_path).convert("RGB")
    if a.size != b.size:
        b = b.resize(a.size)
    diff = np.asarray(ImageChops.difference(a, b)).astype("int32").sum(axis=2)
    return round(float((diff > thresh).mean()), 4)
