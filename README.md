<h1 align="center">✨ Glamour</h1>

<p align="center">
  <em>A web-design critique arena &amp; synthetic-data engine —<br/>
  teaching a model to see a website, judge it like a designer, and drive the fix.</em>
</p>

---

> In folklore a **glamour** is a witch's spell that makes something *appear*
> beautiful. Glamour judges the glamour of websites: does this page actually
> look good, or is it just faking it?

**Glamour** is a [Schneewolf Labs](https://huggingface.co/schneewolflabs)
project that builds toward a focused vision-language specialist which closes the
front-end QA loop:

**see** a rendering → **critique** it (alignment, contrast, readability,
hierarchy, spacing, responsiveness) → **instruct** a coding agent to fix it →
**re-evaluate** → **loop.**

## How

- **Synthetic data, exact labels.** Web UI is code → render, so we manufacture
  training data: take a clean component, inject a defect, render both with a
  headless browser. We know precisely what's wrong, where, and the fix — and
  the DOM gives bounding boxes + measured severity (real contrast ratios, pixel
  offsets) to anchor every critique.
- **Closed-loop, for free.** After a fix, re-render and diff against the clean
  original → an automatic reward signal for multi-turn training and eval.
- **The arena.** Models compete on prompts, outputs are shuffled blind, and
  humans (and judge models) rate the renderings — feeding preference data and a
  leaderboard.

## Status

🌱 Early. See [`CLAUDE.md`](./CLAUDE.md) for the full design.

## License

TBD.
