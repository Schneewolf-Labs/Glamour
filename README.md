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
- **The arena.** Models compete on the same prompt, outputs are shuffled blind,
  and humans (and judge models) rate the renderings — feeding preference data
  and a leaderboard.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and an [OpenRouter](https://openrouter.ai) API key.

```bash
git clone https://github.com/Schneewolf-Labs/Glamour
cd Glamour

# Install deps + Chromium
uv sync
uv run playwright install chromium

# Add your key
echo "OPENROUTER_API_KEY=sk-or-..." > .env
```

### Arena

Pit two models against the same prompt and compare their rendered designs side by side:

```bash
uv run python -m glamour.server
# → http://localhost:7700
```

Pick a model for each panel, enter a design prompt (or generate one), hit **Generate**.

### Synthetic corpus builder

Render a corpus of clean + defect-injected components with ground-truth labels:

```bash
uv run python -m glamour.build        # full run (~21 examples)
uv run python -m glamour.build --n 3  # quick smoke test
# → output/corpus.jsonl + output/renders/*.png
```

Each example includes the screenshot, DOM bounding boxes, measured defect
severity, a grounded critique, and a fix instruction. The closed-loop verifier
confirms the reward signal separates broken from clean on 100% of examples.

**Defect taxonomy:** contrast · alignment · type hierarchy · cramped spacing

## Layout

```
glamour/
  openrouter.py   OpenRouter client (stdlib only, multimodal)
  server.py       Flask arena server
  render.py       Playwright renderer + pixel-diff verifier
  build.py        Corpus builder (3 components × 4 defect types × variants)
  synth_poc.py    End-to-end PoC pipeline
static/
  index.html      Arena UI (single file, no framework)
output/           Generated renders + JSONL corpus (gitignored)
```

## Status

🌿 MVP arena shipped. Corpus builder running. Next: judge model integration,
preference pairs, and leaderboard. See [`CLAUDE.md`](./CLAUDE.md) for the full
design.

## License

TBD.
