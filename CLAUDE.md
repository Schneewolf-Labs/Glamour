# Glamour

**A web-design critique arena + synthetic-data engine** for training a
vision-language model to *look at a web rendering, judge it like a designer,
and drive a coding agent to fix it — then re-evaluate, and loop.*

> In folklore a **glamour** is a witch's spell that makes something *appear*
> beautiful — an illusion of visual allure. Glamour the tool judges the
> glamour of websites: does this page actually look good, or is it faking it?

This is a Schneewolf Labs project. It's greenfield — this file is the seed of
the idea, written so a future session or collaborator picks up the full vision.

---

## What it's for

The end goal is a **focused VLM specialist** (built on the A-series + the
Artemis vision graft) that closes this loop:

1. **See** a web rendering (screenshot).
2. **Critique** it like a designer/QA: "the *Subscribe* button's subtext is
   `#9a9a9a` on white — 2.1:1 contrast, below WCAG AA; and it sits 6px off the
   card's right edge." Visual, UX, accessibility, layout, hierarchy.
3. **Instruct** a coding agent to make the change (concrete, executable).
4. **Re-evaluate** the new rendering — did it fix the issue without regressing
   anything else?
5. **Loop.**

The A3-Doc experiment (a focused doc/chart FFT of the A-series that measurably
beat the generalist) is the evidence this works: a narrow FFT makes the model
*genuinely good* at one thing rather than mediocre at everything. Glamour is
the same bet, aimed at front-end visual QA.

---

## Two halves

### 1. The synthetic-data engine (the core trick)

Web UI is **code → render**, so we have ground truth for free. We don't scrape
and hand-label — we **manufacture** the dataset at scale with exact labels:

- **Mutation-based generation.** Start from a *clean* UI (component or page),
  programmatically **inject a defect**, render *both* with a headless browser
  (Playwright / Chromium). Because we injected it, we know exactly **what's
  wrong, where, and the fix** (the fix is reverting the mutation).
- **DOM ground-truth.** From the same render, pull `getBoundingClientRect()`
  for every element + computed styles. So we can (a) attach precise bounding
  boxes to the critique (real visual grounding), and (b) *measure* the defect
  (actual contrast ratio, pixel misalignment). Critiques are anchored to
  numbers, not vibes.
- **Closed-loop verification — for free.** After the coding agent's change,
  re-render and **diff against the original clean version** (pixel diff /
  re-measure). That's an automatic reward/eval signal — enabling multi-turn
  see→fix→re-evaluate training traces, and later **RL via grimoire** using the
  re-render as reward, not just SFT.
- **Natural critique, grounded labels.** Generate the critique *prose* with a
  strong VLM but **anchor it to the known defect + measurements** — natural
  designer voice that's still factually correct, never hallucinated.

**Defect taxonomy** (the coverage/difficulty lever — each is a programmatic
mutator + a measurable ground-truth):
- alignment (margin/padding/flex/grid offsets)
- contrast / readability (color tweaks that drop WCAG contrast)
- type hierarchy (subtext too small, weak heading scale)
- spacing / overflow (cramped gaps, text overflow, clipping)
- style inconsistency (mismatched radii, button styles, color drift)
- responsive breakage (render at a width that breaks the layout)

**Output schema (per example, draft):**
`{ screenshot, viewport, dom_bboxes, computed_styles, injected_defect,
   measured_severity, critique_text, fix_instruction, fix_diff }`

**Clean-UI sources to mutate:** component libraries (shadcn/ui, Tailwind UI,
Material, Bootstrap examples), real-page HTML, and LLM-generated components.

### 2. The arena (preference data + human-in-the-loop)

A judging arena that produces preference data and keeps humans in the loop:

- Models **compete on prompts** (generate a UI, or critique/fix a rendering).
- Outputs are **shuffled blind** (model identities hidden) so judging isn't
  biased by reputation.
- **Humans judge the renderings** (and/or a strong judge model) → preference
  pairs that feed back into training (ORPO/DPO) and into the eval leaderboard.
- Doubles as the **benchmark**: which model produces / fixes the better page.

---

## Model + training stack (Schneewolf Labs)

- **Base:** A-series VLM — Qwen3-VL ViT (frozen) + learned projector + A2/
  Mistral decoder, via the **Artemis** graft (`Schneewolf-Labs/Artemis`).
- **Training:** **Merlina** (SFT for the critique format + visual grounding;
  ORPO/DPO for critique quality) and **grimoire** (RL once the re-render reward
  is wired).
- **Recipe note:** tool/structured output is format-fragile, so this wants
  **SFT first** (learn the exact critique + fix-instruction format + grounding),
  then preference optimization for quality — unlike pure-DPO creative runs.
- Stack neighbors: Hemlock → Witchgrid (deploy) → Merlina (train) → Artemis
  (VLM lib) → Scry (eval) → Grimoire (RL) → Atelier (diffusion).

---

## Status

**Early — the synthesis engine works end-to-end.** Built so far:

- `glamour/synth_poc.py` — the PoC: clean component → inject contrast/alignment
  defect → Playwright render → DOM-measured severity → grounded critique.
- `glamour/build.py` + `render.py` — corpus builder: 3 components × the defect
  taxonomy (contrast, alignment, type-hierarchy, spacing) × severity variants,
  with **closed-loop verification** (re-measure clean + pixel-diff broken-vs-clean
  → confirms the reward signal separates broken from clean).
- `glamour/openrouter.py` — multimodal LLM client (prompt-gen / competitors /
  vision judge) for the arena and critique synthesis.
- `glamour/critique.py` — **grounded critique synthesis**: rewrites the
  templated critiques in a natural designer voice via a VLM, *anchored* to the
  measured ground-truth (the model is told the defect + numbers, not asked to
  find them) with a measured-number guard + template fallback. Runs as a
  decoupled post-pass over the corpus; tested in `tests/`.

The data build is CPU/headless-browser work (no GPU), so it runs alongside
training jobs.

**Next concrete steps:**
1. **SFT export** — turn `corpus.enriched.jsonl` into Merlina-ready chat traces
   (system + screenshot user turn + grounded critique/fix assistant turn, with
   bbox grounding and a leakage-safe train/val split).
2. **Expand the taxonomy** — style inconsistency (radii/button drift), overflow/
   clipping, responsive breakage (render at a width that breaks layout).
3. **Closed-loop fix agent** — apply the agent's CSS edit, re-render, score
   against the clean baseline (the reward `render.pixel_diff` already provides).
4. **The arena** — blind model competition + judging over the OpenRouter client,
   emitting preference pairs.
