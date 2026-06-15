"""Glamour Arena — Flask web server.

    uv run python -m glamour.server

Opens at http://localhost:7700
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import random
import re
import time
import urllib.request
import uuid
from collections import deque
from datetime import datetime, timezone
from threading import Lock

from flask import Flask, Response, jsonify, request, send_from_directory

from .openrouter import OpenRouter, OpenRouterError, system_message, user_message

logging.basicConfig(
    level=os.environ.get("GLAMOUR_LOG", "DEBUG").upper(),
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("glamour")

ROOT = pathlib.Path(__file__).parent.parent
STATIC = str(ROOT / "static")
OUT = ROOT / "output"
JOBS_PATH = OUT / "jobs.jsonl"
VOTES_PATH = OUT / "votes.jsonl"
SETTINGS_PATH = OUT / "settings.json"
app = Flask(__name__)

# --------------------------------------------------------------------------- #
# settings — persisted to output/settings.json
# --------------------------------------------------------------------------- #
DEFAULT_SETTINGS = {"initial_token_limit": 8192, "favorites": []}
_settings_lock = Lock()


def _load_settings() -> dict:
    with _settings_lock:
        data = {}
        if SETTINGS_PATH.exists():
            try:
                data = json.loads(SETTINGS_PATH.read_text())
            except (json.JSONDecodeError, OSError):
                data = {}
        return {**DEFAULT_SETTINGS, **data}


def _save_settings(data: dict) -> dict:
    with _settings_lock:
        OUT.mkdir(exist_ok=True)
        merged = {**DEFAULT_SETTINGS, **data}
        SETTINGS_PATH.write_text(json.dumps(merged, indent=2))
        return merged

# The client loads .env via _load_dotenv() on import.
_client = OpenRouter()

# --------------------------------------------------------------------------- #
# job persistence — every generation is appended to output/jobs.jsonl, which
# doubles as the dataset Glamour is meant to produce.
# --------------------------------------------------------------------------- #
_jobs_lock = Lock()


def _append_job(record: dict) -> None:
    with _jobs_lock:
        OUT.mkdir(exist_ok=True)
        with open(JOBS_PATH, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_vote(record: dict) -> None:
    with _jobs_lock:
        OUT.mkdir(exist_ok=True)
        with open(VOTES_PATH, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jobs() -> list[dict]:
    if not JOBS_PATH.exists():
        return []
    rows = []
    with _jobs_lock, open(JOBS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _render_doc(html: str, css: str, js: str) -> str:
    """Assemble a standalone HTML document from html/css/js blocks."""
    safe_js = (js or "").replace("</script>", "<\\/script>")
    script = f"<script>{safe_js}</script>" if js else ""
    return (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        f"<style>{css or ''}</style>\n</head>\n<body>\n{html or ''}\n{script}\n"
        "</body>\n</html>"
    )

# --------------------------------------------------------------------------- #
# models cache
# --------------------------------------------------------------------------- #
_models_cache: list[dict] | None = None
_models_fetched_at: float = 0.0
_models_lock = Lock()
MODELS_TTL = 3600


def _fetch_models() -> list[dict]:
    global _models_cache, _models_fetched_at
    with _models_lock:
        if _models_cache is not None and (time.time() - _models_fetched_at) < MODELS_TTL:
            return _models_cache
        url = f"{_client.base_url}/models"
        req = urllib.request.Request(url, headers=_client._headers())
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        models = [
            {"id": m["id"], "name": m.get("name") or m["id"]}
            for m in body.get("data", [])
        ]
        models.sort(key=lambda m: m["name"].lower())
        _models_cache = models
        _models_fetched_at = time.time()
        log.info("fetched %d models from OpenRouter", len(models))
        return models


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #
DESIGN_SYSTEM = """\
You are competing in a live web design arena. Multiple AI models receive the same \
prompt; human judges vote for the best design.

Respond with exactly three fenced code blocks and nothing else — no commentary \
before, between, or after them:

```html
(content for <body> only — no <html>, <head>, <style>, or <script> wrappers)
```

```css
(all CSS — injected into <head><style>)
```

```js
(optional JavaScript — runs after the DOM loads; leave the block empty if unused)
```

Design rules:
- NO external URLs, CDN links, or @import — everything must be self-contained
- Cover the full viewport (min-height: 100vh or equivalent)
- Include rich, realistic content: real headlines, copy, data — no Lorem Ipsum
- Make it visually striking: commit to a strong aesthetic with color, type, and layout
- Use CSS gradients, inline SVG, or Unicode for imagery/icons
"""

# Model used to invent design briefs (separate from the A/B competitors).
PROMPT_MODEL = "~anthropic/claude-sonnet-latest"

PROMPT_SYSTEM = """\
You are a creative web design director running a design competition.
Generate a single design brief for a web page or UI component using the seeds
the user gives you. Lean into them, but make the concrete subject specific,
surprising, and fresh — avoid the obvious or overused interpretation.
Be evocative: name a real domain, a distinct visual mood, and the page type.
Return only the brief text — 2-3 sentences, no preamble, no title.
"""

# Random axes sampled per request to break the model's tendency to converge on
# the same handful of themes. Page-type × aesthetic × a loosely-interpreted
# "spark" word puts each brief in a different region of the design space.
PAGE_TYPES = [
    "landing page", "marketing hero section", "pricing page", "product detail page",
    "analytics dashboard", "login/signup screen", "onboarding flow", "404 page",
    "settings panel", "portfolio homepage", "blog post layout", "documentation page",
    "e-commerce storefront", "checkout flow", "email newsletter", "event/conference page",
    "restaurant menu", "mobile app screen", "music player UI", "photo gallery",
    "user profile page", "FAQ / help center", "coming-soon waitlist page",
    "data-visualization report", "kanban board", "calendar UI", "chat interface",
    "podcast episode page", "job listing board", "real-estate listing",
]
AESTHETICS = [
    "brutalist", "stark minimalist", "maximalist", "retro 80s", "art deco",
    "Swiss / international typographic", "cyberpunk neon", "glassmorphism",
    "neumorphism", "editorial magazine", "playful cartoonish", "corporate clean",
    "vaporwave", "Y2K", "hand-drawn sketchy", "dark-mode luxe", "soft pastel",
    "high-contrast monochrome", "organic / natural", "retro-futuristic",
    "grunge", "Bauhaus", "Memphis design", "claymorphism", "newspaper print",
]
SPARKS = [
    "tide charts", "mushroom foraging", "vintage synthesizers", "competitive yo-yo",
    "antique cartography", "deep-space telemetry", "artisan cheese", "urban beekeeping",
    "noir detective fiction", "tropical houseplants", "vinyl record pressing",
    "high-altitude ballooning", "medieval manuscripts", "tide-pool ecology",
    "model trains", "fermentation", "desert geology", "competitive crossword",
    "lighthouse keeping", "origami engineering", "street food carts", "glacier monitoring",
    "vintage motorsport", "botanical perfume", "amateur radio", "tea ceremony",
    "brutalist architecture tours", "bioluminescent plankton", "typewriter repair",
    "storm chasing", "heirloom seeds", "analog photography", "cave diving",
]

# In-memory recent briefs, fed back as an avoid-list so consecutive prompts
# don't cluster. Resets on server restart.
_recent_prompts: deque[str] = deque(maxlen=12)


# --------------------------------------------------------------------------- #
# fenced-block extraction — robust to quotes/newlines that break JSON
# --------------------------------------------------------------------------- #
_BLOCK_RE = re.compile(r"```[ \t]*(\w+)?[ \t]*\r?\n(.*?)```", re.DOTALL)
# An unterminated trailing fence — happens when the output is truncated
# mid-block. Salvaging it lets the user Continue instead of getting an error.
_OPEN_BLOCK_RE = re.compile(r"```[ \t]*(\w+)?[ \t]*\r?\n(.*)$", re.DOTALL)
_ALIASES = {"html": "html", "css": "css", "js": "js", "javascript": "js"}


def _extract_blocks(text: str) -> dict:
    """Pull the html/css/js fenced code blocks out of a model response.

    First block of each language wins. A final, unterminated block (from a
    truncated response) is salvaged too. Returns {"html","css","js"}.
    """
    found: dict[str, str] = {}
    last_end = 0
    for m in _BLOCK_RE.finditer(text):
        lang = _ALIASES.get((m.group(1) or "").lower())
        if lang and lang not in found:
            found[lang] = m.group(2).rstrip("\n")
        last_end = m.end()
    # Salvage an unterminated block after the last closed one.
    tail = text[last_end:]
    om = _OPEN_BLOCK_RE.search(tail)
    if om:
        lang = _ALIASES.get((om.group(1) or "").lower())
        if lang and lang not in found:
            found[lang] = om.group(2).rstrip("\n")
    return {"html": found.get("html", ""),
            "css": found.get("css", ""),
            "js": found.get("js", "")}


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.route("/api/models")
def api_models():
    try:
        return jsonify(_fetch_models())
    except Exception as e:
        log.exception("models fetch failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/settings", methods=["GET"])
def api_get_settings():
    return jsonify(_load_settings())


@app.route("/api/settings", methods=["POST"])
def api_set_settings():
    body = request.get_json(force=True)
    cur = _load_settings()
    if "initial_token_limit" in body:
        try:
            n = int(body["initial_token_limit"])
        except (ValueError, TypeError):
            return jsonify({"error": "initial_token_limit must be an integer"}), 400
        cur["initial_token_limit"] = max(256, min(n, 32000))
    if "favorites" in body:
        if not isinstance(body["favorites"], list):
            return jsonify({"error": "favorites must be a list"}), 400
        # de-dup, preserve order
        seen, favs = set(), []
        for x in body["favorites"]:
            s = str(x)
            if s and s not in seen:
                seen.add(s)
                favs.append(s)
        cur["favorites"] = favs
    saved = _save_settings(cur)
    log.info("settings saved: token_limit=%s favorites=%d",
             saved["initial_token_limit"], len(saved["favorites"]))
    return jsonify(saved)


@app.route("/api/generate-prompt", methods=["POST"])
def api_generate_prompt():
    page = random.choice(PAGE_TYPES)
    aesthetic = random.choice(AESTHETICS)
    spark = random.choice(SPARKS)
    user = (
        f"Seeds for this brief:\n"
        f"- Page type: {page}\n"
        f"- Visual direction: {aesthetic}\n"
        f"- Subject spark (interpret loosely, don't take it literally): {spark}\n\n"
        f"Write the brief."
    )
    if _recent_prompts:
        avoid = "\n".join(f"- {p}" for p in _recent_prompts)
        user += (f"\n\nDo NOT reuse the subject, brand, or mood of these recent "
                 f"briefs:\n{avoid}")

    log.info("generate-prompt seeds: page=%r aesthetic=%r spark=%r", page, aesthetic, spark)
    try:
        resp = _client.chat(
            [system_message(PROMPT_SYSTEM), user_message(user)],
            model=PROMPT_MODEL,
            temperature=1.0,
            max_tokens=200,
        )
        prompt = resp.text.strip()
        _recent_prompts.append(prompt[:100])
        log.info("generate-prompt OK [%s] -> %r", resp.model, prompt)
        return jsonify({"prompt": prompt})
    except OpenRouterError as e:
        log.error("generate-prompt failed: %s", e)
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        log.exception("generate-prompt crashed")
        return jsonify({"error": f"server error: {e}"}), 500


CONTINUE_MAX_TOKENS = 2000


@app.route("/api/generate", methods=["POST"])
def api_generate():
    body = request.get_json(force=True)
    model = (body.get("model") or "").strip()
    prompt = (body.get("prompt") or "").strip()
    partial = body.get("partial") or ""  # set when continuing a truncated design
    batch = (body.get("batch") or "").strip()  # groups the A/B pair of a battle
    side = (body.get("side") or "").strip()
    if not model:
        return jsonify({"error": "model is required"}), 400
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    messages = [system_message(DESIGN_SYSTEM), user_message(prompt)]
    if partial:
        # Prefill: end on the assistant's partial text so the model continues it.
        messages.append({"role": "assistant", "content": partial})
        max_tokens = CONTINUE_MAX_TOKENS
        mode = "continue"
    else:
        max_tokens = _load_settings()["initial_token_limit"]
        mode = "generate"

    log.info("%s [%s] prompt=%r%s", mode, model, prompt[:80],
             f" (+{len(partial)}b partial)" if partial else "")
    t0 = time.time()
    try:
        resp = _client.chat(messages, model=model, max_tokens=max_tokens,
                            temperature=0.9)
    except OpenRouterError as e:
        log.error("%s [%s] API error: %s", mode, model, e)
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        log.exception("%s [%s] crashed", mode, model)
        return jsonify({"error": f"server error: {e}"}), 500

    dt = round(time.time() - t0, 1)
    raw = partial + resp.text
    truncated = resp.finish_reason == "length"
    usage = resp.usage or {}
    log.debug("%s [%s] %ss finish=%s tokens=%s resp_chars=%d",
              mode, model, dt, resp.finish_reason, usage.get("total_tokens"),
              len(resp.text))

    blocks = _extract_blocks(raw)
    if not blocks["html"].strip() and not blocks["css"].strip():
        log.warning("%s [%s] no html/css blocks; raw head: %r",
                    mode, model, raw[:200])
        return jsonify({"error": "No html/css code blocks found in response",
                        "raw": raw[:600]}), 422

    log.info("%s [%s] OK %ss html=%db css=%db js=%db trunc=%s",
             mode, model, dt, len(blocks["html"]), len(blocks["css"]),
             len(blocks["js"]), truncated)

    job_id = uuid.uuid4().hex[:12]
    _append_job({
        "id": job_id,
        "batch": batch or job_id,
        "side": side,
        "ts": time.time(),
        "iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "prompt": prompt,
        "elapsed": dt,
        "truncated": truncated,
        **blocks,
    })

    return jsonify({**blocks, "id": job_id, "raw": raw,
                    "truncated": truncated, "elapsed": dt})


# --------------------------------------------------------------------------- #
# history
# --------------------------------------------------------------------------- #
@app.route("/history")
def history():
    return send_from_directory(STATIC, "history.html")


@app.route("/api/jobs")
def api_jobs():
    """Battles (A/B pairs), newest first. Lightweight metadata only — the
    designs themselves are fetched per-job from /job/<id>."""
    rows = _read_jobs()
    # Keep only the latest row per (batch, side) so continuations supersede.
    latest: dict[tuple, dict] = {}
    for r in rows:
        latest[(r.get("batch"), r.get("side"))] = r

    battles: dict[str, dict] = {}
    for (batch, side), r in latest.items():
        b = battles.setdefault(batch, {"batch": batch, "ts": 0.0,
                                       "iso": r.get("iso"), "prompt": r.get("prompt"),
                                       "sides": {}})
        b["sides"][side or "?"] = {
            "id": r.get("id"), "model": r.get("model"),
            "elapsed": r.get("elapsed"), "truncated": r.get("truncated"),
        }
        if r.get("ts", 0) > b["ts"]:
            b["ts"] = r.get("ts", 0)
            b["iso"] = r.get("iso")
            b["prompt"] = r.get("prompt")

    ordered = sorted(battles.values(), key=lambda b: b["ts"], reverse=True)
    return jsonify(ordered)


@app.route("/api/vote", methods=["POST"])
def api_vote():
    """Record a blind preference: which side's design is better. This is the
    arena's preference-pair signal (chosen vs rejected) for ORPO/DPO."""
    body = request.get_json(force=True)
    winner = (body.get("winner") or "").strip()
    sides = body.get("sides") or {}
    if not winner or winner not in sides:
        return jsonify({"error": "winner must be one of the competing sides"}), 400
    record = {
        "ts": time.time(),
        "iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "batch": body.get("batch"),
        "prompt": body.get("prompt"),
        "winner": winner,
        "sides": sides,  # {a:{id,model}, b:{id,model}, …}
    }
    _append_vote(record)
    log.info("vote batch=%s winner=%s", record["batch"], winner)
    return jsonify({"ok": True})


@app.route("/job/<job_id>")
def job_render(job_id: str):
    """Serve a single stored design as a standalone HTML document."""
    for r in _read_jobs():
        if r.get("id") == job_id:
            doc = _render_doc(r.get("html", ""), r.get("css", ""), r.get("js", ""))
            return Response(doc, mimetype="text/html")
    return Response("job not found", status=404)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    port = int(os.environ.get("PORT", 7700))
    print(f"Glamour Arena → http://localhost:{port}")
    app.run(host="127.0.0.1", port=port, debug=True, threaded=True)


if __name__ == "__main__":
    main()
