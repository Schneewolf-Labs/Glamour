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
from threading import Lock, Thread

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
DATASET_PATH = OUT / "dataset.jsonl"
SETTINGS_PATH = OUT / "settings.json"
CURATE_SESSION_PATH = OUT / "curate_session.json"
app = Flask(__name__)

# --------------------------------------------------------------------------- #
# settings — persisted to output/settings.json
# --------------------------------------------------------------------------- #
DEFAULT_SETTINGS = {
    "initial_token_limit": 8192,
    "request_timeout": 120,   # seconds to wait for an OpenRouter response
    "temperature": 0.9,       # sampling temperature for design generation
    "prompt_model": "~anthropic/claude-sonnet-latest",  # model that writes briefs
    "hf_repo": "",            # HuggingFace dataset repo for curate uploads
    "favorites": [],
}
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


def _append_dataset(record: dict) -> None:
    with _jobs_lock:
        OUT.mkdir(exist_ok=True)
        with open(DATASET_PATH, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _dataset_count() -> int:
    if not DATASET_PATH.exists():
        return 0
    with _jobs_lock, open(DATASET_PATH) as f:
        return sum(1 for line in f if line.strip())


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


def _fetch_models(force: bool = False) -> list[dict]:
    global _models_cache, _models_fetched_at
    with _models_lock:
        fresh = _models_cache is not None and (time.time() - _models_fetched_at) < MODELS_TTL
        if fresh and not force:
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
    force = request.args.get("refresh") in ("1", "true", "yes")
    try:
        return jsonify(_fetch_models(force=force))
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
    if "request_timeout" in body:
        try:
            n = int(body["request_timeout"])
        except (ValueError, TypeError):
            return jsonify({"error": "request_timeout must be an integer"}), 400
        cur["request_timeout"] = max(10, min(n, 600))
    if "temperature" in body:
        try:
            t = float(body["temperature"])
        except (ValueError, TypeError):
            return jsonify({"error": "temperature must be a number"}), 400
        cur["temperature"] = max(0.0, min(t, 2.0))
    if "prompt_model" in body:
        pm = str(body["prompt_model"]).strip()
        if pm:
            cur["prompt_model"] = pm
    if "hf_repo" in body:
        cur["hf_repo"] = str(body["hf_repo"]).strip()
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
    log.info("settings saved: token_limit=%s timeout=%ss temp=%s prompt_model=%s favorites=%d",
             saved["initial_token_limit"], saved["request_timeout"],
             saved["temperature"], saved["prompt_model"], len(saved["favorites"]))
    return jsonify(saved)


def _generate_brief() -> str:
    """Generate one seeded design brief and record it for anti-repetition."""
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
    s = _load_settings()
    log.info("brief seeds: page=%r aesthetic=%r spark=%r", page, aesthetic, spark)
    resp = _client.chat(
        [system_message(PROMPT_SYSTEM), user_message(user)],
        model=s["prompt_model"], temperature=1.0, max_tokens=200,
    )
    prompt = resp.text.strip()
    _recent_prompts.append(prompt[:100])
    return prompt


@app.route("/api/generate-prompt", methods=["POST"])
def api_generate_prompt():
    try:
        prompt = _generate_brief()
        log.info("generate-prompt OK -> %r", prompt)
        return jsonify({"prompt": prompt})
    except OpenRouterError as e:
        log.error("generate-prompt failed: %s", e)
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        log.exception("generate-prompt crashed")
        return jsonify({"error": f"server error: {e}"}), 500


CONTINUE_MAX_TOKENS = 2000


def _is_prefill_unsupported(e: OpenRouterError) -> bool:
    """True if the error is a provider rejecting assistant-message prefill
    (e.g. newest Claude: 'must end with a user message')."""
    blob = f"{e} {getattr(e, 'body', '')}".lower()
    return "prefill" in blob or "must end with a user message" in blob


@app.route("/api/generate", methods=["POST"])
def api_generate():
    body = request.get_json(force=True)
    model = (body.get("model") or "").strip()
    prompt = (body.get("prompt") or "").strip()
    partial = body.get("partial") or ""  # set when continuing a truncated design
    batch = (body.get("batch") or "").strip()  # groups the A/B pair of a battle
    side = (body.get("side") or "").strip()
    store = body.get("store", True)  # curate sets False to skip the arena jobs log
    if not model:
        return jsonify({"error": "model is required"}), 400
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    s = _load_settings()
    temp = float(s["temperature"])
    base = [system_message(DESIGN_SYSTEM), user_message(prompt)]

    log.info("%s [%s] prompt=%r%s", "continue" if partial else "generate", model,
             prompt[:80], f" (+{len(partial)}b partial)" if partial else "")
    t0 = time.time()
    # Per-request client so the configured timeout applies without mutating the
    # shared client under threads.
    client = OpenRouter(timeout=float(s["request_timeout"]))
    try:
        if partial:
            mode = "continue"
            try:
                # Prefill: end on the assistant's partial so the model extends it.
                resp = client.chat(base + [{"role": "assistant", "content": partial}],
                                   model=model, max_tokens=CONTINUE_MAX_TOKENS,
                                   temperature=temp)
                raw = partial + resp.text
            except OpenRouterError as e:
                if not _is_prefill_unsupported(e):
                    raise
                # Provider rejects prefill (e.g. newest Claude) — regenerate the
                # whole design with a bigger budget instead of extending in place.
                mode = "continue-regen"
                budget = min(32000, len(partial) // 4 + 4000)
                log.info("continue [%s]: prefill unsupported, regenerating @ %d tokens",
                         model, budget)
                resp = client.chat(base, model=model, max_tokens=budget, temperature=temp)
                raw = resp.text
        else:
            mode = "generate"
            resp = client.chat(base, model=model,
                               max_tokens=s["initial_token_limit"], temperature=temp)
            raw = resp.text
    except OpenRouterError as e:
        log.error("generate [%s] API error: %s", model, e)
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        log.exception("generate [%s] crashed", model)
        return jsonify({"error": f"server error: {e}"}), 500

    dt = round(time.time() - t0, 1)
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
    if store:
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
    """Record a per-competitor rating (thumbs up/down). Each up beats each down
    within a battle — the arena's preference signal (chosen vs rejected) for
    ORPO/DPO. One record per rating event; latest per (batch, side) wins."""
    body = request.get_json(force=True)
    rating = (body.get("rating") or "").strip()
    side = (body.get("side") or "").strip()
    if rating not in ("up", "down", "none"):
        return jsonify({"error": "rating must be 'up', 'down', or 'none'"}), 400
    record = {
        "ts": time.time(),
        "iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "batch": body.get("batch"),
        "prompt": body.get("prompt"),
        "side": side,
        "rating": rating,
        "model": body.get("model"),
        "id": body.get("id"),
    }
    _append_vote(record)
    log.info("vote batch=%s side=%s rating=%s model=%s",
             record["batch"], side, rating, record["model"])
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
# curate — queue auto-prompts, one model generates, approve/reject → dataset
# --------------------------------------------------------------------------- #
@app.route("/curate")
def curate():
    return send_from_directory(STATIC, "curate.html")


# Server-owned curate session: the queue + generated-but-unreviewed designs live
# on disk (output/curate_session.json) and generation runs in background threads,
# so closing the tab or restarting the server resumes exactly where you left off.
_curate_lock = Lock()
_curate: dict = {"active": False, "models": [], "parallel": 3,
                 "queue": [], "idx": 0, "approved": 0, "results": {},
                 "target": 0, "queuing": False}
_curate_resumed = False


def _curate_model_for(i: int) -> str:
    """Round-robin the model mix across designs. Caller holds _curate_lock."""
    ms = _curate.get("models") or []
    return ms[i % len(ms)] if ms else ""


def _save_curate_locked() -> None:
    OUT.mkdir(exist_ok=True)
    CURATE_SESSION_PATH.write_text(json.dumps(_curate))


def _load_curate() -> None:
    global _curate
    if not CURATE_SESSION_PATH.exists():
        return
    try:
        data = json.loads(CURATE_SESSION_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return
    base = {"active": False, "models": [], "parallel": 3,
            "queue": [], "idx": 0, "approved": 0, "results": {},
            "target": 0, "queuing": False}
    base.update({k: data[k] for k in base if k in data})
    # Back-compat: older sessions stored a single "model".
    if not base["models"] and data.get("model"):
        base["models"] = [data["model"]]
    # Drop generations that were interrupted mid-flight; they'll regenerate.
    base["results"] = {k: v for k, v in base["results"].items()
                       if v.get("status") in ("done", "error")}
    _curate = base


def _generate_site(model: str, prompt: str) -> dict:
    """One-shot design generation for curate. Returns blocks + truncated + id."""
    s = _load_settings()
    client = OpenRouter(timeout=float(s["request_timeout"]))
    resp = client.chat([system_message(DESIGN_SYSTEM), user_message(prompt)],
                       model=model, max_tokens=s["initial_token_limit"],
                       temperature=float(s["temperature"]))
    blocks = _extract_blocks(resp.text)
    if not blocks["html"].strip() and not blocks["css"].strip():
        raise OpenRouterError("model returned no html/css blocks")
    return {**blocks, "truncated": resp.finish_reason == "length",
            "id": uuid.uuid4().hex[:12]}


def _curate_pump() -> None:
    """Keep up to `parallel` generations in flight, working ahead of `idx`."""
    to_start = []
    with _curate_lock:
        if not _curate["active"]:
            return
        n = _curate["parallel"]
        q, res = _curate["queue"], _curate["results"]
        inflight = sum(1 for v in res.values() if v.get("status") == "pending")
        i = _curate["idx"]
        while inflight + len(to_start) < n and i < len(q):
            if str(i) not in res:
                res[str(i)] = {"status": "pending"}
                to_start.append(i)
            i += 1
        if to_start:
            _save_curate_locked()
    for i in to_start:
        Thread(target=_curate_worker, args=(i,), daemon=True).start()


def _curate_worker(i: int) -> None:
    key = str(i)
    with _curate_lock:
        if not _curate["active"] or _curate["results"].get(key, {}).get("status") != "pending":
            return
        model, prompt = _curate_model_for(i), _curate["queue"][i]
    try:
        site = _generate_site(model, prompt)
        result = {"status": "done", "data": {"prompt": prompt, "model": model, **site}}
        log.info("curate: generated #%d [%s]", i, model)
    except Exception as e:
        result = {"status": "error", "error": str(e)}
        log.warning("curate: gen #%d failed: %s", i, e)
    with _curate_lock:
        if _curate["active"] and key in _curate["results"]:
            _curate["results"][key] = result
            _save_curate_locked()
    _curate_pump()


def _curate_producer() -> None:
    """Generate briefs in the background, appending to the queue as they land so
    review can start on the first one without waiting for the whole batch."""
    while True:
        with _curate_lock:
            if not _curate["active"] or not _curate["queuing"]:
                return
            if len(_curate["queue"]) >= _curate["target"]:
                _curate["queuing"] = False
                _save_curate_locked()
                return
        try:
            brief = _generate_brief()
        except Exception as e:
            log.warning("curate: brief generation failed, stopping queue: %s", e)
            with _curate_lock:
                _curate["queuing"] = False
                _save_curate_locked()
            return
        with _curate_lock:
            if not _curate["active"]:
                return
            _curate["queue"].append(brief)
            if len(_curate["queue"]) >= _curate["target"]:
                _curate["queuing"] = False
            _save_curate_locked()
        _curate_pump()  # designs can start the moment a prompt exists


def _session_view() -> dict:
    with _curate_lock:
        q, idx, res = _curate["queue"], _curate["idx"], _curate["results"]
        statuses = [res.get(str(i), {}).get("status", "todo") for i in range(len(q))]
        ready_ahead = sum(1 for i in range(idx + 1, len(q))
                          if res.get(str(i), {}).get("status") == "done")
        current = None
        if idx < len(q):
            cur = res.get(str(idx))
            if not cur:
                current = {"status": "todo", "prompt": q[idx], "model": _curate_model_for(idx)}
            elif cur["status"] == "done":
                current = {"status": "done", "prompt": q[idx], **cur["data"]}
            elif cur["status"] == "error":
                current = {"status": "error", "prompt": q[idx],
                           "model": _curate_model_for(idx), "error": cur.get("error")}
            else:
                current = {"status": "pending", "prompt": q[idx], "model": _curate_model_for(idx)}
        queuing = _curate.get("queuing", False)
        return {
            "active": _curate["active"], "models": _curate.get("models", []),
            "parallel": _curate["parallel"], "total": len(q), "idx": idx,
            "approved": _curate["approved"], "statuses": statuses,
            "ready_ahead": ready_ahead, "queuing": queuing,
            "target": _curate.get("target", len(q)),
            "done": _curate["active"] and not queuing and idx >= len(q),
            "current": current,
        }


def _maybe_resume() -> None:
    """Resume background generation after a server restart, lazily on first hit."""
    global _curate_resumed
    if _curate_resumed:
        return
    _curate_resumed = True
    if _curate.get("active"):
        log.info("curate: resuming session (idx=%d/%d)", _curate["idx"], len(_curate["queue"]))
        _curate_pump()
        if _curate.get("queuing") and len(_curate["queue"]) < _curate.get("target", 0):
            Thread(target=_curate_producer, daemon=True).start()


@app.route("/api/curate/start", methods=["POST"])
def api_curate_start():
    body = request.get_json(force=True)
    raw = body.get("models")
    if isinstance(raw, list):
        models = [str(m).strip() for m in raw if str(m).strip()]
    else:
        single = (body.get("model") or "").strip()
        models = [single] if single else []
    if not models:
        return jsonify({"error": "at least one model is required"}), 400
    try:
        count = max(1, min(int(body.get("count", 10)), 200))
        parallel = max(1, min(int(body.get("parallel", 3)), 8))
    except (ValueError, TypeError):
        return jsonify({"error": "count/parallel must be integers"}), 400

    with _curate_lock:
        _curate.update(active=True, models=models, parallel=parallel,
                       queue=[], idx=0, approved=0, results={},
                       target=count, queuing=True)
        _save_curate_locked()
    log.info("curate: started target=%d parallel=%d models=%s", count, parallel, models)
    # Briefs generate in the background; returns instantly so review can begin
    # the moment the first design is ready.
    Thread(target=_curate_producer, daemon=True).start()
    return jsonify(_session_view())


@app.route("/api/curate/session")
def api_curate_session():
    _maybe_resume()
    return jsonify(_session_view())


@app.route("/api/curate/approve", methods=["POST"])
def api_curate_approve():
    with _curate_lock:
        idx = _curate["idx"]
        cur = _curate["results"].get(str(idx))
        if not cur or cur.get("status") != "done":
            return jsonify({"error": "current design is not ready"}), 400
        d = cur["data"]
    rec = {
        "id": d.get("id") or uuid.uuid4().hex[:12],
        "ts": time.time(),
        "iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prompt": d["prompt"], "model": d.get("model", ""),
        "html": d.get("html", ""), "css": d.get("css", ""), "js": d.get("js", ""),
    }
    _append_dataset(rec)
    count = _dataset_count()
    with _curate_lock:
        _curate["approved"] += 1
        _curate["results"].pop(str(_curate["idx"]), None)
        _curate["idx"] += 1
        _save_curate_locked()
    log.info("curate: approved -> dataset (%d total)", count)
    _curate_pump()
    view = _session_view()
    view["dataset_count"] = count
    return jsonify(view)


@app.route("/api/curate/reject", methods=["POST"])
def api_curate_reject():
    with _curate_lock:
        _curate["results"].pop(str(_curate["idx"]), None)
        _curate["idx"] += 1
        _save_curate_locked()
    _curate_pump()
    return jsonify(_session_view())


@app.route("/api/curate/clear", methods=["POST"])
def api_curate_clear():
    with _curate_lock:
        _curate.update(active=False, queue=[], results={}, idx=0, approved=0)
        _save_curate_locked()
    return jsonify(_session_view())


@app.route("/api/curate/stats")
def api_curate_stats():
    return jsonify({"count": _dataset_count(),
                    "hf_repo": _load_settings()["hf_repo"]})


@app.route("/dataset.jsonl")
def dataset_download():
    if not DATASET_PATH.exists():
        return Response("no dataset yet", status=404)
    return send_from_directory(str(OUT), "dataset.jsonl",
                               as_attachment=True, mimetype="application/x-ndjson")


@app.route("/api/curate/upload", methods=["POST"])
def api_curate_upload():
    """Push the curated dataset.jsonl to a HuggingFace dataset repo.
    Token comes from $HF_TOKEN (loadable via .env)."""
    body = request.get_json(force=True)
    repo = (body.get("repo") or _load_settings()["hf_repo"]).strip()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not repo:
        return jsonify({"error": "no HuggingFace repo specified (owner/name)"}), 400
    if not token:
        return jsonify({"error": "no HF token: set HF_TOKEN in your .env"}), 400
    if _dataset_count() == 0:
        return jsonify({"error": "dataset is empty — approve some designs first"}), 400

    # Remember the repo for next time.
    s = _load_settings()
    s["hf_repo"] = repo
    _save_settings(s)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        return jsonify({"error": "huggingface_hub not installed (uv sync)"}), 500

    try:
        api = HfApi(token=token)
        api.create_repo(repo, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(DATASET_PATH),
            path_in_repo="data/dataset.jsonl",
            repo_id=repo,
            repo_type="dataset",
        )
    except Exception as e:
        log.exception("HF upload failed")
        return jsonify({"error": f"upload failed: {e}"}), 502

    url = f"https://huggingface.co/datasets/{repo}"
    n = _dataset_count()
    log.info("curate: uploaded %d rows to %s", n, repo)
    return jsonify({"ok": True, "url": url, "count": n})


# Restore any in-progress curate session from disk (generation resumes lazily
# on the first /api/curate/session request, so the reloader doesn't double-run it).
_load_curate()


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    port = int(os.environ.get("PORT", 7700))
    print(f"Glamour Arena → http://localhost:{port}")
    app.run(host="127.0.0.1", port=port, debug=True, threaded=True)


if __name__ == "__main__":
    main()
