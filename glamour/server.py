"""Glamour Arena — Flask web server.

    uv run python -m glamour.server

Opens at http://localhost:5000
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import time
import urllib.request
from threading import Lock

from flask import Flask, jsonify, request, send_from_directory

from .openrouter import OpenRouter, OpenRouterError, system_message, user_message

STATIC = str(pathlib.Path(__file__).parent.parent / "static")
app = Flask(__name__)

# The client loads .env via _load_dotenv() on import.
_client = OpenRouter()

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
        return models


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #
DESIGN_SYSTEM = """\
You are competing in a live web design arena. Multiple AI models receive the same \
prompt; human judges vote for the best design.

Your entire response must be a single raw JSON object — no markdown fences, \
no explanation, nothing outside the JSON:

{"html": "...", "css": "...", "js": "..."}

Rules:
- html: content for <body> only — no <html>, <head>, <style>, or <script> wrappers
- css:  all CSS injected into <head><style> — use resets, variables, whatever you need
- js:   optional JavaScript; runs after DOM loads — empty string if unused
- NO external URLs, CDN links, or @import — everything must be self-contained
- Cover the full viewport (min-height: 100vh or equivalent)
- Include rich, realistic content: real headlines, copy, data — no Lorem Ipsum
- Make it visually striking: commit to a strong aesthetic with color, type, and layout
- Use CSS gradients, inline SVG, or Unicode for imagery/icons
"""

PROMPT_SYSTEM = """\
You are a creative web design director running a design competition.
Generate a single design brief for a web page or UI component.
Be specific and evocative: name a real domain, a distinct visual mood, \
and the type of page (landing page, dashboard, checkout, hero section, etc.).
Return only the brief text — 2-3 sentences, no preamble, no title.
"""


# --------------------------------------------------------------------------- #
# JSON extraction (handles models that wrap output in code fences)
# --------------------------------------------------------------------------- #
def _extract_json(text: str) -> dict:
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ```
    fence = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # If there's still surrounding text, grab first {...}
    if not text.startswith("{"):
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)
    return json.loads(text)


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
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate-prompt", methods=["POST"])
def api_generate_prompt():
    try:
        resp = _client.chat(
            [system_message(PROMPT_SYSTEM),
             user_message("Generate a creative web design brief.")],
            temperature=1.0,
            max_tokens=200,
        )
        return jsonify({"prompt": resp.text.strip()})
    except OpenRouterError as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/generate", methods=["POST"])
def api_generate():
    body = request.get_json(force=True)
    model = (body.get("model") or "").strip()
    prompt = (body.get("prompt") or "").strip()
    if not model:
        return jsonify({"error": "model is required"}), 400
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    try:
        resp = _client.chat(
            [system_message(DESIGN_SYSTEM), user_message(prompt)],
            model=model,
            max_tokens=8192,
            temperature=0.9,
        )
    except OpenRouterError as e:
        return jsonify({"error": str(e)}), 502

    try:
        result = _extract_json(resp.text)
    except (json.JSONDecodeError, AttributeError) as e:
        return jsonify({"error": f"Model returned invalid JSON: {e}",
                        "raw": resp.text[:600]}), 422

    return jsonify({
        "html": result.get("html", ""),
        "css":  result.get("css", ""),
        "js":   result.get("js", ""),
    })


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    port = int(os.environ.get("PORT", 7700))
    print(f"Glamour Arena → http://localhost:{port}")
    app.run(host="127.0.0.1", port=port, debug=True, threaded=True)


if __name__ == "__main__":
    main()
