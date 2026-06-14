"""Glamour OpenRouter client — the LLM access layer for the arena + engine.

One thin client over the OpenRouter chat-completions API (OpenAI-compatible),
used in three places in Glamour:

  * **prompt generation** — ask a strong model to invent clean-UI briefs /
    component specs to seed the synthesis pipeline.
  * **competitor output** — have one or more models generate or fix a UI; their
    renders are what the arena judges blind.
  * **judging** — show a model the *rendering* (a screenshot) and have it
    critique / pick a winner. This is why the client is multimodal first-class:
    Glamour judges pixels, not just prose, so `chat` accepts image parts and
    `image_part()` turns a local PNG into the base64 data URI the API wants.

Stdlib only (urllib) — no extra deps to install for a headless CPU data box.
The key is read from $OPENROUTER_API_KEY by default.

    from glamour.openrouter import OpenRouter, user_message, image_part

    or = OpenRouter(model="anthropic/claude-sonnet-4.6")
    print(or.complete("Give me a one-line brief for a clean pricing card."))

    # vision judge: show the model the render + ask for a verdict
    msg = user_message("Critique this UI's visual quality.",
                       images=["output/renders/subscribe-card__clean.png"])
    print(or.chat([msg]).text)
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import pathlib
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

def _load_dotenv():
    """Load a .env file from the repo root (two levels up from this file) if present."""
    env_path = pathlib.Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

_load_dotenv()

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"
# Sent so this app shows up on OpenRouter's leaderboard / can be allow-listed.
DEFAULT_REFERER = "https://github.com/Schneewolf-Labs/Glamour"
DEFAULT_TITLE = "Glamour"


class OpenRouterError(RuntimeError):
    """An API/transport error from OpenRouter.

    `status` is the HTTP code (None for transport-level failures) and `body`
    is the raw response payload, kept so callers can inspect rate-limit or
    moderation details rather than just a flattened message.
    """

    def __init__(self, message: str, *, status: int | None = None, body: Any = None):
        super().__init__(message)
        self.status = status
        self.body = body


# --------------------------------------------------------------------------- #
# message construction — helpers so callers don't hand-build the content schema
# --------------------------------------------------------------------------- #
def text_part(text: str) -> dict:
    """A text content block."""
    return {"type": "text", "text": text}


def image_part(src: str, *, detail: str | None = None) -> dict:
    """An image content block.

    `src` is either an http(s) URL (passed through) or a local image path,
    which is read and inlined as a base64 data URI — the form the judge uses
    for freshly rendered screenshots that aren't hosted anywhere. `detail`
    ("low"/"high"/"auto") is forwarded when the model supports it.
    """
    if src.startswith(("http://", "https://", "data:")):
        url = src
    else:
        mime = mimetypes.guess_type(src)[0] or "image/png"
        with open(src, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        url = f"data:{mime};base64,{b64}"
    image_url: dict[str, Any] = {"url": url}
    if detail:
        image_url["detail"] = detail
    return {"type": "image_url", "image_url": image_url}


def user_message(text: str | None = None, *, images: Iterable[str] = ()) -> dict:
    """Build a user turn from optional text + any number of images.

    If there are no images we keep the simple string form (content is a plain
    str); otherwise we emit the multimodal content-parts array.
    """
    imgs = list(images)
    if not imgs:
        return {"role": "user", "content": text or ""}
    parts: list[dict] = []
    if text:
        parts.append(text_part(text))
    parts.extend(image_part(p) for p in imgs)
    return {"role": "user", "content": parts}


def system_message(text: str) -> dict:
    """A system turn."""
    return {"role": "system", "content": text}


# --------------------------------------------------------------------------- #
# response wrapper
# --------------------------------------------------------------------------- #
@dataclass
class ChatResponse:
    """A non-streamed completion, with `.text` as the common case and `.raw`
    kept for tool calls, logprobs, provider metadata, etc."""

    text: str
    model: str
    finish_reason: str | None = None
    usage: dict | None = None
    raw: dict = field(default_factory=dict)

    def __str__(self) -> str:  # so `print(resp)` does the obvious thing
        return self.text


# --------------------------------------------------------------------------- #
# client
# --------------------------------------------------------------------------- #
class OpenRouter:
    """A reusable OpenRouter chat client.

    Defaults (model, sampling) set here are per-instance; any can be overridden
    per `chat()` call so one client can drive the generator and the judge with
    different models.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        referer: str | None = DEFAULT_REFERER,
        title: str | None = DEFAULT_TITLE,
        timeout: float = 120.0,
        max_retries: int = 4,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise OpenRouterError(
                "No API key: pass api_key= or set $OPENROUTER_API_KEY."
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.referer = referer
        self.title = title
        self.timeout = timeout
        self.max_retries = max_retries

    # -- public API -------------------------------------------------------- #
    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        stream: bool = False,
        extra_body: dict | None = None,
    ) -> ChatResponse | Iterator[str]:
        """Run a chat completion.

        Returns a `ChatResponse` normally, or — when `stream=True` — an
        iterator of text deltas (handy for surfacing a judge's verdict live in
        the arena). `response_format={"type": "json_object"}` (or a json_schema
        spec) coerces structured output, which the engine wants for machine-read
        critiques. `extra_body` is merged in for OpenRouter-specific knobs
        (`provider`, `models` fallback list, `reasoning`, etc.).
        """
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": stream,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        if extra_body:
            payload.update(extra_body)

        if stream:
            return self._stream(payload)
        return self._request(payload)

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        images: Iterable[str] = (),
        **kwargs: Any,
    ) -> str:
        """One-shot convenience: (optional system) + a user turn -> text.

        Accepts `images=` for the vision path and forwards the rest to `chat`.
        """
        messages: list[dict] = []
        if system:
            messages.append(system_message(system))
        messages.append(user_message(prompt, images=images))
        resp = self.chat(messages, stream=False, **kwargs)
        assert isinstance(resp, ChatResponse)
        return resp.text

    # -- transport --------------------------------------------------------- #
    def _headers(self) -> dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.referer:
            h["HTTP-Referer"] = self.referer
        if self.title:
            h["X-Title"] = self.title
        return h

    def _open(self, payload: dict):
        """POST /chat/completions with retry/backoff, returning the live
        response object (caller reads it as a whole or streams it)."""
        url = f"{self.base_url}/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(
                url, data=data, headers=self._headers(), method="POST"
            )
            try:
                return urllib.request.urlopen(req, timeout=self.timeout)
            except urllib.error.HTTPError as e:
                body = _read_error_body(e)
                # Retry transient server / rate-limit statuses; fail fast on the
                # rest (bad key, bad request) since retrying won't help.
                if e.code in (408, 409, 429, 500, 502, 503, 504) and attempt < self.max_retries:
                    last_err = OpenRouterError(
                        f"HTTP {e.code}: {body}", status=e.code, body=body
                    )
                    time.sleep(_backoff(attempt))
                    continue
                raise OpenRouterError(
                    f"HTTP {e.code}: {body}", status=e.code, body=body
                ) from e
            except urllib.error.URLError as e:
                last_err = OpenRouterError(f"transport error: {e.reason}")
                if attempt < self.max_retries:
                    time.sleep(_backoff(attempt))
                    continue
                raise last_err from e
        # Exhausted retries on a transient error.
        raise last_err or OpenRouterError("request failed")

    def _request(self, payload: dict) -> ChatResponse:
        with self._open(payload) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        # OpenRouter can return a top-level error object even on HTTP 200.
        if isinstance(body.get("error"), dict):
            err = body["error"]
            raise OpenRouterError(
                err.get("message", "unknown error"),
                status=err.get("code"),
                body=err,
            )
        choice = (body.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        return ChatResponse(
            text=msg.get("content") or "",
            model=body.get("model", payload["model"]),
            finish_reason=choice.get("finish_reason"),
            usage=body.get("usage"),
            raw=body,
        )

    def _stream(self, payload: dict) -> Iterator[str]:
        """Yield content deltas from the SSE stream."""
        with self._open(payload) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue  # OpenRouter sends ": OPENROUTER PROCESSING" pings
                if isinstance(chunk.get("error"), dict):
                    raise OpenRouterError(
                        chunk["error"].get("message", "stream error"),
                        body=chunk["error"],
                    )
                delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    yield piece


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _backoff(attempt: int) -> float:
    """Exponential backoff: 2s, 4s, 8s, 16s …"""
    return 2.0 * (2 ** attempt)


def _read_error_body(e: urllib.error.HTTPError) -> Any:
    try:
        raw = e.read().decode("utf-8")
    except Exception:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


if __name__ == "__main__":  # tiny smoke test: `python -m glamour.openrouter`
    import sys

    client = OpenRouter()
    prompt = " ".join(sys.argv[1:]) or "In one sentence, what makes a UI look 'glamoured'?"
    print(f"[model={client.model}]")
    for token in client.chat([user_message(prompt)], stream=True):
        print(token, end="", flush=True)
    print()
