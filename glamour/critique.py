"""Glamour — grounded critique synthesis (engine stage 2).

`build.py` renders defects and writes *templated* critiques anchored to DOM
measurements. They're factually correct but read like a linter. This stage
rewrites them in a natural designer/QA voice **without losing the grounding**:
the VLM is never asked to *find* the defect (that hallucinates) — it's handed
the exact defect kind, the measured numbers, the target element + bounding box,
and shown the render, then asked only to phrase a crisp critique + concrete fix
that references those facts.

The anti-hallucination guard: the headline measured number (contrast ratio,
pixel offset, font size, gap) must survive into the model's prose. If it drifts,
we keep the deterministic template. So the worst case is "still grounded, just
less natural" — never "natural but wrong".

Runs as a post-pass over the corpus, decoupled from the browser:

    python -m glamour.critique                       # output/corpus.jsonl -> *.enriched.jsonl
    python -m glamour.critique --model anthropic/claude-sonnet-4.6 --limit 20

With no $OPENROUTER_API_KEY it passes records through with the template
critique (marked critique_source="template"), so it's always safe to run.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable

from .openrouter import OpenRouter, OpenRouterError, image_part, system_message

OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")

SYSTEM = (
    "You are a senior product designer doing front-end visual QA. You are given "
    "the GROUND TRUTH for a single, known defect in a UI component — its kind, "
    "the exact measurement that proves it, the target element and its bounding "
    "box — and the rendered screenshot. Your job is NOT to hunt for problems: "
    "describe only the defect you are given. Write 1-2 sentences in a natural, "
    "specific designer voice, name the affected element, and cite the measured "
    "number verbatim so the critique stays anchored to fact. Then give one "
    "concrete, executable fix. Never invent issues that aren't in the ground "
    'truth. Respond as JSON: {"critique": "...", "fix": "..."}.'
)

# Per defect kind: which measured value is the "headline" number that must
# survive into the prose, and how to render it as a string (matching how the
# templates in build.py print it, so validation and template stay consistent).
_HEADLINE: dict[str, tuple[str, Callable[[Any], str]]] = {
    "contrast":   ("contrast_ratio",      lambda v: f"{float(v):.1f}"),
    "alignment":  ("right_edge_offset_px", lambda v: str(int(round(float(v))))),
    "type_small": ("font_size_px",         lambda v: f"{float(v):.0f}"),
    "cramped":    ("gap_px",               lambda v: f"{float(v):.0f}"),
}


def headline_number(record: dict) -> str | None:
    """The string form of the measurement that must appear in a grounded
    critique (e.g. "2.3" for a contrast ratio). None if the defect kind has no
    registered headline metric."""
    spec = _HEADLINE.get(record.get("injected_defect"))
    if not spec:
        return None
    key, fmt = spec
    sev = record.get("measured_severity") or {}
    if key not in sev:
        return None
    try:
        return fmt(sev[key])
    except (TypeError, ValueError):
        return None


def _facts_block(record: dict) -> str:
    """Render the ground-truth the model must stay faithful to."""
    sev = record.get("measured_severity") or {}
    meas = ", ".join(f"{k}={v}" for k, v in sev.items())
    return (
        "DEFECT (ground truth — do not second-guess it):\n"
        f"  kind:        {record.get('injected_defect')}\n"
        f"  component:   {record.get('component')}\n"
        f"  element:     {record.get('target_element')} "
        f"(bbox x,y,w,h = {record.get('target_bbox')})\n"
        f"  measurement: {meas}\n"
        f"  known fix:   {record.get('fix_instruction')}"
    )


def build_messages(record: dict, *, image_root: str = OUT) -> list[dict]:
    """System + user(text facts + screenshot). The screenshot path in a corpus
    record is stored relative to the output dir."""
    text = (
        _facts_block(record)
        + "\n\nRewrite this as a natural designer critique + concrete fix, "
        "citing the measured number."
    )
    img = os.path.join(image_root, record["screenshot"])
    user = {"role": "user", "content": [{"type": "text", "text": text}, image_part(img)]}
    return [system_message(SYSTEM), user]


def _parse_json(text: str) -> dict | None:
    """Best-effort JSON extraction (models sometimes fence or pad it)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1].lstrip("json").strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


def is_grounded(critique: str, fix: str, record: dict) -> bool:
    """The anti-hallucination guard: the headline measurement must appear in the
    generated prose. Kinds without a headline number are accepted as-is."""
    num = headline_number(record)
    if num is None:
        return True
    blob = f"{critique} {fix}"
    return num in blob


def enrich_record(
    record: dict,
    client: OpenRouter | None,
    *,
    model: str | None = None,
    temperature: float = 0.4,
    max_tokens: int = 400,
    image_root: str = OUT,
) -> dict:
    """Return a copy with a natural-voice critique when a grounded one is
    obtained, else the original template (annotated via `critique_source`).
    Failures (no client, API error, bad JSON, ungrounded) degrade gracefully."""
    out = dict(record)
    out.setdefault("template_critique", record.get("critique_text"))
    if client is None:
        out["critique_source"] = "template"
        return out
    try:
        messages = build_messages(record, image_root=image_root)
        resp = client.chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        obj = _parse_json(getattr(resp, "text", "") or "")
    except (OpenRouterError, OSError, ValueError):
        obj = None

    if obj and obj.get("critique") and is_grounded(
        str(obj.get("critique", "")), str(obj.get("fix", "")), record
    ):
        out["critique_text"] = str(obj["critique"]).strip()
        if obj.get("fix"):
            out["fix_instruction"] = str(obj["fix"]).strip()
        out["critique_source"] = "llm"
    else:
        out["critique_source"] = "template"
    return out


def enrich_file(
    in_path: str,
    out_path: str,
    client: OpenRouter | None,
    *,
    model: str | None = None,
    limit: int = 0,
    image_root: str = OUT,
) -> dict:
    """Stream a corpus jsonl through enrichment; return a small stats dict."""
    stats = {"total": 0, "llm": 0, "template": 0}
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if limit and stats["total"] >= limit:
                break
            rec = enrich_record(json.loads(line), client, model=model, image_root=image_root)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["total"] += 1
            stats[rec["critique_source"]] += 1
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Grounded VLM critique enrichment.")
    ap.add_argument("--in", dest="in_path", default=os.path.join(OUT, "corpus.jsonl"))
    ap.add_argument("--out", dest="out_path", default=os.path.join(OUT, "corpus.enriched.jsonl"))
    ap.add_argument("--model", default=None, help="OpenRouter model id (default: client default)")
    ap.add_argument("--limit", type=int, default=0, help="cap records (0 = all)")
    args = ap.parse_args()

    client: OpenRouter | None
    try:
        client = OpenRouter(model=args.model) if args.model else OpenRouter()
    except OpenRouterError as e:
        print(f"note: {e}\n      passing records through with template critiques.")
        client = None

    stats = enrich_file(args.in_path, args.out_path, client, model=args.model, limit=args.limit)
    print(
        f"enriched {stats['total']} records -> {args.out_path}\n"
        f"  llm-grounded: {stats['llm']}   template fallback: {stats['template']}"
    )


if __name__ == "__main__":
    main()
