"""Tests for the grounded critique-enrichment stage.

These run without a browser or an API key: the VLM is replaced by a stub client,
so we exercise prompt construction, JSON parsing, the grounding guard, and the
graceful-degradation paths deterministically.
"""
import json
import os

from PIL import Image

from glamour import critique
from glamour.openrouter import ChatResponse

# A corpus record shaped exactly like a line from build.py's corpus.jsonl.
RECORD = {
    "id": "subscribe-card-contrast-0",
    "component": "subscribe-card",
    "injected_defect": "contrast",
    "screenshot": "renders/subscribe-card__contrast0.png",
    "target_element": "subtext",
    "target_bbox": [24, 60, 312, 20],
    "measured_severity": {"contrast_ratio": 2.34, "wcag_aa_pass": False},
    "critique_text": "The subtext is rgb(207, 207, 207) on rgb(255, 255, 255) ...",
    "fix_instruction": "Darken the subtext to meet WCAG AA (>= 4.5:1).",
}


class StubClient:
    """Stands in for OpenRouter; returns a fixed JSON body as the model output."""

    def __init__(self, body: str):
        self.body = body
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return ChatResponse(text=self.body, model="stub")


def _png(tmp_path):
    """Write a real PNG where the record's screenshot points, return image_root."""
    root = str(tmp_path)
    path = os.path.join(root, RECORD["screenshot"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (8, 8), (255, 255, 255)).save(path)
    return root


def test_headline_number():
    assert critique.headline_number(RECORD) == "2.3"
    assert critique.headline_number({"injected_defect": "alignment",
                                     "measured_severity": {"right_edge_offset_px": 46}}) == "46"
    assert critique.headline_number({"injected_defect": "nope",
                                     "measured_severity": {}}) is None


def test_build_messages_includes_facts_and_image(tmp_path):
    root = _png(tmp_path)
    msgs = critique.build_messages(RECORD, image_root=root)
    assert msgs[0]["role"] == "system"
    parts = msgs[1]["content"]
    text = parts[0]["text"]
    assert "contrast_ratio=2.34" in text and "subtext" in text
    assert any(p.get("type") == "image_url" for p in parts)


def test_enrich_uses_grounded_llm_output(tmp_path):
    root = _png(tmp_path)
    client = StubClient(json.dumps({
        "critique": "The subtext barely registers at 2.3:1 against the white card "
                    "— it dips under WCAG AA and reads as disabled.",
        "fix": "Bump the subtext to #4b5563 for a comfortable ~7:1.",
    }))
    out = critique.enrich_record(RECORD, client, image_root=root)
    assert out["critique_source"] == "llm"
    assert "2.3" in out["critique_text"]
    assert out["template_critique"] == RECORD["critique_text"]
    # the json_object response_format must be requested
    assert client.calls[0][1]["response_format"] == {"type": "json_object"}


def test_ungrounded_output_falls_back_to_template(tmp_path):
    root = _png(tmp_path)
    # plausible prose but the measured number was dropped -> reject
    client = StubClient(json.dumps({
        "critique": "The subtext is far too light against the card and fails AA.",
        "fix": "Use a darker gray.",
    }))
    out = critique.enrich_record(RECORD, client, image_root=root)
    assert out["critique_source"] == "template"
    assert out["critique_text"] == RECORD["critique_text"]


def test_no_client_passes_through_as_template():
    out = critique.enrich_record(RECORD, None)
    assert out["critique_source"] == "template"
    assert out["critique_text"] == RECORD["critique_text"]


def test_bad_json_falls_back(tmp_path):
    root = _png(tmp_path)
    out = critique.enrich_record(RECORD, StubClient("not json at all"), image_root=root)
    assert out["critique_source"] == "template"


def test_parse_json_handles_fenced_block():
    obj = critique._parse_json('```json\n{"critique": "x", "fix": "y"}\n```')
    assert obj == {"critique": "x", "fix": "y"}


def test_enrich_file_roundtrip(tmp_path):
    root = _png(tmp_path)
    in_path = os.path.join(root, "corpus.jsonl")
    out_path = os.path.join(root, "corpus.enriched.jsonl")
    with open(in_path, "w") as f:
        f.write(json.dumps(RECORD) + "\n")
        f.write(json.dumps({**RECORD, "id": "x2"}) + "\n")
    client = StubClient(json.dumps({"critique": "Only 2.3:1 here.", "fix": "Darken it."}))
    stats = critique.enrich_file(in_path, out_path, client, image_root=root)
    assert stats == {"total": 2, "llm": 2, "template": 0}
    lines = [json.loads(l) for l in open(out_path)]
    assert len(lines) == 2 and all(r["critique_source"] == "llm" for r in lines)
