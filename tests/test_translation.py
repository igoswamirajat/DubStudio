"""Translation: OpenAI-compatible API path (mocked httpx) + hard failure.

This file used to assert that a dead translator "falls back to demo". That was
the bug, not the contract: the demo map swapped three words out of sixteen and
still marked the segment translated, so the dub shipped in the source language
with a synthetic voice on top. A dead translator must now fail loudly.
"""

from __future__ import annotations

import pytest

import dubstudio.pipeline.translation as tr
from dubstudio.settings import settings


class _FakeResp:
    status_code = 200

    def __init__(self, content: str):
        self._content = content

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def test_openai_translator_used(monkeypatch):
    monkeypatch.setattr(settings, "translator", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "test-key")

    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs.get("headers", {})
        captured["json"] = kwargs.get("json", {})
        return _FakeResp("नमस्ते दुनिया")

    monkeypatch.setattr(tr.httpx, "post", fake_post)

    segs = [{"segment_id": "seg_0", "source_text": "Hello world", "target_duration_ms": 1500}]
    out = tr.translate_segments(segs, source_language="en", target_language="hi")

    assert out[0]["translated_text"] == "नमस्ते दुनिया"
    assert out[0]["status"] == "translated"
    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"].get("Authorization") == "Bearer test-key"


def test_prompt_sent_to_the_api_carries_a_word_budget(monkeypatch):
    monkeypatch.setattr(settings, "translator", "openai")

    captured = {}

    def fake_post(url, **kwargs):
        captured.setdefault("messages", kwargs.get("json", {}).get("messages", []))
        return _FakeResp("नमस्ते दुनिया")

    monkeypatch.setattr(tr.httpx, "post", fake_post)
    tr.translate_segments(
        [{"segment_id": "seg_0", "source_text": "Hello world", "target_duration_ms": 1500}],
        source_language="en", target_language="hi",
    )

    system = captured["messages"][0]["content"]
    assert "words" in system
    assert "Hindi" in system


def test_openai_failure_raises_instead_of_word_mapping(monkeypatch):
    monkeypatch.setattr(settings, "translator", "openai")

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(tr.httpx, "post", boom)
    segs = [{"segment_id": "seg_0", "source_text": "hello", "target_duration_ms": 1000}]

    with pytest.raises(tr.TranslationError) as exc:
        tr.translate_segments(segs, source_language="en", target_language="hi")

    assert "unreachable" in str(exc.value).lower()
    assert not segs[0].get("translated_text")
