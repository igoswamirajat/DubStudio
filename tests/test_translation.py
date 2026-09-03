"""Translation: OpenAI-compatible API path (mocked httpx) + demo fallback."""

from __future__ import annotations

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
        return _FakeResp("नमस्ते दुनिया")

    monkeypatch.setattr(tr.httpx, "post", fake_post)

    segs = [{"segment_id": "seg_0", "source_text": "Hello world", "target_duration_ms": 1500}]
    out = tr.translate_segments(segs, source_language="en", target_language="hi")

    assert out[0]["translated_text"] == "नमस्ते दुनिया"
    assert out[0]["status"] == "translated"
    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"].get("Authorization") == "Bearer test-key"


def test_openai_failure_falls_back_to_demo(monkeypatch):
    monkeypatch.setattr(settings, "translator", "openai")

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(tr.httpx, "post", boom)
    segs = [{"segment_id": "seg_0", "source_text": "hello", "target_duration_ms": 1000}]
    out = tr.translate_segments(segs, source_language="en", target_language="hi")
    # demo map translates "hello" -> Hindi, never crashes
    assert out[0]["translated_text"]
    assert out[0]["status"] == "translated"
