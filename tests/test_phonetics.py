from __future__ import annotations

from dubstudio.util.phonetics import normalize_hinglish


def test_normalize_hinglish_technical_terms():
    text = "यह Free AI Tool आपको ScrapeGraphAI और Python से GitHub पर scrape करने देता है"
    out = normalize_hinglish(text, "hi")
    assert "स्क्रैपग्राफ़ एआई" in out
    assert "पायथन" in out
    assert "गिटहब" in out
    assert "फ़्री" in out
    assert "टूल" in out


def test_normalize_hinglish_acronyms():
    text = "इस API और LLM का URL चेक करें"
    out = normalize_hinglish(text, "hi")
    assert "ए.पी.आई." in out
    assert "एल.एल.एम." in out
    assert "यू.आर.एल." in out


def test_normalize_hinglish_non_hi():
    text = "This is Python on GitHub"
    assert normalize_hinglish(text, "en") == text
