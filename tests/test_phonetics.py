from __future__ import annotations

import pytest

from dubstudio.util.phonetics import normalize_hinglish


def test_normalize_hinglish_technical_terms():
    text = "यह Free AI Tool आपको ScrapeGraphAI और Python से GitHub पर scrape करने देता है"
    out = normalize_hinglish(text, "hi")
    assert "ScrapeGraphAI" in out
    assert "पायथन" in out
    assert "GitHub" in out
    assert "फ़्री" in out
    assert "टूल" in out
    assert "स्क्रैप" in out


def test_normalize_hinglish_acronyms():
    text = "इस API और LLM का URL चेक करें"
    out = normalize_hinglish(text, "hi")
    assert "ए.पी.आई." in out
    assert "एल.एल.एम." in out
    assert "यू.आर.एल." in out


def test_normalize_hinglish_non_hi():
    text = "This is Python on GitHub"
    assert normalize_hinglish(text, "en") == text


@pytest.mark.parametrize(
    "name",
    [
        "DeepSeek", "GitHub", "Harness", "Cloud Code", "Claude Code",
        "ScrapeGraphAI", "ScrapeGraph", "ChatGPT", "OpenAI", "Claude",
        "Gemini", "Ollama", "HuggingFace", "Hugging Face", "Groq",
        "Anthropic", "Docker", "Chrome", "GPT-4", "GPT-4o",
        "deepseek", "GITHUB", "cloud code",
    ],
)
def test_normalize_hinglish_preserves_known_product_names(name):
    assert normalize_hinglish(f"({name}) इस्तेमाल करें।") == f"({name}) इस्तेमाल करें।"


def test_normalize_hinglish_protects_multiword_names_not_common_words():
    text = "Cloud Code और Claude Code से cloud पर code install करें"
    assert normalize_hinglish(text) == (
        "Cloud Code और Claude Code से क्लाउड पर कोड इंस्टॉल करें"
    )


def test_normalize_hinglish_multiword_name_whitespace():
    assert normalize_hinglish("Cloud\tCode और Hugging  Face") == (
        "Cloud Code और Hugging Face"
    )


def test_normalize_hinglish_preserves_versioned_names_before_number_expansion():
    text = "GPT-4 और GPT-4o के 5 API calls की कीमत $50 है"
    assert normalize_hinglish(text) == (
        "GPT-4 और GPT-4o के पांच ए.पी.आई. calls की कीमत पचास डॉलर है"
    )


def test_normalize_hinglish_product_name_boundaries():
    text = "myGitHub GitHubber DeepSeekish GitHub's"
    assert normalize_hinglish(text) == text


def test_normalize_hinglish_repeated_pass_preserves_names():
    text = "GitHub पर DeepSeek और GPT-4o का API install करें"
    once = normalize_hinglish(text)
    assert once == "GitHub पर DeepSeek और GPT-4o का ए.पी.आई. इंस्टॉल करें"
    assert normalize_hinglish(once) == once


def test_normalize_hinglish_common_vocabulary_unchanged():
    assert normalize_hinglish("install cloud code download upload API LLM URL") == (
        "इंस्टॉल क्लाउड कोड डाउनलोड अपलोड ए.पी.आई. एल.एल.एम. यू.आर.एल."
    )


def test_normalize_hinglish_numbers_and_whitespace_unchanged():
    assert normalize_hinglish("  30,000\tfiles और $50 तथा 5%  ") == (
        "तीस हज़ार फ़ाइल्स और पचास डॉलर तथा पांच प्रतिशत"
    )


def test_normalize_hinglish_empty():
    assert normalize_hinglish("") == ""


def test_normalize_hinglish_non_hi_preserves_numbers_and_whitespace():
    text = "  GitHub GPT-4 Cloud Code 50%\tAPI  "
    assert normalize_hinglish(text, "en") == text
