from __future__ import annotations

from dubstudio.util.phonetics import PROPER_NOUNS, normalize_hinglish


def test_normalize_hinglish_generic_terms_are_transliterated():
    text = "यह Free AI Tool आपको Python से GitHub पर scrape करने देता है"
    out = normalize_hinglish(text, "hi")
    # Generic technical vocabulary keeps its Devanagari spelling: it is what the
    # TTS engine pronounces cleanly, with no code-switching hesitation.
    assert "फ़्री" in out
    assert "टूल" in out
    assert "ए.आई." in out
    assert "स्क्रैप" in out


def test_normalize_hinglish_keeps_proper_nouns_in_latin():
    """Names are read, not decoded.

    "डीपसीक" and "गिटहब" are mis-transliterations a viewer has to work backwards
    from; the brand spells itself DeepSeek and GitHub.
    """
    text = "DeepSeek ने Cloud Code launch किया और GitHub पर Harness भी"
    out = normalize_hinglish(text, "hi")
    for name in ("DeepSeek", "Cloud Code", "GitHub", "Harness"):
        assert name in out, out
    for mangled in ("डीपसीक", "गिटहब", "हार्नेस", "क्लाउड कोड"):
        assert mangled not in out, out


def test_reported_brands_are_all_covered():
    # The four names from the bug report must be in the preserve list, or the
    # dictionary will transliterate them again the moment they appear.
    for name in ("deepseek", "github", "harness", "cloud code"):
        assert name in PROPER_NOUNS


def test_proper_nouns_keep_their_original_spelling():
    # Protection must not normalise case or strip possessives.
    assert normalize_hinglish("GITHUB और github", "hi") == "GITHUB और github"
    assert "GitHub's" in normalize_hinglish("GitHub's repo देखें", "hi")


def test_a_name_containing_a_digit_is_not_read_as_a_quantity():
    # "gpt-4" is a name, not a number. Expanding its digit produced "gpt-चार".
    assert normalize_hinglish("gpt-4 aur GPT-4", "hi") == "gpt-4 aur GPT-4"
    assert normalize_hinglish("Cloud Code 2.0", "hi").startswith("Cloud Code")


def test_numbers_outside_a_name_still_expand():
    out = normalize_hinglish("DeepSeek ne 180,000 stars liye", "hi")
    assert "DeepSeek" in out
    assert "एक लाख अस्सी हज़ार" in out


def test_normalize_hinglish_acronyms():
    text = "इस API और LLM का URL चेक करें"
    out = normalize_hinglish(text, "hi")
    assert "ए.पी.आई." in out
    assert "एल.एल.एम." in out
    assert "यू.आर.एल." in out


def test_normalize_hinglish_non_hi():
    text = "This is Python on GitHub"
    assert normalize_hinglish(text, "en") == text
