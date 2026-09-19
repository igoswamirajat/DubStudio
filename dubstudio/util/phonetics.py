from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Two kinds of English word come back from the translator and they need opposite
# treatment before the TTS sees them.
#
#   * NAMES - products, brands, companies and languages (see PROPER_NOUNS
#     below). These stay in Latin script, exactly as their owners write them: a
#     viewer reads "DeepSeek", never "डीपसीक", and "गिटहब" reads as a
#     mis-transliteration rather than as GitHub. The Latin words the translator
#     already leaves alone (install, plugin, version) reach the engine fine, so
#     names ride that same path.
#
#   * GENERIC technical vocabulary - install, cloud, plugin, API. These have
#     settled Devanagari spellings that Indian neural TTS engines pronounce with
#     authentic native prosody, zero code-switching hesitation and no robotic
#     glitches, so they are transliterated below.
# ---------------------------------------------------------------------------
TECH_TERMS_HI: dict[str, str] = {
    # Generic technical vocabulary
    "api": "ए.पी.आई.",
    "apis": "ए.पी.आई.ज़",
    "llm": "एल.एल.एम.",
    "llms": "एल.एल.एम.ज़",
    "ai": "ए.आई.",
    "ml": "एम.एल.",
    "open source": "ओपन सोर्स",
    "open-source": "ओपन सोर्स",
    "opensource": "ओपन सोर्स",
    "framework": "फ्रेमवर्क",
    "library": "लाइब्रेरी",
    "libraries": "लाइब्रेरीज",
    "code": "कोड",
    "coding": "कोडिंग",
    "developer": "डेवलपर",
    "developers": "डेवलपर्स",
    "repo": "रेपो",
    "repository": "रिपॉज़िटरी",
    "database": "डेटाबेस",
    "data": "डेटा",
    "dataset": "डेटासेट",
    "scrape": "स्क्रैप",
    "scraper": "स्क्रैपर",
    "scraping": "स्क्रैपिंग",
    "web": "वेब",
    "website": "वेबसाइट",
    "websites": "वेबसाइट्स",
    "internet": "इंटरनेट",
    "online": "ऑनलाइन",
    "offline": "ऑफ़लाइन",
    "free": "फ़्री",
    "unlimited": "अनलिमिटेड",
    "tool": "टूल",
    "tools": "टूल्स",
    "automation": "ऑटोमेशन",
    "automate": "ऑटोमेट",
    "install": "इंस्टॉल",
    "installation": "इंस्टॉलेशन",
    "setup": "सेटअप",
    "pipeline": "पाइपलाइन",
    "server": "सर्वर",
    "cloud": "क्लाउड",
    "download": "डाउनलोड",
    "upload": "अपलोड",
    "file": "फ़ाइल",
    "files": "फ़ाइल्स",
    "link": "लिंक",
    "browser": "ब्राउज़र",
    "extension": "एक्सटेंशन",
    "prompt": "प्रॉम्प्ट",
    "prompts": "प्रॉम्प्ट्स",
    "model": "मॉडल",
    "models": "मॉडल्स",
    "workflow": "वर्कफ़्लो",
    "platform": "प्लेटफ़ॉर्म",
    "token": "टोकन",
    "tokens": "टोकन्स",
    "app": "ऐप",
    "apps": "ऐप्स",
    "application": "एप्लिकेशन",
}

# Proper nouns: product, brand, company and language names. These must survive
# in Latin script - transliterating them is what turned "DeepSeek" into
# "डीपसीक" and "GitHub" into "गिटहब" in the dub. Matching is word-bounded and
# case-insensitive, and the original spelling is handed back untouched, so
# "GitHub" stays "GitHub" and "GitHub's" keeps its apostrophe.
PROPER_NOUNS: frozenset[str] = frozenset({
    # AI models, tools and the companies behind them
    "scrapegraphai", "scrapegraph",
    "chatgpt", "gpt-4", "gpt-4o", "gpt",
    "openai", "claude", "gemini", "ollama", "huggingface", "hugging face",
    "deepseek", "groq", "anthropic", "harness", "copilot", "cursor", "codex",
    # Developer platforms, tools and languages
    "github", "git", "docker", "chrome",
    "python", "javascript", "typescript",
    # Multi-word product names - the longest match wins, so these beat the
    # generic entries they contain ("cloud", "code").
    "cloud code", "vs code", "visual studio code", "google cloud",
})

_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE), repl)
    for term, repl in sorted(TECH_TERMS_HI.items(), key=lambda x: -len(x[0]))
]

_PROPER_NOUN_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
    for name in sorted(PROPER_NOUNS, key=lambda x: -len(x))
]

# Sentinels used to park a protected name while the other passes run. They come
# from the Unicode private use area: not letters (so the \b boundaries around
# the terms still being matched are undisturbed) and not digits (so the number
# expander cannot rewrite a parked slot index, which is exactly how "gpt-4"
# used to come out as "gpt-चार").
_HOLD_BASE = 0xE000

_ACRONYMS = {
    "URL": "यू.आर.एल.",
    "JSON": "जेसन",
    "HTML": "एच.टी.एम.एल.",
    "CSS": "सी.एस.एस.",
    "SQL": "एस.क्यू.एल.",
    "HTTP": "एच.टी.टी.पी.",
    "HTTPS": "एच.टी.टी.पी.एस.",
    "UI": "यू.आई.",
    "UX": "यू.एक्स.",
    "TTS": "टी.टी.एस.",
    "ASR": "ए.एस.आर.",
    "STT": "एस.टी.टी.",
    "BGM": "बी.जी.एम.",
    "SFX": "एस.एफ़.एक्स.",
    "VRAM": "वी-रैम",
    "RAM": "रैम",
    "CPU": "सी.पी.यू.",
    "GPU": "जी.पी.यू.",
}

_ACRONYM_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b" + re.escape(acr) + r"\b", re.IGNORECASE), repl)
    for acr, repl in sorted(_ACRONYMS.items(), key=lambda x: -len(x[0]))
]


_ONES = {
    0: "शून्य", 1: "एक", 2: "दो", 3: "तीन", 4: "चार", 5: "पांच",
    6: "छह", 7: "सात", 8: "आठ", 9: "नौ", 10: "दस",
    11: "ग्यारह", 12: "बारह", 13: "तेरह", 14: "चौदह", 15: "पंद्रह",
    16: "सोलह", 17: "सत्रह", 18: "अठारह", 19: "उन्नीस", 20: "बीस",
    21: "इक्कीस", 22: "बाईस", 23: "तेईस", 24: "चौबीस", 25: "पच्चीस",
    26: "छब्बीस", 27: "सत्ताईस", 28: "अट्ठाईस", 29: "उनतीस", 30: "तीस",
    31: "इकत्तीस", 32: "बत्तीस", 33: "तैंतीस", 34: "चौंतीस", 35: "पैंतीस",
    36: "छत्तीस", 37: "सैंतीस", 38: "अड़तीस", 39: "उनतालीस", 40: "चालीस",
    41: "इकतालीस", 42: "बयालीस", 43: "तैंतालीस", 44: "चवालीस", 45: "पैंतालीस",
    46: "छियालीस", 47: "सैंतालीस", 48: "अड़तालीस", 49: "उनचास", 50: "पचास",
    51: "इक्यावन", 52: "बावन", 53: "तिरेपन", 54: "चौवन", 55: "पचपन",
    56: "छप्पन", 57: "सत्तावन", 58: "अट्ठावन", 59: "उनसठ", 60: "साठ",
    61: "इकसठ", 62: "बासठ", 63: "तिरेसठ", 64: "चौंसठ", 65: "पैंसठ",
    66: "छियासठ", 67: "सरसठ", 68: "अड़सठ", 69: "उनहत्तर", 70: "सत्तर",
    71: "इकहत्तर", 72: "बहत्तर", 73: "तिहत्तर", 74: "चौहत्तर", 75: "पचहत्तर",
    76: "छिहत्तर", 77: "सतहत्तर", 78: "अठहत्तर", 79: "उनासी", 80: "अस्सी",
    81: "इक्यासी", 82: "बयासी", 83: "तिरासी", 84: "चौरासी", 85: "पचासी",
    86: "छियासी", 87: "सत्तासी", 88: "अट्ठासी", 89: "नवासी", 90: "नब्बे",
    91: "इक्यानवे", 92: "बानवे", 93: "तिरानवे", 94: "चौरानवे", 95: "पंचानवे",
    96: "छियानवे", 97: "सत्तानवे", 98: "अट्ठानवे", 99: "निन्यानवे", 100: "सौ",
}


def int_to_hindi(n: int) -> str:
    """Convert an integer to clean Hindi spoken words."""
    if n in _ONES:
        return _ONES[n]
    if n < 1000:
        h = n // 100
        rem = n % 100
        prefix = "एक सौ" if h == 1 else f"{_ONES.get(h, str(h))} सौ"
        return f"{prefix} {_ONES[rem]}" if rem else prefix
    if n < 100000:
        th = n // 1000
        rem = n % 1000
        prefix = f"{int_to_hindi(th)} हज़ार"
        return f"{prefix} {int_to_hindi(rem)}" if rem else prefix
    if n < 10000000:
        lakh = n // 100000
        rem = n % 100000
        prefix = f"{int_to_hindi(lakh)} लाख"
        return f"{prefix} {int_to_hindi(rem)}" if rem else prefix
    cr = n // 10000000
    rem = n % 10000000
    prefix = f"{int_to_hindi(cr)} करोड़"
    return f"{prefix} {int_to_hindi(rem)}" if rem else prefix


def expand_hindi_numbers(text: str) -> str:
    """Expand digits and numbers (with optional commas) into spoken Hindi words."""
    if not text:
        return ""

    # Replace currency & symbols first
    text = re.sub(r"\$(\d+(?:,\d+)*(?:\.\d+)?)", r"\1 डॉलर", text)
    text = re.sub(r"(\d+(?:,\d+)*(?:\.\d+)?)\s*%", r"\1 प्रतिशत", text)

    def _replace_num(match: re.Match) -> str:
        raw = match.group(0).replace(",", "")
        try:
            val = int(raw)
            return int_to_hindi(val)
        except Exception:
            return match.group(0)

    # Match numbers with optional commas like 30,000 or 100
    return re.sub(r"\b\d{1,3}(?:,\d{3})+\b|\b\d+\b", _replace_num, text)


def normalize_hinglish(text: str, target_lang: str = "hi") -> str:
    """Transliterate technical and English loan words to clean phonetic Devanagari.

    This ensures Indian TTS models (such as Maya Research Veena) pronounce technical
    terms and English code-switching smoothly without awkward accent shifts, hesitation,
    or robotic distortion.

    Proper nouns are the exception: names stay in Latin script (see PROPER_NOUNS).
    """
    if not text or target_lang != "hi":
        return text or ""

    # 1. Park proper nouns first, so none of the passes below can touch them.
    #    This has to happen before number expansion: "gpt-4" is a name, not a
    #    quantity, and expanding its digit would leave "gpt-चार" behind.
    held: list[str] = []
    result = text

    def _park(match: re.Match) -> str:
        held.append(match.group(0))
        return chr(_HOLD_BASE + len(held) - 1)

    for pat in _PROPER_NOUN_PATTERNS:
        result = pat.sub(_park, result)

    # 2. Expand numbers to spoken Hindi words
    result = expand_hindi_numbers(result)

    # 3. Transliterate technical terms and loan words
    for pat, repl in _PATTERNS:
        result = pat.sub(repl, result)

    # 4. Transliterate acronyms
    for pat, repl in _ACRONYM_PATTERNS:
        result = pat.sub(repl, result)

    # 5. Hand the names back exactly as they were written.
    for idx, name in enumerate(held):
        result = result.replace(chr(_HOLD_BASE + idx), name)

    return re.sub(r"[ \t]+", " ", result).strip()
