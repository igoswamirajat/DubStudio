from __future__ import annotations

import re

# Comprehensive phonetic mappings for technical, programming, AI, and web terms
# mapped to clean Devanagari representations that Indian neural TTS engines pronounce
# with authentic native prosody, zero code-switching hesitation, and no robotic glitches.
TECH_TERMS_HI: dict[str, str] = {
    # AI & Tools
    "scrapegraphai": "स्क्रैपग्राफ़ एआई",
    "scrapegraph": "स्क्रैपग्राफ़",
    "chatgpt": "चैट जीपीटी",
    "gpt-4": "जीपीटी फोर",
    "gpt-4o": "जीपीटी फोर ओ",
    "gpt": "जीपीटी",
    "openai": "ओपन एआई",
    "claude": "क्लॉड",
    "gemini": "जेमिनी",
    "ollama": "ओलामा",
    "huggingface": "हगिंगफ़ेस",
    "hugging face": "हगिंगफ़ेस",
    "deepseek": "डीपसीक",
    "groq": "ग्रॉक",
    "anthropic": "एंथ्रोपिक",

    # Programming & Tech
    "python": "पायथन",
    "javascript": "जावास्क्रिप्ट",
    "typescript": "टाइपस्क्रिप्ट",
    "github": "गिटहब",
    "git": "गिट",
    "docker": "डॉकर",
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
    "chrome": "क्रोम",
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

# Known Latin product/brand names take precedence over the legacy phonetic map.
# This is an explicit allowlist, not general named-entity detection. Keep source
# spelling/case; do not guess reverse transliterations of Devanagari input.
DO_NOT_TRANSLITERATE: frozenset[str] = frozenset({
    "scrapegraphai", "scrapegraph", "chatgpt", "gpt-4", "gpt-4o",
    "openai", "claude", "claude code", "gemini", "ollama",
    "huggingface", "hugging face", "deepseek", "groq", "anthropic",
    "github", "docker", "chrome", "harness", "cloud code",
})

# Capture complete names longest-first, including horizontal whitespace variants.
# Splitting keeps protected spans out of number, term, and acronym replacement
# without placeholders that could collide with user text or be normalized.
_PROTECTED_NAME_PATTERN = re.compile(
    r"\b("
    + "|".join(
        r"[ \t]+".join(re.escape(word) for word in name.split())
        for name in sorted(DO_NOT_TRANSLITERATE, key=lambda name: (-len(name), name))
    )
    + r")\b",
    re.IGNORECASE,
)

_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE), repl)
    for term, repl in sorted(TECH_TERMS_HI.items(), key=lambda x: -len(x[0]))
]

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
    """Normalize Hindi loan words while preserving known Latin product names.

    Common technical vocabulary and acronyms retain their phonetic Devanagari
    forms for TTS. Protected names keep their source spelling and case, including
    digits in listed model names. Other languages are returned unchanged.
    """
    if not text or target_lang != "hi":
        return text or ""

    parts = _PROTECTED_NAME_PATTERN.split(text)
    # Captured names occupy odd positions and bypass every normalization stage.
    for index in range(0, len(parts), 2):
        result = expand_hindi_numbers(parts[index])
        for pat, repl in _PATTERNS:
            result = pat.sub(repl, result)
        for pat, repl in _ACRONYM_PATTERNS:
            result = pat.sub(repl, result)
        parts[index] = result

    return re.sub(r"[ \t]+", " ", "".join(parts)).strip()
