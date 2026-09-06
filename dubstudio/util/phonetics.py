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


def normalize_hinglish(text: str, target_lang: str = "hi") -> str:
    """Transliterate technical and English loan words to clean phonetic Devanagari.

    This ensures Indian TTS models (such as Maya Research Veena) pronounce technical
    terms and English code-switching smoothly without awkward accent shifts, hesitation,
    or robotic distortion.
    """
    if not text or target_lang != "hi":
        return text or ""

    result = text
    for pat, repl in _PATTERNS:
        result = pat.sub(repl, result)

    for pat, repl in _ACRONYM_PATTERNS:
        result = pat.sub(repl, result)

    return re.sub(r"[ \t]+", " ", result).strip()
