from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Literal

Script = Literal[
    "latin",
    "cyrillic",
    "arabic",
    "han",
    "japanese",
    "hangul",
    "devanagari",
    "mixed",
    "unknown",
]

_WHITESPACE = re.compile(r"\s+")
_SEARCH_PUNCTUATION = re.compile(r"[^\w.\- '\u0600-\u06ff\u3040-\u30ff\u3400-\u9fff]+")
_BIDI_AND_FORMAT = {
    "\u061c",
    "\u200b",
    "\u200c",
    "\u200d",
    "\u200e",
    "\u200f",
    "\u202a",
    "\u202b",
    "\u202c",
    "\u202d",
    "\u202e",
    "\u2066",
    "\u2067",
    "\u2068",
    "\u2069",
    "\ufeff",
}
_ARABIC_TRANSLATION = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
        "ـ": "",
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
    }
)


def sanitize_unicode(text: str, *, max_length: int = 512) -> str:
    """Normalize untrusted OCR without interpreting it as instructions."""

    normalized = unicodedata.normalize("NFKC", text[:max_length])
    safe = "".join(
        character
        for character in normalized
        if character not in _BIDI_AND_FORMAT
        and (unicodedata.category(character) not in {"Cc", "Cs"} or character in "\n\t")
    )
    return _WHITESPACE.sub(" ", safe).strip()


def normalize_search_text(text: str) -> str:
    text = sanitize_unicode(text).translate(_ARABIC_TRANSLATION)
    text = "".join(
        character
        for character in text
        if not ("\u064b" <= character <= "\u065f" or character == "\u0670")
    )
    text = text.casefold().replace("i\u0307", "i")
    text = _SEARCH_PUNCTUATION.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip().strip(".-'").strip()


def secondary_search_key(text: str) -> str:
    """A lower-priority accent/Turkish-I tolerant key; never replaces the exact key."""

    normalized = normalize_search_text(text)
    normalized = normalized.replace("ı", "i").replace("i\u0307", "i")
    decomposed = unicodedata.normalize("NFKD", normalized)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _character_script(character: str) -> str | None:
    codepoint = ord(character)
    if 0x0041 <= codepoint <= 0x024F or 0x1E00 <= codepoint <= 0x1EFF:
        return "latin"
    if 0x0400 <= codepoint <= 0x052F or 0x2DE0 <= codepoint <= 0x2DFF:
        return "cyrillic"
    if 0x0600 <= codepoint <= 0x06FF or 0x0750 <= codepoint <= 0x08FF:
        return "arabic"
    if 0x3040 <= codepoint <= 0x309F:
        return "hiragana"
    if 0x30A0 <= codepoint <= 0x30FF:
        return "katakana"
    if 0x3400 <= codepoint <= 0x4DBF or 0x4E00 <= codepoint <= 0x9FFF:
        return "han"
    if 0xAC00 <= codepoint <= 0xD7AF or 0x1100 <= codepoint <= 0x11FF:
        return "hangul"
    if 0x0900 <= codepoint <= 0x097F:
        return "devanagari"
    return None


def detect_script(text: str) -> Script:
    counts = Counter(
        script
        for character in sanitize_unicode(text)
        if (script := _character_script(character)) is not None
    )
    if not counts:
        return "unknown"
    if counts["hiragana"] or counts["katakana"]:
        japanese_count = counts["hiragana"] + counts["katakana"] + counts["han"]
        other_count = sum(counts.values()) - japanese_count
        return "japanese" if other_count == 0 else "mixed"
    populated = [script for script, count in counts.items() if count > 0]
    if len(populated) > 1:
        largest, largest_count = counts.most_common(1)[0]
        if largest_count / sum(counts.values()) < 0.85:
            return "mixed"
        return largest  # type: ignore[return-value]
    return populated[0]  # type: ignore[return-value]


def language_hints(script: Script, profile: str) -> list[str]:
    profile_key = profile.casefold()
    profile_hints: tuple[str, ...] = ()
    if "arab" in profile_key:
        profile_hints = ("ar", "fa", "ur")
    elif "cyr" in profile_key:
        profile_hints = ("ru", "uk", "bg")
    elif "korean" in profile_key:
        profile_hints = ("ko",)
    elif "latin" in profile_key or "multi" in profile_key or "ppocrv6" in profile_key:
        profile_hints = ("und-Latn",)
    script_hints = {
        "latin": ("und-Latn",),
        "cyrillic": ("und-Cyrl",),
        "arabic": ("und-Arab",),
        "han": ("zh",),
        "japanese": ("ja",),
        "hangul": ("ko",),
        "devanagari": ("und-Deva",),
    }.get(script, ())
    return list(dict.fromkeys((*profile_hints, *script_hints)))[:6]
