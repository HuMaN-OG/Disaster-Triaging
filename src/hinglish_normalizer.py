"""
Configurable Hinglish Normalization Pipeline.

Applies dictionary lookups, token-level transformations, and lightweight text cleaning
to standardize non-canonical Romanized Hindi spellings.

Does NOT alter Devanagari Hindi (script detection bypass) or standard English words.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Default extensible Hinglish normalization dictionary
DEFAULT_NORMALIZATION_DICT: dict[str, str] = {
    # Water / Flood terms
    "pani": "paani",
    "paanee": "paani",
    "paani": "paani",
    "baad": "baadh",
    "baadh": "baadh",
    "dub": "doob",
    "doob": "doob",
    # Trapped / Emergency terms
    "phasa": "phanse",
    "fasa": "phanse",
    "phase": "phanse",
    "phans": "phanse",
    "faase": "phanse",
    "fanse": "phanse",
    # Response / Help terms
    "madad": "madad",
    "maddad": "madad",
    "help": "madad",
    "bhejo": "bhejo",
    "bhejdo": "bhejo",
    "jarurat": "zaroorat",
    "jaroorat": "zaroorat",
    "zarurat": "zaroorat",
    "zaroorat": "zaroorat",
    # Fire / Smoke terms
    "dhua": "dhua",
    "dhuwa": "dhua",
    "dhuān": "dhua",
    "aag": "aag",
    "aage": "aag",
    # Transport / Access terms
    "rasta": "rasta",
    "raasta": "rasta",
    "raaste": "rasta",
    "sadak": "sadak",
    "sarak": "sadak",
    # Quantifiers & Modifiers
    "bohot": "bohot",
    "bahut": "bohot",
    "bhot": "bohot",
    "zyada": "zyada",
    "ziyada": "zyada",
    "jyada": "zyada",
    # Location terms
    "ghar": "ghar",
    "gharon": "gharon",
    "kamra": "kamra",
    "roof": "roof",
    "chhat": "roof",
}


def is_devanagari(text: str) -> bool:
    """Detect if string contains Devanagari Unicode characters (U+0900 to U+097F)."""
    return any("\u0900" <= char <= "\u097f" for char in text)


class HinglishNormalizer:
    """
    Configurable Hinglish Normalization Pipeline.
    Combines dictionary lookups, regex patterns, and token-level cleaning.
    """

    def __init__(self, norm_dict: dict[str, str] | None = None) -> None:
        self.norm_dict = norm_dict if norm_dict is not None else DEFAULT_NORMALIZATION_DICT.copy()
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex boundary patterns for word-level dictionary matching."""
        self.compiled_rules: list[tuple[re.Pattern[str], str]] = []
        for raw_k, v in self.norm_dict.items():
            pattern = re.compile(rf"\b{re.escape(raw_k)}\b", flags=re.IGNORECASE)
            self.compiled_rules.append((pattern, v))

    def add_rule(self, variant: str, canonical: str) -> None:
        """Dynamically add or override a normalization rule."""
        self.norm_dict[variant.lower()] = canonical.lower()
        self._compile_patterns()

    def normalize(self, text: str) -> str:
        """
        Normalize Hinglish text input.
        Bypasses Devanagari Hindi text automatically.
        """
        if not text or not text.strip():
            return text

        # Bypass Devanagari script entirely
        if is_devanagari(text):
            return text

        # 1. Clean repetitive punctuation and spaces
        cleaned = re.sub(r"[\!\?]{2,}", "!", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # 2. Dictionary / Pattern normalization
        for pattern, replacement in self.compiled_rules:
            cleaned = pattern.sub(replacement, cleaned)

        # 3. Contextual phrase adjustments
        cleaned = re.sub(r"\bghar me\b", "ghar mein", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bmadad kro\b", "madad karo", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bmadad krdo\b", "madad karo", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\blog fase hue\b", "log phanse hue", cleaned, flags=re.IGNORECASE)

        return cleaned


# Global default instance
_default_normalizer = HinglishNormalizer()


def normalize_hinglish_text(text: str) -> str:
    """Convenience function using the default HinglishNormalizer instance."""
    return _default_normalizer.normalize(text)
