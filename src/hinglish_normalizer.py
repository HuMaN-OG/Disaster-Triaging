"""
Modular 4-Stage Hinglish Preprocessing & Normalization Pipeline.

Pipeline Stages:
1. `clean()`: Lowercasing, noise reduction, repetitive punctuation/whitespace stripping.
2. `normalize()`: Dictionary lookups, spelling variant mapping, connector token normalization.
3. `canonicalize()`: Number-to-word conversion and disaster keyword anchoring.
4. `preprocess_for_embedding()`: Orchestrates clean() -> normalize() -> canonicalize().
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
NORM_DICT_PATH: Path = PROJECT_ROOT / "src" / "hinglish_normalization_dict.json"

NUMBER_MAP: dict[str, str] = {
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
    "15": "fifteen",
    "20": "twenty",
    "25": "twenty five",
    "50": "fifty",
    "100": "one hundred",
}

CANONICAL_ANCHORS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(paani|water|flood_water|baadh|flood)\b", re.IGNORECASE), "flood_water"),
    (re.compile(r"\b(aag|flame|fire)\b", re.IGNORECASE), "fire_outbreak"),
    (re.compile(r"\b(dhua|smoke)\b", re.IGNORECASE), "toxic_smoke"),
    (re.compile(r"\b(phansa|phanse|trapped|fasa|phase)\b", re.IGNORECASE), "people_trapped"),
    (re.compile(r"\b(madad|help|rescue)\b", re.IGNORECASE), "rescue_help"),
    (re.compile(r"\b(bhejo|send_help)\b", re.IGNORECASE), "send_rescue"),
    (re.compile(r"\b(ghayal|injured)\b", re.IGNORECASE), "injured_people"),
    (re.compile(r"\b(chhat|roof)\b", re.IGNORECASE), "building_roof"),
    (re.compile(r"\b(manzil|floor)\b", re.IGNORECASE), "building_floor"),
]


def load_dict_rules(dict_path: Path = NORM_DICT_PATH) -> dict[str, str]:
    """Load flat rule map from JSON dictionary."""
    if not dict_path.exists():
        return {}
    try:
        data = json.loads(dict_path.read_text(encoding="utf-8"))
        if "rules" in data:
            return data["rules"]
        # Fallback if nested categories
        rules: dict[str, str] = {}
        for cat, map_dict in data.items():
            if isinstance(map_dict, dict):
                rules.update(map_dict)
        return rules
    except Exception:
        return {}


def is_devanagari(text: str) -> bool:
    """Detect if text contains Devanagari Unicode characters (U+0900 to U+097F)."""
    return any("\u0900" <= char <= "\u097f" for char in text)


class HinglishNormalizer:
    """
    Modular Preprocessing Engine executing:
    clean() -> normalize() -> canonicalize() -> preprocess_for_embedding()
    """

    def __init__(self, norm_dict: dict[str, str] | None = None) -> None:
        self.norm_dict = norm_dict if norm_dict is not None else load_dict_rules()
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile word boundary regex patterns for high-speed replacement."""
        self.compiled_rules: list[tuple[re.Pattern[str], str]] = []
        # Sort keys by length descending to match multi-word phrases first
        sorted_keys = sorted(self.norm_dict.keys(), key=lambda k: len(k), reverse=True)
        for raw_k in sorted_keys:
            if not raw_k.startswith("_"):
                v = self.norm_dict[raw_k]
                pattern = re.compile(rf"\b{re.escape(raw_k)}\b", flags=re.IGNORECASE)
                self.compiled_rules.append((pattern, v))

    def clean(self, text: str) -> str:
        """Stage 1: Lowercasing, noise reduction, repetitive punctuation/whitespace stripping."""
        if not text or not text.strip():
            return text
        if is_devanagari(text):
            return text.strip()

        cleaned = text.lower()
        cleaned = re.sub(r"[\!\?]{2,}", "!", cleaned)
        cleaned = re.sub(r"[\.,\-;:]{2,}", ".", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def normalize(self, text: str) -> str:
        """Stage 2: Dictionary lookups, spelling variant mapping, connector token normalization."""
        if not text or is_devanagari(text):
            return text

        normalized = text
        for pattern, replacement in self.compiled_rules:
            normalized = pattern.sub(replacement, normalized)

        return normalized

    def canonicalize(self, text: str) -> str:
        """Stage 3: Number-to-word conversion and disaster keyword anchoring."""
        if not text or is_devanagari(text):
            return text

        tokens = text.split()
        canonicalized_tokens = []
        for token in tokens:
            word = NUMBER_MAP.get(token, token)
            canonicalized_tokens.append(word)

        res = " ".join(canonicalized_tokens)

        # Apply canonical semantic anchor tokens
        for pattern, anchor in CANONICAL_ANCHORS:
            res = pattern.sub(anchor, res)

        return res

    def preprocess_for_embedding(self, text: str) -> str:
        """Stage 4: Safe Preprocessing Execution: clean() -> normalize()."""
        if not text or not text.strip():
            return ""
        if is_devanagari(text):
            return text.strip()

        step1 = self.clean(text)
        step2 = self.normalize(step1)
        return step2


# Global default instance
_default_normalizer = HinglishNormalizer()


def normalize_hinglish_text(text: str) -> str:
    """Convenience wrapper executing the complete 4-stage preprocessing pipeline."""
    return _default_normalizer.preprocess_for_embedding(text)
