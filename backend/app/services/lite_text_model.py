"""Low-memory, transparent text-only ML for a FREE-tier portfolio prototype.

This is a supervised nearest-neighbor model fitted to 2,000 labeled text reports
in disaster_dataset.json. It is NOT the project's CatBoost/SentenceTransformer
research ensemble. It makes no image classifications and has not been clinically
or operationally validated. No emergency dispatch is performed.
"""
from collections import Counter, defaultdict
from heapq import nlargest
from math import log, sqrt
from pathlib import Path
import json
import re

from fastapi import HTTPException, status
from app.utils.language import detect_language


ROOT = Path(__file__).resolve().parents[3]
REPORTS_PATH = ROOT / "disaster_dataset.json"
COMMON = frozenset(
    "the and for are was with from this that when where what your have has they their "
    "those these then than there here near into around under above about please "
    "report reporting citizen case reference ref people area place local road "
    "after before during along just already also but not yet many some our "
    "you them did its him her she his the a an on in of to at by is it as "
    "we will can may be being been i my us our their do".split()
)
# Recognize a few common disaster terms in transliterated Hindi / Devanagari.
# This is dictionary normalization only and is NOT trained multilingual ML.
LEXICON = {
    "paani": "water", "pani": "water", "baadh": "flood", "barish": "rain",
    "aag": "fire", "dhua": "smoke", "madad": "help", "bhejo": "send",
    "fas": "trapped", "fasa": "trapped", "fasgaya": "trapped",
    "phas": "trapped", "phanse": "trapped", "ghayal": "injured",
    "bhukamp": "earthquake", "bhookamp": "earthquake",
    "बाढ़": "flood", "बाढ़": "flood", "पानी": "water",
    "आग": "fire", "धुआँ": "smoke", "धुआं": "smoke",
    "भूकंप": "earthquake", "मदद": "help", "फंसे": "trapped",
    "घायल": "injured", "बचाओ": "rescue",
}
NUM_PATTERN = re.compile(r"\b(?:case|ref|reference)\s+[A-Z]{1,4}\s*\d+\b", re.I)
TOKENS = re.compile(r"[\w]+", re.UNICODE)


def _terms(text: str) -> frozenset[str]:
    clean = NUM_PATTERN.sub(" ", text.lower())
    # Python's \w tokenization can split Devanagari vowel marks, so normalize
    # known Hindi phrases BEFORE tokenization instead of relying on exact tokens.
    for word, english in LEXICON.items():
        if any(ord(char) > 127 for char in word):
            clean = clean.replace(word, " " + english + " ")
    tokens = [LEXICON.get(w, w) for w in TOKENS.findall(clean)]
    words = [w for w in tokens if len(w) >= 3 and not w.isdigit() and w not in COMMON]
    bigrams = [words[i] + "_" + words[i + 1] for i in range(len(words) - 1)]
    return frozenset(words + bigrams)


def _score_band(score: float) -> str:
    if score <= 20: return "Very Low"
    if score <= 40: return "Low"
    if score <= 60: return "Medium"
    if score <= 80: return "High"
    return "Critical"


class FreeTierTextPredictor:
    """Actual supervised similarity estimator, bounded for 512 MB / 0.1 CPU."""

    def __init__(self, dataset: Path = REPORTS_PATH):
        records = json.loads(dataset.read_text(encoding="utf-8"))
        entries: list[tuple[frozenset[str], float]] = []
        for row in records:
            body = row.get("citizen_report")
            if not isinstance(body, str) or not body.strip():
                continue
            try:
                score = float(row["priority_score"])
            except (KeyError, ValueError, TypeError):
                continue
            if not 0 <= score <= 100:
                continue
            words = _terms(body)
            if words:
                entries.append((words, score))
        if len(entries) < 100:
            raise ValueError("Insufficient labeled reports for free-tier estimator")

        df = Counter(w for words, _ in entries for w in words)
        n = len(entries)
        self.idf = {w: log((n + 1) / (freq + 1)) + 1 for w, freq in df.items()}
        self.scores = [score for _, score in entries]
        self.index: dict[str, list[tuple[int, float]]] = defaultdict(list)

        for i, (words, _) in enumerate(entries):
            norm = sqrt(sum(self.idf[w] ** 2 for w in words))
            if norm:
                for word in words:
                    self.index[word].append((i, self.idf[word] / norm))

    def predict(self, report: str) -> tuple[float, str, str]:
        if not report or not report.strip():
            raise ValueError("A nonempty report is required")
        lang = detect_language(report)
        tokens = _terms(report)
        matched = [w for w in tokens if w in self.index]
        if not matched:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="This free research model has no vocabulary overlap with the report. Please try a more detailed English description.",
            )
        norm = sqrt(sum(self.idf[w] ** 2 for w in matched))
        similarities = defaultdict(float)
        for w in matched:
            weight = self.idf[w] / norm
            for idx, doc_weight in self.index[w]:
                similarities[idx] += weight * doc_weight
        top = nlargest(9, similarities.items(), key=lambda pair: pair[1])
        if not top or top[0][1] < 0.04:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Insufficient similarity to training reports. No trustworthy priority estimate is available.",
            )
        weighted = [(self.scores[idx], sim ** 3) for idx, sim in top]
        denom = sum(weight for _, weight in weighted)
        score = max(0, min(100, sum(score * weight for score, weight in weighted) / denom))
        return round(score, 1), _score_band(score), lang
