"""
Phase 3 v2: High-Quality Hinglish Augmentation.

Key improvements over v1:
- Natural code-mixed sentence structures (not just find-and-replace)
- 5 structurally distinct Hinglish writing styles per sample
- External JSON normalization dictionary (configurable without code changes)
- Embedding-based quality control with configurable cosine threshold
- Keeps ALL good variants (not just best one) to maximise diversity
- Logs quality statistics
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.text_dataset import DEFAULT_JSON_PATH, RANDOM_SEED, RESULTS_DIR

MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
COSINE_SIM_THRESHOLD: float = 0.72   # Slightly lower to allow structural variation
MAX_VARIANTS_PER_SAMPLE: int = 3     # Keep best 3 distinct variants per sample
NORM_DICT_PATH: Path = Path(__file__).parent / "hinglish_normalization_dict.json"
OUT_PATH: Path = RESULTS_DIR / "disaster_dataset_hinglish_v2.json"


# ---------------------------------------------------------------------------
# 1. External dictionary-based normalizer
# ---------------------------------------------------------------------------

def load_norm_dict(path: Path) -> dict[str, str]:
    """Load and flatten the nested JSON normalization dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    flat: dict[str, str] = {}
    for section, rules in raw.items():
        if section.startswith("_"):
            continue
        for variant, canonical in rules.items():
            flat[variant.lower()] = canonical.lower()
    return flat


def normalize(text: str, flat_dict: dict[str, str]) -> str:
    """Apply word-boundary normalization from external dictionary."""
    if not text:
        return text
    result = text
    for variant, canonical in flat_dict.items():
        pattern = re.compile(rf"\b{re.escape(variant)}\b", flags=re.IGNORECASE)
        result = pattern.sub(canonical, result)
    result = re.sub(r"\s+", " ", result).strip()
    return result


# ---------------------------------------------------------------------------
# 2. Natural Hinglish variant generator
# ---------------------------------------------------------------------------

# Vocabulary pools for natural code-mixing
WATER_WORDS = ["paani", "water", "barsaat ka paani", "flood ka paani"]
FIRE_WORDS = ["aag", "fire", "aag aur dhua", "jwala"]
TRAPPED_PHRASES = [
    "phanse hue hain", "trapped hain", "nikal nahi pa rahe",
    "bahar nahi nikal sakte", "andar band hain"
]
HELP_PHRASES = [
    "madad chahiye", "help bhejo", "rescue team bhejo",
    "jaldi aao", "emergency hai", "urgent help chahiye"
]
QUANTITY_WORDS = {
    "many": ["bohot saare", "kai", "kaafi", "multiple"],
    "very": ["bohot", "bahut", "kaafi", "zyada"],
    "immediately": ["jaldi", "abhi", "turant", "seedha"],
    "severe": ["bohot bura", "khatarnak", "severe", "critical"],
    "inside": ["andar", "ghar ke andar", "building mein"],
    "floor": ["floor", "manzil", "zameen"],
}

STRUCTURAL_TEMPLATES = {
    # Flood
    "flood_trapped": [
        "{place} mein paani aa gaya hai. {count} log {floor} par phanse hue hain. Madad chahiye.",
        "Yahan {place} par baadh ka paani bhar gaya. {count} log andar trapped hain, jaldi rescue bhejo.",
        "{place} mein paani bahut upar aa gaya. {count} logon ko nikalna hai, help urgent hai.",
        "Ghar mein paani aa gaya hai {place} mein. Rooftop par {count} log hain. Koi aa sakta hai?",
        "Flood situation serious hai {place} mein. Paani ghar ke andar hai, {count} log phanse.",
    ],
    "flood_minor": [
        "{place} mein thoda paani aa gaya hai lekin situation control mein hai.",
        "Paani aaya hai {place} mein lekin hum safe hain. Update ke liye bata rahe hain.",
        "{place} par paani hai thoda, abhi kisi ki zaroorat nahi but inform kar rahe hain.",
    ],
    "fire_trapped": [
        "{place} mein aag lagi hai. {count} log andar phanse hain. Jaldi fire brigade bhejo.",
        "Aag ki situation bohot serious hai {place} mein. {count} log nikalna chahte hain, help karo.",
        "{place} mein building mein aag hai. Dhua bohot zyada hai. {count} log trapped.",
        "Emergency! {place} aag aur dhua. {count} log roof par hain, nikal nahi sakte.",
    ],
    "fire_smoke": [
        "{place} ke aas paas dhua aa raha hai. Situation monitor kar rahe hain.",
        "Dhua {place} mein bhar raha hai, aag door hai abhi.",
        "{place} mein dhuaan hai lekin aag seedha nahi. Careful raho.",
    ],
    "medical": [
        "{place} mein {condition} hai. Medical team ki zaroorat hai.",
        "{condition} ke liye {place} mein doctor chahiye, jaldi bhejo.",
        "Medical emergency {place} mein. {condition}. Ambulance bhejo abhi.",
    ],
    "infrastructure": [
        "{place} mein rasta band ho gaya hai. {count} logon ko nahi nikal sakte.",
        "Sadak block hai {place} mein. Help chahiye rasta kholne ke liye.",
        "{place} ka road kharab ho gaya, access nahi. Team bhejo.",
    ],
    "generic_low_priority": [
        "{place} se update: Situation stable hai. Koi immediate help ki zaroorat nahi.",
        "{place} mein sab theek hai abhi. Bas inform kar rahe hain.",
        "Update from {place}: situation normal hai, koi action nahi chahiye.",
    ],
}

# Disaster type patterns  
DISASTER_PATTERNS = {
    "flood": re.compile(r"\b(flood|water|inundation|submerged|waterlog)\b", re.IGNORECASE),
    "fire": re.compile(r"\b(fire|smoke|wildfire|blaze|burn|flame)\b", re.IGNORECASE),
    "earthquake": re.compile(r"\b(earthquake|tremor|seismic|aftershock|building.*tilt|tilt.*building|collapse)\b", re.IGNORECASE),
    "landslide": re.compile(r"\b(landslide|mudslide|rockfall|debris|mud.*buried|buried.*mud)\b", re.IGNORECASE),
    "medical": re.compile(r"\b(medical|doctor|hospital|ambulance|injured|wound|pregnancy|pregnant|dialysis|insulin)\b", re.IGNORECASE),
}

PLACE_EXTRACTORS = [
    re.compile(r"\bat ([A-Z][A-Za-z\s]+?)(?:\.|,|\s+where|\s+the\s|\s+who|\s+with)", re.MULTILINE),
    re.compile(r"\bnear ([A-Z][A-Za-z\s]+?)(?:\.|,|\s+where|\s+the\s|\s+who|\s+with)", re.MULTILINE),
    re.compile(r"\bfrom ([A-Z][A-Za-z\s]+?)(?:\.|,|\s+where|\s+the\s|\s+who|\s+with)", re.MULTILINE),
]

COUNT_PATTERN = re.compile(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:people|persons|adults|children|residents|individuals|survivors|workers|family)", re.IGNORECASE)

CONDITION_PATTERNS = {
    re.compile(r"\bpregnant\b", re.IGNORECASE): "pregnant woman",
    re.compile(r"\bdialysis\b", re.IGNORECASE): "dialysis patient",
    re.compile(r"\binjur", re.IGNORECASE): "ghayal log",
    re.compile(r"\belderly\b", re.IGNORECASE): "bujurg patient",
    re.compile(r"\binfant\b", re.IGNORECASE): "chota bachcha",
}


def extract_context(text: str) -> dict[str, str]:
    """Extract place, count, condition from English text."""
    ctx: dict[str, str] = {}

    # Extract place
    for rx in PLACE_EXTRACTORS:
        m = rx.search(text)
        if m:
            ctx["place"] = m.group(1).strip()[:30]
            break
    ctx.setdefault("place", "yahan")

    # Extract count
    m = COUNT_PATTERN.search(text)
    if m:
        ctx["count"] = m.group(1)
    else:
        ctx["count"] = "kuch"

    # Extract floor
    floor_m = re.search(r"\b(first|second|third|\d+(?:st|nd|rd|th))\s+floor\b", text, re.IGNORECASE)
    ctx["floor"] = floor_m.group(0) if floor_m else "upar"

    # Extract medical condition
    ctx["condition"] = "emergency situation"
    for rx, label in CONDITION_PATTERNS.items():
        if rx.search(text):
            ctx["condition"] = label
            break

    return ctx


def detect_disaster_type(text: str) -> str:
    """Detect primary disaster type from English text."""
    for dtype, rx in DISASTER_PATTERNS.items():
        if rx.search(text):
            return dtype
    return "generic"


def score_to_priority_band(score: float) -> str:
    """Map priority score to high/medium/low band."""
    if score >= 70:
        return "high"
    elif score >= 35:
        return "medium"
    else:
        return "low"


def generate_natural_hinglish_variants(text: str, score: float, flat_dict: dict[str, str]) -> list[str]:
    """
    Generate 5 structurally distinct, natural code-mixed Hinglish variants.
    Completely rewritten to avoid find-and-replace monotony.
    """
    disaster = detect_disaster_type(text)
    ctx = extract_context(text)
    band = score_to_priority_band(score)
    variants: list[str] = []

    # ---------- Strategy 1: Template-based structural variant ----------
    templates: list[str] = []
    if disaster == "flood":
        if band == "high":
            templates = STRUCTURAL_TEMPLATES["flood_trapped"]
        else:
            templates = STRUCTURAL_TEMPLATES["flood_minor"]
    elif disaster == "fire":
        if band == "high":
            templates = STRUCTURAL_TEMPLATES["fire_trapped"]
        else:
            templates = STRUCTURAL_TEMPLATES["fire_smoke"]
    elif disaster == "landslide":
        templates = STRUCTURAL_TEMPLATES["infrastructure"]
    elif disaster == "medical":
        templates = STRUCTURAL_TEMPLATES["medical"]
    else:
        if band == "low":
            templates = STRUCTURAL_TEMPLATES["generic_low_priority"]
        else:
            templates = STRUCTURAL_TEMPLATES["infrastructure"]

    if templates:
        tmpl = random.choice(templates)
        try:
            v = tmpl.format(**ctx)
            variants.append(normalize(v, flat_dict))
        except (KeyError, IndexError):
            pass

    # ---------- Strategy 2: Moderate code-mix (English nouns, Hindi verbs) ----------
    v2 = text
    replacements_mod = [
        (r"\bwater\b", "paani"), (r"\bfire\b", "aag"), (r"\bflood\b", "baadh"),
        (r"\bsmoke\b", "dhua"), (r"\btrapped\b", "phanse"), (r"\binjured\b", "ghayal"),
        (r"\broof\b", "chhat"), (r"\bneed help\b", "madad chahiye"),
        (r"\bsend help\b", "madad bhejo"), (r"\bimmediately\b", "jaldi"),
        (r"\burgently\b", "urgent"), (r"\bpeople\b", "log"),
        (r"\brescue\b", "rescue karo"), (r"\bsafe\b", "safe hai"),
    ]
    for pattern, repl in replacements_mod:
        v2 = re.sub(pattern, repl, v2, flags=re.IGNORECASE)
    variants.append(normalize(v2, flat_dict))

    # ---------- Strategy 3: Heavy code-mix (mostly Hindi with English technical terms) ----------
    v3 = text
    replacements_heavy = [
        (r"\bThe situation is\b", "Situation hai"), (r"\bWe are\b", "Hum hain"),
        (r"\bwater has reached\b", "paani aa gaya hai"), (r"\bwater is\b", "paani hai"),
        (r"\bFloodwater is creeping higher\b", "Baadh ka paani badh raha hai"),
        (r"\bfire is spreading\b", "aag fail rahi hai"),
        (r"\bsmoke is dense\b", "dhua bohot zyada hai"),
        (r"\bare trapped\b", "phanse hue hain"), (r"\bis trapped\b", "phanse hua hai"),
        (r"\bneed medical\b", "medical madad chahiye"),
        (r"\bsend rescue\b", "rescue bhejo"), (r"\bno help needed\b", "koi zaroorat nahi"),
        (r"\bpeople\b", "log"), (r"\bchildren\b", "bachche"),
        (r"\belderly\b", "bujurg"), (r"\binjured\b", "ghayal"),
        (r"\bwater\b", "paani"), (r"\bfire\b", "aag"), (r"\bflood\b", "baadh"),
        (r"\bsmoke\b", "dhua"), (r"\broad\b", "sadak"), (r"\bhome\b", "ghar"),
        (r"\bhouse\b", "ghar"), (r"\bbuilding\b", "building"),
        (r"\bimmediately\b", "abhi"), (r"\burgently\b", "jaldi se"),
        (r"\bhelp\b", "madad"),
    ]
    for pattern, repl in replacements_heavy:
        v3 = re.sub(pattern, repl, v3, flags=re.IGNORECASE)
    variants.append(normalize(v3, flat_dict))

    # ---------- Strategy 4: Short SMS / WhatsApp style ----------
    # Extract key info and compress
    key_facts: list[str] = []
    if disaster == "flood":
        key_facts.append("Paani aa gaya")
    elif disaster == "fire":
        key_facts.append("Aag lagi hai")
    elif disaster == "landslide":
        key_facts.append("Rasta band")
    else:
        key_facts.append("Emergency")

    m = COUNT_PATTERN.search(text)
    if m:
        key_facts.append(f"{m.group(1)} log phanse")
    if disaster == "medical" or re.search(r"\binjured\b|\bmedical\b", text, re.IGNORECASE):
        key_facts.append("Medical help chahiye")

    if band == "high":
        key_facts.append("URGENT madad chahiye")
    elif band == "low":
        key_facts.append("Koi zaroorat nahi abhi")

    if ctx["place"] != "yahan":
        key_facts.insert(0, f"{ctx['place']} se report")

    v4 = ". ".join(key_facts) + "."
    variants.append(normalize(v4, flat_dict))

    # ---------- Strategy 5: Question/request style (common in WhatsApp disaster reports) ----------
    if band == "high":
        if disaster == "flood":
            v5 = f"Kya koi aa sakta hai? {ctx['place']} mein paani bhar gaya aur {ctx['count']} log phanse hain. Jaldi madad chahiye."
        elif disaster == "fire":
            v5 = f"Aag lagi hai {ctx['place']} mein! {ctx['count']} log andar hain. Fire brigade ko batao please."
        else:
            v5 = f"Emergency hai {ctx['place']} mein! {ctx['count']} log phanse hain. Koi madad kar sakta hai?"
    else:
        v5 = f"Update: {ctx['place']} se bata rahe hain. Situation {band} hai. Agar zaroorat padi to batayenge."

    variants.append(normalize(v5, flat_dict))

    # Deduplicate and clean
    seen: set[str] = set()
    clean_variants: list[str] = []
    for v in variants:
        v_clean = v.strip()
        if v_clean and v_clean not in seen and len(v_clean) > 10:
            seen.add(v_clean)
            clean_variants.append(v_clean)

    return clean_variants


# ---------------------------------------------------------------------------
# 3. Cosine similarity utility
# ---------------------------------------------------------------------------

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = float(np.dot(v1, v2))
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    return dot / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0


# ---------------------------------------------------------------------------
# 4. Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(RANDOM_SEED)

    print(f"Loading original English dataset: {DEFAULT_JSON_PATH}")
    with open(DEFAULT_JSON_PATH, "r", encoding="utf-8") as f:
        english_data = json.load(f)

    print(f"Loading external normalization dictionary: {NORM_DICT_PATH}")
    flat_dict = load_norm_dict(NORM_DICT_PATH)
    print(f"  Loaded {len(flat_dict)} normalization rules.")

    print(f"\nLoading multilingual encoder: {MODEL_NAME}...")
    encoder = SentenceTransformer(MODEL_NAME)

    print("\nGenerating high-quality Hinglish variants...")

    # Encode all English texts upfront
    eng_texts = [item["citizen_report"] for item in english_data]
    print(f"  Encoding {len(eng_texts)} English reports...")
    eng_embs = encoder.encode(eng_texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)

    # Generate and collect all candidates
    all_candidates: list[str] = []
    candidate_meta: list[tuple[int, int]] = []  # (sample_idx, candidate_offset_start)

    for idx, item in enumerate(english_data):
        variants = generate_natural_hinglish_variants(
            item["citizen_report"], float(item["priority_score"]), flat_dict
        )
        start = len(all_candidates)
        all_candidates.extend(variants)
        candidate_meta.append((idx, start, start + len(variants)))

    print(f"  Total candidates to encode: {len(all_candidates)}")
    all_cand_embs = encoder.encode(all_candidates, batch_size=64, show_progress_bar=False, convert_to_numpy=True)

    # Select best variants per sample
    augmented_data: list[dict[str, Any]] = []
    accepted = 0
    rejected = 0
    total_variants_kept = 0

    # Start with all 2,000 English originals
    for item in english_data:
        augmented_data.append({
            "citizen_report": item["citizen_report"],
            "priority_score": float(item["priority_score"]),
            "language": "english",
        })

    for idx, item in enumerate(english_data):
        _, start, end = candidate_meta[idx]
        eng_emb = eng_embs[idx]

        scored_variants: list[tuple[float, str]] = []
        for ci in range(start, end):
            sim = cosine_similarity(eng_emb, all_cand_embs[ci])
            if sim >= COSINE_SIM_THRESHOLD:
                scored_variants.append((sim, all_candidates[ci]))
                accepted += 1
            else:
                rejected += 1

        # Sort by cosine similarity descending, take best MAX_VARIANTS_PER_SAMPLE
        scored_variants.sort(key=lambda x: x[0], reverse=True)
        kept = scored_variants[:MAX_VARIANTS_PER_SAMPLE]

        if not kept:
            # Fallback: keep the moderate code-mix style even if sim is low
            best_idx = start + 1 if (end - start) > 1 else start
            best_sim = cosine_similarity(eng_emb, all_cand_embs[best_idx])
            kept = [(best_sim, all_candidates[best_idx])]
            rejected -= 1
            accepted += 1

        for sim, variant in kept:
            augmented_data.append({
                "citizen_report": variant,
                "priority_score": float(item["priority_score"]),
                "language": "hinglish",
                "cosine_similarity": round(sim, 4),
            })
            total_variants_kept += 1

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(augmented_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 85)
    print("HIGH-QUALITY HINGLISH AUGMENTATION v2 SUMMARY")
    print("=" * 85)
    print(f"Original English Samples    : {len(english_data)}")
    print(f"Accepted Hinglish variants  : {accepted}")
    print(f"Rejected Hinglish variants  : {rejected}")
    print(f"Total Hinglish samples kept : {total_variants_kept}")
    print(f"Avg variants per English    : {total_variants_kept / len(english_data):.2f}")
    print(f"Total augmented dataset     : {len(augmented_data)} samples")
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
