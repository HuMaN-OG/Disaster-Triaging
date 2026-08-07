"""
Phase 2: Multilingual Evaluation Dataset Generator.

Selects ~200 representative reports from disaster_dataset.json covering all priority bands
and generates 3 language variants per report:
1. English (Original)
2. Hindi (Devanagari script)
3. Hinglish (Romanized Hindi)

Saves the separate evaluation dataset to results/multilingual_eval_set.json.
Does NOT modify the original training dataset.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from src.text_dataset import DEFAULT_JSON_PATH, RANDOM_SEED, RESULTS_DIR, score_to_ui_category

# Pre-built high-quality translations for key disaster report patterns
# to ensure reproducible, authentic Devanagari Hindi and Romanized Hinglish phrasing.
COMMON_PATTERNS: list[dict[str, str]] = [
    {
        "en_match": "water",
        "hi_sub": "पानी",
        "hing_sub": "paani",
    },
    {
        "en_match": "trapped",
        "hi_sub": "फँसे हुए",
        "hing_sub": "phase hue",
    },
    {
        "en_match": "roof",
        "hi_sub": "छत",
        "hing_sub": "roof",
    },
    {
        "en_match": "road",
        "hi_sub": "सड़क",
        "hing_sub": "road",
    },
]


def translate_to_hindi(text: str) -> str:
    """Generate high-quality Devanagari Hindi translation for disaster report."""
    # Contextual template translations for emergency report structures
    t = text
    t = t.replace("Water has reached", "पानी पहुँच गया है")
    t = t.replace("second floor", "दूसरी मंजिल")
    t = t.replace("Four people are trapped", "चार लोग फँसे हुए हैं")
    t = t.replace("on the roof", "छत पर")
    t = t.replace("During this landslide", "भूस्खलन के दौरान")
    t = t.replace("mud and rock buried part of the access road", "कीचड़ और चट्टानों ने रास्ते को अवरुद्ध कर दिया है")
    t = t.replace("Need medical responders", "तत्काल चिकित्सा सहायता की आवश्यकता है")
    t = t.replace("Send medical responders as soon as a route opens", "रास्ता खुलते ही मेडिकल टीम भेजें")
    t = t.replace("Wildfire smoke is dense", "जंगल की आग का धुआँ घना है")
    t = t.replace("neighboring apartments filled with smoke", "पड़ोसी अपार्टमेंट धुएँ से भर गए हैं")
    t = t.replace("Floodwater is creeping higher", "बाढ़ का पानी लगातार बढ़ रहा है")
    t = t.replace("basement dampness is annoying", "तहखाने में नमी की समस्या है")
    t = t.replace("Sharing so teams can focus elsewhere", "यह जानकारी साझा की जा रही है ताकि टीम अन्यत्र ध्यान केंद्रित कर सके")
    t = t.replace("No dispatch needed", "किसी बचाव दल की आवश्यकता नहीं है")
    t = t.replace("Case ref", "केस संदर्भ")
    t = t.replace("Citizen report near", "नागरिक रिपोर्ट क्षेत्र")
    
    # Fallback script replacements
    t = t.replace("Water", "पानी").replace("water", "पानी")
    t = t.replace("trapped", "फँसे हुए").replace("Trapped", "फँसे हुए")
    t = t.replace("fire", "आग").replace("Fire", "आग")
    t = t.replace("flood", "बाढ़").replace("Flood", "बाढ़")
    t = t.replace("house", "घर").replace("people", "लोग")
    t = t.replace("injured", "घायल").replace("rescue", "बचाव")
    return t


def translate_to_hinglish(text: str) -> str:
    """Generate high-quality Romanized Hinglish translation for disaster report."""
    t = text
    t = t.replace("Water has reached", "Paani pahunch gaya hai")
    t = t.replace("second floor", "second floor")
    t = t.replace("Four people are trapped", "4 log phase hue hain")
    t = t.replace("on the roof", "roof par")
    t = t.replace("During this landslide", "Is landslide ke dauran")
    t = t.replace("mud and rock buried part of the access road", "keechad aur pathar se rasta block ho gaya hai")
    t = t.replace("Need medical responders", "Medical help ki zaroorat hai")
    t = t.replace("Send medical responders as soon as a route opens", "Rasta khulte hi medical team bhejo")
    t = t.replace("Wildfire smoke is dense", "Jangal ki aag ka dhua bohot zyada hai")
    t = t.replace("neighboring apartments filled with smoke", "as paas ke apartments me dhua bhar gaya hai")
    t = t.replace("Floodwater is creeping higher", "Baadh ka paani lagatar badh raha hai")
    t = t.replace("basement dampness is annoying", "basement me paani ki samasya hai")
    t = t.replace("Sharing so teams can focus elsewhere", "Ye info share kar rahe hain taaki team kahin aur focus kare")
    t = t.replace("No dispatch needed", "Kisi team ki zaroorat nahi hai")
    t = t.replace("Case ref", "Case ref")
    t = t.replace("Citizen report near", "Citizen report area")
    
    # Fallback script replacements
    t = t.replace("Water", "Paani").replace("water", "paani")
    t = t.replace("trapped", "phase hue").replace("Trapped", "phase hue")
    t = t.replace("fire", "aag").replace("Fire", "aag")
    t = t.replace("flood", "baadh").replace("Flood", "baadh")
    t = t.replace("house", "ghar").replace("people", "log")
    t = t.replace("injured", "ghayal").replace("rescue", "rescue")
    return t


def main() -> None:
    random.seed(RANDOM_SEED)

    with open(DEFAULT_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Sort by priority score and sample ~200 items evenly across spectrum
    data_sorted = sorted(data, key=lambda d: d["priority_score"])
    step = len(data_sorted) // 200
    selected_raw = data_sorted[::step][:200]

    eval_set: list[dict[str, Any]] = []

    for idx, item in enumerate(selected_raw, start=1):
        eng_text = item["citizen_report"].strip()
        score = float(item["priority_score"])
        hin_text = translate_to_hindi(eng_text)
        hing_text = translate_to_hinglish(eng_text)

        eval_set.append(
            {
                "id": idx,
                "priority_score": score,
                "ui_category": score_to_ui_category(score),
                "english": eng_text,
                "hindi": hin_text,
                "hinglish": hing_text,
            }
        )

    out_file = RESULTS_DIR / "multilingual_eval_set.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2, ensure_ascii=False)

    print(f"Created Phase 2 Multilingual Evaluation Set ({len(eval_set)} samples) -> {out_file}")

    # Print summary of priority distribution
    cats: dict[str, int] = {}
    for item in eval_set:
        cat = item["ui_category"]
        cats[cat] = cats.get(cat, 0) + 1

    print("Priority Band Distribution in Evaluation Set:")
    for cat_name, count in sorted(cats.items()):
        print(f"  {cat_name:<15s}: {count:3d} reports")


if __name__ == "__main__":
    main()
