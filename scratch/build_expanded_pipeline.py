"""
Expanded Hinglish Normalization Dictionary with Utility & Negation Context
"""

import json
from pathlib import Path

# Expanded dictionary with safe context rules
SAFE_EXPANDED_DICT = {
    # ---------------------------------------------------------
    # 1. Utility & Low-Priority Contexts (Prevent False Positives)
    # ---------------------------------------------------------
    "paani supply": "water supply",
    "pani supply": "water supply",
    "paanee supply": "water supply",
    "halka paani": "shallow water",
    "chhoti si aag": "minor spark",
    "choti aag": "minor spark",
    "situation control": "situation under control",
    "control mein": "under control",
    "control me": "under control",
    "koi injured nahi": "no injuries reported",
    "injured nahi": "no injuries",
    "madad nahi chahiye": "no rescue required",
    "kisi ko madad nahi": "no help needed",
    "ghar safe": "houses safe",
    "sab safe": "everyone safe",

    # ---------------------------------------------------------
    # 2. General Vocabulary & Spelling Normalization
    # ---------------------------------------------------------
    "paani": "water",
    "pani": "water",
    "paanee": "water",
    "pni": "water",
    "baadh": "flood",
    "baad": "flood",
    "barish": "rain",
    "baarish": "rain",
    "barsaat": "rain",
    "doob": "submerged",
    "dooba": "submerged",

    "phansa": "trapped",
    "phase": "trapped",
    "phanse": "trapped",
    "phasa": "trapped",
    "fasa": "trapped",
    "fase": "trapped",
    "fanse": "trapped",
    "faase": "trapped",
    "phans": "trapped",
    "atka": "trapped",
    "atke": "trapped",

    "aag": "fire",
    "aage": "fire",
    "dhua": "smoke",
    "dhuwa": "smoke",
    "dhuan": "smoke",

    "bhukamp": "earthquake",
    "bhookamp": "earthquake",
    "jhatka": "earthquake",
    "jhatke": "earthquake",

    "keechad": "mudslide",
    "kichad": "mudslide",
    "pathar": "rockfall",
    "drought": "drought",
    "sookha": "drought",
    "sukha": "drought",

    "madad": "help",
    "maddad": "help",
    "bhejo": "send",
    "bhejdo": "send",
    "chahiye": "needed",
    "chahie": "needed",
    "zaroorat": "needed",
    "zarurat": "needed",
    "jaroorat": "needed",

    "ghar": "house",
    "gharon": "houses",
    "makan": "house",
    "chhat": "roof",
    "chhaton": "roof",
    "kamra": "room",
    "kamre": "rooms",
    "manzil": "floor",

    "log": "people",
    "bande": "people",
    "aadmi": "people",
    "aurat": "women",
    "bacha": "child",
    "bachche": "children",
    "bache": "children",
    "bujurg": "elderly",
    "ghayal": "injured",
    "behosh": "unconscious",
    "behoosh": "unconscious",
    "bimaar": "sick",
    "marez": "patient",
    "maut": "fatalities",

    "bohot": "heavy",
    "bahut": "heavy",
    "bhot": "heavy",
    "zyada": "extreme",
    "jyada": "extreme",
    "ziyada": "extreme",
    "bada": "large",
    "badi": "large",
    "bade": "large",

    "rasta": "road",
    "raasta": "road",
    "sadak": "road",
    "sarak": "road",
    "band": "closed",
    "gir": "falling",
    "gira": "fallen",
    "gire": "fallen",

    "hai": "is",
    "h": "is",
    "he": "is",
    "hain": "are",
    "hn": "are",
    "me": "in",
    "mein": "in",
    "mai": "in",
    "par": "on",
    "pr": "on",
    "pe": "on",

    "jaldi": "urgently",
    "jald": "urgently",
    "pls": "please",
    "turant": "immediately",
    "abhi": "now",

    "tak": "until",
    "ko": "to",
    "se": "from",
    "bhi": "also",
    "ki": "of",
    "ka": "of",
    "ke": "of",
    "le": "take",
    "lo": "take",
    "ne": "by",
    "to": "then",
    "aur": "and",
    "ya": "or",
    "gaya": "has",
    "gayi": "has",
    "gaye": "have",
    "hua": "been",
    "hue": "been",
    "hui": "been",
    "raha": "is",
    "rahi": "is",
    "rahe": "are"
}

json_path = Path("src/hinglish_normalization_dict.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({"_comment": "Safe Expanded Hinglish Normalization Dictionary with Utility & Negation Rules", "rules": SAFE_EXPANDED_DICT}, f, indent=2)

print(f"Updated {json_path} with {len(SAFE_EXPANDED_DICT)} entries.")
