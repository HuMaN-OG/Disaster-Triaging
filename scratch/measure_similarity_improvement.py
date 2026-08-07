"""
Measure Cosine Similarity and Validation Performance Improvement
"""

import sys
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append("backend")
sys.path.append(".")
from src.hinglish_normalizer import normalize_hinglish_text, HinglishNormalizer

encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

test_pairs = [
    (
        "Water has reached the second floor. Four people are trapped on the roof.",
        "Paani second floor tak aa gaya hai. 4 log roof par phase hue hain."
    ),
    (
        "Fire broke out in the building, send rescue team urgently!",
        "Building me aag lag gayi hai, jaldi rescue team bhejo!"
    ),
    (
        "Heavy flood water entered houses, people trapped on roof.",
        "Bohot paani ghar me ghus gaya hai, log chhat par phanse hain."
    ),
    (
        "Landslide blocked the road, rocks falling from hill.",
        "Rasta band ho gaya hai keechad aur pathar gir rahe hain."
    )
]

def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

normalizer = HinglishNormalizer()

print("=" * 80)
print("COSINE SIMILARITY EVALUATION (English vs Hinglish)")
print("=" * 80)

sims_before = []
sims_after = []

for idx, (eng, hng) in enumerate(test_pairs, 1):
    hng_clean_before = hng.lower().strip()
    hng_clean_after = normalizer.preprocess_for_embedding(hng)
    
    emb_eng = encoder.encode([eng])[0]
    emb_hng_before = encoder.encode([hng_clean_before])[0]
    emb_hng_after = encoder.encode([hng_clean_after])[0]
    
    sim_b = cos_sim(emb_eng, emb_hng_before)
    sim_a = cos_sim(emb_eng, emb_hng_after)
    
    sims_before.append(sim_b)
    sims_after.append(sim_a)
    
    print(f"Pair #{idx}:")
    print(f"  English         : \"{eng}\"")
    print(f"  Hinglish Raw    : \"{hng}\" (Sim: {sim_b:.4f})")
    print(f"  Hinglish Preproc: \"{hng_clean_after}\" (Sim: {sim_a:.4f})")
    print(f"  Gain            : +{(sim_a - sim_b):.4f}\n")

print("-" * 80)
print(f"Average Similarity BEFORE Pipeline : {np.mean(sims_before):.4f}")
print(f"Average Similarity AFTER Pipeline  : {np.mean(sims_after):.4f}")
print(f"Median Similarity  AFTER Pipeline  : {np.median(sims_after):.4f}")
print(f"Worst-case Similarity              : {np.min(sims_after):.4f}")
print("=" * 80)
