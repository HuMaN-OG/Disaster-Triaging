"""
Full 9-Phase Audit Script for Multilingual Priority Model Pipeline
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import joblib
import urllib.request
import urllib.parse

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append("backend")

from sentence_transformers import SentenceTransformer
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.hinglish_normalizer import normalize_hinglish_text
from src.inference.predict import detect_language, MultimodalPredictor
from src.predict_text import predict_priority_score, load_pipeline
from src.text_dataset import (
    DEFAULT_JSON_PATH,
    CATBOOST_PATH,
    MODEL_META_PATH,
    MODELS_DIR,
    load_text_dataset,
    train_val_split,
    samples_to_arrays,
    clip_score,
)

print("=" * 80)
print("PHASE 1 — VERIFY MODEL LOADING")
print("=" * 80)

print(f"CatBoost Checkpoint Path : {CATBOOST_PATH} (Exists: {CATBOOST_PATH.exists()})")
print(f"Model Meta Path          : {MODEL_META_PATH} (Exists: {MODEL_META_PATH.exists()})")

meta = joblib.load(MODEL_META_PATH) if MODEL_META_PATH.exists() else {}
sentence_model_name = meta.get("sentence_model", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
print(f"SentenceTransformer Name : {sentence_model_name}")

norm_dict_path = Path("src/hinglish_normalization_dict.json")
print(f"Normalization Dict Path  : {norm_dict_path} (Exists: {norm_dict_path.exists()})")

cbm_model = CatBoostRegressor()
cbm_model.load_model(str(CATBOOST_PATH))
print(f"CatBoost Model Loaded    : Yes (Feature dim: {cbm_model.n_features_in_ if hasattr(cbm_model, 'n_features_in_') else 'Dynamic 768'})")

encoder = SentenceTransformer(sentence_model_name)
print(f"SentenceTransformer Loaded: Yes (Dim: {encoder.get_sentence_embedding_dimension()})")


print("\n" + "=" * 80)
print("PHASE 2 — VERIFY PREPROCESSING PIPELINE TRACE")
print("=" * 80)

sample_text = "Paani second floor tak aa gaya hai. 4 log roof par phase hue hain."
print(f"1. Original Text      : \"{sample_text}\"")

lang = detect_language(sample_text)
print(f"2. Detected Language  : {lang}")

norm_text = normalize_hinglish_text(sample_text)
print(f"3. Normalized Text    : \"{norm_text}\"")

emb = encoder.encode([norm_text], convert_to_numpy=True, show_progress_bar=False)
print(f"4. Embedding Model    : {sentence_model_name} (Shape: {emb.shape})")
print(f"5. Checkpoint Path    : {CATBOOST_PATH}")

raw_pred = float(cbm_model.predict(emb)[0])
final_score = clip_score(raw_pred)
print(f"6. Final Prediction   : Score = {final_score:.1f} (Raw: {raw_pred:.2f})")


print("\n" + "=" * 80)
print("PHASE 3 — VERIFY LANGUAGE DETECTION")
print("=" * 80)

test_cases = {
    "English": "Water has reached the second floor. Four people are trapped on the roof.",
    "Hindi": "पानी दूसरी मंजिल तक पहुँच गया है। चार लोग छत पर फँसे हुए हैं।",
    "Hinglish": "Paani second floor tak aa gaya hai. 4 log roof par phase hue hain."
}

for expected_lang, text in test_cases.items():
    det = detect_language(text)
    print(f"Expected: {expected_lang:<10} | Detected: {det:<10} | Match: {expected_lang == det} | Input: \"{text[:50]}...\"")


print("\n" + "=" * 80)
print("PHASE 4 — VERIFY HINGLISH NORMALIZATION")
print("=" * 80)

dict_data = json.loads(norm_dict_path.read_text(encoding="utf-8"))
print(f"Loaded Dictionary Keys  : {len(dict_data)} replacement pairs.")

norm_tests = [
    ("Bohot PAANI aa gaya hai", "bohot paani aa gaya hai"),
    ("Phase hue log, jaldi BHEJO", "phanse hue log, jaldi bhejo"),
    ("Ghar me keechad aur aag", "ghar me keechad aur aag")
]

for orig, expected in norm_tests:
    res = normalize_hinglish_text(orig)
    print(f"Original: \"{orig}\"\n  -> Normalized: \"{res}\"")


print("\n" + "=" * 80)
print("PHASE 5 — VERIFY EMBEDDINGS")
print("=" * 80)

emb_eng = encoder.encode([normalize_hinglish_text(test_cases["English"])], convert_to_numpy=True)[0]
emb_hin = encoder.encode([normalize_hinglish_text(test_cases["Hindi"])], convert_to_numpy=True)[0]
emb_hng = encoder.encode([normalize_hinglish_text(test_cases["Hinglish"])], convert_to_numpy=True)[0]

def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

print(f"Vector Dimension      : {emb_eng.shape[0]}")
print(f"Cosine Sim Eng vs Hin : {cos_sim(emb_eng, emb_hin):.4f}")
print(f"Cosine Sim Eng vs Hng : {cos_sim(emb_eng, emb_hng):.4f}")
print(f"Cosine Sim Hin vs Hng : {cos_sim(emb_hin, emb_hng):.4f}")
print(f"Embedding Statistics  : Min={emb_hng.min():.4f}, Max={emb_hng.max():.4f}, Mean={emb_hng.mean():.4f}, Std={emb_hng.std():.4f}")


print("\n" + "=" * 80)
print("PHASE 6 — CLI VS BACKEND REST API COMPARISON")
print("=" * 80)

test_input = "Aag bohot fail gayi hai building me, rescue team jaldi bhejo!"

# CLI Execution
cli_res = predict_priority_score(test_input, cbm_model, encoder)
cli_score = round(cli_res["priority_score"], 1)

# Backend API Execution
data = urllib.parse.urlencode({"text_report": test_input}).encode("utf-8")
req = urllib.request.Request("http://127.0.0.1:8000/predict", data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})
try:
    res = urllib.request.urlopen(req)
    api_res = json.loads(res.read())
    api_score = api_res["priority_score"]
    print(f"CLI Predicted Score     : {cli_score}")
    print(f"Backend API Score       : {api_score}")
    print(f"Exact Match             : {cli_score == api_score}")
except Exception as e:
    print(f"Backend API Call Error  : {e}")


print("\n" + "=" * 80)
print("PHASE 7 & 8 — BENCHMARK REPRODUCTION & MULTILINGUAL ERROR ANALYSIS")
print("=" * 80)

samples = load_text_dataset(DEFAULT_JSON_PATH)
_, val_samples = train_val_split(samples, seed=42)

val_texts, y_true = samples_to_arrays(val_samples)

# Categorize validation samples into English, Hindi, Hinglish
eng_samples, hin_samples, hng_samples = [], [], []

for s in val_samples:
    l = detect_language(s.text)
    if l == "Hindi":
        hin_samples.append(s)
    elif l == "Hinglish":
        hng_samples.append(s)
    else:
        eng_samples.append(s)

print(f"Validation Split Breakdown ({len(val_samples)} total):")
print(f"  English  : {len(eng_samples)} samples")
print(f"  Hindi    : {len(hin_samples)} samples")
print(f"  Hinglish : {len(hng_samples)} samples")

def evaluate_subset(name, subset):
    if not subset:
        return 0.0, 0.0, 0.0
    texts = [s.text for s in subset]
    y_t = np.array([s.score for s in subset], dtype=np.float32)
    
    # Preprocess & Predict
    norm_t = [normalize_hinglish_text(t) for t in texts]
    embs = encoder.encode(norm_t, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    preds = np.clip(cbm_model.predict(embs), 0.0, 100.0)
    
    r2 = r2_score(y_t, preds)
    mae = mean_absolute_error(y_t, preds)
    rmse = np.sqrt(mean_squared_error(y_t, preds))
    return r2, mae, rmse

r2_eng, mae_eng, rmse_eng = evaluate_subset("English", eng_samples)
r2_hin, mae_hin, rmse_hin = evaluate_subset("Hindi", hin_samples)
r2_hng, mae_hng, rmse_hng = evaluate_subset("Hinglish", hng_samples)

print("\n=== Validation Split Multilingual Performance ===")
print(f"English  (N={len(eng_samples):<3}) -> R²: {r2_eng*100:6.2f}% | MAE: {mae_eng:5.2f} pts | RMSE: {rmse_eng:5.2f}")
print(f"Hindi    (N={len(hin_samples):<3}) -> R²: {r2_hin*100:6.2f}% | MAE: {mae_hin:5.2f} pts | RMSE: {rmse_hin:5.2f}")
print(f"Hinglish (N={len(hng_samples):<3}) -> R²: {r2_hng*100:6.2f}% | MAE: {mae_hng:5.2f} pts | RMSE: {rmse_hng:5.2f}")

print("\nAudit completed cleanly.")
