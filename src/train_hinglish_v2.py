"""
Phase 4 v2: Retrain CatBoost on High-Quality Hinglish v2 Dataset.

Backbone  : sentence-transformers/paraphrase-multilingual-mpnet-base-v2
Dataset   : disaster_dataset_hinglish_v2.json (5,484 samples: 2,000 Eng + 3,484 Hinglish)
Normalizer: External JSON-based HinglishNormalizer (109 rules)
HP        : Fixed as per tuning experiments (depth=4, lr=0.0714, iters=500)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from catboost import CatBoostRegressor
from sentence_transformers import SentenceTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.hinglish_normalizer import normalize_hinglish_text
from src.text_dataset import (
    MODEL_META_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    TextSample,
    build_model_meta,
    ensure_output_dirs,
    train_val_split,
)

V2_DATASET_PATH: Path = RESULTS_DIR / "disaster_dataset_hinglish_v2.json"
MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CACHE_DIR: Path = RESULTS_DIR / "multilingual_embedding_cache"

# Best hyperparameters from Optuna tuning (depth=4 wins consistently)
CATBOOST_PARAMS = dict(
    iterations=500,
    depth=4,
    learning_rate=0.0714,
    l2_leaf_reg=1.75,
    random_seed=RANDOM_SEED,
    verbose=0,
)


def main() -> None:
    ensure_output_dirs()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print("PHASE 4 v2: RETRAINING ON HIGH-QUALITY HINGLISH v2 DATASET")
    print("=" * 85)
    print(f"Loading dataset: {V2_DATASET_PATH}")

    with open(V2_DATASET_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    lang_counts: dict[str, int] = {}
    for item in raw_data:
        lang = item.get("language", "unknown")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    print(f"Dataset breakdown:")
    for lang, count in lang_counts.items():
        print(f"  {lang:<20}: {count}")
    print(f"  {'TOTAL':<20}: {len(raw_data)}")

    # Normalize all texts with external-dict normalizer
    texts = [normalize_hinglish_text(item["citizen_report"]) for item in raw_data]
    scores = np.array([float(item["priority_score"]) for item in raw_data], dtype=np.float32)

    # 80/20 stratified split
    sample_objs = [TextSample(t, s) for t, s in zip(texts, scores)]
    tr_objs, va_objs = train_val_split(sample_objs, seed=RANDOM_SEED)

    train_texts = [s.text for s in tr_objs]
    y_train = np.array([s.score for s in tr_objs], dtype=np.float32)
    val_texts = [s.text for s in va_objs]
    y_val = np.array([s.score for s in va_objs], dtype=np.float32)

    print(f"\nTrain samples: {len(train_texts)} | Val samples: {len(val_texts)}")

    # Embeddings (never cache-collide with v1 cache)
    cache_key = f"mpnet_hinglish_v2_{len(raw_data)}"
    train_cache = CACHE_DIR / f"{cache_key}_train.npy"
    val_cache = CACHE_DIR / f"{cache_key}_val.npy"

    if train_cache.exists() and val_cache.exists():
        print(f"  [Cache Hit] Loading precomputed embeddings...")
        x_train = np.load(train_cache)
        x_val = np.load(val_cache)
        print(f"  x_train: {x_train.shape}  x_val: {x_val.shape}")
    else:
        print(f"  [Encoding] Extracting 768-d embeddings with {MODEL_NAME}...")
        t0 = time.perf_counter()
        encoder = SentenceTransformer(MODEL_NAME)
        x_train = encoder.encode(train_texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
        x_val = encoder.encode(val_texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
        t_enc = time.perf_counter() - t0

        x_train = np.asarray(x_train, dtype=np.float32)
        x_val = np.asarray(x_val, dtype=np.float32)

        np.save(train_cache, x_train)
        np.save(val_cache, x_val)
        print(f"  [Cache Saved] {x_train.shape} in {t_enc:.1f}s", flush=True)

    # CatBoost with tuned params
    print(f"\nFitting CatBoostRegressor (tuned params)...")
    print(f"  {CATBOOST_PARAMS}", flush=True)
    cat_model = CatBoostRegressor(**CATBOOST_PARAMS)
    t0_fit = time.perf_counter()
    cat_model.fit(x_train, y_train)
    t_fit = time.perf_counter() - t0_fit
    print(f"Fit complete in {t_fit:.1f}s", flush=True)

    # Validation metrics
    y_pred = np.clip(cat_model.predict(x_val), 0.0, 100.0)
    r2 = float(r2_score(y_val, y_pred))
    mae = float(mean_absolute_error(y_val, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))

    print("\n" + "=" * 85)
    print("VALIDATION METRICS (v2 Augmented Dataset, Mixed English + Hinglish Val Set)")
    print("=" * 85)
    print(f"  R2 Score                 : {r2 * 100:6.2f}%")
    print(f"  Mean Absolute Error (MAE): {mae:6.4f} points")
    print(f"  Root Mean Squared Error  : {rmse:6.4f}")
    print(f"  Params: {CATBOOST_PARAMS}")

    # Save checkpoints
    out_ckpt = MODELS_DIR / "catboost_multilingual_hinglish_augmented.cbm"
    cat_model.save_model(str(out_ckpt))
    cat_model.save_model(str(MODELS_DIR / "priority_catboost.cbm"))
    meta = build_model_meta(MODEL_NAME)
    joblib.dump(meta, MODEL_META_PATH)

    print(f"\nSaved model -> {out_ckpt}")
    print(f"Saved production checkpoint -> {MODELS_DIR / 'priority_catboost.cbm'}")


if __name__ == "__main__":
    main()
