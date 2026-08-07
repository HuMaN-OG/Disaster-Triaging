"""
Phase 1: Multilingual Embedding Model Benchmark Suite & Manual Consistency Check.

Evaluates 4 Sentence Embedding Models on the ORIGINAL English dataset:
1. BAAI/bge-base-en-v1.5 (English baseline reference)
2. sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (768-d)
3. BAAI/bge-m3 (1024-d)
4. intfloat/multilingual-e5-base (768-d, using 'query: ' prefix)

Trains independent CatBoostRegressor models, caches embeddings, and performs
a mini English-Hindi-Hinglish cross-lingual consistency test.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import psutil
from catboost import CatBoostRegressor
from sentence_transformers import SentenceTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.text_dataset import (
    DEFAULT_JSON_PATH,
    MODEL_META_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    build_model_meta,
    clip_score,
    ensure_output_dirs,
    load_text_dataset,
    samples_to_arrays,
    train_val_split,
)

MULTILINGUAL_CANDIDATES: list[dict[str, Any]] = [
    {"name": "BAAI/bge-base-en-v1.5", "display": "bge-base-v1.5 (English Ref)", "prefix": ""},
    {"name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "display": "multilingual-mpnet-v2 (768-d)", "prefix": ""},
    {"name": "BAAI/bge-m3", "display": "bge-m3 (1024-d)", "prefix": ""},
    {"name": "intfloat/multilingual-e5-base", "display": "multilingual-e5-base (768-d)", "prefix": "query: "},
]

CACHE_DIR: Path = RESULTS_DIR / "multilingual_embedding_cache"


def get_memory_usage_mb() -> float:
    """Return process resident memory in Megabytes."""
    process = psutil.Process(os.getpid())
    return float(process.memory_info().rss / (1024 * 1024))


def prepare_texts(texts: list[str], prefix: str) -> list[str]:
    """Add prompt prefix if required by embedding architecture (e.g. e5)."""
    if not prefix:
        return texts
    return [f"{prefix}{t}" for t in texts]


def encode_and_cache(
    model_name: str,
    prefix: str,
    train_texts: list[str],
    val_texts: list[str],
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Encode texts using SentenceTransformer and cache numpy arrays."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_")
    train_cache = CACHE_DIR / f"{safe_name}_train.npy"
    val_cache = CACHE_DIR / f"{safe_name}_val.npy"

    if train_cache.exists() and val_cache.exists():
        print(f"  [Cache Hit] Loading precomputed embeddings for {model_name}...")
        x_train = np.load(train_cache)
        x_val = np.load(val_cache)
        return x_train, x_val, 0.0

    print(f"  [Encoding] Extracting dense embeddings with {model_name}...")
    t0 = time.perf_counter()
    encoder = SentenceTransformer(model_name)
    
    tr_formatted = prepare_texts(train_texts, prefix)
    va_formatted = prepare_texts(val_texts, prefix)

    x_train = encoder.encode(tr_formatted, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
    x_val = encoder.encode(va_formatted, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
    t_enc = time.perf_counter() - t0

    x_train = np.asarray(x_train, dtype=np.float32)
    x_val = np.asarray(x_val, dtype=np.float32)

    np.save(train_cache, x_train)
    np.save(val_cache, x_val)
    print(f"  [Cache Saved] {train_cache.name} ({x_train.shape}), {val_cache.name} ({x_val.shape}) in {t_enc:.1f}s")
    return x_train, x_val, t_enc


def measure_inference_latency(regressor: CatBoostRegressor, encoder: SentenceTransformer, prefix: str, sample_text: str) -> float:
    """Measure single report inference latency in milliseconds."""
    latencies: list[float] = []
    formatted = f"{prefix}{sample_text}" if prefix else sample_text
    for _ in range(10):
        t0 = time.perf_counter()
        emb = encoder.encode([formatted], convert_to_numpy=True, show_progress_bar=False)
        _ = regressor.predict(np.asarray(emb, dtype=np.float32))
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(latencies[2:]))


def mini_cross_lingual_test(encoder: SentenceTransformer, regressor: CatBoostRegressor, prefix: str) -> dict[str, float]:
    """Test zero-shot cross-lingual prediction consistency on a sample report across Eng/Hin/Hinglish."""
    test_cases = {
        "english": "Water has reached the second floor. Four people are trapped on the roof.",
        "hindi": "पानी दूसरी मंजिल तक पहुँच गया है। चार लोग छत पर फँसे हुए हैं।",
        "hinglish": "Paani second floor tak aa gaya hai. 4 log roof par phase hue hain.",
    }

    preds: dict[str, float] = {}
    for lang, text in test_cases.items():
        formatted = f"{prefix}{text}" if prefix else text
        emb = encoder.encode([formatted], convert_to_numpy=True, show_progress_bar=False)
        pred = clip_score(float(regressor.predict(np.asarray(emb, dtype=np.float32))[0]))
        preds[lang] = pred

    return preds


def main() -> None:
    ensure_output_dirs()

    print(f"Loading dataset: {DEFAULT_JSON_PATH}")
    samples = load_text_dataset(DEFAULT_JSON_PATH)
    train_samples, val_samples = train_val_split(samples, seed=RANDOM_SEED)

    train_texts, y_train = samples_to_arrays(train_samples)
    val_texts, y_val = samples_to_arrays(val_samples)

    print("\n" + "=" * 90)
    print("PHASE 1: MULTILINGUAL EMBEDDING MODEL BENCHMARK (CatBoost Regressor)")
    print("=" * 90)

    benchmark_results: list[dict[str, Any]] = []

    for cand in MULTILINGUAL_CANDIDATES:
        name = cand["name"]
        display = cand["display"]
        prefix = cand["prefix"]
        print(f"\nEvaluating Backbone: {display} ({name})")

        mem_start = get_memory_usage_mb()

        # Step A: Embedding Extraction
        x_train, x_val, t_enc = encode_and_cache(name, prefix, train_texts, val_texts)
        encoder = SentenceTransformer(name)

        # Step B: CatBoost Model Fit
        cat_model = CatBoostRegressor(
            iterations=700,
            depth=6,
            learning_rate=0.04,
            l2_leaf_reg=5.0,
            random_seed=RANDOM_SEED,
            verbose=0,
        )

        t0_fit = time.perf_counter()
        cat_model.fit(x_train, y_train)
        t_fit = time.perf_counter() - t0_fit

        # Save independent CatBoost model checkpoint
        safe_name = name.replace("/", "_")
        model_ckpt = MODELS_DIR / f"catboost_{safe_name}.cbm"
        cat_model.save_model(str(model_ckpt))

        # Step C: Evaluation Metrics on English Validation Set
        y_pred = np.clip(cat_model.predict(x_val), 0.0, 100.0)
        r2 = float(r2_score(y_val, y_pred))
        mae = float(mean_absolute_error(y_val, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))

        # Step D: Inference Latency & Memory
        latency_ms = measure_inference_latency(cat_model, encoder, prefix, val_texts[0])
        mem_peak = get_memory_usage_mb() - mem_start

        # Step E: Mini Cross-Lingual Test
        cross_ling_preds = mini_cross_lingual_test(encoder, cat_model, prefix)
        eng_p = cross_ling_preds["english"]
        hin_p = cross_ling_preds["hindi"]
        hing_p = cross_ling_preds["hinglish"]
        delta_hin = abs(eng_p - hin_p)
        delta_hing = abs(eng_p - hing_p)

        print(f"  -> English R² Score         : {r2 * 100:6.2f}%")
        print(f"  -> English MAE              : {mae:6.4f} points")
        print(f"  -> English RMSE             : {rmse:6.4f}")
        print(f"  -> Embedding Time           : {t_enc:6.1f} s")
        print(f"  -> Fit Time                 : {t_fit:6.1f} s")
        print(f"  -> Latency (per report)     : {latency_ms:6.1f} ms")
        print(f"  -> Mini Cross-Lingual Test  : English={eng_p:.1f} | Hindi={hin_p:.1f} (d_Hindi={delta_hin:.1f}) | Hinglish={hing_p:.1f} (d_Hinglish={delta_hing:.1f})")

        benchmark_results.append(
            {
                "model_name": name,
                "display_name": display,
                "embedding_dim": x_train.shape[1],
                "r2": r2,
                "mae": mae,
                "rmse": rmse,
                "embedding_time_sec": t_enc,
                "fit_time_sec": t_fit,
                "latency_ms": latency_ms,
                "memory_mb": round(mem_peak, 1),
                "mini_cross_lingual_preds": cross_ling_preds,
                "delta_hindi": round(delta_hin, 2),
                "delta_hinglish": round(delta_hing, 2),
                "checkpoint_path": str(model_ckpt),
            }
        )

    # Sort benchmark results by R² score descending
    benchmark_results.sort(key=lambda r: r["r2"], reverse=True)

    print("\n" + "=" * 90)
    print("PHASE 1 MULTILINGUAL EMBEDDING LEADERBOARD (Original English Dataset)")
    print("=" * 90)
    print(f"{'Model Display Name':<32s} {'Dim':<6s} {'R² Score':<10s} {'MAE':<8s} {'RMSE':<8s} {'Latency':<10s} {'Hindi Dev':<10s}")
    print("-" * 90)
    for r in benchmark_results:
        print(
            f"{r['display_name']:<32s} {r['embedding_dim']:<6d} "
            f"{r['r2']*100:6.2f}%    {r['mae']:6.4f}   {r['rmse']:6.4f}   {r['latency_ms']:6.1f}ms   {r['delta_hindi']:6.1f} pts"
        )
    print("=" * 90)

    best_result = benchmark_results[0]
    print(f"\n[WINNER] WINNING EMBEDDING MODEL: {best_result['display_name']}")
    print(f"  English R²: {best_result['r2']*100:.2f}% | MAE: {best_result['mae']:.4f} points")
    print(f"  Zero-Shot Mini Consistency: English={best_result['mini_cross_lingual_preds']['english']:.1f} | Hindi={best_result['mini_cross_lingual_preds']['hindi']:.1f} | Hinglish={best_result['mini_cross_lingual_preds']['hinglish']:.1f}")

    out_json = RESULTS_DIR / "multilingual_phase1_benchmark.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "english_baseline_r2": 0.9217,
                "english_baseline_mae": 6.0380,
                "winning_model": best_result,
                "leaderboard": benchmark_results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved Phase 1 benchmark results -> {out_json}")


if __name__ == "__main__":
    main()
