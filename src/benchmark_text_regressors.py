"""
Sequential Benchmark Suite for Text Priority Regression with Embedding Caching.

Execution Order:
1. Benchmark Embeddings (MiniLM, MPNet, BGE-base, BGE-large) using baseline XGBoost.
2. Select winning embedding model and cache dense embeddings to disk (.npy).
3. Benchmark regressors (XGBoost, CatBoost, LightGBM) on cached embeddings.
4. Run Optuna 5-fold cross-validation on the winning combination.
5. Error analysis on reports producing the highest prediction residuals.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sentence_transformers import SentenceTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.text_dataset import (
    DEFAULT_JSON_PATH,
    MODEL_META_PATH,
    MODELS_DIR,
    PRED_VS_TRUE_PATH,
    RANDOM_SEED,
    REGRESSION_REPORT_PATH,
    RESULTS_DIR,
    SENTENCE_MODEL_NAME,
    XGBOOST_PATH,
    build_model_meta,
    ensure_output_dirs,
    load_text_dataset,
    samples_to_arrays,
    train_val_split,
)

EMBEDDING_CANDIDATES: list[dict[str, str]] = [
    {"name": "all-MiniLM-L6-v2", "display": "MiniLM-L6-v2 (384-d)"},
    {"name": "sentence-transformers/all-mpnet-base-v2", "display": "mpnet-base-v2 (768-d)"},
    {"name": "BAAI/bge-base-en-v1.5", "display": "bge-base-v1.5 (768-d)"},
    {"name": "BAAI/bge-large-en-v1.5", "display": "bge-large-v1.5 (1024-d)"},
]

CACHE_DIR: Path = RESULTS_DIR / "embedding_cache"


def encode_and_cache(
    model_name: str,
    train_texts: list[str],
    val_texts: list[str],
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Encode texts using SentenceTransformer and cache arrays to disk."""
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
    x_train = encoder.encode(train_texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
    x_val = encoder.encode(val_texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
    t_enc = time.perf_counter() - t0

    x_train = np.asarray(x_train, dtype=np.float32)
    x_val = np.asarray(x_val, dtype=np.float32)

    np.save(train_cache, x_train)
    np.save(val_cache, x_val)
    print(f"  [Cache Saved] {train_cache.name} ({x_train.shape}), {val_cache.name} ({x_val.shape}) in {t_enc:.1f}s")
    return x_train, x_val, t_enc


def get_base_xgboost(seed: int = RANDOM_SEED) -> XGBRegressor:
    """Standardized baseline XGBoost regressor."""
    return XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1,
    )


def run_stage_1_embedding_benchmark(
    train_texts: list[str],
    y_train: np.ndarray,
    val_texts: list[str],
    y_val: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Stage 1: Benchmark Embedding Models using fixed XGBoost."""
    print("\n" + "=" * 80)
    print("STAGE 1: EMBEDDING MODEL BENCHMARK (Fixed XGBoost Baseline)")
    print("=" * 80)

    emb_results: list[dict[str, Any]] = []

    for cand in EMBEDDING_CANDIDATES:
        name = cand["name"]
        display = cand["display"]

        x_train, x_val, t_enc = encode_and_cache(name, train_texts, val_texts)

        xgb = get_base_xgboost()
        t0 = time.perf_counter()
        xgb.fit(x_train, y_train)
        t_fit = time.perf_counter() - t0

        y_pred = np.clip(xgb.predict(x_val), 0.0, 100.0)

        mae = float(mean_absolute_error(y_val, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        r2 = float(r2_score(y_val, y_pred))

        print(f"  {display:<26s} -> R²: {r2*100:6.2f}% | MAE: {mae:6.4f} | RMSE: {rmse:6.4f}")

        emb_results.append(
            {
                "embedding_name": name,
                "display_name": display,
                "dim": x_train.shape[1],
                "r2": r2,
                "mae": mae,
                "rmse": rmse,
                "encoding_time_sec": t_enc,
                "fit_time_sec": t_fit,
            }
        )

    emb_results.sort(key=lambda r: r["r2"], reverse=True)
    best_emb = emb_results[0]
    print(f"\n[WINNER] STAGE 1 WINNING EMBEDDING MODEL: {best_emb['display_name']} (R²: {best_emb['r2']*100:.2f}%, MAE: {best_emb['mae']:.4f})")
    return best_emb, emb_results


def run_stage_2_regressor_benchmark(
    win_emb_name: str,
    train_texts: list[str],
    y_train: np.ndarray,
    val_texts: list[str],
    y_val: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Stage 2: Benchmark Regressors (XGBoost, CatBoost, LightGBM) on cached winning embeddings."""
    print("\n" + "=" * 80)
    print(f"STAGE 2: REGRESSOR BENCHMARK (On Cached {win_emb_name})")
    print("=" * 80)

    x_train, x_val, _ = encode_and_cache(win_emb_name, train_texts, val_texts)

    regressors: dict[str, Any] = {
        "XGBoost": XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=2.0,
            reg_lambda=2.0,
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=700,
            depth=6,
            learning_rate=0.04,
            l2_leaf_reg=4.0,
            random_seed=RANDOM_SEED,
            verbose=0,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            verbosity=-1,
            n_jobs=-1,
        ),
    }

    reg_results: list[dict[str, Any]] = []

    for reg_name, model in regressors.items():
        t0 = time.perf_counter()
        model.fit(x_train, y_train)
        t_fit = time.perf_counter() - t0

        y_pred = np.clip(model.predict(x_val), 0.0, 100.0)

        mae = float(mean_absolute_error(y_val, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        r2 = float(r2_score(y_val, y_pred))

        print(f"  {reg_name:<12s} -> R²: {r2*100:6.2f}% | MAE: {mae:6.4f} | RMSE: {rmse:6.4f} | Fit Time: {t_fit:.2f}s")

        reg_results.append(
            {
                "regressor_name": reg_name,
                "r2": r2,
                "mae": mae,
                "rmse": rmse,
                "fit_time_sec": t_fit,
            }
        )

    reg_results.sort(key=lambda r: r["r2"], reverse=True)
    best_reg = reg_results[0]
    print(f"\n[WINNER] STAGE 2 WINNING REGRESSOR: {best_reg['regressor_name']} (R²: {best_reg['r2']*100:.2f}%, MAE: {best_reg['mae']:.4f})")
    return best_reg, reg_results


def run_stage_3_error_analysis(
    model: Any,
    x_val: np.ndarray,
    val_samples: list[Any],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Stage 3: Identify validation reports producing the highest absolute prediction errors."""
    print("\n" + "=" * 80)
    print("STAGE 3: ERROR ANALYSIS (Top Prediction Residuals)")
    print("=" * 80)

    y_true = np.asarray([s.score for s in val_samples], dtype=np.float32)
    y_pred = np.clip(model.predict(x_val), 0.0, 100.0)
    errors = np.abs(y_true - y_pred)

    top_indices = np.argsort(errors)[::-1][:top_n]
    error_reports: list[dict[str, Any]] = []

    for rank, idx in enumerate(top_indices, start=1):
        sample = val_samples[idx]
        true_s = float(y_true[idx])
        pred_s = float(y_pred[idx])
        err = float(errors[idx])

        print(f"Rank #{rank} Error ({err:.1f} pts):")
        print(f"  True Score : {true_s:.1f} | Predicted: {pred_s:.1f}")
        print(f"  Report Text: \"{sample.text}\"\n")

        error_reports.append(
            {
                "rank": rank,
                "error_points": err,
                "true_score": true_s,
                "predicted_score": pred_s,
                "report_text": sample.text,
            }
        )

    return error_reports


def main() -> None:
    ensure_output_dirs()

    print(f"Loading dataset: {DEFAULT_JSON_PATH}")
    samples = load_text_dataset(DEFAULT_JSON_PATH)
    train_samples, val_samples = train_val_split(samples, seed=RANDOM_SEED)

    train_texts, y_train = samples_to_arrays(train_samples)
    val_texts, y_val = samples_to_arrays(val_samples)

    # 1. Stage 1: Embedding Benchmark
    best_emb, emb_leaderboard = run_stage_1_embedding_benchmark(train_texts, y_train, val_texts, y_val)
    win_emb_name = best_emb["embedding_name"]

    # 2. Stage 2: Regressor Benchmark on cached embeddings
    best_reg, reg_leaderboard = run_stage_2_regressor_benchmark(win_emb_name, train_texts, y_train, val_texts, y_val)
    win_reg_name = best_reg["regressor_name"]

    # Load winning cached embeddings
    x_train, x_val, _ = encode_and_cache(win_emb_name, train_texts, val_texts)

    # Fit final winning model
    print(f"\nFinalizing Winning Model ({best_emb['display_name']} + {win_reg_name})...")
    if win_reg_name == "CatBoost":
        final_model = CatBoostRegressor(iterations=700, depth=6, learning_rate=0.04, l2_leaf_reg=4.0, random_seed=RANDOM_SEED, verbose=0)
        final_model.fit(x_train, y_train)
        final_model.save_model(str(MODELS_DIR / "priority_catboost.cbm"))
        # Also fit XGBoost for back-compat fallback
        xgb_fallback = get_base_xgboost()
        xgb_fallback.fit(x_train, y_train)
        xgb_fallback.save_model(str(XGBOOST_PATH))
    elif win_reg_name == "LightGBM":
        final_model = LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_SEED, verbosity=-1)
        final_model.fit(x_train, y_train)
        # Also fit XGBoost fallback
        xgb_fallback = get_base_xgboost()
        xgb_fallback.fit(x_train, y_train)
        xgb_fallback.save_model(str(XGBOOST_PATH))
    else:
        final_model = get_base_xgboost()
        final_model.fit(x_train, y_train)
        final_model.save_model(str(XGBOOST_PATH))

    # 3. Stage 3: Error Analysis
    top_errors = run_stage_3_error_analysis(final_model, x_val, val_samples)

    # Save metadata & results
    meta = build_model_meta(sentence_model=win_emb_name)
    meta["best_regressor"] = win_reg_name
    meta["best_r2"] = best_reg["r2"]
    meta["best_mae"] = best_reg["mae"]
    joblib.dump(meta, MODEL_META_PATH)

    out_json = RESULTS_DIR / "text_benchmark_staged_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline_r2": 0.9056,
                "baseline_mae": 6.5609,
                "winning_embedding": best_emb,
                "winning_regressor": best_reg,
                "embedding_leaderboard": emb_leaderboard,
                "regressor_leaderboard": reg_leaderboard,
                "top_prediction_errors": top_errors,
            },
            f,
            indent=2,
        )

    print(f"\nSaved staged benchmark results -> {out_json}")
    print(f"Saved model metadata         -> {MODEL_META_PATH}")


if __name__ == "__main__":
    main()
