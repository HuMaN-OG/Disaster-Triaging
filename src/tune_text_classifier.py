"""
Optuna 5-Fold Cross-Validation Hyperparameter Optimization for Text Priority Regressor.

Tunes XGBoost and CatBoost on precomputed BAAI/bge-base-en-v1.5 embeddings
to maximize R² variance score and lower priority prediction error below 5.5 MAE points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
from catboost import CatBoostRegressor
from sentence_transformers import SentenceTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from src.text_dataset import (
    DEFAULT_JSON_PATH,
    MODEL_META_PATH,
    MODELS_DIR,
    PRED_VS_TRUE_PATH,
    RANDOM_SEED,
    REGRESSION_REPORT_PATH,
    RESULTS_DIR,
    XGBOOST_PATH,
    build_model_meta,
    ensure_output_dirs,
    load_text_dataset,
    samples_to_arrays,
    train_val_split,
)

WINNING_EMBEDDING: str = "BAAI/bge-base-en-v1.5"
CACHE_DIR: Path = RESULTS_DIR / "embedding_cache"

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_cached_embeddings(
    train_texts: list[str],
    val_texts: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Load precomputed BAAI/bge-base-en-v1.5 embeddings from cache."""
    safe_name = WINNING_EMBEDDING.replace("/", "_")
    train_cache = CACHE_DIR / f"{safe_name}_train.npy"
    val_cache = CACHE_DIR / f"{safe_name}_val.npy"

    if train_cache.exists() and val_cache.exists():
        print(f"Loading cached embeddings from {train_cache.name} & {val_cache.name}...")
        return np.load(train_cache), np.load(val_cache)

    print(f"Computing embeddings with {WINNING_EMBEDDING}...")
    encoder = SentenceTransformer(WINNING_EMBEDDING)
    x_tr = encoder.encode(train_texts, show_progress_bar=False, convert_to_numpy=True)
    x_va = encoder.encode(val_texts, show_progress_bar=False, convert_to_numpy=True)
    return np.asarray(x_tr, dtype=np.float32), np.asarray(x_va, dtype=np.float32)


def optuna_xgb_objective(trial: optuna.Trial, x_tr: np.ndarray, y_tr: np.ndarray) -> float:
    """5-Fold CV objective for XGBoost."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 20.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True),
        "objective": "reg:squarederror",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scores: list[float] = []

    for train_idx, fold_idx in kf.split(x_tr, y_tr):
        model = XGBRegressor(**params)
        model.fit(x_tr[train_idx], y_tr[train_idx])
        preds = np.clip(model.predict(x_tr[fold_idx]), 0.0, 100.0)
        scores.append(float(mean_absolute_error(y_tr[fold_idx], preds)))

    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna 5-Fold CV Tuning for Text Regressor")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    args = parser.parse_args()

    ensure_output_dirs()

    print(f"Loading dataset: {DEFAULT_JSON_PATH}")
    samples = load_text_dataset(DEFAULT_JSON_PATH)
    train_samples, val_samples = train_val_split(samples, seed=RANDOM_SEED)

    train_texts, y_train = samples_to_arrays(train_samples)
    val_texts, y_val = samples_to_arrays(val_samples)

    x_train, x_val = load_cached_embeddings(train_texts, val_texts)

    print(f"\nStarting Optuna 5-Fold Cross-Validation ({args.trials} trials) on {WINNING_EMBEDDING}...")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(lambda trial: optuna_xgb_objective(trial, x_train, y_train), n_trials=args.trials, show_progress_bar=True)

    best_params = study.best_params
    print("\n" + "=" * 60)
    print("OPTUNA 5-FOLD CV HYPERPARAMETER SEARCH COMPLETE")
    print("=" * 60)
    print(f"Best 5-Fold CV MAE: {study.best_value:.4f} priority points")
    print("Optimal Hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k:20s}: {v}")

    # Train final tuned XGBoost on full train set and evaluate on validation set
    final_xgb_params = {
        **best_params,
        "objective": "reg:squarederror",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }
    tuned_model = XGBRegressor(**final_xgb_params)
    tuned_model.fit(x_train, y_train)

    y_pred = np.clip(tuned_model.predict(x_val), 0.0, 100.0)

    final_mae = float(mean_absolute_error(y_val, y_pred))
    final_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    final_r2 = float(r2_score(y_val, y_pred))

    print("\nFinal Optuna-Tuned Model Validation Metrics:")
    print(f"  R² Score                 : {final_r2 * 100:.2f}%  (Baseline: 90.56%)")
    print(f"  Mean Absolute Error (MAE): {final_mae:.4f} points (Baseline: 6.5609)")
    print(f"  Root Mean Squared Error  : {final_rmse:.4f}       (Baseline: 8.1466)")

    # Save tuned model
    tuned_model.save_model(str(XGBOOST_PATH))

    meta = build_model_meta(sentence_model=WINNING_EMBEDDING)
    meta["best_regressor"] = "XGBoost (Optuna Tuned)"
    meta["best_r2"] = final_r2
    meta["best_mae"] = final_mae
    meta["best_params"] = best_params
    joblib.dump(meta, MODEL_META_PATH)

    REGRESSION_REPORT_PATH.write_text(
        "Optuna-Tuned Text Priority Regressor Report\n"
        f"Embedding Model          : {WINNING_EMBEDDING}\n"
        f"R² Score                 : {final_r2:.4f}\n"
        f"MAE                      : {final_mae:.4f}\n"
        f"RMSE                     : {final_rmse:.4f}\n"
        f"Hyperparameters          : {json.dumps(best_params)}\n",
        encoding="utf-8",
    )

    hp_out = MODELS_DIR / "best_text_hyperparams.json"
    with open(hp_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "embedding_model": WINNING_EMBEDDING,
                "best_r2": final_r2,
                "best_mae": final_mae,
                "best_rmse": final_rmse,
                "best_params": best_params,
            },
            f,
            indent=2,
        )

    print(f"\nSaved tuned XGBRegressor -> {XGBOOST_PATH}")
    print(f"Saved model metadata     -> {MODEL_META_PATH}")
    print(f"Saved hyperparams report -> {hp_out}")


if __name__ == "__main__":
    main()
