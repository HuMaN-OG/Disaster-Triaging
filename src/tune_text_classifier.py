"""
Automated Hyperparameter Optimization for Text Priority Regressor.

Uses RandomizedSearchCV with scikit-learn & XGBoost to find optimal tree parameters
(n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
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


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 64,
) -> np.ndarray:
    """Embed citizen reports into a (N, 384) float32 matrix."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune Text Priority Regressor")
    parser.add_argument("--n-iter", type=int, default=30, help="Number of random parameter combinations")
    args = parser.parse_args()

    ensure_output_dirs()

    print(f"Loading dataset: {DEFAULT_JSON_PATH}")
    samples = load_text_dataset(DEFAULT_JSON_PATH)
    train_samples, val_samples = train_val_split(samples, seed=RANDOM_SEED)

    train_texts, y_train = samples_to_arrays(train_samples)
    val_texts, y_val = samples_to_arrays(val_samples)

    print(f"Encoding texts with {SENTENCE_MODEL_NAME}...")
    encoder = SentenceTransformer(SENTENCE_MODEL_NAME)
    x_train = encode_texts(encoder, train_texts)
    x_val = encode_texts(encoder, val_texts)

    print(f"Starting Randomized Hyperparameter Search ({args.n_iter} iterations)...")

    param_distributions = {
        "n_estimators": [300, 400, 500, 600, 800],
        "max_depth": [4, 5, 6, 7, 8, 9],
        "learning_rate": [0.015, 0.03, 0.05, 0.08, 0.1],
        "subsample": [0.7, 0.8, 0.85, 0.9, 0.95],
        "colsample_bytree": [0.7, 0.8, 0.85, 0.9, 0.95],
        "min_child_weight": [1, 2, 3, 5],
        "reg_alpha": [0.0, 0.01, 0.1, 0.5, 1.0, 5.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
    }

    base_xgb = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=param_distributions,
        n_iter=args.n_iter,
        scoring="neg_mean_absolute_error",
        cv=4,
        random_state=RANDOM_SEED,
        verbose=1,
        n_jobs=-1,
    )

    search.fit(x_train, y_train)

    best_xgb: XGBRegressor = search.best_estimator_
    best_params = search.best_params_

    print("\n" + "=" * 60)
    print("HYPERPARAMETER OPTIMIZATION COMPLETE")
    print("=" * 60)
    print("Optimal Hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k:20s}: {v}")

    # Evaluate best model on validation set
    y_pred = np.clip(best_xgb.predict(x_val), 0.0, 100.0)
    mae = float(mean_absolute_error(y_val, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    r2 = float(r2_score(y_val, y_pred))

    print("\nFinal Tuned Model Validation Metrics:")
    print(f"  MAE                      : {mae:.4f}")
    print(f"  RMSE                     : {rmse:.4f}")
    print(f"  R²                       : {r2:.4f}")

    # Save tuned model
    best_xgb.save_model(str(XGBOOST_PATH))
    joblib.dump(build_model_meta(SENTENCE_MODEL_NAME), MODEL_META_PATH)

    REGRESSION_REPORT_PATH.write_text(
        "Tuned Text Priority Regressor — Validation Report\n"
        f"Sentence model           : {SENTENCE_MODEL_NAME}\n"
        f"MAE                      : {mae:.4f}\n"
        f"RMSE                     : {rmse:.4f}\n"
        f"R²                       : {r2:.4f}\n"
        f"Best Hyperparameters     : {json.dumps(best_params)}\n",
        encoding="utf-8",
    )

    hp_out = MODELS_DIR / "best_text_hyperparams.json"
    with open(hp_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_mae": mae,
                "best_r2": r2,
                "params": best_params,
            },
            f,
            indent=2,
        )

    print(f"\nSaved tuned XGBRegressor -> {XGBOOST_PATH}")
    print(f"Saved model metadata     -> {MODEL_META_PATH}")
    print(f"Saved hyperparams report -> {hp_out}")


if __name__ == "__main__":
    main()
