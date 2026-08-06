"""
Train Sentence-Transformer embeddings + XGBoost priority regressor.

Usage
-----
    python -m src.train_text_classifier
    python -m src.train_text_classifier --json disaster_dataset.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.text_dataset import (
    DEFAULT_JSON_PATH,
    EMBEDDING_DIM,
    MODEL_META_PATH,
    PRED_VS_TRUE_PATH,
    RANDOM_SEED,
    REGRESSION_REPORT_PATH,
    SENTENCE_MODEL_NAME,
    XGBOOST_PATH,
    build_model_meta,
    ensure_output_dirs,
    load_text_dataset,
    samples_to_arrays,
    train_val_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train text priority regressor")
    parser.add_argument("--json", type=str, default=str(DEFAULT_JSON_PATH))
    parser.add_argument("--sentence-model", type=str, default=SENTENCE_MODEL_NAME)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for SentenceTransformer.encode",
    )
    return parser.parse_args()


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 64,
) -> np.ndarray:
    """Embed citizen reports into a (N, 384) float32 matrix."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Expected embeddings of shape (N, {EMBEDDING_DIM}), got {embeddings.shape}"
        )
    return embeddings


def build_xgboost(seed: int = RANDOM_SEED) -> XGBRegressor:
    """Configure an XGBoost regressor for continuous priority scores in [0, 100]."""
    return XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=seed,
        n_jobs=-1,
    )


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE, R², and mean absolute priority-point error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "avg_abs_priority_error": mae,  # same as MAE; priority points on 0–100 scale
    }


def save_pred_vs_true_plot(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.45, s=18, edgecolors="none")
    lims = [0, 100]
    ax.plot(lims, lims, "r--", linewidth=1, label="Ideal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True priority score")
    ax.set_ylabel("Predicted priority score")
    ax.set_title("Text Priority Regressor — Pred vs True")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train(args: argparse.Namespace) -> dict[str, Any]:
    ensure_output_dirs()

    print(f"Loading dataset: {args.json}")
    samples = load_text_dataset(args.json)
    train_samples, val_samples = train_val_split(samples, seed=args.seed)

    train_texts, y_train = samples_to_arrays(train_samples)
    val_texts, y_val = samples_to_arrays(val_samples)

    print(
        f"Samples: total={len(samples)} train={len(train_samples)} val={len(val_samples)}"
    )
    print(
        f"  train score mean/std/min/max = "
        f"{y_train.mean():.1f}/{y_train.std():.1f}/{y_train.min():.1f}/{y_train.max():.1f}"
    )
    print(
        f"  val   score mean/std/min/max = "
        f"{y_val.mean():.1f}/{y_val.std():.1f}/{y_val.min():.1f}/{y_val.max():.1f}"
    )

    print(f"Loading Sentence Transformer: {args.sentence_model}")
    encoder = SentenceTransformer(args.sentence_model)

    print("Encoding train texts…")
    x_train = encode_texts(encoder, train_texts, batch_size=args.batch_size)
    print("Encoding validation texts…")
    x_val = encode_texts(encoder, val_texts, batch_size=args.batch_size)

    print("Training XGBRegressor…")
    model = build_xgboost(seed=args.seed)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

    y_pred = np.clip(model.predict(x_val), 0.0, 100.0)
    metrics = compute_regression_metrics(y_val, y_pred)

    print("\n=== Validation metrics ===")
    print(f"MAE                      : {metrics['mae']:.4f}")
    print(f"RMSE                     : {metrics['rmse']:.4f}")
    print(f"R²                       : {metrics['r2']:.4f}")
    print(f"Avg abs priority error   : {metrics['avg_abs_priority_error']:.4f} points")

    model.save_model(str(XGBOOST_PATH))
    joblib.dump(build_model_meta(args.sentence_model), MODEL_META_PATH)

    save_pred_vs_true_plot(y_val, y_pred, PRED_VS_TRUE_PATH)
    REGRESSION_REPORT_PATH.write_text(
        "Text Priority Regressor — Validation Report\n"
        f"Sentence model: {args.sentence_model}\n"
        f"MAE                    : {metrics['mae']:.4f}\n"
        f"RMSE                   : {metrics['rmse']:.4f}\n"
        f"R²                     : {metrics['r2']:.4f}\n"
        f"Avg abs priority error : {metrics['avg_abs_priority_error']:.4f} points\n",
        encoding="utf-8",
    )

    print(f"Saved XGBRegressor   → {XGBOOST_PATH}")
    print(f"Saved model meta     → {MODEL_META_PATH}")
    print(f"Saved pred-vs-true   → {PRED_VS_TRUE_PATH}")
    print(f"Saved report         → {REGRESSION_REPORT_PATH}")

    return metrics


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
