"""
Evaluate the trained text priority regressor on the validation split.

Usage
-----
    python -m src.evaluate_text
    python -m src.evaluate_text --json disaster_dataset.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from xgboost import XGBRegressor

from src.text_dataset import (
    DEFAULT_JSON_PATH,
    MODEL_META_PATH,
    PRED_VS_TRUE_PATH,
    RANDOM_SEED,
    REGRESSION_REPORT_PATH,
    SENTENCE_MODEL_NAME,
    XGBOOST_PATH,
    ensure_output_dirs,
    load_text_dataset,
    samples_to_arrays,
    train_val_split,
)
from src.train_text_classifier import compute_regression_metrics, encode_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate text priority regressor")
    parser.add_argument("--json", type=str, default=str(DEFAULT_JSON_PATH))
    parser.add_argument("--model", type=str, default=str(XGBOOST_PATH))
    parser.add_argument(
        "--meta",
        type=str,
        default=str(MODEL_META_PATH),
        help="Path to priority_model_meta.pkl (sentence-model name, score bounds).",
    )
    # Back-compat alias for older --label-encoder flag
    parser.add_argument("--label-encoder", type=str, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def load_artifacts(
    model_path: Path,
    meta_path: Path,
) -> tuple[XGBRegressor, dict[str, Any], SentenceTransformer]:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"XGBoost model not found: {model_path}. Run: python -m src.train_text_classifier"
        )

    meta: dict[str, Any] = {}
    if meta_path.is_file():
        meta = joblib.load(meta_path)
    else:
        print(f"[evaluate_text] Meta not found at {meta_path}; using defaults.")

    sentence_model_name = meta.get("sentence_model", SENTENCE_MODEL_NAME)
    model = XGBRegressor()
    model.load_model(str(model_path))
    encoder = SentenceTransformer(sentence_model_name)
    return model, meta, encoder


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    ensure_output_dirs()

    meta_path = Path(args.label_encoder) if args.label_encoder else Path(args.meta)
    model, _meta, encoder = load_artifacts(Path(args.model), meta_path)

    samples = load_text_dataset(args.json)
    _train_samples, val_samples = train_val_split(samples, seed=args.seed)
    val_texts, y_true = samples_to_arrays(val_samples)

    print(f"Evaluating on validation split ({len(val_samples)} reports)…")
    x_val = encode_texts(encoder, val_texts, batch_size=args.batch_size)
    y_pred = np.clip(model.predict(x_val), 0.0, 100.0)
    metrics = compute_regression_metrics(y_true, y_pred)

    print("\n=== Evaluation results ===")
    print(f"MAE                      : {metrics['mae']:.4f}")
    print(f"RMSE                     : {metrics['rmse']:.4f}")
    print(f"R²                       : {metrics['r2']:.4f}")
    print(f"Avg abs priority error   : {metrics['avg_abs_priority_error']:.4f} points")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.45, s=18, edgecolors="none")
    ax.plot([0, 100], [0, 100], "r--", linewidth=1, label="Ideal")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("True priority score")
    ax.set_ylabel("Predicted priority score")
    ax.set_title("Text Priority Regressor — Pred vs True")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PRED_VS_TRUE_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    REGRESSION_REPORT_PATH.write_text(
        "Text Priority Regressor — Evaluation Report\n"
        f"MAE                    : {metrics['mae']:.4f}\n"
        f"RMSE                   : {metrics['rmse']:.4f}\n"
        f"R²                     : {metrics['r2']:.4f}\n"
        f"Avg abs priority error : {metrics['avg_abs_priority_error']:.4f} points\n",
        encoding="utf-8",
    )

    print(f"Pred-vs-true plot → {PRED_VS_TRUE_PATH}")
    print(f"Regression report → {REGRESSION_REPORT_PATH}")
    return metrics


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
