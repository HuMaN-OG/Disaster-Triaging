"""
Predict a continuous rescue priority score from a citizen report.

Usage
-----
    python -m src.predict_text --text "There are four people trapped on the roof."
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from xgboost import XGBRegressor

from src.text_dataset import (
    MODEL_META_PATH,
    SENTENCE_MODEL_NAME,
    XGBOOST_PATH,
    clip_score,
    score_to_ui_category,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict continuous priority score from citizen report text"
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Citizen report text to score",
    )
    parser.add_argument("--model", type=str, default=str(XGBOOST_PATH))
    parser.add_argument("--meta", type=str, default=str(MODEL_META_PATH))
    parser.add_argument("--label-encoder", type=str, default=None)
    return parser.parse_args()


def load_pipeline(
    model_path: Path,
    meta_path: Path,
) -> tuple[XGBRegressor, SentenceTransformer]:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train first with: python -m src.train_text_classifier"
        )

    meta: dict[str, Any] = {}
    if meta_path.is_file():
        meta = joblib.load(meta_path)

    sentence_model_name = meta.get("sentence_model", SENTENCE_MODEL_NAME)
    model = XGBRegressor()
    model.load_model(str(model_path))
    encoder = SentenceTransformer(sentence_model_name)
    return model, encoder


def predict_priority_score(
    text: str,
    model: XGBRegressor,
    encoder: SentenceTransformer,
) -> dict[str, Any]:
    """
    Embed ``text`` and return a clipped priority score plus UI-only category.
    """
    if not text or not text.strip():
        raise ValueError("Input text must be a non-empty citizen report.")

    embedding = encoder.encode(
        [text.strip()],
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    embedding = np.asarray(embedding, dtype=np.float32)
    raw = float(model.predict(embedding)[0])
    score = clip_score(raw)

    return {
        "priority_score": score,
        "ui_category": score_to_ui_category(score),
    }


def format_prediction(result: dict[str, Any]) -> str:
    return (
        f"Priority Score: {result['priority_score']:.1f}\n"
        f"UI Category: {result['ui_category']}"
    )


def main() -> None:
    args = parse_args()
    meta_path = Path(args.label_encoder) if args.label_encoder else Path(args.meta)
    model, encoder = load_pipeline(Path(args.model), meta_path)
    result = predict_priority_score(args.text, model, encoder)
    print(format_prediction(result))


if __name__ == "__main__":
    main()
