"""
Phase 3: Cross-Language Prediction Consistency & Benchmark Evaluation.

Evaluates the winning Phase 1 model:
Backbone: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
Regressor: CatBoostRegressor (trained on original English dataset)

Evaluates on the 200-sample multilingual evaluation set (English, Hindi, Hinglish).
Calculates per-language R2, MAE, RMSE, and cross-lingual score deviations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from catboost import CatBoostRegressor
from sentence_transformers import SentenceTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.text_dataset import MODELS_DIR, RESULTS_DIR, clip_score

from src.hinglish_normalizer import normalize_hinglish_text

EVAL_SET_PATH: Path = RESULTS_DIR / "multilingual_eval_set.json"
WINNING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
MODEL_CKPT_PATH: Path = MODELS_DIR / "catboost_multilingual_hinglish_augmented.cbm"


def main() -> None:
    print("=" * 85)
    print("PHASE 5: RETRAINED CROSS-LANGUAGE PREDICTION CONSISTENCY BENCHMARK")
    print("=" * 85)
    print(f"Loading evaluation dataset: {EVAL_SET_PATH}")

    if not EVAL_SET_PATH.exists():
        raise FileNotFoundError(f"Evaluation set missing: {EVAL_SET_PATH}")
    ckpt_to_use = MODEL_CKPT_PATH if MODEL_CKPT_PATH.exists() else (MODELS_DIR / "priority_catboost.cbm")
    if not ckpt_to_use.exists():
        raise FileNotFoundError(f"Trained CatBoost model missing: {ckpt_to_use}")

    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_items = json.load(f)

    y_true = np.array([item["priority_score"] for item in eval_items], dtype=np.float32)
    eng_texts = [normalize_hinglish_text(item["english"]) for item in eval_items]
    hin_texts = [normalize_hinglish_text(item["hindi"]) for item in eval_items]
    hing_texts = [normalize_hinglish_text(item["hinglish"]) for item in eval_items]

    print(f"\nLoading Multilingual Encoder: {WINNING_MODEL_NAME}...")
    encoder = SentenceTransformer(WINNING_MODEL_NAME)

    print(f"Loading CatBoost Regressor: {ckpt_to_use.name}...")
    cat_model = CatBoostRegressor()
    cat_model.load_model(str(ckpt_to_use))

    print("\nExtracting Embeddings (with Hinglish Normalization)...")
    eng_emb = encoder.encode(eng_texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    hin_emb = encoder.encode(hin_texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    hing_emb = encoder.encode(hing_texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)

    y_pred_eng = np.clip(cat_model.predict(eng_emb), 0.0, 100.0)
    y_pred_hin = np.clip(cat_model.predict(hin_emb), 0.0, 100.0)
    y_pred_hing = np.clip(cat_model.predict(hing_emb), 0.0, 100.0)

    # Calculate metrics vs Ground Truth priority score
    metrics: dict[str, dict[str, float]] = {}
    for lang_code, preds in [("english", y_pred_eng), ("hindi", y_pred_hin), ("hinglish", y_pred_hing)]:
        r2 = float(r2_score(y_true, preds))
        mae = float(mean_absolute_error(y_true, preds))
        rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
        metrics[lang_code] = {"r2": r2, "mae": mae, "rmse": rmse}

    # Calculate cross-lingual score deviations relative to English prediction
    delta_hin = np.abs(y_pred_eng - y_pred_hin)
    delta_hing = np.abs(y_pred_eng - y_pred_hing)

    outliers: list[dict[str, Any]] = []
    for i, item in enumerate(eval_items):
        diff_hin = float(delta_hin[i])
        diff_hing = float(delta_hing[i])
        if diff_hin > 5.0 or diff_hing > 10.0:
            outliers.append(
                {
                    "id": item["id"],
                    "priority_score": item["priority_score"],
                    "english_pred": round(float(y_pred_eng[i]), 2),
                    "hindi_pred": round(float(y_pred_hin[i]), 2),
                    "hinglish_pred": round(float(y_pred_hing[i]), 2),
                    "delta_hindi": round(diff_hin, 2),
                    "delta_hinglish": round(diff_hing, 2),
                    "english_text": item["english"],
                    "hindi_text": item["hindi"],
                    "hinglish_text": item["hinglish"],
                }
            )

    outliers.sort(key=lambda o: max(o["delta_hindi"], o["delta_hinglish"]), reverse=True)

    print("\n" + "=" * 85)
    print("PER-LANGUAGE EVALUATION METRICS (vs Ground Truth Priority Score)")
    print("=" * 85)
    print(f"{'Language':<15s} {'R² Score':<12s} {'MAE':<10s} {'RMSE':<10s}")
    print("-" * 85)
    for lang, m in metrics.items():
        print(f"{lang.capitalize():<15s} {m['r2']*100:6.2f}%       {m['mae']:6.4f}     {m['rmse']:6.4f}")

    print("\n" + "=" * 85)
    print("CROSS-LINGUAL PREDICTION CONSISTENCY (Relative to English Prediction)")
    print("=" * 85)
    print(f"Hindi (Devanagari) Avg Deviation : {np.mean(delta_hin):.4f} priority points (Max: {np.max(delta_hin):.2f})")
    print(f"Hinglish (Romanized) Avg Deviation: {np.mean(delta_hing):.4f} priority points (Max: {np.max(delta_hing):.2f})")
    print(f"Total Outliers (Score Deviation > 5.0 pts): {len(outliers)}")

    if outliers:
        print("\nTop Outlier Cases:")
        for o in outliers[:3]:
            print(f"  [ID {o['id']}] Target: {o['priority_score']} | Eng: {o['english_pred']} | Hin: {o['hindi_pred']} (d={o['delta_hindi']}) | Hing: {o['hinglish_pred']} (d={o['delta_hinglish']})")

    # Save complete Phase 3 report
    report_out = RESULTS_DIR / "multilingual_consistency_report.json"
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "winning_model": WINNING_MODEL_NAME,
                "per_language_metrics": metrics,
                "hindi_avg_deviation": round(float(np.mean(delta_hin)), 4),
                "hindi_max_deviation": round(float(np.max(delta_hin)), 2),
                "hinglish_avg_deviation": round(float(np.mean(delta_hing)), 4),
                "hinglish_max_deviation": round(float(np.max(delta_hing)), 2),
                "total_outliers": len(outliers),
                "outliers": outliers,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nSaved Phase 3 consistency report -> {report_out}")


if __name__ == "__main__":
    main()
