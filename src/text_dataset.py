"""
Text priority dataset utilities (regression).

Loads citizen reports and continuous priority scores from a JSON array.
Image-related and other unused fields are ignored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_JSON_PATH: Path = PROJECT_ROOT / "disaster_dataset.json"
MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

XGBOOST_PATH: Path = MODELS_DIR / "priority_xgboost.json"
MODEL_META_PATH: Path = MODELS_DIR / "priority_model_meta.pkl"
# Kept as an alias so older CLI flags / docs referring to label_encoder still resolve.
LABEL_ENCODER_PATH: Path = MODEL_META_PATH

REGRESSION_REPORT_PATH: Path = RESULTS_DIR / "text_regression_report.txt"
PRED_VS_TRUE_PATH: Path = RESULTS_DIR / "text_pred_vs_true.png"

SENTENCE_MODEL_NAME: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384

SCORE_MIN: float = 0.0
SCORE_MAX: float = 100.0

TRAIN_RATIO: float = 0.80
RANDOM_SEED: int = 42

# UI-only display bands (model never predicts these directly)
UI_PRIORITY_BANDS: list[tuple[float, float, str]] = [
    (0, 20, "Very Low"),
    (21, 40, "Low"),
    (41, 60, "Medium"),
    (61, 80, "High"),
    (81, 100, "Critical"),
]


@dataclass(frozen=True)
class TextSample:
    """Single training example: report text + continuous priority score."""

    text: str
    score: float


def ensure_output_dirs() -> None:
    """Create models/ and results/ if missing."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def score_to_ui_category(score: float) -> str:
    """
    Map a numeric priority score to a human-readable UI label.

    Used for display only — the model predicts the continuous score.
    """
    s = float(np.clip(score, SCORE_MIN, SCORE_MAX))
    for low, high, name in UI_PRIORITY_BANDS:
        if low <= s <= high:
            return name
    return "Critical" if s > 80 else "Very Low"


def clip_score(score: float) -> float:
    """Clamp a predicted / target score to [0, 100]."""
    return float(np.clip(score, SCORE_MIN, SCORE_MAX))


def load_text_records(json_path: Path | str | None = None) -> list[dict[str, Any]]:
    """
    Load the disaster JSON array from disk.

    Only ``citizen_report`` and ``priority_score`` are required per object.
    """
    path = Path(json_path) if json_path is not None else DEFAULT_JSON_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset JSON not found: {path}\n"
            "Expected a JSON array of objects with at least "
            "'citizen_report' and 'priority_score'."
        )

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")

    return data


def _parse_priority_score(raw: Any) -> float | None:
    """Validate and normalize a priority_score field to float in [0, 100]."""
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
    elif isinstance(raw, str):
        try:
            value = float(raw.strip())
        except ValueError:
            return None
    else:
        return None
    if not np.isfinite(value):
        return None
    return clip_score(value)


def parse_text_samples(records: list[dict[str, Any]]) -> list[TextSample]:
    """
    Extract (citizen_report, priority_score) pairs.

    Skips rows with missing/empty text or invalid scores.
    """
    samples: list[TextSample] = []
    skipped = 0

    for i, row in enumerate(records):
        text = row.get("citizen_report")
        score = _parse_priority_score(row.get("priority_score"))

        if not isinstance(text, str) or not text.strip():
            skipped += 1
            continue
        if score is None:
            skipped += 1
            print(f"[text_dataset] Skipping row {i}: invalid priority_score {row.get('priority_score')!r}")
            continue

        samples.append(TextSample(text=text.strip(), score=score))

    if not samples:
        raise ValueError(
            "No valid text samples found (need citizen_report + priority_score)."
        )

    if skipped:
        print(f"[text_dataset] Skipped {skipped} invalid row(s).")

    return samples


def load_text_dataset(json_path: Path | str | None = None) -> list[TextSample]:
    """Load JSON and return validated text samples."""
    return parse_text_samples(load_text_records(json_path))


def train_val_split(
    samples: list[TextSample],
    train_ratio: float = TRAIN_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[list[TextSample], list[TextSample]]:
    """
    80/20 train/validation split.

    Scores are binned into quintiles so the split stays roughly balanced
    across the priority range (regression-friendly stratification).
    """
    scores = np.asarray([s.score for s in samples], dtype=np.float64)
    try:
        bins = np.quantile(scores, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        # Unique edges required by digitize
        bins = np.unique(bins)
        if len(bins) < 2:
            stratify = None
        else:
            stratify = np.digitize(scores, bins[1:-1], right=True)
    except ValueError:
        stratify = None

    train_samples, val_samples = train_test_split(
        samples,
        train_size=train_ratio,
        stratify=stratify,
        random_state=seed,
    )
    return train_samples, val_samples


# Back-compat alias used by older call sites
stratified_split = train_val_split


def samples_to_arrays(samples: list[TextSample]) -> tuple[list[str], np.ndarray]:
    """Unpack samples into text list and float score array."""
    texts = [s.text for s in samples]
    scores = np.asarray([s.score for s in samples], dtype=np.float32)
    return texts, scores


def build_model_meta(sentence_model: str = SENTENCE_MODEL_NAME) -> dict[str, Any]:
    """Metadata saved alongside the regressor checkpoint."""
    return {
        "task": "regression",
        "target": "priority_score",
        "score_min": SCORE_MIN,
        "score_max": SCORE_MAX,
        "sentence_model": sentence_model,
        "embedding_dim": EMBEDDING_DIM,
        "ui_bands": [
            {"min": lo, "max": hi, "label": name} for lo, hi, name in UI_PRIORITY_BANDS
        ],
    }
