"""
AI Image Authenticity Detection Package.
Provides modules for dataset management, auditing, model training, evaluation,
inference, benchmarking, and visualization.
"""

from .config import AuthenticityConfig
from .dataset import AuthenticityDataset, get_authenticity_dataloaders
from .audit import audit_authenticity_dataset
from .model import AuthenticityClassifier
from .trainer import AuthenticityTrainer
from .evaluator import AuthenticityEvaluator
from .predictor import AuthenticityPredictor
from .benchmark import run_authenticity_benchmark
from .visualization import plot_authenticity_metrics
from .checkpoint import CheckpointManager

__all__ = [
    "AuthenticityConfig",
    "AuthenticityDataset",
    "get_authenticity_dataloaders",
    "audit_authenticity_dataset",
    "AuthenticityClassifier",
    "AuthenticityTrainer",
    "AuthenticityEvaluator",
    "AuthenticityPredictor",
    "run_authenticity_benchmark",
    "plot_authenticity_metrics",
    "CheckpointManager",
]
