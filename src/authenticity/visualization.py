import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve
from typing import Dict, Any, List
from .config import AuthenticityConfig
from .logger import get_logger

logger = get_logger("authenticity.visualization")

def plot_authenticity_metrics(metrics: Dict[str, Any], config: AuthenticityConfig):
    config.create_dirs()
    if "raw_predictions" not in metrics:
        logger.warning("No raw_predictions in metrics, skipping curve plots.")
        return

    raw = metrics["raw_predictions"]
    probs = np.array(raw["probabilities"])
    labels = np.array(raw["true_labels"])
    preds = (probs >= 0.5).astype(int)

    sns.set_theme(style="whitegrid")

    # 1. Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real Photo", "AI Generated"],
                yticklabels=["Real Photo", "AI Generated"])
    plt.title("Authenticity Detector Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # 2. ROC Curve
    plt.figure(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {metrics['roc_auc']:.4f})", color="darkorange", lw=2)
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, "roc_curve.png"), dpi=300)
    plt.close()

    # 3. Precision-Recall Curve
    plt.figure(figsize=(6, 5))
    precision, recall, _ = precision_recall_curve(labels, probs)
    plt.plot(recall, precision, label=f"PR Curve (AUC = {metrics['pr_auc']:.4f})", color="purple", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, "precision_recall_curve.png"), dpi=300)
    plt.close()

    # 4. Calibration Curve
    plt.figure(figsize=(6, 5))
    prob_true, prob_pred = calibration_curve(labels, probs, n_bins=10)
    plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label=f"ECE = {metrics['expected_calibration_error']:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve (Reliability Diagram)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, "calibration_curve.png"), dpi=300)
    plt.close()

    # 5. Confidence Histogram
    plt.figure(figsize=(7, 5))
    real_probs = probs[labels == 0]
    ai_probs = probs[labels == 1]
    plt.hist(real_probs, bins=25, alpha=0.6, label="Real Photograph (Label 0)", color="blue", density=True)
    plt.hist(ai_probs, bins=25, alpha=0.6, label="AI Generated (Label 1)", color="red", density=True)
    plt.axvline(x=config.thresholds.pass_threshold, color="green", linestyle="--", label=f"PASS Threshold ({config.thresholds.pass_threshold})")
    plt.axvline(x=config.thresholds.flagged_threshold, color="darkred", linestyle="--", label=f"FLAGGED Threshold ({config.thresholds.flagged_threshold})")
    plt.xlabel("AI Probability Score")
    plt.ylabel("Density")
    plt.title("Prediction Confidence Distribution")
    plt.legend(loc="upper center")
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, "confidence_histogram.png"), dpi=300)
    plt.close()

    logger.info(f"Generated all evaluation metric plots in {config.results_dir}")

def plot_training_history(history: Dict[str, List[float]], config: AuthenticityConfig):
    config.create_dirs()
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], "r-o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss History")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_acc"], "g-o", label="Val Accuracy")
    plt.plot(epochs, history.get("val_f1", []), "m-o", label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Performance")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, "training_history.png"), dpi=300)
    plt.savefig(os.path.join(config.results_dir, "training_curves.png"), dpi=300)
    plt.close()
    logger.info(f"Saved training curves plot to {os.path.join(config.results_dir, 'training_curves.png')}")
