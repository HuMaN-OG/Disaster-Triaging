import os
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)
from .config import AuthenticityConfig
from .logger import get_logger

logger = get_logger("authenticity.evaluator")

def calculate_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i+1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin] == (probs[in_bin] >= 0.5))
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    return float(ece)

class AuthenticityEvaluator:
    def __init__(self, config: AuthenticityConfig):
        self.config = config

    def evaluate_model(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device = torch.device("cpu")
    ) -> Dict[str, Any]:
        model.eval()
        model.to(device)

        all_probs = []
        all_labels = []
        all_paths = []

        logger.info("Evaluating model on evaluation dataset...")
        with torch.no_grad():
            for images, labels, paths in dataloader:
                images = images.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy()

                all_probs.extend(probs)
                all_labels.extend(labels.numpy())
                all_paths.extend(paths)

        probs_arr = np.array(all_probs)
        labels_arr = np.array(all_labels)
        preds_arr = (probs_arr >= 0.5).astype(int)

        # Standard Metrics
        acc = float(accuracy_score(labels_arr, preds_arr))
        prec = float(precision_score(labels_arr, preds_arr, zero_division=0))
        rec = float(recall_score(labels_arr, preds_arr, zero_division=0))
        f1 = float(f1_score(labels_arr, preds_arr, zero_division=0))

        try:
            roc_auc = float(roc_auc_score(labels_arr, probs_arr))
        except Exception:
            roc_auc = 0.5

        precision_curve, recall_curve, _ = precision_recall_curve(labels_arr, probs_arr)
        pr_auc = float(auc(recall_curve, precision_curve))
        ece = float(calculate_ece(probs_arr, labels_arr))

        cm = confusion_matrix(labels_arr, preds_arr)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # Error Analysis Extraction
        false_positives = []
        false_negatives = []
        all_sample_results = []

        for path, prob, label, pred in zip(all_paths, probs_arr, labels_arr, preds_arr):
            error_margin = abs(float(prob) - float(label))
            uncertainty = abs(float(prob) - 0.5)
            
            sample_info = {
                "image_path": path,
                "ai_probability": float(prob),
                "true_label": int(label),
                "predicted_label": int(pred),
                "error_margin": float(error_margin),
                "uncertainty_margin": float(uncertainty)
            }
            all_sample_results.append(sample_info)

            if label == 0 and pred == 1:
                false_positives.append(sample_info)
            elif label == 1 and pred == 0:
                false_negatives.append(sample_info)

        # Worst predictions (highest error margin)
        worst_predictions = sorted(all_sample_results, key=lambda x: x["error_margin"], reverse=True)[:15]
        
        # Most uncertain predictions (closest to 0.5)
        most_uncertain = sorted(all_sample_results, key=lambda x: x["uncertainty_margin"])[:15]

        metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "expected_calibration_error": ece,
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp)
            },
            "false_positives_count": len(false_positives),
            "false_negatives_count": len(false_negatives),
            "worst_predictions": worst_predictions,
            "most_uncertain_predictions": most_uncertain,
            "raw_predictions": {
                "paths": all_paths,
                "probabilities": probs_arr.tolist(),
                "true_labels": labels_arr.tolist()
            }
        }

        self.save_evaluation_report(metrics, labels_arr, preds_arr)
        return metrics

    def save_evaluation_report(self, metrics: Dict[str, Any], labels_arr: np.ndarray = None, preds_arr: np.ndarray = None):
        from sklearn.metrics import classification_report
        self.config.create_dirs()
        report_json_path = os.path.join(self.config.results_dir, "evaluation_metrics.json")
        metrics_json_path = os.path.join(self.config.results_dir, "metrics.json")
        
        save_metrics = {k: v for k, v in metrics.items() if k != "raw_predictions"}
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(save_metrics, f, indent=2)

        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(save_metrics, f, indent=2)

        if labels_arr is not None and preds_arr is not None:
            clf_report = classification_report(
                labels_arr,
                preds_arr,
                target_names=["Real Photograph", "AI Generated"],
                output_dict=True,
                zero_division=0
            )
            report_clf_path = os.path.join(self.config.results_dir, "classification_report.json")
            with open(report_clf_path, "w", encoding="utf-8") as f:
                json.dump(clf_report, f, indent=2)

        report_txt_path = os.path.join(self.config.results_dir, "evaluation_report.txt")
        with open(report_txt_path, "w", encoding="utf-8") as f:
            f.write("="*60 + "\n")
            f.write("AI IMAGE AUTHENTICITY EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Accuracy:                    {metrics['accuracy']:.4f}\n")
            f.write(f"Precision (AI Class):         {metrics['precision']:.4f}\n")
            f.write(f"Recall (AI Class):            {metrics['recall']:.4f}\n")
            f.write(f"F1 Score:                    {metrics['f1_score']:.4f}\n")
            f.write(f"ROC-AUC:                     {metrics['roc_auc']:.4f}\n")
            f.write(f"PR-AUC:                      {metrics['pr_auc']:.4f}\n")
            f.write(f"Expected Calibration Error:  {metrics['expected_calibration_error']:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            cm = metrics['confusion_matrix']
            f.write(f"  True Negatives (Real -> Real): {cm['true_negative']}\n")
            f.write(f"  False Positives (Real -> AI):  {cm['false_positive']}\n")
            f.write(f"  False Negatives (AI -> Real): {cm['false_negative']}\n")
            f.write(f"  True Positives (AI -> AI):   {cm['true_positive']}\n\n")

        logger.info(f"Saved evaluation metrics to {report_json_path}, {metrics_json_path} and report to {report_txt_path}")
