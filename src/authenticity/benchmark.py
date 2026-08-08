import os
import json
import copy
from typing import Dict, Any
from .config import AuthenticityConfig, ModelConfig
from .trainer import AuthenticityTrainer
from .logger import get_logger

logger = get_logger("authenticity.benchmark")

def run_authenticity_benchmark(config: AuthenticityConfig = None) -> Dict[str, Any]:
    base_config = config or AuthenticityConfig()
    base_config.create_dirs()

    architectures = ["convnext_tiny", "swin_t"]
    results = {}

    logger.info("Starting AI Authenticity Architecture Benchmarking...")

    for arch in architectures:
        logger.info(f"\n{'='*50}\nBenchmarking Architecture: {arch}\n{'='*50}")
        cfg = copy.deepcopy(base_config)
        cfg.model.architecture = arch
        cfg.checkpoints_dir = os.path.join(base_config.checkpoints_dir, arch)
        cfg.models_dir = os.path.join(base_config.models_dir, arch)
        cfg.results_dir = os.path.join(base_config.results_dir, arch)

        trainer = AuthenticityTrainer(cfg)
        metrics = trainer.train(resume=True)

        results[arch] = {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "expected_calibration_error": metrics["expected_calibration_error"],
            "confusion_matrix": metrics["confusion_matrix"]
        }

    # Compare results and make recommendation
    convnext_auc = results.get("convnext_tiny", {}).get("roc_auc", 0.0)
    swin_auc = results.get("swin_t", {}).get("roc_auc", 0.0)
    
    convnext_f1 = results.get("convnext_tiny", {}).get("f1_score", 0.0)
    swin_f1 = results.get("swin_t", {}).get("f1_score", 0.0)

    if swin_auc > convnext_auc + 0.02 or swin_f1 > convnext_f1 + 0.02:
        recommended_model = "swin_t"
        rationale = f"Swin-T outperformed ConvNeXt-Tiny with ROC-AUC {swin_auc:.4f} vs {convnext_auc:.4f} and F1 {swin_f1:.4f} vs {convnext_f1:.4f}."
    else:
        recommended_model = "convnext_tiny"
        rationale = f"ConvNeXt-Tiny achieves competitive performance (ROC-AUC {convnext_auc:.4f}, F1 {convnext_f1:.4f}) with lower computational overhead."

    summary = {
        "architectures": results,
        "recommended_model": recommended_model,
        "recommendation_rationale": rationale
    }

    # Save benchmark outputs
    summary_path = os.path.join(base_config.results_dir, "benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_path = os.path.join(base_config.results_dir, "benchmark_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# AI Authenticity Model Benchmark Report\n\n")
        f.write(f"**Recommended Model**: `{recommended_model}`\n")
        f.write(f"**Rationale**: {rationale}\n\n")
        f.write("| Architecture | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | ECE |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for arch, m in results.items():
            f.write(f"| `{arch}` | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1_score']:.4f} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['expected_calibration_error']:.4f} |\n")

    logger.info(f"Saved benchmark summary to {summary_path} and report to {report_path}")
    return summary
