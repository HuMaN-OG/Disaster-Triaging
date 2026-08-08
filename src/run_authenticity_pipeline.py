import os
import json
import argparse
from src.authenticity.config import AuthenticityConfig
from src.authenticity.audit import audit_authenticity_dataset
from src.authenticity.trainer import AuthenticityTrainer
from src.authenticity.benchmark import run_authenticity_benchmark
from src.authenticity.logger import get_logger

logger = get_logger("authenticity.pipeline")

def main():
    parser = argparse.ArgumentParser(description="Run AI Authenticity Training and Evaluation Pipeline")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--architecture", type=str, default="convnext_tiny", help="Model architecture")
    parser.add_argument("--benchmark", action="store_true", help="Run comparative benchmark with Swin-T")
    args = parser.parse_args()

    config = AuthenticityConfig()
    config.train.num_epochs = args.epochs
    config.train.batch_size = args.batch_size
    config.model.architecture = args.architecture

    logger.info("=== STEP 1: Running Dataset Audit ===")
    audit_results = audit_authenticity_dataset(config)

    logger.info("=== STEP 2: Running Model Training & Evaluation ===")
    trainer = AuthenticityTrainer(config)
    test_metrics = trainer.train(resume=True)

    logger.info("\n" + "="*60)
    logger.info("CONVNEXT-TINY BASELINE RESULTS:")
    logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Recall: {test_metrics['recall']:.4f}")
    logger.info(f"F1 Score: {test_metrics['f1_score']:.4f}")
    logger.info(f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
    logger.info(f"PR-AUC: {test_metrics['pr_auc']:.4f}")
    logger.info(f"ECE: {test_metrics['expected_calibration_error']:.4f}")
    logger.info("="*60 + "\n")

    if args.benchmark:
        logger.info("=== STEP 3: Running Architecture Benchmark (ConvNeXt-Tiny vs Swin-T) ===")
        benchmark_summary = run_authenticity_benchmark(config)
        logger.info(f"Benchmark Recommendation: {benchmark_summary['recommended_model']}")

    logger.info("Pipeline execution finished successfully!")

if __name__ == "__main__":
    main()
