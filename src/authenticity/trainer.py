import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Any, Tuple
from .config import AuthenticityConfig
from .dataset import get_authenticity_dataloaders
from .model import AuthenticityClassifier
from .evaluator import AuthenticityEvaluator
from .checkpoint import CheckpointManager
from .visualization import plot_training_history, plot_authenticity_metrics
from .logger import get_logger

logger = get_logger("authenticity.trainer")

class AuthenticityTrainer:
    def __init__(self, config: AuthenticityConfig = None):
        self.config = config or AuthenticityConfig()
        self.config.create_dirs()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if self.device.type == "cpu":
            num_threads = min(8, os.cpu_count() or 4)
            torch.set_num_threads(num_threads)
            logger.info(f"Using device for training: {self.device} (threads={num_threads})")
        else:
            logger.info(f"Using device for training: {self.device}")

        self.checkpoint_manager = CheckpointManager(self.config.checkpoints_dir)
        self.evaluator = AuthenticityEvaluator(self.config)

    def train(self, resume: bool = True) -> Dict[str, Any]:
        train_loader, val_loader, test_loader, weights = get_authenticity_dataloaders(self.config)
        model = AuthenticityClassifier(self.config.model).to(self.device)

        # Loss function with pos_weight for class balance
        if self.config.train.use_class_weights and weights.get("pos_weight"):
            pos_w = torch.tensor([weights["pos_weight"]], dtype=torch.float32).to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
            logger.info(f"Using BCEWithLogitsLoss with pos_weight={weights['pos_weight']:.3f}")
        else:
            criterion = nn.BCEWithLogitsLoss()

        optimizer = AdamW(
            model.parameters(),
            lr=self.config.train.learning_rate,
            weight_decay=self.config.train.weight_decay
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

        start_epoch = 1
        best_metric = -1.0
        patience_counter = 0

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "val_auc": []
        }

        # Check for existing checkpoint to resume
        if resume:
            ckpt = self.checkpoint_manager.load_latest(self.device)
            if ckpt is not None:
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                start_epoch = ckpt["epoch"] + 1
                best_metric = ckpt.get("best_metric", -1.0)
                history = ckpt.get("history", history)
                logger.info(f"Resumed training from epoch {start_epoch} with best_metric={best_metric:.4f}")

        if start_epoch > self.config.train.num_epochs:
            logger.info(f"Training already completed ({start_epoch - 1}/{self.config.train.num_epochs} epochs). Skipping loop.")
            best_ckpt = self.checkpoint_manager.load_best(self.device)
            if best_ckpt:
                model.load_state_dict(best_ckpt["model_state_dict"])
            test_metrics = self.evaluator.evaluate_model(model, test_loader, self.device)
            plot_authenticity_metrics(test_metrics, self.config)
            return test_metrics

        logger.info(f"Starting training for {self.config.train.num_epochs - start_epoch + 1} remaining epochs...")

        for epoch in range(start_epoch, self.config.train.num_epochs + 1):
            t0 = time.time()
            model.train()
            total_train_loss = 0.0

            for step, (images, labels, _) in enumerate(train_loader, 1):
                images = images.to(self.device)
                labels = labels.float().to(self.device)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * images.size(0)

                if step % 5 == 0 or step == len(train_loader):
                    logger.info(f"Epoch [{epoch}/{self.config.train.num_epochs}] Step [{step}/{len(train_loader)}] Batch Loss: {loss.item():.4f}")

            avg_train_loss = total_train_loss / len(train_loader.dataset)

            # Validation step
            val_metrics = self.evaluator.evaluate_model(model, val_loader, self.device)
            val_loss = 1.0 - val_metrics["f1_score"] # surrogate or evaluate BCE loss
            val_acc = val_metrics["accuracy"]
            val_f1 = val_metrics["f1_score"]
            val_auc = val_metrics["roc_auc"]

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)
            history["val_auc"].append(val_auc)

            scheduler.step(val_auc)
            epoch_time = time.time() - t0

            logger.info(
                f"Epoch [{epoch}/{self.config.train.num_epochs}] ({epoch_time:.1f}s) - "
                f"Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}"
            )

            is_best = val_auc > best_metric
            if is_best:
                best_metric = val_auc
                patience_counter = 0
            else:
                patience_counter += 1

            ckpt_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_metric": best_metric,
                "history": history,
                "config": self.config
            }
            self.checkpoint_manager.save(ckpt_state, is_best=is_best, filename=f"epoch_{epoch}.pt")
            self.checkpoint_manager.save(ckpt_state, is_best=False, filename="latest_checkpoint.pt")

            if patience_counter >= self.config.train.patience:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        plot_training_history(history, self.config)

        # Final evaluation on Test Set using best model
        logger.info("Loading best model checkpoint for final Test set evaluation...")
        best_ckpt = self.checkpoint_manager.load_best(self.device)
        if best_ckpt:
            model.load_state_dict(best_ckpt["model_state_dict"])
            
        test_metrics = self.evaluator.evaluate_model(model, test_loader, self.device)
        plot_authenticity_metrics(test_metrics, self.config)

        # Save final PyTorch model weights to models/authenticity/best_model.pth
        best_model_path = os.path.join(self.config.models_dir, "best_model.pth")
        model_save_path = os.path.join(self.config.models_dir, f"{self.config.model.architecture}_authenticity.pt")
        torch.save(model.state_dict(), best_model_path)
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"Saved production model state_dict to {best_model_path} and {model_save_path}")

        return test_metrics
