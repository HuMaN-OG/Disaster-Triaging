import os
import torch
from typing import Dict, Any, Optional
from .config import AuthenticityConfig
from .logger import get_logger

logger = get_logger("authenticity.checkpoint")

class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        state: Dict[str, Any],
        is_best: bool = False,
        filename: str = "latest_checkpoint.pt"
    ) -> str:
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(state, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(state, best_path)
            logger.info(f"Saved best model checkpoint to {best_path}")
            
        return filepath

    def load_latest(self, device: torch.device = torch.device("cpu")) -> Optional[Dict[str, Any]]:
        latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        if os.path.exists(latest_path):
            try:
                logger.info(f"Loading latest checkpoint from {latest_path}")
                return torch.load(latest_path, map_location=device)
            except Exception as e:
                logger.warning(f"Failed to load {latest_path} (possibly corrupted): {e}")

        # Fallback to scanning for epoch_*.pt checkpoints
        epoch_files = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith("epoch_") and f.endswith(".pt")
        ]
        if epoch_files:
            def extract_epoch(fname):
                try:
                    return int(fname.split("_")[1].split(".")[0])
                except Exception:
                    return 0
            epoch_files.sort(key=extract_epoch, reverse=True)
            for fname in epoch_files:
                latest_file = os.path.join(self.checkpoint_dir, fname)
                try:
                    logger.info(f"Loading epoch checkpoint from {latest_file}")
                    return torch.load(latest_file, map_location=device)
                except Exception as e:
                    logger.warning(f"Checkpoint file {latest_file} is corrupted/unreadable: {e}")

        return None

    def load_best(self, device: torch.device = torch.device("cpu")) -> Optional[Dict[str, Any]]:
        best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        if os.path.exists(best_path):
            try:
                logger.info(f"Loading best checkpoint from {best_path}")
                return torch.load(best_path, map_location=device)
            except Exception as e:
                logger.warning(f"Failed to load {best_path}: {e}")
        return self.load_latest(device)
