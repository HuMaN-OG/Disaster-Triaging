import os
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class ModelConfig:
    architecture: str = "convnext_tiny"  # options: convnext_tiny, swin_t
    pretrained: bool = True
    num_classes: int = 1  # Binary classification: 0 = Real, 1 = AI-Generated
    dropout_rate: float = 0.2
    img_size: int = 224

@dataclass
class TrainConfig:
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    val_check_interval: int = 1
    patience: int = 4
    use_class_weights: bool = True
    seed: int = 42

@dataclass
class DecisionThresholds:
    pass_threshold: float = 0.30     # Prob(AI) < 0.30 -> PASS (Real Photograph)
    flagged_threshold: float = 0.70  # Prob(AI) > 0.70 -> FLAGGED (AI Generated)
    # 0.30 <= Prob(AI) <= 0.70 -> UNCERTAIN

@dataclass
class AuthenticityConfig:
    root_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir: str = field(default_factory=lambda: os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "data", "authenticity"))
    dataset_dir: str = field(default_factory=lambda: os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "dataset"))
    models_dir: str = field(default_factory=lambda: os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "models", "authenticity"))
    results_dir: str = field(default_factory=lambda: os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "results", "authenticity"))
    logs_dir: str = field(default_factory=lambda: os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "logs", "authenticity"))
    checkpoints_dir: str = field(default_factory=lambda: os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "checkpoints", "authenticity"))
    
    real_subdirs: List[str] = field(default_factory=lambda: [
        "Damaged_Infrastructure",
        "Fire_Disaster",
        "Land_Disaster",
        "Water_Disaster"
    ])
    ai_subdir: str = "AI_Images"
    
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    thresholds: DecisionThresholds = field(default_factory=DecisionThresholds)

    def __post_init__(self):
        if self.root_dir:
            self.data_dir = os.path.join(self.root_dir, "data", "authenticity")
            self.dataset_dir = os.path.join(self.root_dir, "dataset")
            self.models_dir = os.path.join(self.root_dir, "models", "authenticity")
            self.results_dir = os.path.join(self.root_dir, "results", "authenticity")
            self.logs_dir = os.path.join(self.root_dir, "logs", "authenticity")
            self.checkpoints_dir = os.path.join(self.root_dir, "checkpoints", "authenticity")

    def create_dirs(self):
        for path in [
            self.data_dir,
            self.models_dir,
            self.results_dir,
            self.logs_dir,
            self.checkpoints_dir
        ]:
            os.makedirs(path, exist_ok=True)
