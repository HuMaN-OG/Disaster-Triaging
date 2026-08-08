import os
import pytest
import torch
from PIL import Image
import numpy as np

from src.authenticity.config import AuthenticityConfig, ModelConfig
from src.authenticity.dataset import AuthenticityDataset, get_transforms
from src.authenticity.model import AuthenticityClassifier
from src.authenticity.predictor import AuthenticityPredictor
from src.authenticity.evaluator import calculate_ece, AuthenticityEvaluator
from src.authenticity.checkpoint import CheckpointManager

def test_authenticity_config():
    config = AuthenticityConfig()
    assert config.model.architecture == "convnext_tiny"
    assert config.thresholds.pass_threshold == 0.30
    assert config.thresholds.flagged_threshold == 0.70

def test_dataset_transforms(tmp_path):
    img_path = str(tmp_path / "test.jpg")
    img = Image.new("RGB", (100, 100), (255, 0, 0))
    img.save(img_path)

    _, eval_transform = get_transforms(224)
    ds = AuthenticityDataset([(img_path, 0)], transform=eval_transform)
    assert len(ds) == 1

    tensor, label, path = ds[0]
    assert tensor.shape == (3, 224, 224)
    assert label == 0
    assert path == img_path

def test_model_forward():
    config = ModelConfig(architecture="convnext_tiny", pretrained=False)
    model = AuthenticityClassifier(config)
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)
    assert logits.shape == (2,)

def test_predictor_decision_thresholds(tmp_path):
    img_path = str(tmp_path / "sample.jpg")
    img = Image.new("RGB", (224, 224), (128, 128, 128))
    img.save(img_path)

    config = AuthenticityConfig()
    model = AuthenticityClassifier(config.model)
    predictor = AuthenticityPredictor(config=config, model=model)

    res = predictor.predict_image(img_path)
    assert res["assessment"] in ["PASS", "FLAGGED", "UNCERTAIN"]
    assert 0.0 <= res["ai_probability"] <= 1.0
    assert "explanation" in res

def test_calculate_ece():
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    ece = calculate_ece(probs, labels)
    assert isinstance(ece, float)
    assert 0.0 <= ece <= 1.0

def test_checkpoint_manager(tmp_path):
    mgr = CheckpointManager(str(tmp_path))
    dummy_state = {"epoch": 1, "best_metric": 0.95}
    filepath = mgr.save(dummy_state, is_best=True, filename="latest.pt")
    
    assert os.path.exists(filepath)
    assert os.path.exists(os.path.join(str(tmp_path), "best_model.pt"))
    
    loaded = mgr.load_best()
    assert loaded["epoch"] == 1
    assert loaded["best_metric"] == 0.95
