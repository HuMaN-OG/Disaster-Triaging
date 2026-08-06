# Image Classifier — Project Structure & Usage

This document explains the disaster **image classification** pipeline (independent of the text model).

## Project structure

```
disaster-triaging/
├── dataset/                          # Image data (unchanged on disk)
│   ├── Damaged_Infrastructure/
│   │   ├── Earthquake/               → label: Earthquake
│   │   └── Infrastructure/           → label: Earthquake  (mapped in loader)
│   ├── Fire_Disaster/
│   │   ├── Urban_Fire/               → label: Urban Fire
│   │   └── Wild_Fire/                → label: Wild Fire
│   ├── Land_Disaster/
│   │   ├── Drought/                  → label: Drought
│   │   └── Land_Slide/               → label: Landslide
│   └── Water_Disaster/               → label: Flood
├── src/
│   ├── dataset.py                    # Custom Dataset + path→label mapping
│   ├── model.py                      # EfficientNet-B0 classifier
│   ├── train_image_classifier.py     # Training loop, early stopping, checkpoints
│   ├── evaluate.py                   # Metrics + confusion matrix
│   └── utils/
│       └── image_classifier.py       # Config, metrics, plotting helpers
│                                     # (repo already had a src/utils/ package,
│                                      so a top-level src/utils.py cannot coexist)
├── models/
│   └── disaster_classifier.pth       # Best checkpoint (created after training)
└── results/
    ├── confusion_matrix.png
    └── training_history.png
```

## Purpose of each file

| File | Role |
|------|------|
| `src/utils/image_classifier.py` | Shared constants (class names, paths, hyper-parameters), seed helpers, metric computation, and plotting (`confusion_matrix.png`, `training_history.png`). |
| `src/dataset.py` | Recursively scans `dataset/`, maps folder paths to the 6 labels (Infrastructure → Earthquake), stratified 80/20 split, train/val transforms. |
| `src/model.py` | EfficientNet-B0 (ImageNet pretrained) with a 6-class head; `freeze_backbone()` / `unfreeze_backbone()` for staged fine-tuning; checkpoint save/load. |
| `src/train_image_classifier.py` | End-to-end training: CrossEntropyLoss, AdamW (1e-4), ReduceLROnPlateau, early stopping, saves best weights to `models/disaster_classifier.pth`. |
| `src/evaluate.py` | Loads a checkpoint, reports loss/accuracy/precision/recall/F1 and writes the confusion matrix. |

## Label mapping (implemented in the loader — files are not moved)

| Folder path | Training label |
|-------------|----------------|
| `Water_Disaster/` | Flood |
| `Damaged_Infrastructure/Earthquake/` | Earthquake |
| `Damaged_Infrastructure/Infrastructure/` | Earthquake |
| `Fire_Disaster/Urban_Fire/` | Urban Fire |
| `Fire_Disaster/Wild_Fire/` | Wild Fire |
| `Land_Disaster/Land_Slide/` | Landslide |
| `Land_Disaster/Drought/` | Drought |

## How to train

From the project root (with the venv activated and dependencies installed):

```bash
python -m src.train_image_classifier
```

Useful options:

```bash
python -m src.train_image_classifier --epochs 20 --batch-size 32 --lr 1e-4
python -m src.train_image_classifier --unfreeze-backbone   # fine-tune full network
```

Outputs:

- Best model → `models/disaster_classifier.pth`
- Curves → `results/training_history.png`
- Confusion matrix → `results/confusion_matrix.png`

By default the EfficientNet backbone is **frozen** and only the classifier head is trained.

## How to evaluate

```bash
python -m src.evaluate
python -m src.evaluate --checkpoint models/disaster_classifier.pth --split val
python -m src.evaluate --split all
```

Reports validation (or full-dataset) loss, accuracy, precision, recall, F1, and refreshes `results/confusion_matrix.png`.

## Training defaults

- Split: 80% train / 20% val (stratified)
- Loss: CrossEntropyLoss
- Optimizer: AdamW (lr=1e-4)
- Scheduler: ReduceLROnPlateau
- Epochs: 20 (with early stopping patience=5)
- Model: EfficientNet-B0, backbone frozen initially
