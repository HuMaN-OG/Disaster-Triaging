# Disaster Triaging System

A multimodal disaster triaging system that combines computer vision and natural language processing to classify disaster images and estimate rescue priority from citizen reports.

The repository includes machine-learning pipelines, a FastAPI backend prototype (`backend/`) and a React/Vite frontend (`frontend/`). Their source code is present, but full end-to-end deployment and production readiness have not been independently verified.

## Demonstration and team contributions

This is an AlgOlympia team project. My resume describes work on multilingual citizen-report priority scoring and combining text predictions with disaster-image classification; it does not imply sole authorship of the entire repository.

**Inspect or reproduce:** Follow the installation instructions below. Run `python -m pytest -q` to execute available tests. The supplied inference examples require compatible model checkpoints and dependencies. Model training can take time and may require datasets or weights that are not immediately available.

**Current status:** The model pipelines and evaluation are documented below; the API, dashboard, incident database and interactive map remain in progress. This repository is a research prototype, not a deployed public emergency service.

**Screenshots to add after running the code:** (1) anonymized incident report with language and 0–100 priority output, (2) image inference with predicted disaster class, and (3) a genuine evaluation plot. Keep real citizen information out of the repository. No verified UI screenshots are currently included here.

**Reproducibility note:** Text-model code includes both XGBoost and CatBoost paths. Confirm which training script, checkpoint and evaluation split generated each reported metric before attributing results to one model.

## Features

### Implemented

- Disaster image classification using an ensemble of vision models
- Multilingual priority prediction for English, Hindi, and Hinglish
- Multimodal inference supporting image-only, text-only, and combined predictions
- Continuous priority scoring (0–100) with categorical priority levels
- Model benchmarking and evaluation pipeline
- Training and inference scripts for both image and text models

### Application prototype present (requires end-to-end verification)

- FastAPI backend routes for authentication, prediction, incidents and dashboard
- React/Vite frontend with citizen portal, rescue dashboard and map components
- Database models for incidents, users, teams and status history
- Route tests under `backend/app/tests/`

These are code-presence claims, not proof of production readiness. The current frontend is React/Vite, not Next.js. Do not deploy publicly with the demo credentials or JWT secret in `backend/app/core/config.py`.

## Repository Structure

```text
.
├── models/
│   ├── image/
│   ├── text/
│   └── archive/
├── results/
│   ├── image/
│   ├── text/
│   └── archive/
├── src/
│   ├── inference/
│   ├── utils/
│   ├── dataset.py
│   ├── model.py
│   ├── train_image_classifier.py
│   └── benchmark_models.py
├── tests/
├── requirements.txt
└── README.md
```

## Models

### Image Classification

Production image classifier:

- Swin Transformer-Tiny
- ConvNeXt-Tiny
- Vision Transformer (ViT-B/16)
- EfficientNetV2-S

These models are combined using a weighted ensemble with Test-Time Augmentation.

Performance:

| Metric | Value |
|-------|------:|
| Accuracy | 85.40% |
| Macro F1 | 85.15% |

Supported classes:

- Flood
- Earthquake
- Urban Fire
- Wild Fire
- Drought

---

### Text Priority Prediction

Pipeline:

```text
Citizen Report
        │
SentenceTransformer
(paraphrase-multilingual-mpnet-base-v2)
        │
CatBoost Regressor
        │
Priority Score (0–100)
```

Performance:

| Language | R² |
|----------|----:|
| English | 95.54% |
| Hindi | 89.58% |
| Hinglish | 86.93% |

## Setup

Clone the repository.

```bash
git clone https://github.com/HuMaN-OG/Disaster-Triaging.git
cd Disaster-Triaging
```

Create a virtual environment.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

## Running the Models

### Image Model

```bash
python -m src.train_image_classifier
```

### Text Model

```bash
python -m src.train_text_classifier
```

### Evaluate the Ensemble

```bash
python -m src.evaluate_ensemble
```

## Usage

### Image Prediction

```python
from PIL import Image
from src.inference.predict import ImagePredictor

predictor = ImagePredictor()

result = predictor.predict(
    Image.open("sample.jpg")
)

print(result.disaster_type)
print(result.confidence)
```

### Multimodal Prediction

```python
from PIL import Image
from src.inference.predict import MultimodalPredictor

predictor = MultimodalPredictor()

result = predictor.predict(
    image_input=Image.open("sample.jpg"),
    text_report="Water has reached the second floor. Four people are trapped on the roof."
)

print(result.to_dict())
```

## Architecture

The project is divided into two independent machine learning pipelines.

### Image Pipeline

```
Image

↓

Vision Ensemble

↓

Disaster Classification
```

### Text Pipeline

```
Citizen Report

↓

SentenceTransformer

↓

CatBoost

↓

Priority Score
```

### Multimodal Pipeline

```
Image
      \
       \
        MultimodalPredictor
       /
Text  /

↓

Unified Prediction
```

Keeping both pipelines independent allows either model to be updated without retraining the other.

## Evaluation

### Image Model

- Accuracy: 85.40%
- Macro F1: 85.15%

### Text Model

| Language | R² |
|----------|----:|
| English | 95.54% |
| Hindi | 89.58% |
| Hinglish | 86.93% |

Benchmark reports, confusion matrices, and evaluation outputs are available in the `results/` directory.

## Current Status

Completed:

- Image classification pipeline
- Multilingual text priority prediction
- Multimodal inference pipeline
- Model benchmarking
- Hyperparameter tuning
- Model evaluation
- Automated testing

Implementation / verification pending:

- Run end-to-end tests for the present FastAPI and React code
- Confirm production deployment, data persistence and authentication configuration
- Verify live inference and incident-management workflows with sanitized demo data
- Replace demo credentials and hard-coded signing keys before public deployment

## Known Limitations

- Image classification performs scene classification only; object detection is not implemented.
- Performance depends on image quality and lighting conditions.
- Hinglish performance is lower than English due to inconsistent Romanized spellings.
- Priority prediction currently relies only on image and text inputs.
- Disaster-specific technical indicators such as Richter magnitude, flood water level, and Air Quality Index are not yet incorporated.
- Backend and frontend source code exists, but a running end-to-end deployment was not verified in this review.
- Demo authentication credentials and a hard-coded JWT signing key must be replaced before any public deployment.