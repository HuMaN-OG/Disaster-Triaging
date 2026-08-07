# Disaster Triaging System

A multimodal disaster triaging system that combines computer vision and natural language processing to classify disaster images and estimate rescue priority from citizen reports.

The repository currently contains the complete machine learning pipelines for image classification and multilingual text priority prediction. The backend API, frontend application, and rescue operations dashboard are under active development.

## Features

### Implemented

- Disaster image classification using an ensemble of vision models
- Multilingual priority prediction for English, Hindi, and Hinglish
- Multimodal inference supporting image-only, text-only, and combined predictions
- Continuous priority scoring (0–100) with categorical priority levels
- Model benchmarking and evaluation pipeline
- Training and inference scripts for both image and text models

### In Progress

- FastAPI backend
- Next.js frontend
- Rescue operations dashboard
- Incident database
- Interactive map interface
- Live request management
- REST API for production inference

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
git clone https://github.com/<your-username>/disaster-triaging.git
cd disaster-triaging
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

In Progress:

- FastAPI backend
- Next.js frontend
- Rescue operations dashboard
- Incident database
- Live API endpoints
- Interactive map interface

## Known Limitations

- Image classification performs scene classification only; object detection is not implemented.
- Performance depends on image quality and lighting conditions.
- Hinglish performance is lower than English due to inconsistent Romanized spellings.
- Priority prediction currently relies only on image and text inputs.
- Disaster-specific technical indicators such as Richter magnitude, flood water level, and Air Quality Index are not yet incorporated.
- Backend services and web interface are still under development.
- Rescue request management and live dashboard functionality are not yet available.