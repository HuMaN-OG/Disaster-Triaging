# Multimodal AI Disaster Triaging System — Comprehensive Technical Project Report

**Author / Team**: AI Disaster Triage System Development Team  
**Date**: August 2026  
**Target Domain**: Emergency Dispatch, Disaster Response, Multilingual Triage Automation  

---

## Executive Summary

During natural and urban disasters (floods, wildfires, earthquakes, landslides), emergency call centers and first responders are flooded with unstructured, multi-lingual citizen reports and crowd-sourced imagery. Existing triage protocols rely on manual categorization, leading to critical bottlenecks and delayed dispatch to life-threatening emergency sites.

This project delivers a **multimodal, multilingual disaster triaging system** that combines:
1. A **Vision Ensemble** (Swin-T, ConvNeXt-Tiny, ViT-B/16, EfficientNet-V2-S) predicting 6 disaster types with **85.4% accuracy**.
2. A **Multilingual Priority Regressor** (`paraphrase-multilingual-mpnet-base-v2` + CatBoost) predicting continuous priority scores ($0-100$) across **English ($95.5\% R^2$)**, **Devanagari Hindi ($89.6\% R^2$)**, and **Romanized Hinglish ($86.9\% R^2$)**.

---

## 1. Problem Statement & Operational Context

Emergency response coordinators face two fundamental challenges during catastrophe events:
1. **Visual Classification Bottleneck**: Distinguishing between disaster categories (e.g., Wildfire vs. Urban Fire vs. Flood) from aerial/ground photos under varying illumination and resolution conditions.
2. **Multilingual Code-Mixed Text Ambiguity**: Citizens in India and multicultural regions report life-threatening situations using code-mixed Hinglish (e.g., *"Paani ghar ke andar aa gaya hai, 4 log phase hue hain"*). Standard English NLP models fail on Hinglish text, yielding false low-priority predictions that endanger lives.

---

## 2. Dataset Overview

### Vision Dataset
- **Taxonomy**: 6 Disaster Classes: `Flood`, `Earthquake`, `Urban Fire`, `Wild Fire`, `Landslide`, `Drought`.
- **Pre-processing**: Albumentations / PyTorch transforms (Random horizontal flip, rotation $\pm 15^\circ$, color jitter, ImageNet standard normalization to $224 \times 224$).

### Multilingual Text Priority Dataset
- **Original Dataset**: 2,000 English emergency reports labeled with continuous priority scores ($0.0 - 100.0$).
- **Hinglish v2 High-Quality Augmentation**: 9,521 candidate variants generated across 5 structural writing styles (Code-Mixed, Phonetic SMS, Disaster-Aware) and filtered using sentence-transformer cosine similarity ($\ge 0.72$).
- **Final Production Dataset**: **5,484 samples** (2,000 English + 3,484 accepted Hinglish variants).

---

## 3. Methodology & Architecture Design

### Computer Vision Pipeline
We evaluated multiple modern vision backbones using transfer learning with ImageNet pre-training:
- **Feature Extractors**: Swin-Transformer (Swin-T), ConvNeXt-Tiny, Vision Transformer (ViT-B/16), EfficientNet-V2-S.
- **Ensemble Fusion**: Weighted Softmax probability averaging:
  $$P_{\text{ensemble}}(c) = 0.30 P_{\text{swin}}(c) + 0.20 P_{\text{convnext}}(c) + 0.20 P_{\text{vit}}(c) + 0.30 P_{\text{effnet}}(c)$$

### Multilingual Priority Regression Pipeline
- **Normalizer**: External 109-rule JSON normalization dictionary standardizing erratic Hinglish spellings (e.g., `wareghar` $\rightarrow$ `warehouse`, `phasa` $\rightarrow$ `phanse`, `paani` $\rightarrow$ `water/paani`).
- **Embedding Backbone**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` generating 768-dimensional dense semantic representations.
- **Gradient Boosted Head**: `CatBoostRegressor` tuned via Optuna (`depth=4`, `learning_rate=0.0714`, `iterations=500`, `l2_leaf_reg=1.75`).

---

## 4. Benchmark Comparisons & Rejection Analysis

### Vision Backbone Benchmark Comparison

| Model Architecture | Parameters | Val Accuracy | Macro F1 | Status |
|---|:---:|:---:|:---:|:---:|
| EfficientNet-B0 | 5.3 M | 79.20% | 78.80% | Rejected (Underfitting) |
| EfficientNet-B2 | 9.1 M | 81.40% | 81.10% | Rejected (Superseded by V2) |
| ViT-B/16 | 86.6 M | 82.90% | 82.40% | Accepted in Ensemble |
| ConvNeXt-Tiny | 28.6 M | 83.75% | 83.50% | Accepted in Ensemble |
| Swin-T | 28.3 M | 84.12% | 83.95% | Accepted in Ensemble |
| EfficientNet-V2-S | 21.5 M | 84.50% | 84.10% | Accepted in Ensemble |
| **Weighted Vision Ensemble** | — | **85.40%** | **85.15%** | **WINNER (Production)** |

**Why Single Models Were Rejected**: While Swin-T and EfficientNet-V2-S performed strongly individually, single vision models exhibited higher variance on low-lighting urban fire vs. wildfire edge cases. The weighted ensemble reduced error variance by **18.2%**.

### Text Embedding Model Benchmark Comparison

| Embedding Model | Dimension | English $R^2$ | Hindi $R^2$ | Hinglish $R^2$ | Status |
|---|:---:|:---:|:---:|:---:|:---:|
| BAAI/bge-base-en-v1.5 | 768 | 94.10% | 42.10% | 38.50% | Rejected (English-Only) |
| intfloat/multilingual-e5-base | 768 | 91.80% | 87.20% | 71.40% | Rejected (Lower Hinglish $R^2$) |
| BAAI/bge-m3 | 1024 | 93.50% | 88.90% | 73.20% | Rejected (Higher Latency) |
| **paraphrase-multilingual-mpnet-base-v2** | 768 | **95.54%** | **89.58%** | **86.93%** | **WINNER (Production)** |

---

## 5. Final Production Metrics Summary

| Language Target | Baseline $R^2$ | v1 Augmentation $R^2$ | **v2 Production $R^2$** | MAE (Points) | RMSE (Points) |
|---|:---:|:---:|:---:|:---:|:---:|
| **English** | 92.33% | 96.72% | **95.54%** | **4.63** | **5.57** |
| **Hindi (Devanagari)** | 89.34% | 91.07% | **89.58%** | **6.68** | **8.51** |
| **Hinglish (Romanized)** | 75.97% | 79.50% | **86.93%** | **7.42** | **9.53** |

---

## 6. Deployment Architecture & Production API

The inference engine is unified under `MultimodalPredictor` in `src/inference/predict.py`.

```
                        ┌───────────────────────────────┐
                        │   MultimodalPredictor API     │
                        └───────────────┬───────────────┘
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
     [ Image Only ]              [ Text Only ]             [ Multimodal ]
             │                          │                          │
  ImagePredictor.predict()    TextPredictor.predict()       Executes Both &
             │                          │                   Combines Decision
             ▼                          ▼                          ▼
    Disaster Type &            Priority Score (0-100)      Fused Triage Dict
    Class Probabilities        & Detected Language         with Metadata
```

### End-to-End Latency Profile
- **Image Ensemble Inference**: ~14.5 ms
- **Hinglish Normalization & MPNet Embedding**: ~7.8 ms
- **CatBoost Prediction**: ~0.7 ms
- **Total Multimodal Latency**: **~23.0 ms per report** (suitable for high-throughput API gateways).

---

## 7. Future Expansion Roadmap

1. **Edge Deployment**: Quantize MPNet encoder to ONNX INT8 to run on mobile first-responder tablets.
2. **Speech-to-Text Integration**: Connect Whisper-v3 for automatic voice report transcription in audio emergency calls.
3. **Geospatial Prioritization**: Overlay real-time weather and flood map layers to adjust priority scores dynamically.
