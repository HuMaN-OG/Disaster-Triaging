# 🚨 Flood Rescue Priority Prediction

> A multimodal machine learning system that triages citizen flood reports by fusing text and image signals to predict emergency severity scores and priority classes in real time.

---

## 📌 Project Objective

During flood disasters, emergency response teams are overwhelmed by thousands of simultaneous citizen reports submitted via text messages, social media posts, and uploaded photographs. Manual triage is too slow and inconsistent under high-stress conditions.

This project builds an **automated rescue priority prediction system** that:

- Ingests a citizen's **written report** (e.g., "Water reached roof level, elderly woman trapped, no food since yesterday") alongside a **submitted photograph**.
- Produces an **Emergency Severity Score (0–100)** using a multimodal ML pipeline.
- Maps the score to a discrete **Priority Class** (e.g., Critical, High, Medium, Low) to help dispatchers allocate rescue teams efficiently.

---

## 🔭 Scope

| In Scope | Out of Scope |
|---|---|
| Multimodal feature extraction (text + image) | Real-time video stream processing |
| XGBoost classifier training & evaluation | Mobile / web frontend |
| Severity scoring (0–100 regression) | Live GPS dispatch routing |
| Priority class labelling (multi-class) | Full production API server |
| Offline batch inference pipeline | Continuous online learning |
| Dataset curation & augmentation utilities | Third-party alert integrations |

---

## 🏗️ Planned ML Pipeline

```
Citizen Text Report          Citizen Uploaded Image
       │                              │
       ▼                              ▼
┌─────────────────────┐   ┌──────────────────────────┐
│  Sentence Transformer│   │  CLIP Image Encoder      │
│  (pretrained, frozen)│   │  (pretrained, frozen)    │
│  e.g. all-MiniLM-L6 │   │  e.g. openai/clip-vit-b32│
└──────────┬──────────┘   └────────────┬─────────────┘
           │                           │
           │   Text Embedding (384-d)  │  Image Embedding (512-d)
           └──────────┬────────────────┘
                      ▼
          ┌───────────────────────┐
          │   Feature Fusion      │
          │  (concat + optional   │
          │   MLP projection)     │
          └──────────┬────────────┘
                     │  Fused Feature Vector (896-d)
                     ▼
          ┌───────────────────────┐
          │   XGBoost Classifier  │
          │   (trained by us)     │
          └──────────┬────────────┘
                     │
           ┌─────────┴──────────┐
           ▼                    ▼
  Emergency Severity      Priority Class
    Score  (0–100)     (Critical / High /
                        Medium / Low)
```

### Pipeline Stages

| Stage | Description | Status |
|---|---|---|
| **Text Encoding** | Extract semantic embeddings from citizen text using a pretrained Sentence Transformer | 🔲 Planned |
| **Image Encoding** | Extract visual embeddings from uploaded photo using pretrained CLIP | 🔲 Planned |
| **Feature Fusion** | Concatenate embeddings; optionally apply a lightweight MLP projection head | 🔲 Planned |
| **XGBoost Training** | Train gradient-boosted classifier on fused features + labelled severity scores | 🔲 Planned |
| **Severity Scoring** | Output a continuous score 0–100 representing emergency urgency | 🔲 Planned |
| **Priority Mapping** | Threshold-based mapping from score → priority class | 🔲 Planned |

---

## 📁 Repository Structure

```
disaster-triaging/
│
├── data/
│   ├── raw/            # Original, unprocessed citizen reports (text + metadata)
│   ├── processed/      # Cleaned and feature-engineered datasets (CSV/Parquet)
│   ├── images/         # Raw and preprocessed citizen-uploaded images
│   └── labels/         # Ground-truth severity scores and priority class labels
│
├── notebooks/          # Exploratory data analysis and experiment notebooks
│
├── src/
│   ├── data/           # Dataset loading, cleaning, augmentation utilities
│   ├── features/       # Text & image embedding extraction, feature fusion
│   ├── models/         # XGBoost model definition, training, evaluation
│   ├── inference/      # Offline & online inference pipeline
│   └── utils/          # Logging, config management, common helpers
│
├── models/             # Serialised trained model artefacts (.pkl, .json, etc.)
│
├── tests/              # Unit and integration tests
│
├── README.md
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

---

## ⚙️ Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-org/disaster-triaging.git
cd disaster-triaging

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📓 Notebooks

| Notebook | Purpose |
|---|---|
| `01_eda.ipynb` | Exploratory data analysis on raw reports and images |
| `02_embedding_exploration.ipynb` | Visualise text & image embeddings (UMAP/t-SNE) |
| `03_feature_fusion.ipynb` | Prototype fusion strategies |
| `04_xgboost_training.ipynb` | Model training, cross-validation, hyperparameter search |
| `05_evaluation.ipynb` | Final evaluation metrics and confusion matrix |

---

## 🗺️ Roadmap

- [x] Project scaffolding & documentation
- [ ] Data collection and labelling schema design
- [ ] Text embedding module (`src/features/text_encoder.py`)
- [ ] Image embedding module (`src/features/image_encoder.py`)
- [ ] Feature fusion module (`src/features/fusion.py`)
- [ ] XGBoost training script (`src/models/train.py`)
- [ ] Evaluation metrics (`src/models/evaluate.py`)
- [ ] Inference pipeline (`src/inference/predict.py`)
- [ ] Unit tests for all modules
- [ ] Experiment tracking integration (MLflow / W&B)

---

## 👥 Team

Built for a hackathon. Contributions welcome.

---

## 📄 License

MIT