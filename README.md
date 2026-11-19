# Disaster Triaging System

**Live frontend preview:** [DisasterSignal on Vercel](https://disastersignal-bhuvan.vercel.app/) — deployed in the `human-ogs-projects` Vercel team from the `main` source snapshot. The FastAPI/ML API is **not yet connected** to this deployment; incident submission, database-backed command view and server-side analytics require a separately configured persistent backend. **Do not treat this preview as operational emergency dispatch.**

A multimodal disaster triaging system that combines computer vision and natural language processing to classify disaster images and estimate rescue priority from citizen reports.

The repository includes machine-learning pipelines, a FastAPI backend prototype (`backend/`) and a React/Vite frontend (`frontend/`). Their source code is present, but full end-to-end deployment and production readiness have not been independently verified.

**Vercel setup and live-backend verification:** [Deployment guide](docs/VERCEL_DEPLOYMENT.md). The production domain must use the correct connected Git repository, Vite root directory, and current `main` commit.

## DisasterSignal — redesigned interactive frontend (2026)

**DisasterSignal** is a responsive research prototype that brings the citizen report workflow, responder review and incident analytics together in one interface. Its navy/coral visual system replaces the older bare-bones UI.

### Screens and working integration paths

- **Citizen reporting:** multilingual free-text samples (English, Hindi, Hinglish), optional JPG/PNG/WEBP uploads (10 MB limit), **explicit incident coordinates** using GPS, map or manual input, API submission and stored model-derived triage result including confidence and priority
- **Response command:** live-from-database statistics, searchable/filterable incident queue, map pins, CSV summary export, 30-second refresh and per-incident inspection
- **Case details:** model scores, evidence attachments, location, status audit, structured response brief with downloadable text and **role-authorized** responder transitions
- **Intelligence hub:** backend-provided daily volume, predicted classifications, priority, status, language distributions and average recorded confidence
- **Authentication:** normal sign-in and public **Citizen-only** registration. Admin and Responder accounts must be provisioned securely and may not be created through public registration

> **Important:** This is an academic prototype; it **does not automatically dispatch emergency services**. Model predictions and suggested actions are unverified estimates and must not be used as emergency instructions. Call actual emergency services when help is needed.

### Local Docker demo (frontend + FastAPI)

1. Clone the repo, make a **private** local `.env` from `.env.example` (do not commit it).
2. Generate a new signing secret using `python -c "import secrets; print(secrets.token_urlsafe(48))"` and set `JWT_SECRET` in `.env`. On a real deployment use secure secrets management.
3. Run `docker compose up --build` from the repository root. Initial installation downloads model dependencies and may require substantial memory/storage/time.
4. Open **http://localhost** for the frontend. FastAPI documentation is at **http://localhost:8000/docs**. The browser calls the same-origin `/api` gateway through Nginx.
5. For a **private local demonstration only**, optionally set `DEMO_ACCOUNTS_ENABLED=true` and supply unique `DEMO_ADMIN_PASSWORD` and `DEMO_RESPONDER_PASSWORD`. Never enable public demo accounts or publish these secrets. If the database contains old accounts, review and rotate them manually.

For frontend development, run FastAPI on port 8000 and `cd frontend && npm ci && npm run dev`. Vite development proxies `/api` to `http://127.0.0.1:8000`. If hosting the frontend separately, set `VITE_API_URL` to your real, appropriately CORS-enabled backend origin.

### Verification

- `.github/workflows/ci.yml` validates TypeScript types, builds the React/Vite frontend and checks backend Python syntax.
- `.github/workflows/ui-smoke.yml` uses Chromium and **synthetic, mocked API data** to test navigation, form validation/report submission, dashboard queue, map, response briefs, sign-in, responder actions, charts and mobile layout. Screenshots are saved as GitHub Actions artifacts. This does **not** prove live ML inference or production integration.
- Run backend integration tests in a properly configured environment: `python -m pytest backend/app/tests -v`.
- Before deploying, validate the actual ML model files, external map tiles, database persistence, API connectivity, attachments, and authorized responder account settings.

### Portfolio contribution

The project is collaborative. My contributions include multilingual ML scoring, system integration, front-end response workflow refinements, secure UI flows, responsive design and testing. See git history and commit attribution for individual ownership.

---

## Demonstration and team contributions

This is an AlgOlympia team project. My resume describes work on multilingual citizen-report priority scoring and combining text predictions with disaster-image classification; it does not imply sole authorship of the entire repository.

**Inspect or reproduce:** Follow the installation instructions below. Run `python -m pytest -q` to execute available tests. The supplied inference examples require compatible model checkpoints and dependencies. Model training can take time and may require datasets or weights that are not immediately available.

**Current status:** The model pipelines and evaluation are documented below; the UI, API, incident database and map are implemented as a prototype; real end-to-end deployment still requires verification. This repository is a research prototype, not a deployed public emergency service.

**Useful real screenshots to capture after running the code:** (1) anonymized incident report with language and 0–100 priority output, (2) image inference with predicted disaster class, and (3) a genuine evaluation plot. Keep real citizen information out of the repository. No verified UI screenshots are currently included here.

**Reproducibility note:** Text-model code includes both XGBoost and CatBoost paths. Confirm which training script, checkpoint and evaluation split generated each reported metric before attributing results to one model.

## Security and deployment checklist

The repository is a research prototype. Do not publish a deployment with test accounts or sensitive incident data.

- **Public registration is Citizen-only.** Creating Admin and Responder users is restricted to authenticated admin operations.
- **JWT signing key:** Set `JWT_SECRET` to a fresh, unpredictable value (at least 32 characters) in the deployment environment. Without it, non-Docker development uses a random in-memory key, so tokens are invalidated on process restart. Docker Compose requires `JWT_SECRET` explicitly.
- **Demo accounts:** Seeding is disabled by default. For private local tests only, set `DEMO_ACCOUNTS_ENABLED=true` and supply unique `DEMO_ADMIN_PASSWORD` and `DEMO_RESPONDER_PASSWORD` values. Never enable test accounts on a public server.
- **Existing databases:** Previously seeded admin/responder accounts are **not automatically removed** by this change. Inspect your current database, remove unused demo accounts, rotate any known passwords, and invalidate old sessions before making a deployment public.
- **CORS:** Replace allowed origins with your actual frontend URL in `CORS_ORIGINS`. Do not use wildcard origins for authenticated production applications.
- **Testing:** Run the tests under `backend/app/tests` with a private test database and model dependencies. A passing frontend build or Python syntax check is not a complete security review.

### Hand-off to deployment owner

Create a local `.env` from `.env.example`, generate a fresh key with `python -c "import secrets; print(secrets.token_urlsafe(48))"`, set `JWT_SECRET`, then run `docker compose config` and verify values are present **without sharing secrets or the rendered config**. Follow with private development smoke tests for register/login, role permissions, and incident lifecycle.

---

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