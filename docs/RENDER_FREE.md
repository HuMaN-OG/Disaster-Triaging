# DisasterSignal on Render Free (portfolio demonstration)

**Live deployment:** [DisasterSignal frontend on Vercel](https://disastersignal-bhuvan.vercel.app) + [Render Free API](https://disastersignal-free-api.onrender.com/health). The Vercel production variable `VITE_API_URL` is set to `https://disastersignal-free-api.onrender.com`. Live smoke tests on 19 November 2026 verified API health/CORS and submitted fictional incidents that appeared in the response queue, map and analytics. Public Citizen registration/login and write restrictions also passed. **This remains a text-only experimental demo with temporary data.**

The checked-in **[render.yaml](../render.yaml)** configures one Render Free Python web service with no paid disk or instance.

## Limitations you must accept

- **512 MB RAM / 0.1 CPU.** The original PyTorch ensemble, Sentence Transformers and CatBoost are intentionally *not loaded* on this service.
- **Sleeps after 15 minutes idle.** Waking up can take about a minute. The browser's health check is allowed extra time.
- **No durable database or uploads.** A local SQLite database and account registrations can disappear after sleep, restart, or redeployment. Do not store real personal details or operational data.
- **Free is for demos, not real incident response.** This app does not contact emergency services.
- An optional Render Free Postgres database is not a permanent solution: it expires after 30 days. This Blueprint does not create one.

Official details: https://render.com/docs/free

## What genuinely works

- The API is real FastAPI with SQLite and role-based authentication.
- The **text-priority demo** uses a small supervised nearest-neighbor estimator fitted to the repository's **2,000 labeled citizen reports** in `disaster_dataset.json`. This is experimental, not the published multimodal benchmark, and its performance has *not* been validated for real emergency use.
- The training reports are English. A small transparent Hindi/Hinglish disaster-word dictionary provides limited vocabulary normalization, but this is **not** a validated multilingual ML model.
- Text-only submissions can generate priority scores (0–100), create research incidents, populate charts and maps, and support responder case status in the current running session.
- Text-only classifications remain `Unknown`, `confidence: 0.0`, and empty image-class probabilities because the lite estimator does not classify scenes.
- **Image-only and image+text requests return HTTP 503** because the trained vision model checkpoint files are not in the repository. The frontend labels/disables image upload while the free-tier API is detected.
- Health reports `model_variant: "free-tier-supervised-retrieval"`, `storage_persistence: "ephemeral"`, `text_model_loaded: true`, and `vision_model_loaded: false`.

## Deploy

1. Connect your **Render** account to GitHub, granting access to `HuMaN-OG/Disaster-Triaging`.
2. Open https://dashboard.render.com/select-repo?type=blueprint or Render Dashboard → **New → Blueprint**.
3. Select `HuMaN-OG/Disaster-Triaging` (branch `main`). Render reads root `render.yaml`.
4. Confirm **plan: Free**, **runtime: Python**, service `disastersignal-free-api`, and **no disk/no paid database** before creating the service.
5. Render generates a random `JWT_SECRET` automatically. Do not share or commit it.
6. For this project the live API is **https://disastersignal-free-api.onrender.com/health**. If deploying a new instance, substitute its real address. Check `database: connected`, `text_model_loaded: true`, and `available_inference_modes: ["text"]`. `status: degraded` is **expected** because the original image ensemble is missing.
7. **Already configured:** Vercel project `disastersignal-bhuvan` has production `VITE_API_URL=https://disastersignal-free-api.onrender.com`. Update this only if you change Render hosts.
8. **Already redeployed and verified:** the browser JavaScript contains the Render URL. On future backend URL changes, redeploy Vercel; Vite embeds `VITE_*` at build time.
9. Test with **invented incident details only** (not personal information): submit an English text report and test coordinates; verify result, response queue, analytics and route navigation. The incident will disappear after a Render Free restart. Sign-up/account data also resets.

The frontend may take over a minute to show API status when the free service wakes. Do not continuously ping the API to defeat free-tier idling.

## Local smoke check

To test the same lightweight backend on a local Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements-free.txt
export PYTHONPATH=backend
export LITE_MODEL_ENABLED=true
export JWT_SECRET='generate-a-new-private-secret-locally'
export DATABASE_URL='sqlite:///./free_demo.db'
export UPLOAD_DIR='./free_demo_uploads'
uvicorn app.main:app --port 8000
```

On Windows PowerShell, set environment variables with `$env:NAME="value"` and activate `.venv\Scripts\Activate.ps1`.

Run `python -m pytest backend/app/tests -q` after also installing the extra development dependencies in `backend/requirements-test.txt`. CI validates API workflows with **clearly separate test fixtures**, plus tests the real lite estimator.

## Full multimodal AI and persistence

Those capabilities require a larger paid instance or another approved inference host, the real trained vision checkpoints, and reliable database/object storage. They are **not** silently simulated in the public free demo.
