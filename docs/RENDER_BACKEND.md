# DisasterSignal — Render backend deployment

**Architecture:** Vercel serves the React/Vite frontend at [disastersignal-bhuvan.vercel.app](https://disastersignal-bhuvan.vercel.app/). Render runs the separate FastAPI/ML backend. This is an academic research prototype; it **does not dispatch emergency services**.

## 1. Before creating Render resources

Read the official [Render pricing](https://render.com/pricing) and [persistent disk limitations](https://render.com/docs/disks). The checked-in [render.yaml](../render.yaml) proposes **one paid Standard (2 GB RAM) Docker web service plus a 5 GB persistent disk**. Creating it can incur ongoing charges; it is **not** a free-tier deployment. Do not provision it until you approve your hosting budget. The 2 GB memory tier is a starting estimate, **not a benchmark**: model loading may require upgrading to a larger instance.

You need access to the GitHub repository [HuMaN-OG/Disaster-Triaging](https://github.com/HuMaN-OG/Disaster-Triaging) in Render. Use the *Blueprints* interface to import the repo's root `render.yaml`. The region is set to Singapore; the backend uses the repository-root `backend.Dockerfile` with the root as build context.

Render will generate `JWT_SECRET` (never check secrets into Git). Public admin/demo accounts remain disabled. The 5 GB disk at `/app/data` stores:
- `/app/data/disaster_triage.db` — the SQLite research database
- `/app/data/uploads` — uploaded evidence files
- `/app/data/huggingface` — downloaded pretrained text encoder cache

**Keep one application instance** when using SQLite on a service-attached disk. Scale up memory/CPU rather than horizontally. Plan database backups and do not use this prototype for sensitive real-world reports.

## 2. Model limitations: what the repo actually contains

The text-priority CatBoost checkpoint and metadata are in `models/text/`. The text encoder named by the metadata is retrieved using Sentence Transformers during startup if it is not already cached. This requires internet access, adequate RAM and persistent model-cache space. Verify the actual downloaded encoder and model dimensions before accepting inference results.

The trained vision ensemble `.pth` checkpoints **are not included in the GitHub repository**. Therefore:
- If the real text model loads, a **text-only** report can produce its trained priority score and reported language.
- It **cannot** truthfully classify an image or assign class probabilities. Text-only results explicitly return `disaster_type: "Unknown"`, `confidence: 0.0` and empty `class_probabilities`.
- Image-only or image+text inference returns **HTTP 503** until real, compatible vision checkpoints are deployed. It will not invent “Flood: 85%”.
- If the text encoder/model also fails to load, text inference returns **HTTP 503** instead of a fabricated score.
- The health response exposes `text_model_loaded`, `vision_model_loaded` and `available_inference_modes`. With text-only inference, `status` is `degraded` because the full multimodal system is incomplete.

**Do not claim full image or multimodal inference is live until real trained weights are supplied and end-to-end tested.**

## 3. Connect the deployed API to Vercel

After Render completes building and starts the service:

1. Open `https://<your-real-Render-hostname>.onrender.com/health`. Check it reports database connected and inspect `available_inference_modes`. The URL here is intentionally a placeholder, not an existing deployment.
2. Open **Vercel → disastersignal-bhuvan → Settings → Environment Variables**.
3. Set `VITE_API_URL` in **Production** to the actual HTTPS Render service origin, **without a trailing `/api` or slash**. For example, if your real API is `https://your-service.onrender.com`, use exactly that origin.
4. **Redeploy production.** Vite compiles `VITE_*` variables into client JavaScript at build time. Never put `JWT_SECRET` or database credentials into a `VITE_` variable.
5. The Render Blueprint already sets `CORS_ORIGINS` to `["https://disastersignal-bhuvan.vercel.app"]`. If you add another real frontend domain, update this exact allowlist and redeploy Render.

## 4. Verify the real end-to-end behavior

1. Verify `GET /health` returns expected connection + actual model modes. A successful HTTP response with `status: "degraded"` does **not** indicate full-model readiness.
2. Use only **synthetic demonstration information**: submit a text-only English/Hindi/Hinglish report with explicit test coordinates. Check the API returns a real incident ID, an honest model score, and it appears in the queue after refresh.
3. Inspect `/dashboard/stats`, `/dashboard/map`, and `/dashboard/analytics`; check they reflect the incident saved to disk.
4. Test normal citizen registration/login. Public users must be forbidden from uploading responder evidence or changing incident status.
5. Admin/Responder roles must be provisioned through a secure backend flow; do not publish or enable default demo credentials.
6. Check the original report image upload path only after the trained vision model has been supplied. Verify persistence across service restart, access controls, CORS and upload limits.
7. Run `python -m pytest backend/app/tests -q` in the configured development environment. In CI these tests use **explicit synthetic fixtures** for workflow logic; they are not a live-model accuracy test.

The Vercel site and the Render API should both show the real operational state. A UI screenshot or a passing build alone does not establish production readiness.

## 5. Other production limitations

A single SQLite file + local uploads on one Render disk are appropriate for a small, supervised portfolio demonstration, not for multi-instance scaling. If future traffic or privacy needs grow, migrate to managed PostgreSQL and private object storage, and use a separate managed inference service if your trained vision model requires substantially more compute.

Render persistent disks are mounted only at service runtime; they are not available in Docker builds. This Dockerfile creates its local cache directory but the actual encoder is downloaded/cached at runtime. Service startup can be slow the first time.
