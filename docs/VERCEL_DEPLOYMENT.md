# Deploy DisasterSignal on Vercel (frontend) + a separate FastAPI host

**Current code:** https://github.com/HuMaN-OG/Disaster-Triaging (branch `main`). **Live frontend:** https://disastersignal-bhuvan.vercel.app · **Live Render API:** https://disastersignal-free-api.onrender.com/health. The frontend is connected using the production Vite environment variable. Original multimodal vision checkpoints and persistent storage are not available on the free backend.

The latest application is **DisasterSignal v3** (navy-and-coral response-studio UI), NOT the old black/orange “DISASTER TRIAGE v2.0” website. GitHub Actions verifies the Vite build, TypeScript checks and browser UI flows with explicit synthetic test fixtures.

## Vercel settings for this monorepo

Open the Vercel project that **actually owns your production domain**. If you cannot see the project, switch Vercel accounts or team scopes. Do **not** create a duplicate and assume the old domain has moved.

| Setting | Correct value |
| --- | --- |
| Connected Git repository | `HuMaN-OG/Disaster-Triaging` |
| Production branch | `main` |
| Framework | `Vite` |
| Root Directory | `frontend` |
| Install command | `npm ci` |
| Build command | `npm run build` |
| Output directory | `dist` |
| Production API | `VITE_API_URL=https://disastersignal-free-api.onrender.com` (already set and rebuilt; Render Free text-only research backend) |

The repository also provides `frontend/vercel.json` with Vite build/output defaults. The Vercel project's **Root Directory** must still be `frontend` in Settings.

**Do not add secrets** or JWT signing keys to `VITE_*`: Vite client variables are embedded into the public JavaScript bundle.

## Verify it deployed the actual new source

1. Compare Vercel's deployment Git SHA to GitHub `main`'s latest commit.
2. Deploy the current `main` commit; turn off Build Cache for a suspiciously stale build.
3. Open the production URL in a private window, or hard refresh.
4. The browser title should be **“DisasterSignal — Incident Reporting & Triage Research”**. View-source also contains `disastersignal-ui-version=3.0`.
5. The interface should have a dark left sidebar, a research-prototype caution banner and a three-step citizen-reporting form. If it still says “DISASTER TRIAGE v2.0”, Vercel is serving a different/old project.
6. Open DevTools → Network. The browser should request `/health` from the actual `VITE_API_URL`. An HTML response or 404 to `/api/health` means the backend proxy/base URL isn't configured.

## Backend is a separate deployment

The original FastAPI backend's full ML ensemble is heavy. The currently deployed **Render Free** variant intentionally uses a small supervised text estimator, not PyTorch/Sentence Transformers/CatBoost. The Vercel frontend serves static React assets only.

Deploy the backend to a suitable Python host with the project's dependencies/model assets. Configure **strong private** `JWT_SECRET`, secure storage, database persistence, rate limits, upload limits, and restricted production `CORS_ORIGINS`. `DEMO_ACCOUNTS_ENABLED` should remain **false** on a public installation.

Use its real HTTPS endpoint for `VITE_API_URL` (exclude any trailing `/api` unless the backend is explicitly hosted under that path). For a locally Docker-hosted frontend, the built-in Nginx reverse proxy connects `/api/` to `backend:8000`.

## Acceptance checks

- Browser loads the correct new layout after hard refresh.
- `GET /health` reports database and model status; a `degraded` status is **not** a healthy inference service.
- Submit a **synthetic, non-sensitive** English/Hindi/Hinglish test report with explicitly entered map coordinates. Check persisted incident ID, prediction and server logs. Prototype reports do **not** notify real emergency services.
- Sign in as a securely provisioned Responder; verify only authorized users can upload evidence and update incident status.
- Verify incident map, filters, analytics, response brief, attachment upload/download and a mobile viewport.
- Run `npm ci && npm run build` under `frontend/` and check [GitHub Actions](https://github.com/HuMaN-OG/Disaster-Triaging/actions). These checks validate the code and synthetic browser flows, **not** a live deployment.

**Known deployment blocker as of 19 November 2026:** The previously connected Vercel team named `human-ogs-projects` listed **zero projects and zero deployments** and did not recognize the existing `disaster-triaging.vercel.app` hostname. The original deployment must be shared with or connected to the correct Vercel account/team before its settings or production domain can be changed.
