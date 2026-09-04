# AIP Dashboard (Next.js)

Three screens over the M9 API. The browser computes nothing: every number, verdict and
evidence ID on screen came from the backend.

| Route | Screen | What it shows |
|---|---|---|
| `/` | Project overview | dashboard tiles, projects, telemetry files with quality verdicts, upload, run gates + metrics, reports and approval status |
| `/incidents/[fileId]` | Incident explorer | signal plots with event markers, event timeline, metrics table, **verified** AI diagnosis with clickable evidence IDs that resolve to metrics, events or requirement chunks; create report |
| `/reports/[reportId]` | Report and approval | the rendered §27.3 report, approval form (reviewer, decision, reason) that re-renders the report's approval section |

## Run

```bash
# backend (from backend/)
$env:LLM_PROVIDER="fake"; .venv\Scripts\python.exe -m uvicorn app.api.main:app --port 8000

# frontend (from frontend/)
npm install
npm run dev          # http://localhost:3000
```

`NEXT_PUBLIC_API_URL` (see `.env.example`) points at the API; default `http://127.0.0.1:8000`.
The backend allows the dev-server origins via `CORS_ORIGINS`.

With `LLM_PROVIDER=fake` the backend has no model to answer, so "Ask the agent" returns 503;
everything else (upload, gates, metrics, metrics-only reports, approvals) works offline.

## Checks

```bash
npm run lint
npx tsc --noEmit
npm run build
```

Stack: Next.js 16 (App Router), TypeScript, Tailwind v4, Recharts, react-markdown.
