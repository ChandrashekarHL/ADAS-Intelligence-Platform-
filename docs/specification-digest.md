# AIP Specification Digest

Condensed from `ADAS_Intelligence_Platform_Full_Project_Document.pdf` (37 pages, v1.0,
July 2026). Section references (§) point into the PDF. Extract PDF text with Python
`pymupdf`/`fitz` — `pdftoppm` is not installed, so page rendering is unavailable.
The 13 architecture figures are raster images with no text layer and are not recoverable
from the PDF text; their captions survive, their contents do not.

**Everything in this document is product specification (planned), not implemented fact,**
unless the repository code says otherwise.

## 1. Project identity

ADAS Intelligence Platform (AIP): an AI-powered engineering workbench for ADAS
validation, simulation, diagnostics, telemetry analysis and safety evidence. Explicitly
NOT a self-driving stack and NOT a perception model. Portfolio project targeting AI
Systems / ADAS AI / LLM-Agent engineering roles (OEMs, Tier-1s, motorsport, robotics).

Core flow: engineer uploads ADAS data (CSV telemetry, CAN/DBC, video, ROS bag metadata,
CARLA output, requirement documents) → platform time-aligns and indexes → AI agents
investigate, retrieve knowledge, run simulations/analytics → evidence-backed root-cause
reports, safety metrics, recommendations. Every output traceable to logs, documents,
metrics or simulation.

Mission (§2.2): "Turn raw ADAS data and engineering knowledge into traceable, measurable
and explainable validation intelligence."

## 2. Architectural principles (§6.3)

1. **Evidence-first AI** — every claim points to logs, metrics, documents or simulation output.
2. **Human-supervised autonomy** — AI investigates/recommends; risky claims or actions need human approval.
3. **Simulation before deployment** — hypotheses validated in sim/offline first.
4. **Traceability by design** — requirements ↔ scenarios ↔ logs ↔ metrics ↔ reports linked by IDs.
5. **Modular AI** — LLMs, perception models, detectors, generators are swappable.
6. **Production observability** — latency, cost, tool calls, failures, safety gates logged from day one.

## 3. Product modules (§3.4)

Data Ingestion → Automotive Knowledge (RAG) → Agent Orchestration → Simulation
(CARLA/ROS 2) → Analytics → Verification → Dashboard → DevOps.

## 4. Functional requirements (§7, abridged)

Must-have: auth/workspaces (FR-01), file upload/management (FR-02), telemetry parsing
with timestamp alignment (FR-03), RAG over engineering docs (FR-05), agent root-cause
workflow (FR-06), ADAS metrics (FR-09), evidence-backed reports with citations (FR-11),
LLM/agent evaluation dashboard (FR-13), admin dashboard (FR-15).
Should-have: DBC decoding (FR-04), scenario generation (FR-07), CARLA execution (FR-08),
perception evaluation (FR-10), human approval queue (FR-12), report export (FR-14).

Non-functional highlights (§8): no data loss + retryable jobs, traceable reports,
RBAC/audit/prompt-injection protection, independent scaling, <10 s interactive queries
(long jobs async), cost control via routing/caching, reproducibility of every run.

## 5. Agent architecture (§11)

Seven agents, each with responsibility, tools, permissions and evaluation criteria:

| Agent | Responsibility | Tools |
|---|---|---|
| Planner | Break query into steps, choose tools | Task graph, policy engine, memory |
| Telemetry | Time-series, CAN signals, ADAS state transitions | Pandas, TimescaleDB, plotting, decoders |
| Perception | Detection output, ground truth, frames, confidence | OpenCV, FiftyOne, PyTorch metrics |
| Simulation | Generate/replay CARLA scenarios | CARLA API, ROS 2 bridge, scenario runner |
| RAG | Retrieve requirements, manuals, DBC, issue history | Vector DB, keyword search, parser |
| Safety Critic | Verify claims supported; gate approvals | Evidence verifier, policies, OWASP guardrails |
| Report | Final report: metrics, charts, references | PDF/HTML generator, templates |

Execution state machine (§11.3): classify intent (diagnostics / simulation / metrics /
knowledge / report) → plan evidence sources → execute tools under policy → store
intermediate artifacts → verifier rejects unsupported claims → human approval if risky →
final report with confidence + evidence links.

Agent output JSON schema (§11.4) — fixed contract:

```json
{
  "observations": [],
  "hypotheses": [{ "cause": "", "evidence_ids": [], "confidence": 0.0 }],
  "missing_evidence": [],
  "recommended_next_tests": []
}
```

Prompt rules (§11.4): use only provided telemetry windows / decoded signals; never claim
a cause without timestamped evidence; return uncertainty when evidence is missing.

Model routing (§11.5): strong LLM for reasoning/reports; medium LLM + retrieval for RAG;
classical ML for signal anomaly; code-specialized LLM for log parsing; PyTorch/OpenCV
(not LLM) computes perception metrics — the LLM explains results.
Current project decision: OpenAI API only, behind a provider interface. No Ollama.

## 6. Data architecture (§9)

Supported inputs: CSV/Parquet telemetry, DBC files, MP4/AVI/JPG/PNG, ROS bag metadata,
CARLA logs/JSON, PDF/MD/DOCX/TXT documents, COCO/KITTI/nuScenes-style annotations.

Time sync (§9.3): normalize to canonical UTC/monotonic time; keep original timestamp and
clock domain; align by nearest-neighbor/interpolation/window; build event windows
(T−5 s → T+5 s around AEB trigger, lane departure, collision); never hide missing,
delayed or interpolated data.

Engineered features (§9.4):

| Feature | Formula / meaning | Use |
|---|---|---|
| TTC | `relative_distance / relative_speed` when closing | AEB / collision risk |
| Lane center offset | lateral position vs lane center | Lane keeping |
| Braking latency | risk detection → brake command | AEB responsiveness |
| Jerk | d(acceleration)/dt | Comfort / smoothness |
| Object confidence drop | confidence decrease across frames | Perception stability |
| Sensor disagreement | camera/radar/lidar estimate delta | Fusion anomaly |
| Scenario severity | weighted: speed, TTC, collision, occlusion, weather | Failure prioritization |

## 7. RAG pipeline (§12)

Parse → semantic chunking (requirement / signal definition / procedure / test case) →
embed → index (vector + metadata) → hybrid retrieval (vector + keyword + metadata
filters) → rerank (task, source trust, recency) → answer with strict citations →
verify every claim against retrieved chunks.

Chunk metadata schema (§12.3): `document_id, chunk_id, source_type (requirement | dbc |
manual | test_spec | issue | release_note), vehicle_platform, feature (AEB | ACC | LKA |
TSR | BSD | DMS), version, valid_from, access_level (public | internal | restricted),
text, embedding_id, related_signal_names, related_scenario_ids`.
Access-level filtering is mandatory: restricted docs never reach unauthorized users.

## 8. Metrics library (§13.3) and failure taxonomy (§13.4)

Metrics: collision rate, minimum TTC, braking latency, false positive/negative rates,
lane invasion count, mean lane center offset, detection mAP, tracking ID switches,
comfort jerk score, scenario coverage, **evidence support rate** (% of LLM claims linked
to evidence; target: unsupported claims < 10%).

Failure classes: sensor limitation, perception error, fusion error, planner/logic error,
control issue, calibration/config issue, data/logging issue — each with typical evidence.

## 9. Safety, confidence and evidence (§14, §28)

Report confidence levels: **High** (multiple independent sources, no contradiction) /
**Medium** (likely cause, some data missing) / **Low** (incomplete — recommend next tests
instead of asserting) / **Blocked** (insufficient data quality).

Confidence penalty rules (§28.1): critical signal missing → Low/Blocked; single evidence
source → max Medium; contradictory evidence → human review; simulation-only evidence →
no real-world claims; outdated document → warning + reduced confidence; unsupported LLM
claim → removed or marked unsupported.

Data-quality gates run BEFORE any AI analysis (§28): timestamp continuity, required
signals, unit consistency, file integrity, scenario completeness, ground-truth quality,
document freshness, evidence sufficiency. Failures block or downgrade — never ignored.

Mandatory disclaimers (§14.4): AIP is engineering assistance, not certified safety
tooling; safety-critical conclusions need qualified engineer review; simulation results
do not transfer to road safety without validation. Context standards: ISO 26262
(functional safety), SOTIF/ISO 21448 (intended-functionality limits) — AIP aligns with,
never certifies against.

## 10. Security and governance (§15)

OWASP LLM Top 10 driven: prompt injection → input guards, tool allowlists, instruction
hierarchy, output verification. Leakage → RBAC, access filters, redaction, audit logs.
Excessive agency → human approval, sandboxing, scoped tokens. Hallucination → claim
verification + citations. Poisoning → source trust scoring + versioning. Simulation
abuse → rate limits, quotas, budgets. Everything versioned; workflows reproducible via
stored run IDs; approvals record reviewer, timestamp, decision, reason.

## 11. Database design (§17)

Core tables: `users, projects, log_files, signals, telemetry_windows, events, scenarios,
test_runs, metrics, documents, doc_chunks, agent_runs, reports, approval_tasks`.
Traceability via IDs across all of them.
Project decision: SQLite + SQLAlchemy for MVP; large time-series stays in CSV/Parquet
files, not the DB. PostgreSQL path: docs/postgres-migration.md.

## 12. API specification (§18)

| Method | Endpoint | Purpose |
|---|---|---|
| POST | /api/projects | Create project workspace |
| POST | /api/projects/{id}/files | Upload logs, videos, documents, DBC |
| POST | /api/ingestion/jobs | Start ingestion/indexing job |
| GET | /api/events | List incidents/validation events |
| POST | /api/query | Ask engineering question via RAG/agents |
| POST | /api/scenarios | Create/import scenario definition |
| POST | /api/scenarios/{id}/run | Execute scenario in simulator |
| GET | /api/runs/{id}/metrics | Fetch scenario/test metrics |
| POST | /api/reports | Generate report |
| GET | /api/reports/{id} | Download report |
| GET | /api/dashboard/summary | Summary KPIs |
| POST | /api/approvals/{id}/decision | Approve/reject risky AI action/claim |

Query response contract (§18.2): `{ answer, confidence, evidence_ids,
unsupported_claims, recommended_next_tests }`. API is async-first: ingestion, simulation
and report generation run as jobs.

## 13. Simulation and scenarios (§10) — POST-MVP

First scenario types: lead-vehicle sudden braking (AEB/ACC), cut-in, lane curve low
visibility (LKA), pedestrian crossing with occlusion, traffic sign poor lighting, ghost
radar object. Scenario JSON v1 shape: `{scenario_id, map, weather{rain,fog,sun_altitude},
ego{speed_kmh,lane}, actors[{type,behavior,initial_distance_m,relative_speed_kmh}],
success_criteria{collision,min_ttc_s,max_deceleration_mps2}}`. Every run stores config,
seed, map, weather, model version. v2: OpenSCENARIO/OpenDRIVE import/export.
CARLA integration plan (§10.4). **Fallback mode without GPU/CARLA is a hard requirement:**
pre-recorded simulation logs + synthetic CSV keep the platform demo-ready.

## 14. ADAS feature modules (§26)

- **AEB** (MVP focus): inputs ego_speed, relative_distance, relative_velocity,
  object_class/confidence, brake_command, acceleration, TTC, weather. Metrics: min TTC,
  braking latency, collision, false pos/neg braking, max deceleration, jerk. Failure
  modes: late/no trigger, unnecessary braking, wrong target, confidence drop, planner
  blocked. Report: AEB timeline, risk-threshold crossing, first detection time, brake
  command time, root cause + confidence.
- **ACC**: time headway, speed tracking error, target-switch correctness, comfort jerk,
  min distance, cut-in response time.
- **LKA**: mean/max lateral offset, lane invasion count, confidence drop, steering
  oscillation, curve performance.
- **TSR**: classification accuracy, missed-sign rate, wrong-class rate, confidence
  calibration, detection distance.
- **DMS (optional)**: drowsiness/distraction linked to ADAS events; demo/synthetic data
  only unless consent + privacy controls exist.

## 15. Dashboard / UI (§27) — POST-BACKEND (M10)

Screens: Project Overview, Incident Explorer (timeline + video strip + signal plots +
AI diagnosis + evidence panel + requirements drawer), Scenario Bank, Simulation Runs,
Knowledge Base, Agent Trace Viewer, Safety Evidence Report, Admin/Security.

Report template (§27.3): executive summary → event metadata (run ID, scenario ID, model
version, map, weather, timestamp) → evidence timeline → metrics table (threshold vs
observed vs pass/fail) → ranked root-cause hypotheses with evidence + confidence →
recommended next tests → limitations → approval section.

## 16. Metrics targets and release gates (§19)

Portfolio targets: 5–6 reproducible scenario types; unsupported claim rate < 10%;
every report claim linked to evidence; p50/p95 latency dashboard; model routing +
caching visible; every run replayable from stored metadata.
Release gates: data quality, scenario validity, ADAS metric thresholds, LLM evidence
gate (no high-risk unsupported claims), human approval, regression (no worse on locked
scenario suite).

## 17. Testing strategy (§20)

Unit (parsers, metrics, schemas — pytest/mypy/ruff), integration (upload → ingest →
query → report — testcontainers later), simulation (deterministic seeds), RAG quality
(Ragas/DeepEval/custom), agent (mocked tools, trace checks), security (prompt injection,
unauthorized access), performance (Locust/k6), regression (locked suites in CI).
Demo data plan: CARLA-generated synthetic scenarios, public datasets (nuScenes) where
licensed, synthetic DBC-like definitions, synthetic requirements/manuals, all under
data/demo with README + licenses.

## 18. Roadmap (§22, §30) and scope guardrails

Phases: 0 research/setup → 1 backend foundation → 2 ingestion → 3 RAG → 4 metrics →
5 agent diagnostics → 6 CARLA → 7 scenario generation → 8 security/approvals → 9 polish.
Weekly plan in §30 (repo/docs → FastAPI+uploads → RAG → telemetry → AEB metrics →
planner+telemetry agent → verifier+reports → dashboard → CARLA → scoring → generation →
security → benchmarks → polish).

**Spec MVP (§22.1)**: upload synthetic ADAS log + requirement doc → ask why an event
failed → report with metrics, citations, confidence → run/review one CARLA scenario.
**Current project MVP (approved)**: AEB late-braking diagnostic vertical slice —
synthetic/prerecorded telemetry → data-quality gates → AEB metrics → requirement RAG →
diagnostic agent → evidence verifier → traceable report. No CARLA, ROS 2, Kubernetes,
GPU or Docker. Backend + CLI first; Next.js dashboard afterwards (M10).

**Cut strategy (§26.1/§31)**: under time pressure, go deep on AEB late-braking
diagnostics only — one deep workflow beats many shallow modules.
Out of scope v1: certified autonomy stack, real-road deployment, training foundation
models, replacing safety engineers.

Risk register (§31): CARLA heavy → fallback mode; no real vehicle data → synthetic +
public datasets; scope creep → AEB-first; hallucination → verification + confidence +
approval; dataset licensing → scripts/links not redistribution; sim ≠ reality → state
limitations; complex stack → modular milestones.

## 19. Planned full-scale tech stack (§16) — NOT installed; adopt only when justified

Frontend: Next.js, TypeScript, Tailwind, ShadCN UI, Recharts. Backend: Python 3.11+,
FastAPI, Pydantic, SQLAlchemy. Jobs: Celery/RQ/Dramatiq + Redis. DB: PostgreSQL +
TimescaleDB. Vector: pgvector or Qdrant. Object storage: MinIO/S3. LLM serving:
OpenAI-compatible abstraction (vLLM optional later; Ollama excluded by project
decision). Agents: LangGraph or custom state machine. Simulation: CARLA + ROS 2 bridge.
Analytics: PyTorch, OpenCV, NumPy, Pandas, SciPy, scikit-learn, FiftyOne. DevOps:
Docker Compose, GitHub Actions, K8s optional. Monitoring: OpenTelemetry, Prometheus,
Grafana, Loki/Jaeger.

Planned repo layout (§23): `backend/app/{api, auth, ingestion, telemetry, rag, agents,
simulation, evaluation, security, reports, observability}` + `backend/tests/`,
`frontend/{app, components, dashboards}`, `simulation/{carla_runner, scenarios, maps,
scripts}`, `data/{demo_logs, demo_docs, demo_scenarios, licenses}`, `docs/`, `infra/`,
`notebooks/`, `benchmarks/`, `scripts/`.

Spec's local dev commands (§24) — aspirational until implemented and verified:
docker compose for infra services; uvicorn backend; npm run dev frontend;
`scripts/upload_demo_dataset.py`; `scripts/query.py`; CARLA runner + scorer;
fallback mode without GPU.

## 20. Extensions (§29) and positioning (§25)

Motorsport/F1 extension: lap telemetry analysis, setup comparison, race-strategy agents,
driver coaching, reliability monitoring, sim-vs-real correlation.
Positioning: "I built the engineering intelligence layer ADAS teams need to validate,
debug and improve ADAS behavior" — targeting AI systems, ADAS software, LLM/agent,
validation, simulation, MLOps and telemetry engineering roles.

## 21. References (§32)

CARLA (carla.org), ASAM OpenSCENARIO DSL, ASAM OpenDRIVE, ROS 2 docs, AUTOSAR Adaptive,
nuScenes/nuPlan, OWASP LLM Top 10, OWASP prompt injection, ISO 26262 overview,
SOTIF/ISO 21448 overview.
