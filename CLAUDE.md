# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ADAS Intelligence Platform (AIP) — evidence-backed ADAS diagnostics.
Current scope: **AEB late-braking diagnostic vertical slice** (no CARLA, ROS 2, GPU,
Docker, Kubernetes). Backend + CLI first; Next.js dashboard only after the CLI slice
passes acceptance tests (M10).

Product detail lives in `docs/specification-digest.md` (condensed from the 37-page spec
PDF — do not load the PDF into context). PostgreSQL plan: `docs/postgres-migration.md`.

## Verified commands

All from `backend/` (venv exists at `backend/.venv`; on Windows call its python directly):

```
.venv\Scripts\python.exe -m pip install -e ".[dev]"   # install/update deps
.venv\Scripts\python.exe -m pytest                    # all tests
.venv\Scripts\python.exe -m pytest tests/test_skeleton.py -k units   # single test
.venv\Scripts\python.exe -m ruff check .              # lint
.venv\Scripts\python.exe -m mypy                      # strict type check
.venv\Scripts\python.exe -m app.synthetic.cli --all --seed 42 --out ../data/demo  # demo AEB CSVs
.venv\Scripts\python.exe -m app.ingestion.cli ../data/demo/aeb_late_braking_seed42/telemetry.csv  # ingest + quality gates
.venv\Scripts\python.exe -m app.metrics.cli ../data/demo/aeb_late_braking_seed42/telemetry.csv    # ingest + gates + AEB metrics
$env:LLM_PROVIDER="fake"; .venv\Scripts\python.exe -m app.rag.cli build ../data/demo_docs --out ../data/index   # offline index
$env:LLM_PROVIDER="fake"; .venv\Scripts\python.exe -m app.rag.cli query ../data/index "brake command latency" --access internal
```

Add commands here only after they have actually been run successfully.

## Architecture rules

- Evidence-first: every generated claim must cite resolvable evidence IDs created via
  `app/core/ids.py` (`metric_*`, `chunk_*`, `window_*`, …). The verifier strips
  anything else.
- Confidence rules: single evidence source → max Medium; missing critical signal →
  Low/Blocked; contradictory evidence → flag for human review.
- Data-quality gates run before any metric or AI analysis; failures raise
  `DataQualityError` or downgrade confidence — never silently ignored.
- LLM access only through the provider protocol in `app/llm/` (OpenAI is the only
  concrete provider; no Ollama; no `openai` imports outside `app/llm/`).
- Agents emit the fixed JSON schema: `observations`, `hypotheses` (cause /
  evidence_ids / confidence), `missing_evidence`, `recommended_next_tests`.
- Persistence: one SQLite database via SQLAlchemy, DB-agnostic (portable types, no
  dialect-specific SQL, connection string from config only). Bulk telemetry stays in
  CSV/Parquet files, never in the DB. No Docker, PostgreSQL, TimescaleDB, Redis,
  vector DB, queues or observability stacks unless an implemented requirement proves
  the need.

## Safety constraints

- AIP is engineering assistance, not safety certification. Every report includes the
  disclaimer and a limitations section.
- Never assert a root cause without timestamped evidence; report `missing_evidence`
  instead.
- Synthetic/simulation-only evidence must never be described as real-world validation.

## Coding conventions

- Python 3.12+ (numpy 2.5 requires it; venv runs 3.12.6), fully typed, `mypy --strict` clean; Pydantic models
  at all boundaries.
- Deterministic core: metrics and quality gates are pure functions; all randomness is
  seeded.
- SI units everywhere internally; conversions happen once at ingestion
  (`app/core/units.py`).

## Testing requirements

- Every module lands with tests. LLM-dependent tests use a FakeProvider; live-API
  tests are skipped when `OPENAI_API_KEY` is unset.
- `pytest`, `ruff check .` and `mypy` must all pass before a milestone is "done".
- The end-to-end demo flow test (once it exists) gates every milestone after it.

## Git and secrets restrictions

- Never commit, push, delete files, run migrations or deploy without explicit user
  approval. Before any commit: show `git status` and the proposed file list, and
  confirm no secrets, datasets, generated files, model weights or `.env` files are
  included.
- Never read or print `.env` values. `.env` is gitignored; `.env.example` holds
  variable names only.
