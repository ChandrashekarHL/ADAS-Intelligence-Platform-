# PostgreSQL Migration Path (production phase)

The MVP uses SQLite through SQLAlchemy. The persistence layer is kept
database-agnostic so PostgreSQL can replace SQLite without rewriting business logic.

## Rules that keep the swap cheap (enforced now)

1. All DB access goes through SQLAlchemy ORM/Core — no raw SQLite-dialect SQL in
   application code.
2. The connection string is configuration only: `DATABASE_URL` in
   `backend/app/core/config.py`. Code never assumes a file-based database.
3. Portable column types only (String, Integer, Float, Boolean, DateTime, JSON via
   SQLAlchemy's dialect-neutral `JSON` type). No SQLite-specific pragmas in models.
4. Large time-series data lives in CSV/Parquet files on disk, referenced by path/URI
   from the DB — so the DB migration never moves bulk telemetry.
5. IDs are application-generated strings (see `app/core/ids.py`), not DB
   autoincrement — no sequence/identity behavior to migrate.
6. Embeddings are stored as opaque blobs/JSON in the MVP; similarity is computed in
   NumPy. They are the one component expected to move OUT of the relational store.

## Migration steps (when an implemented requirement justifies it)

1. Stand up PostgreSQL 16 (Docker Compose or managed service).
2. Add `psycopg[binary]` to backend dependencies.
3. Introduce Alembic; autogenerate the initial migration from the ORM models and
   verify it against a scratch PostgreSQL database.
4. Set `DATABASE_URL=postgresql+psycopg://...` in the environment.
5. Move embeddings from the SQLite table to `pgvector` (or Qdrant if corpus size and
   query volume demand a dedicated store); replace the NumPy cosine search with a
   vector-index query behind the same retrieval interface.
6. Adopt TimescaleDB hypertables only if telemetry windows move into the DB (i.e.
   file-based Parquet stops being sufficient) — this is not assumed.
7. Run the full test suite against PostgreSQL in CI (testcontainers) before cutover.

## Triggers that justify starting the migration

- Concurrent writers (multiple workers/users) hitting SQLite lock contention.
- Corpus growth making NumPy cosine search a measured bottleneck.
- Deployment to any shared/hosted environment.
