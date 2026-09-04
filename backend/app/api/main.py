"""ASGI entry point: ``uvicorn app.api.main:app --reload`` from ``backend/``."""

from app.api.app import create_app

app = create_app()
