"""Live OpenAI smoke test. Skipped unless OPENAI_API_KEY is set (never in CI by default)."""

import os

import pytest
from pydantic import BaseModel

from app.core.config import Settings
from app.llm.factory import build_provider
from app.llm.schemas import LLMRequest

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set; live test skipped"
)


class Tiny(BaseModel):
    answer: int


def test_live_structured_and_embedding() -> None:
    provider = build_provider(Settings(llm_provider="openai"))
    obj, resp = provider.complete_structured(
        LLMRequest.simple(
            "Return JSON only.", "What is 2 + 2? Put the number in 'answer'.", purpose="live"
        ),
        Tiny,
    )
    assert obj.answer == 4
    assert resp.usage.total_tokens > 0 and resp.latency_s > 0
    emb = provider.embed(["automatic emergency braking"], purpose="live")
    assert emb.dimension > 100
