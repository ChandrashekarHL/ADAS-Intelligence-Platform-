"""M4 LLM layer: contracts, retries, structured parsing, fake + mocked OpenAI providers."""

import ast
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import httpx
import numpy as np
import openai
import pytest
from openai import OpenAI
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.core.errors import ProviderError, StructuredOutputError
from app.llm.factory import build_provider
from app.llm.fake import FakeProvider, FakeTransientError
from app.llm.openai_provider import OpenAIProvider
from app.llm.provider import CallLog, LLMProvider, parse_structured, with_retries
from app.llm.schemas import ChatMessage, LLMRequest, Role, TokenUsage


class Verdict(BaseModel):
    cause: str
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str]


REQ = LLMRequest.simple("You are a test.", "Say hi.", purpose="unit")


# --- contracts -----------------------------------------------------------------------------


def test_request_helpers_and_usage_arithmetic() -> None:
    assert [m.role for m in REQ.messages] == [Role.SYSTEM, Role.USER]
    assert REQ.temperature == 0.0 and REQ.seed == 0 and REQ.purpose == "unit"
    with pytest.raises(ValueError):
        LLMRequest(messages=())
    u = TokenUsage(prompt_tokens=10, completion_tokens=5) + TokenUsage(prompt_tokens=1)
    assert (u.prompt_tokens, u.completion_tokens, u.total_tokens) == (11, 5, 16)


def test_parse_structured_accepts_json_and_fences_and_rejects_garbage() -> None:
    good = '{"cause": "x", "confidence": 0.5, "evidence_ids": ["metric_1"]}'
    assert parse_structured(good, Verdict).cause == "x"
    fenced = f"```json\n{good}\n```"
    assert parse_structured(fenced, Verdict).confidence == 0.5
    with pytest.raises(StructuredOutputError, match="not JSON"):
        parse_structured("Sure! Here is the answer: {", Verdict)
    with pytest.raises(StructuredOutputError, match="does not match Verdict"):
        parse_structured('{"cause": "x", "confidence": 7, "evidence_ids": []}', Verdict)


def test_with_retries_backoff_is_deterministic() -> None:
    sleeps: list[float] = []
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise FakeTransientError("nope")
        return "ok"

    result, attempts = with_retries(
        flaky, attempts=4, retry_on=(FakeTransientError,), sleep=sleeps.append
    )
    assert (result, attempts) == ("ok", 3)
    assert sleeps == [0.5, 1.0]

    with pytest.raises(ProviderError, match="after 2 attempt"):
        with_retries(
            lambda: (_ for _ in ()).throw(FakeTransientError("x")),
            attempts=2,
            retry_on=(FakeTransientError,),
            sleep=sleeps.append,
        )
    # a non-retryable exception propagates untouched
    with pytest.raises(KeyError):
        with_retries(
            lambda: (_ for _ in ()).throw(KeyError("x")),
            attempts=3,
            retry_on=(FakeTransientError,),
            sleep=sleeps.append,
        )


# --- FakeProvider ----------------------------------------------------------------------------


def test_fake_provider_scripted_and_queued() -> None:
    log = CallLog()
    fake = FakeProvider(
        ["first"], scripted=lambda r: f"echo:{r.messages[-1].content}", call_log=log
    )
    assert fake.complete(REQ).text == "first"
    assert fake.complete(REQ).text == "echo:Say hi."
    fake.queue(Verdict(cause="c", confidence=0.9, evidence_ids=["metric_a"]))
    obj, resp = fake.complete_structured(REQ, Verdict)
    assert obj.cause == "c" and resp.provider == "fake" and resp.usage.total_tokens > 0
    assert len(fake.requests) == 3 and fake.requests[0] is REQ
    assert len(log) == 3 and log.failures == 0 and log.usage.total_tokens > 0
    assert log.records[0].purpose == "unit" and log.records[0].kind == "completion"
    with pytest.raises(ProviderError, match="no queued response"):
        FakeProvider().complete(REQ)


def test_fake_provider_structured_rejects_bad_output() -> None:
    fake = FakeProvider(['{"cause": 1}'])
    with pytest.raises(StructuredOutputError):
        fake.complete_structured(REQ, Verdict)


def test_fake_embeddings_are_deterministic_and_normalised() -> None:
    fake = FakeProvider(embedding_dim=16)
    a = fake.embed(["braking latency", "time to collision"], purpose="rag")
    b = FakeProvider(embedding_dim=16).embed(["braking latency"])
    assert a.dimension == 16 and len(a.vectors) == 2
    assert a.vectors[0] == b.vectors[0]  # same text, same vector, across instances
    assert a.vectors[0] != a.vectors[1]
    assert np.linalg.norm(a.vectors[0]) == pytest.approx(1.0)
    assert fake.embedded == [("braking latency", "time to collision")]
    assert fake.call_log.records[0].kind == "embedding"
    assert fake.call_log.records[0].purpose == "rag"


def test_fake_provider_satisfies_protocol() -> None:
    provider: LLMProvider = FakeProvider()
    assert provider.name == "fake" and provider.model == "fake-model"


# --- OpenAIProvider with a mocked SDK client -------------------------------------------------


def _http_error(cls: type[openai.APIStatusError], status: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return cls("boom", response=response, body=None)


def _completion(
    text: str = "hello", *, parsed: object = None, refusal: str | None = None
) -> MagicMock:
    msg = MagicMock()
    msg.content = text
    msg.parsed = parsed
    msg.refusal = refusal
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    comp = MagicMock()
    comp.choices = [choice]
    comp.model = "gpt-test-2026"
    comp.usage = MagicMock(prompt_tokens=12, completion_tokens=3)
    return comp


def _provider(client: MagicMock, **kw: object) -> OpenAIProvider:
    return OpenAIProvider(
        api_key="test-key-not-real",
        model="gpt-test",
        embedding_model="emb-test",
        client=cast(OpenAI, client),
        sleep=lambda _s: None,
        **kw,  # type: ignore[arg-type]
    )


def test_openai_complete_maps_request_and_usage() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _completion("hello there")
    p = _provider(client)
    resp = p.complete(REQ.model_copy(update={"temperature": 0.2, "max_tokens": 50, "seed": 7}))
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-test"
    assert kwargs["messages"] == [
        {"role": "system", "content": "You are a test."},
        {"role": "user", "content": "Say hi."},
    ]
    assert (kwargs["temperature"], kwargs["max_tokens"], kwargs["seed"]) == (0.2, 50, 7)
    assert (
        resp.text == "hello there" and resp.model == "gpt-test-2026" and resp.provider == "openai"
    )
    assert resp.usage == TokenUsage(prompt_tokens=12, completion_tokens=3)
    rec = p.call_log.records[-1]
    assert rec.ok and rec.kind == "completion" and rec.attempts == 1 and rec.purpose == "unit"


def test_openai_retries_transient_then_succeeds() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _http_error(openai.RateLimitError, 429),
        openai.APIConnectionError(request=httpx.Request("POST", "https://x")),
        _completion("third time"),
    ]
    p = _provider(client, max_attempts=3)
    assert p.complete(REQ).text == "third time"
    assert client.chat.completions.create.call_count == 3
    assert p.call_log.records[-1].attempts == 3


def test_openai_gives_up_after_max_attempts() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = _http_error(openai.RateLimitError, 429)
    p = _provider(client, max_attempts=2)
    with pytest.raises(ProviderError, match="after 2 attempt"):
        p.complete(REQ)
    assert client.chat.completions.create.call_count == 2
    rec = p.call_log.records[-1]
    assert not rec.ok and rec.error and "test-key-not-real" not in rec.error


def test_openai_non_retryable_status_fails_fast() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = _http_error(openai.BadRequestError, 400)
    p = _provider(client, max_attempts=3)
    with pytest.raises(ProviderError, match="HTTP 400"):
        p.complete(REQ)
    assert client.chat.completions.create.call_count == 1
    assert p.call_log.failures == 1


def test_openai_structured_parse_refusal_and_unparsed() -> None:
    client = MagicMock()
    verdict = Verdict(cause="late detection", confidence=0.7, evidence_ids=["metric_x"])
    client.chat.completions.parse.return_value = _completion(
        verdict.model_dump_json(), parsed=verdict
    )
    p = _provider(client)
    obj, resp = p.complete_structured(REQ, Verdict)
    assert obj == verdict and resp.text == verdict.model_dump_json()
    assert client.chat.completions.parse.call_args.kwargs["response_format"] is Verdict
    assert p.call_log.records[-1].kind == "structured"

    client.chat.completions.parse.return_value = _completion("", refusal="I cannot do that")
    with pytest.raises(StructuredOutputError, match="refused"):
        p.complete_structured(REQ, Verdict)

    client.chat.completions.parse.return_value = _completion("", parsed=None)
    with pytest.raises(StructuredOutputError, match="no parsed Verdict"):
        p.complete_structured(REQ, Verdict)
    assert p.call_log.failures == 2


def test_openai_embed_preserves_order() -> None:
    client = MagicMock()
    result = MagicMock()
    result.model = "emb-test-v1"
    result.usage = MagicMock(prompt_tokens=5, completion_tokens=0)
    d1, d0 = MagicMock(), MagicMock()
    d0.index, d0.embedding = 0, [1.0, 0.0]
    d1.index, d1.embedding = 1, [0.0, 1.0]
    result.data = [d1, d0]  # deliberately out of order
    client.embeddings.create.return_value = result
    p = _provider(client)
    emb = p.embed(["a", "b"], purpose="rag")
    assert emb.vectors == ((1.0, 0.0), (0.0, 1.0)) and emb.dimension == 2
    assert client.embeddings.create.call_args.kwargs == {"model": "emb-test", "input": ["a", "b"]}
    assert p.call_log.records[-1].model == "emb-test" and p.call_log.records[-1].purpose == "rag"


def test_openai_requires_key_when_no_client() -> None:
    with pytest.raises(ProviderError, match="key is empty"):
        OpenAIProvider(api_key="", model="m", embedding_model="e")


# --- factory ---------------------------------------------------------------------------------


def test_factory() -> None:
    fake = build_provider(Settings(_env_file=None, llm_provider="fake"))
    assert fake.name == "fake"
    with pytest.raises(ProviderError, match="OPENAI_API_KEY is not set"):
        build_provider(Settings(_env_file=None, llm_provider="openai", openai_api_key=None))
    real = build_provider(
        Settings(_env_file=None, llm_provider="openai", openai_api_key="not-a-real-key")
    )
    assert real.name == "openai" and real.model == "gpt-4o-mini"


# --- architecture guard ----------------------------------------------------------------------


def test_openai_is_only_imported_inside_app_llm() -> None:
    app_root = Path(__file__).resolve().parents[1] / "app"
    offenders: list[str] = []
    for py in app_root.rglob("*.py"):
        if py.parent.name == "llm":
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(n == "openai" or n.startswith("openai.") for n in names):
                offenders.append(str(py.relative_to(app_root)))
    assert offenders == []


def test_messages_are_immutable() -> None:
    m = ChatMessage(role=Role.USER, content="x")
    with pytest.raises(ValueError):
        m.content = "y"  # type: ignore[misc]
