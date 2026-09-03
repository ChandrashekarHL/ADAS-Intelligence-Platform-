"""Deterministic in-memory provider for tests and offline demos.

* Completions come from a queue of scripted responses (strings or Pydantic objects) or
  from a callable that inspects the request.
* Embeddings are a deterministic hashed bag-of-words (each token hashed into one of
  ``embedding_dim`` buckets with a hashed sign), unit-normalised. Texts sharing words are
  close, unrelated texts are not, and no network is involved, so offline retrieval demos
  rank sensibly and tests are reproducible.
* ``fail_times`` makes the first N calls raise a transient error to exercise retries.
"""

import hashlib
import re
import time
from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np
from pydantic import BaseModel

from app.core.errors import ProviderError
from app.llm.provider import CallLog, parse_structured
from app.llm.schemas import (
    CallRecord,
    EmbeddingResponse,
    LLMRequest,
    LLMResponse,
    TokenUsage,
)

T = TypeVar("T", bound=BaseModel)
_TOKEN = re.compile(r"[a-z0-9][a-z0-9_]*")


class FakeTransientError(Exception):
    """Stands in for a rate-limit / connection error."""


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class FakeProvider:
    def __init__(
        self,
        responses: Sequence[str | BaseModel] = (),
        *,
        scripted: Callable[[LLMRequest], str] | None = None,
        embedding_dim: int = 32,
        model: str = "fake-model",
        call_log: CallLog | None = None,
        fail_times: int = 0,
    ) -> None:
        self._queue: list[str] = [
            r if isinstance(r, str) else r.model_dump_json() for r in responses
        ]
        self._scripted = scripted
        self._dim = embedding_dim
        self._model = model
        self.call_log = call_log if call_log is not None else CallLog()
        self._fail_remaining = fail_times
        self.requests: list[LLMRequest] = []
        self.embedded: list[tuple[str, ...]] = []

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return self._model

    def queue(self, *responses: str | BaseModel) -> None:
        self._queue.extend(r if isinstance(r, str) else r.model_dump_json() for r in responses)

    def _maybe_fail(self) -> None:
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            raise FakeTransientError("simulated transient provider failure")

    def _next_text(self, request: LLMRequest) -> str:
        if self._queue:
            return self._queue.pop(0)
        if self._scripted is not None:
            return self._scripted(request)
        raise ProviderError("FakeProvider has no queued response and no script")

    def complete(self, request: LLMRequest) -> LLMResponse:
        start = time.perf_counter()
        self.requests.append(request)
        self._maybe_fail()
        text = self._next_text(request)
        usage = TokenUsage(
            prompt_tokens=sum(_approx_tokens(m.content) for m in request.messages),
            completion_tokens=_approx_tokens(text),
        )
        latency = time.perf_counter() - start
        self.call_log.add(
            CallRecord(
                provider=self.name,
                model=self._model,
                kind="completion",
                purpose=request.purpose,
                usage=usage,
                latency_s=latency,
                ok=True,
            )
        )
        return LLMResponse(
            text=text,
            model=self._model,
            provider=self.name,
            usage=usage,
            latency_s=latency,
            finish_reason="stop",
        )

    def complete_structured(self, request: LLMRequest, schema: type[T]) -> tuple[T, LLMResponse]:
        response = self.complete(request)
        return parse_structured(response.text, schema), response

    def embed(self, texts: Sequence[str], *, purpose: str = "") -> EmbeddingResponse:
        start = time.perf_counter()
        self.embedded.append(tuple(texts))
        self._maybe_fail()
        vectors = tuple(self._vector(t) for t in texts)
        usage = TokenUsage(prompt_tokens=sum(_approx_tokens(t) for t in texts))
        latency = time.perf_counter() - start
        self.call_log.add(
            CallRecord(
                provider=self.name,
                model=f"{self._model}-embed",
                kind="embedding",
                purpose=purpose,
                usage=usage,
                latency_s=latency,
                ok=True,
            )
        )
        return EmbeddingResponse(
            vectors=vectors,
            model=f"{self._model}-embed",
            provider=self.name,
            usage=usage,
            latency_s=latency,
        )

    def _vector(self, text: str) -> tuple[float, ...]:
        v = np.zeros(self._dim, dtype="float64")
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dim
            sign = 1.0 if digest[4] & 1 else -1.0
            v[bucket] += sign
        norm = float(np.linalg.norm(v))
        if norm == 0.0:  # no tokens: fall back to a seeded random direction
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
            v = np.random.default_rng(seed).normal(size=self._dim)
            norm = float(np.linalg.norm(v))
        return tuple(float(x) for x in v / norm)
