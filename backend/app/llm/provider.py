"""The provider protocol plus the shared machinery every provider uses.

* :class:`LLMProvider` is the only surface the rest of the codebase may depend on.
* :func:`parse_structured` turns model text into a validated Pydantic object or raises
  :class:`StructuredOutputError`; nothing downstream ever sees half-parsed JSON.
* :func:`with_retries` gives deterministic exponential back-off for transient failures.
* :class:`CallLog` collects :class:`CallRecord` s for latency / token dashboards.
"""

import json
import re
import time
from collections.abc import Callable, Sequence
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from app.core.errors import ProviderError, StructuredOutputError
from app.llm.schemas import CallRecord, EmbeddingResponse, LLMRequest, LLMResponse, TokenUsage

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")


class LLMProvider(Protocol):
    """What an LLM backend must offer. OpenAI and the test fake both satisfy this."""

    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Plain text completion."""
        ...

    def complete_structured(self, request: LLMRequest, schema: type[T]) -> tuple[T, LLMResponse]:
        """Completion constrained to ``schema``; returns the validated object and raw response."""
        ...

    def embed(self, texts: Sequence[str], *, purpose: str = "") -> EmbeddingResponse:
        """One vector per input text, in order."""
        ...


class CallLog:
    """In-memory record of every provider call in this process."""

    def __init__(self) -> None:
        self.records: list[CallRecord] = []

    def add(self, record: CallRecord) -> None:
        self.records.append(record)

    @property
    def usage(self) -> TokenUsage:
        total = TokenUsage()
        for r in self.records:
            total = total + r.usage
        return total

    @property
    def total_latency_s(self) -> float:
        return sum(r.latency_s for r in self.records)

    @property
    def failures(self) -> int:
        return sum(1 for r in self.records if not r.ok)

    def __len__(self) -> int:
        return len(self.records)


_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def parse_structured[T: BaseModel](text: str, schema: type[T]) -> T:
    """Validate model output against ``schema``. Tolerates a ```json fence, nothing else."""
    body = text
    m = _FENCE.match(text)
    if m:
        body = m.group(1)
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(
            f"model output is not JSON ({exc.msg} at pos {exc.pos}): {body[:200]!r}"
        ) from exc
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        raise StructuredOutputError(
            f"model output does not match {schema.__name__}: {exc.error_count()} error(s); "
            f"first: {exc.errors()[0]['msg']} at {list(exc.errors()[0]['loc'])}"
        ) from exc


def with_retries[R](
    fn: Callable[[], R],
    *,
    attempts: int,
    retry_on: tuple[type[BaseException], ...],
    base_delay_s: float = 0.5,
    max_delay_s: float = 8.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[R, int]:
    """Call ``fn`` up to ``attempts`` times on ``retry_on`` exceptions. Returns (result, attempts).

    Back-off is exponential and deterministic (no jitter) so tests can assert on it.
    Anything not in ``retry_on`` propagates immediately.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    last: BaseException | None = None
    for i in range(1, attempts + 1):
        try:
            return fn(), i
        except retry_on as exc:
            last = exc
            if i == attempts:
                break
            sleep(min(max_delay_s, base_delay_s * 2 ** (i - 1)))
    raise ProviderError(f"provider call failed after {attempts} attempt(s): {last!r}") from last
